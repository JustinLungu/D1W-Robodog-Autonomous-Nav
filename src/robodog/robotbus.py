"""Listen to the robot's own ROS 2 topics from the laptop, without ROS installed.

The robot's programs talk over ROS 2 with the Zenoh transport (rmw_zenoh). Its Zenoh
router listens on tcp/<robot>:7447, so the laptop can join as a Zenoh *client* and
receive topics directly. We only SUBSCRIBE: nothing is ever sent to the robot.

Robot side (found by inspection): Zenoh 1.6.2, rmw_zenoh_cpp 0.1.8, ROS_DOMAIN_ID 24.
A topic's data arrives on the Zenoh key "<domain>/<topic>/<type>/<hash>", e.g.
"24/perception/detection2d/vision_msgs::msg::dds_::Detection2DArray_/TypeHashNotSupported".
The payload is the ROS message in CDR encoding; decoders for the few messages we use are
below (CDR is simple enough that this beats depending on a ROS install).

Needs the robot-bus extra: `uv sync --extra robot-bus` (add your vision extra too, e.g.
`uv sync --extra vision-cpu --extra robot-bus`, or `uv sync` removes it again).
"""

from __future__ import annotations

import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

ROS_DOMAIN_ID = 24
ZENOH_PORT = 7447


# ----------------------------------------------------------------------------- CDR decoding
class CdrReader:
    """Reads little-endian CDR (XCDR1), the encoding ROS 2 uses on the wire."""

    def __init__(self, data: bytes):
        if len(data) < 4 or data[1] not in (1, 3):   # 4-byte encapsulation header; 1/3 = little endian
            raise ValueError(f"not little-endian CDR (header {data[:4]!r})")
        self.b = data[4:]   # alignment counts from after the header
        self.o = 0

    def _align(self, n: int):
        self.o += (-self.o) % n

    def _unpack(self, fmt: str, size: int):
        self._align(size)
        v = struct.unpack_from("<" + fmt, self.b, self.o)[0]
        self.o += size
        return v

    def u32(self) -> int:
        return self._unpack("I", 4)

    def i32(self) -> int:
        return self._unpack("i", 4)

    def f64(self) -> float:
        return self._unpack("d", 8)

    def string(self) -> str:
        n = self.u32()   # length including the trailing NUL
        v = self.b[self.o:self.o + max(0, n - 1)].decode("utf-8", errors="replace")
        self.o += n
        return v

    def header(self) -> tuple[float, str]:
        """std_msgs/Header -> (stamp in seconds, frame_id)."""
        sec, nsec = self.i32(), self.u32()
        return sec + nsec * 1e-9, self.string()


@dataclass
class Box:
    track_id: str            # the robot keeps this the same for one person across frames
    label: str               # class name; the vendor leaves it empty (their classes: person, car)
    score: float
    cx: float                # centre and size in pixels of the robot's 1920x1080 camera image
    cy: float
    w: float
    h: float


def decode_detection2d_array(data: bytes) -> tuple[float, list[Box]]:
    """vision_msgs/Detection2DArray (ROS 2 Humble) -> (stamp, boxes)."""
    r = CdrReader(data)
    stamp, _ = r.header()
    boxes = []
    for _ in range(r.u32()):
        r.header()
        hyps = []
        for _ in range(r.u32()):                      # ObjectHypothesisWithPose[]
            label, score = r.string(), r.f64()
            for _ in range(3 + 4 + 36):               # pose: position, orientation, covariance
                r.f64()
            hyps.append((label, score))
        cx, cy, _theta, w, h = r.f64(), r.f64(), r.f64(), r.f64(), r.f64()   # BoundingBox2D
        track_id = r.string()
        label, score = max(hyps, key=lambda x: x[1]) if hyps else ("", 0.0)
        boxes.append(Box(track_id, label, score, cx, cy, w, h))
    return stamp, boxes


# ----------------------------------------------------------------------------- listener
class TopicListener:
    """Subscribes to one ROS 2 topic on the robot and keeps the newest decoded message.

        with TopicListener(robot_ip, "/perception/detection2d", decode_detection2d_array) as t:
            msg, age, rate = t.latest()
    """

    RATE_WINDOW = 2.0   # seconds

    def __init__(self, robot_ip: str, topic: str, decode, domain: int = ROS_DOMAIN_ID, port: int = ZENOH_PORT):
        try:
            import zenoh
        except ImportError as e:
            raise SystemExit("listening to the robot needs the robot-bus extra:\n"
                             "  uv sync --extra robot-bus   (plus your vision extra, if you use one)") from e
        conf = zenoh.Config()
        conf.insert_json5("mode", '"client"')                                  # join the robot's router
        conf.insert_json5("connect/endpoints", f'["tcp/{robot_ip}:{port}"]')
        conf.insert_json5("scouting/multicast/enabled", "false")               # don't look for other peers
        self._decode = decode
        self._lock = threading.Lock()
        self._msg = None
        self._stamp = 0.0
        self._arrivals: deque[float] = deque()   # arrival times in the last RATE_WINDOW seconds
        self._count = 0
        self.errors = 0
        self.key = f"{domain}/{topic.strip('/')}/**"
        self._first = time.monotonic()
        self._session = zenoh.open(conf)
        self._sub = self._session.declare_subscriber(self.key, self._on_sample)

    def _on_sample(self, sample):
        now = time.monotonic()
        try:
            msg = self._decode(sample.payload.to_bytes())
        except Exception:
            self.errors += 1
            return
        with self._lock:
            self._msg, self._stamp = msg, now
            self._count += 1
            self._arrivals.append(now)

    def latest(self):
        """(newest message or None, seconds since it arrived, messages per second)."""
        now = time.monotonic()
        with self._lock:
            # Rate = arrivals in the last RATE_WINDOW seconds (an average over single gaps
            # would be thrown off by the burst of queued messages right after connecting).
            while self._arrivals and now - self._arrivals[0] > self.RATE_WINDOW:
                self._arrivals.popleft()
            span = min(self.RATE_WINDOW, now - self._first) if self._count else 0.0
            rate = len(self._arrivals) / span if span > 0.5 else 0.0
            age = now - self._stamp if self._msg is not None else float("inf")
            return self._msg, age, rate

    @property
    def count(self) -> int:
        return self._count

    def close(self):
        try:
            self._sub.undeclare()
        finally:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
