"""Control the robot the way the vendor's phone app / remote does: UDP JSON to `dog_task`.

Used only by robodog-app-teleop. It never touches the SDK: while an SDK program is connected
to the robot, `dog_task` ignores these commands (and connecting the SDK while crawling makes
the robot fold down), so never run this together with robodog-teleop / robodog-goto.

What the app does (found in the robot's logs, tested 2026-11-15, see README "Crawling"):
- Laptop -> robot UDP 8081:
    {"type":"heartbeat"}                                         registers us as an "app"
    {"type":"remote","button":[14 x 0/1],"joystick":[4 floats],...}   the sticks, ~20x a second.
        Commands only take effect while this stream runs.
    {"cmd":122,"type":"cmd"}  stand     {"cmd":106,...}  crawl     {"cmd":90,...}  dump (lie down)
- Robot -> laptop UDP 8080: state JSON ~10x a second ("feedback" control_mode, "odom_info"
  position / rpy / v_body / omega_world, "navigation_state" ...).

The stick -> speed scale and which stick axis is which are NOT documented; see AXES and
FULL_SCALE below and calibrate with the window's probe keys.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .app_link import APP_PORT, CMD_CRAWL, CMD_DUMP, CMD_STAND

STATE_PORT = 8080          # dog_task sends its state stream to the app's address, this port
STREAM_HZ = 20             # heartbeat + stick messages per second (the real app: ~10-20)

# Our speed (vx, vy, yaw) -> stick axes: (axis index in "joystick", sign). Tested on the robot
# (2026-11-16, +0.05 on each axis): axis 0 + = sideways RIGHT, axis 1 + = BACKWARD, axis 2 = turn
# (A/D turned the right way), axis 3 = nothing. So forward = -axis 1, left = -axis 0.
# Calibration (C key) measures signs + scales and overrides these (see CALIBRATION_FILE).
AXES = {"vx": (1, -1), "vy": (0, -1), "yaw": (2, -1)}

# Speed at full stick deflection (1.0), per axis. Unknown, so we assume the LARGEST values in
# the robot's dog_task speed config (x_speed_max 1.0, y_speed_max 3.0, yaw_speed_max 2.0):
# if the real scale is smaller, the robot moves SLOWER than asked, never faster.
# Replace with measured values (the window shows the robot's own measured speed).
FULL_SCALE = {"vx": 1.0, "vy": 3.0, "yaw": 2.0}

# Never deflect a stick further than this, whatever the speed asked (extra safety net).
MAX_STICK = 0.35

# Written by the C-key calibration in robodog-app-teleop, read at start: {"axes": ..., "full_scale": ...}
CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "app_teleop_calibration.json"


def load_calibration(path=CALIBRATION_FILE):
    """(axes, full_scale) from the calibration file, or the defaults above if there is none."""
    try:
        d = json.loads(Path(path).read_text())
        axes = {k: (int(v[0]), int(v[1])) for k, v in d["axes"].items()}
        scale = {k: float(v) for k, v in d["full_scale"].items()}
        if set(axes) == set(scale) == {"vx", "vy", "yaw"} and min(scale.values()) > 0:
            return axes, scale
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return dict(AXES), dict(FULL_SCALE)


def save_calibration(axes, full_scale, path=CALIBRATION_FILE, note=""):
    Path(path).write_text(json.dumps({"axes": {k: list(v) for k, v in axes.items()},
                                      "full_scale": full_scale, "note": note}, indent=2) + "\n")

STATE_STALE_S = 1.0        # no state from the robot for this long -> sticks to zero


@dataclass
class RobotState:
    """The latest state the robot reported (None = not received yet)."""
    control_mode: str | None = None        # "rl" standing/driving, "stand_down" crawling, "stand_up", "emergency", "init"
    z: float | None = None                 # body height, m
    x: float | None = None
    y: float | None = None
    yaw: float | None = None               # rad
    v_body: tuple[float, float, float] | None = None   # measured speed in the body frame, m/s
    yaw_rate: float | None = None          # rad/s
    battery: int | None = None
    t_rx: float = -math.inf                # monotonic time of the last state message

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.t_rx


def speeds_to_sticks(vx: float, vy: float, yaw: float, axes=AXES, full_scale=FULL_SCALE,
                     max_stick: float = MAX_STICK) -> list[float]:
    """(vx m/s, vy m/s, yaw rad/s) -> the 4 joystick values, clamped to +/-max_stick."""
    sticks = [0.0, 0.0, 0.0, 0.0]
    for name, v in (("vx", vx), ("vy", vy), ("yaw", yaw)):
        idx, sign = axes[name]
        s = sign * v / full_scale[name]
        sticks[idx] = max(-max_stick, min(max_stick, s))
    return sticks


def _parse_into(st: RobotState, m: dict, now: float):
    t = m.get("type")
    if t == "feedback":
        st.control_mode = m.get("control_mode", st.control_mode)
    elif t == "odom_info":
        pos, rpy = m.get("position"), m.get("rpy")
        if pos and len(pos) >= 3:
            st.x, st.y, st.z = float(pos[0]), float(pos[1]), float(pos[2])
        if rpy and len(rpy) >= 3:
            st.yaw = float(rpy[2])
        vb, om = m.get("v_body"), m.get("omega_world")
        if vb and len(vb) >= 3:
            st.v_body = (float(vb[0]), float(vb[1]), float(vb[2]))
        if om and len(om) >= 3:
            st.yaw_rate = float(om[2])
    elif t == "navigation_state":
        pos = (m.get("current_pose") or {}).get("position")
        if pos and len(pos) >= 3:
            st.z = float(pos[2])
    for k in ("battery", "battery_power", "bms_power", "power"):
        if isinstance(m.get(k), (int, float)):
            st.battery = int(m[k])
    st.t_rx = now


class AppRemote:
    """Streams heartbeat + sticks at STREAM_HZ, sends commands, and listens to the robot's state."""

    def __init__(self, robot_ip: str, laptop_ip: str, port: int = APP_PORT, state_port: int = STATE_PORT,
                 log=print):
        self.dst = (robot_ip, port)
        self.laptop_ip = laptop_ip
        self.state_port = state_port
        self.log = log
        self.state = RobotState()
        self._sticks = [0.0, 0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx.bind((laptop_ip, 0))
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx.bind((laptop_ip, state_port))
        self._rx.settimeout(0.2)
        self._seq = 0
        self._threads = [threading.Thread(target=self._send_loop, name="app-send", daemon=True),
                         threading.Thread(target=self._recv_loop, name="app-recv", daemon=True)]
        for t in self._threads:
            t.start()

    # --- what the teleop calls ---
    def set_sticks(self, sticks):
        with self._lock:
            self._sticks = [float(s) for s in sticks]

    def command(self, code: int, repeat: int = 3):
        msg = json.dumps({"cmd": int(code), "type": "cmd"}).encode()
        for _ in range(repeat):
            self._tx.sendto(msg, self.dst)
            time.sleep(0.005)

    def stand(self):
        self.command(CMD_STAND)

    def crawl(self):
        self.command(CMD_CRAWL)

    def dump(self):
        self.command(CMD_DUMP)

    def close(self):
        self.set_sticks([0, 0, 0, 0])
        time.sleep(0.2)                    # a few zero-stick messages go out first
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1)
        self._tx.close()
        self._rx.close()

    # --- threads ---
    def _send_loop(self):
        period = 1.0 / STREAM_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                sticks = list(self._sticks)
            self._seq += 1
            try:
                self._tx.sendto(b'{"type":"heartbeat"}', self.dst)
                self._tx.sendto(json.dumps({"button": [0] * 14, "joystick": sticks, "seq": self._seq,
                                            "time": int(time.time() * 1000), "type": "remote"}).encode(), self.dst)
            except OSError as e:
                self.log(f"app send failed: {e}")
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self._rx.recvfrom(65535)
                m = json.loads(data)
            except (socket.timeout, ValueError, OSError):
                continue
            if isinstance(m, dict):
                with self._lock:
                    _parse_into(self.state, m, time.monotonic())


class FakeAppRemote:
    """Stand-in for --fake and tests: no network. Commands change the posture, sticks 'drive'."""

    def __init__(self, axes=AXES, full_scale=FULL_SCALE, log=print):
        self.axes, self.full_scale, self.log = axes, full_scale, log
        self.state = RobotState(control_mode="init", z=0.14, x=0.0, y=0.0, yaw=0.0,
                                v_body=(0.0, 0.0, 0.0), yaw_rate=0.0, battery=80, t_rx=time.monotonic())
        self.sticks = [0.0, 0.0, 0.0, 0.0]
        self.commands: list[int] = []
        self.alive = True                  # False: pretend the robot stopped talking

    def set_sticks(self, sticks):
        self.sticks = [float(s) for s in sticks]

    def command(self, code: int, repeat: int = 3):
        self.commands.append(int(code))
        mode, z = {CMD_STAND: ("rl", 0.385), CMD_CRAWL: ("stand_down", 0.25), CMD_DUMP: ("emergency", 0.15)}.get(
            int(code), (self.state.control_mode, self.state.z))
        if int(code) == CMD_CRAWL and self.state.control_mode != "rl":
            return                         # like the robot: crawl only from standing
        self.state.control_mode, self.state.z = mode, z

    def stand(self):
        self.command(CMD_STAND)

    def crawl(self):
        self.command(CMD_CRAWL)

    def dump(self):
        self.command(CMD_DUMP)

    def tick(self, dt: float, now: float | None = None):
        """Integrate the commanded sticks into a fake measured speed + pose (now = state time)."""
        if not self.alive:
            return
        st = self.state
        moving = st.control_mode in ("rl", "stand_down")
        v = {}
        for name in ("vx", "vy", "yaw"):
            idx, sign = self.axes[name]
            v[name] = sign * self.sticks[idx] * self.full_scale[name] if moving else 0.0
        st.v_body, st.yaw_rate = (v["vx"], v["vy"], 0.0), v["yaw"]
        st.x += (v["vx"] * math.cos(st.yaw) - v["vy"] * math.sin(st.yaw)) * dt
        st.y += (v["vx"] * math.sin(st.yaw) + v["vy"] * math.cos(st.yaw)) * dt
        st.yaw += v["yaw"] * dt
        st.t_rx = time.monotonic() if now is None else now

    def close(self):
        self.sticks = [0.0, 0.0, 0.0, 0.0]
