"""People: when a person is in the robot's way, ALL motion stops until they've gone.

    camera frame -> small standard YOLO (knows "person") -> person boxes
    obstacle check (depth)  -> per pixel: in the robot's path? and how far (the row it stands on)
    PersonGuard: a person box (grown by people.margin) overlapping the path closer than
                 people.warn_row -> stop everything, until that hasn't been the case for people.clear_s

people.warn_row works like obstacles.warn_row (same scale), but is its own value: people usually get
a bigger margin than boxes. And the reaction is a full stop instead of only blocking forward: the
person may step beside or behind the robot, where the camera can't see. So no turning, backing or
sideways either; standing up / lying down still work.

- When it isn't sure (no picture yet, a picture older than STALE_S, either check crashed): stop too.
- The distance shown next to each box (feet = box bottom-centre on the floor) is for display only.

The model is Ultralytics' small standard YOLO (trained on COCO, which has "person"). The first run
downloads it (~5 MB, from Ultralytics' GitHub releases) to `people.model`; after that it's offline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT

STALE_S = 1.5   # a verdict about a picture older than this says nothing about now: stop


@dataclass
class Person:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in frame pixels
    conf: float
    dist: float | None   # metres on the floor from under the camera (display); None = feet cut off (very
                         # close); inf = feet at/above the horizon (far, or the floor maths can't place them)


def place(box, conf: float, model) -> Person:
    """A person box -> a Person with the distance of its feet (bottom-centre) from under the camera."""
    x0, y0, x1, y1 = box
    if y1 >= model.height * 0.99:      # feet cut off by the picture's bottom: no floor point
        return Person(tuple(box), conf, None)
    floor = model.pixel_to_floor((x0 + x1) / 2, y1)
    if floor is None:                  # "feet" at/above the horizon: far away (can't place them)
        return Person(tuple(box), conf, float("inf"))
    return Person(tuple(box), conf, float(np.hypot(floor[0] - model.forward_offset_m, floor[1])))


class PersonDetector:
    """Ultralytics YOLO, only the "person" class."""

    def __init__(self, model: str, conf: float, imgsz: int, device: str = "auto"):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit("the person check needs the vision extra: uv sync --extra vision-gpu (or vision-cpu)") from e
        path = Path(model)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading the person model to {path} (first run only, needs internet)...")
        self.model = YOLO(str(path))   # downloads Ultralytics' official file if it's missing
        names = {str(n).lower(): i for i, n in dict(self.model.names).items()}
        if "person" not in names:
            raise SystemExit(f"people.model {model} doesn't know 'person' (it knows {sorted(names)[:10]}...)")
        self.person_id = names["person"]
        self.conf, self.imgsz = conf, imgsz
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def detect(self, bgr: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, classes=[self.person_id],
                               device=self.device, verbose=False)[0]
        if r.boxes is None:
            return []
        return [(tuple(float(v) for v in b), float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


class PersonWatcher:
    """Runs the detector in its own thread on the newest camera frame.

    get_frame() returns (frame or None, arrival time (time.monotonic), sequence number), like
    LatestFrameReader.get. make_model(width, height) gives the camera model for that frame size.
    """

    def __init__(self, get_frame, detector, make_model, log=print):
        self.get_frame = get_frame
        self.detector = detector
        self.make_model = make_model
        self.log = log
        self._lock = threading.Lock()
        self._people: list[Person] | None = None   # None until the first frame is done
        self._stamp = 0.0
        self._size = (0, 0)                        # (width, height) of the analysed frame
        self.error: str | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="person-watcher", daemon=True)
        self._thread.start()

    def _run(self):
        last_seq, cam = None, None
        try:
            while self._running:
                frame, stamp, seq = self.get_frame()
                if frame is None or seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                if cam is None or (cam.width, cam.height) != (frame.shape[1], frame.shape[0]):
                    cam = self.make_model(frame.shape[1], frame.shape[0])
                people = [place(b, c, cam) for b, c in self.detector.detect(frame)]
                with self._lock:
                    self._people, self._stamp, self._size = people, stamp, (frame.shape[1], frame.shape[0])
        except Exception as e:   # never take the caller down; report instead (the guard then stops)
            self.error = f"person check stopped: {e!r}"
            self.log(self.error)

    def latest(self) -> tuple[list[Person] | None, float, tuple[int, int]]:
        """(people in the newest analysed frame or None before the first, that frame's age in s,
        its (width, height))."""
        with self._lock:
            age = time.monotonic() - self._stamp if self._people is not None else float("inf")
            return (None if self._people is None else list(self._people)), age, self._size

    def close(self):
        self._running = False
        self._thread.join(timeout=5.0)


class PersonGuard:
    """Decides "stop everything": a person in the obstacle check's red area. Give `guard.stop_reason`
    to SafeDog (dog.stop_guard): every control tick SafeDog asks it, and while it returns a reason,
    every speed is 0, whoever asked for it. Thread-safe: SafeDog's loop and the window both call it.

    obstacles: the ObstacleWatcher. A person is in the way when their FEET (box bottom) are on or
    before `warn_row` (or cut off by the picture's bottom: very close), and their box, grown by
    `margin` (fraction of the picture, for the two checks seeing frames a few hundredths of a second
    apart), has at least `overlap` of its area "red for people": in the robot's path, standing on or
    before `warn_row` (when no floor is visible at all: the covered floor).
    """

    def __init__(self, watcher, obstacles, warn_row: float, margin: float, overlap: float, clear_s: float,
                 clock=time.monotonic):
        self.watcher, self.obstacles = watcher, obstacles
        self.warn_row, self.margin, self.overlap, self.clear_s = warn_row, margin, overlap, clear_s
        self.clock = clock
        self._last_in_way = -float("inf")
        self._lock = threading.Lock()

    def in_the_way(self, p: Person, size: tuple[int, int], mask: np.ndarray) -> bool:
        """Is this person close (their FEET on or before warn_row, or cut off by the picture's bottom)
        and does their box (+ margin) cover enough of the path? Feet first: a person standing BEHIND
        a box has their box overlapping the box's close pixels, but their feet are far away (seen on
        the robot: "PERSON in the way" for someone ~4 m off, behind a cardboard box)."""
        w, h = size
        feet = p.box[3] / h
        if feet < 0.99 and feet < self.warn_row - self.margin:
            return False
        mh, mw = mask.shape
        x0, y0, x1, y1 = p.box
        c0 = int(max(0.0, x0 / w - self.margin) * mw)
        c1 = int(min(1.0, x1 / w + self.margin) * mw) + 1
        r0 = int(max(0.0, y0 / h - self.margin) * mh)
        r1 = int(min(1.0, y1 / h + self.margin) * mh) + 1
        cells = mask[r0:r1, c0:c1]
        return cells.size > 0 and cells.mean() >= self.overlap

    def check(self) -> tuple[str | None, list[Person]]:
        """(why everything must stop, or None; the people seen in the newest frame)."""
        with self._lock:
            people, age, size = self.watcher.latest()
            res, oage, _ = self.obstacles.latest()
            now = self.clock()
            for err in (self.watcher.error, self.obstacles.error):
                if err:
                    return err, []
            if people is None or res is None:
                return "person check: waiting for the first picture", people or []
            if max(age, oage) > STALE_S:
                return f"person check: no fresh picture ({max(age, oage):.1f} s old)", people
            red = self.red(res)
            if red is not None and any(self.in_the_way(p, size, red) for p in people):
                self._last_in_way = now
                return "PERSON in the way", people
            if now - self._last_in_way <= self.clear_s:
                return f"person just left: waiting {self.clear_s - (now - self._last_in_way):.1f} s", people
            return None, people

    def red(self, res) -> np.ndarray | None:
        """The pixels that count as "in the way" for people: in the path, on or before warn_row."""
        if res.status == "blocked" or res.in_path is None or res.foot is None:
            return res.mask     # no floor visible: the covered floor counts
        return res.in_path & (res.foot >= self.warn_row)

    def stop_reason(self) -> str | None:
        return self.check()[0]
