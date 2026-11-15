"""Background obstacle watcher: camera -> depth -> obstacle check, always on the newest frame.

Runs in its own thread so a slow depth model never slows the caller down (teleop keeps
its 30 Hz loop). The caller just asks for the latest verdict and how old it is.

    with ObstacleWatcher(source, obstacle_cfg) as w:
        res, age, fps = w.latest()     # res is None until the first frame is done

`age` is the time since the analysed frame ARRIVED on the laptop. The camera's own
delay (encoding + Wi-Fi, ~0.2-0.5 s) comes on top. A verdict that is old says nothing
about now: treat it as "don't know", not as "clear".
"""

from __future__ import annotations

import threading
import time

from .camera import LatestFrameReader
from .depth import DepthEstimator
from .obstacles import ObstacleChecker, ObstacleConfig, ObstacleResult


class ObstacleWatcher:
    def __init__(self, source: str | None, cfg: ObstacleConfig, model: str = "small", device: str = "auto",
                 size: int | None = None, log=print, camera: LatestFrameReader | None = None):
        self.cfg = cfg
        self.log = log
        self.estimator = DepthEstimator(model, device, size)   # loads the model (a few seconds)
        self.checker = ObstacleChecker(cfg)
        # camera: share a reader that's already open (go-to's), instead of a second video stream.
        self._own_camera = camera is None
        self.camera = camera or LatestFrameReader(source)
        self._lock = threading.Lock()
        self._result: ObstacleResult | None = None
        self._frame = None         # the camera frame the result belongs to (for drawing it)
        self._stamp = 0.0          # arrival time of the frame the result belongs to
        self._fps = 0.0
        self.error: str | None = None
        self._ignore = None        # box (fractions) left out of the check: go-to's target, see ignore()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="obstacle-watcher", daemon=True)
        self._thread.start()

    def _run(self):
        last_seq, t_prev = 0, time.monotonic()
        try:
            while self._running:
                frame, stamp, seq = self.camera.get()
                if frame is None or seq == last_seq:
                    if self.camera.ended:
                        self.error = "camera stream ended"
                        return
                    time.sleep(0.005)
                    continue
                last_seq = seq
                res = self.checker.check(self.estimator.predict(frame), self._ignore)
                now = time.monotonic()
                with self._lock:
                    self._result, self._stamp, self._frame = res, stamp, frame
                    dt = now - t_prev
                    self._fps = 1.0 / dt if self._fps == 0.0 else 0.8 * self._fps + 0.2 / dt
                t_prev = now
        except Exception as e:  # never take the caller down; report instead
            self.error = f"obstacle watcher stopped: {e!r}"
            self.log(self.error)

    def latest(self) -> tuple[ObstacleResult | None, float, float]:
        """(newest verdict or None, its age in seconds, checks per second)."""
        with self._lock:
            age = time.monotonic() - self._stamp if self._result is not None else float("inf")
            return self._result, age, self._fps

    def ignore(self, box: tuple[float, float, float, float] | None):
        """Leave this box (left, top, right, bottom as fractions of the picture) out of the check,
        e.g. the trash piece go-to is driving to. None = check everything."""
        self._ignore = box

    def view(self):
        """(analysed frame, its verdict), both None until the first check: to draw what was decided."""
        with self._lock:
            return self._frame, self._result

    def close(self):
        self._running = False
        self._thread.join(timeout=5.0)
        if self._own_camera:
            self.camera.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
