"""Save a pygame window to an .mp4 while a program runs (robodog-goto --save).

The main loop calls `offer(screen)` after drawing. Frames are encoded by ffmpeg in a separate
thread; if encoding can't keep up, frames are DROPPED (never waited for), so recording can't slow
down the robot's control loop. The video plays at real speed: when the loop is slower than `fps`,
the last frame is repeated.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time


class ScreenRecorder:
    def __init__(self, path: str, size: tuple[int, int], fps: float = 15.0, log=print):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("--save needs ffmpeg: sudo apt install ffmpeg")
        self.path, self.fps, self.log = path, fps, log
        self.w, self.h = size[0] - size[0] % 2, size[1] - size[1] % 2   # x264 wants even sizes
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, start_new_session=True)   # Ctrl+C in the terminal: we still finish the file
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._t0: float | None = None
        self._written = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="screen-record", daemon=True)
        self._thread.start()

    def offer(self, surface, now: float | None = None):
        """Hand over the window after drawing; copies it only when a video frame is due."""
        now = time.monotonic() if now is None else now
        if self._t0 is None:
            self._t0 = now
        due = int((now - self._t0) * self.fps) + 1 - self._written
        if due <= 0:
            return
        import pygame as pg

        rgb = pg.surfarray.array3d(surface)[: self.w, : self.h].swapaxes(0, 1).tobytes()
        for _ in range(min(due, 10)):          # repeat to keep real time after a slow loop
            try:
                self._q.put_nowait(rgb)
            except queue.Full:
                self.dropped += 1
            self._written += 1

    def _run(self):
        while True:
            data = self._q.get()
            if data is None:
                break
            try:
                self._proc.stdin.write(data)
            except (BrokenPipeError, OSError) as e:
                self.log(f"recording stopped: {e}")
                break

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=10)
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait(timeout=30)
        secs = self._written / self.fps
        self.log(f"saved {self.path} ({secs:.0f} s" + (f", {self.dropped} frames dropped" if self.dropped else "") + ")")
