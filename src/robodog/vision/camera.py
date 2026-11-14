"""Camera reader that always hands out the NEWEST frame.

Why a thread: cv2.VideoCapture queues frames. If processing is slower than the camera
(depth on a CPU easily is), reading straight from it gives frames that are older and
older. Here a background thread reads continuously and keeps only the latest one, so
whatever we process is as fresh as the stream allows.

Each frame comes with the time it was received, so callers can see how stale it is.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

# Low-latency RTSP: TCP (no broken frames from lost UDP packets), no input buffering.
# Must be set before the first VideoCapture is created. setdefault: you can override it.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)


class LatestFrameReader:
    """Read a camera/stream/file in the background; get() returns the newest frame.

    source: an RTSP/HTTP URL, a video file path, or a webcam index like "0".
    """

    def __init__(self, source: str):
        self.source = source
        cap_source = int(source) if source.isdigit() else source
        self._cap = cv2.VideoCapture(cap_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honoured by some backends, harmless elsewhere

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0      # time.monotonic() when the newest frame arrived
        self._seq = 0          # increments with every new frame
        self._running = True
        self.ended = False     # True once the stream stops delivering (file end, camera gone)
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.ended = True
                break
            with self._lock:
                self._frame, self._stamp, self._seq = frame, time.monotonic(), self._seq + 1

    def get(self) -> tuple[np.ndarray | None, float, int]:
        """(newest frame or None, monotonic time it arrived, sequence number)."""
        with self._lock:
            return self._frame, self._stamp, self._seq

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
