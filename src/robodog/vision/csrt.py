"""Keep hold of the SELECTED trash piece when YOLO loses it, with OpenCV's CSRT tracker.

YOLO + ByteTrack sometimes drops a piece for a few frames (blur, odd angle, low confidence).
CSRT is a classic appearance tracker: give it a box once and it follows those pixels from
frame to frame, with no idea what the object is. So:

- Every frame where YOLO sees the selected piece: (re)start CSRT on YOLO's box. Its picture
  of the object stays fresh, and YOLO stays in charge whenever it has an answer.
- A frame where YOLO doesn't see it: ask CSRT where those pixels went, and report that box
  under the same track ID, so go-to keeps correcting the goal.
- CSRT drifts off objects over time, so its answers are dropped when: it has been on its own
  for longer than max_s, it says it lost the object, the box jumps too far in one frame, or
  the box gets too small. Then the piece is simply "not seen" (go-to finishes on odometry).
- If YOLO has a box on top of CSRT's (ByteTrack renamed the piece), that box wins: CSRT
  steps aside and the follower adopts the new ID.

Only the one selected piece is tracked this way (CSRT costs ~10-30 ms per box per frame).
Needs opencv-contrib-python (the project's OpenCV).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

JUMP_FRAC = 0.08     # max movement of the box centre per frame, as a fraction of the picture width
MIN_AREA_PX = 100    # boxes smaller than this (10 x 10 px) aren't trusted
OVERLAP_IOU = 0.3    # a YOLO box overlapping CSRT's this much is the same object

Box = tuple[float, float, float, float]   # x0, y0, x1, y1 in frame pixels


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _to_xywh(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def make_csrt():
    """A new CSRT tracker (its factory moved between OpenCV versions)."""
    for factory in (getattr(cv2, "TrackerCSRT_create", None), getattr(getattr(cv2, "TrackerCSRT", None), "create", None),
                    getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)):
        if factory is not None:
            return factory()
    raise RuntimeError("OpenCV has no CSRT tracker: install opencv-contrib-python (`uv sync`)")


class SelectedBoxTracker:
    """CSRT on one box: started from YOLO's box, used only while YOLO doesn't see the piece."""

    def __init__(self, max_s: float = 2.0, make=make_csrt, clock=time.monotonic):
        self.max_s = max_s
        self._make = make
        self._clock = clock
        self._csrt = None
        self._box: Box | None = None
        self._since = 0.0          # when YOLO last saw it (CSRT on its own since then)

    @property
    def active(self) -> bool:
        return self._csrt is not None

    def reset(self):
        self._csrt, self._box = None, None

    def seen(self, frame: np.ndarray, box: Box):
        """YOLO sees the piece in this frame: restart CSRT on its box."""
        if (box[2] - box[0]) * (box[3] - box[1]) < MIN_AREA_PX:
            self.reset()
            return
        h, w = frame.shape[:2]
        csrt = self._make()
        if csrt.init(frame, _to_xywh(box, w, h)) is False:
            self.reset()
            return
        self._csrt, self._box, self._since = csrt, box, self._clock()

    def follow(self, frame: np.ndarray) -> Box | None:
        """YOLO doesn't see it in this frame: CSRT's box, or None if CSRT can't be trusted (any more)."""
        if self._csrt is None:
            return None
        if self._clock() - self._since > self.max_s:
            self.reset()
            return None
        ok, (x, y, bw, bh) = self._csrt.update(frame)
        box = (float(x), float(y), float(x + bw), float(y + bh))
        h, w = frame.shape[:2]
        old = self._box
        jump = max(abs((box[0] + box[2]) - (old[0] + old[2])), abs((box[1] + box[3]) - (old[1] + old[3]))) / 2
        if not ok or bw * bh < MIN_AREA_PX or jump > JUMP_FRAC * w:
            self.reset()
            return None
        self._box = box
        return box
