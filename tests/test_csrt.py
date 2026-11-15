"""CSRT fallback for the selected trash piece (vision/csrt.py + TrashWatcher)."""

import numpy as np
import pytest

from robodog.vision.csrt import SelectedBoxTracker, iou, make_csrt
from robodog.vision.trash import Track, TrashWatcher

W, H = 320, 240
rng = np.random.default_rng(0)
BACKGROUND = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)
PATCH = rng.integers(120, 256, (40, 40, 3), dtype=np.uint8)   # a textured "object"


def frame_with_object_at(x: int, y: int) -> np.ndarray:
    f = BACKGROUND.copy()
    f[y:y + 40, x:x + 40] = PATCH
    return f


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCSRT:
    """Answers whatever the test puts in `answer`."""
    def __init__(self):
        self.answer = (True, (100, 100, 40, 40))

    def init(self, frame, xywh):
        return True

    def update(self, frame):
        return self.answer


def test_real_csrt_follows_a_moving_object():
    make_csrt()   # the project's OpenCV must have it (opencv-contrib-python)
    t = SelectedBoxTracker(max_s=10.0)
    t.seen(frame_with_object_at(100, 100), (100, 100, 140, 140))
    for i in range(1, 11):
        box = t.follow(frame_with_object_at(100 + 4 * i, 100 + 2 * i))
        assert box is not None
    assert box[0] == pytest.approx(140, abs=6) and box[1] == pytest.approx(120, abs=6)


def test_fallback_gives_up_after_max_s():
    clock, fake = Clock(), FakeCSRT()
    t = SelectedBoxTracker(max_s=2.0, make=lambda: fake, clock=clock)
    frame = np.zeros((H, W, 3), np.uint8)
    t.seen(frame, (100, 100, 140, 140))
    clock.t = 1.9
    assert t.follow(frame) is not None
    clock.t = 2.1
    assert t.follow(frame) is None and not t.active


def test_seen_again_restarts_the_clock():
    clock, fake = Clock(), FakeCSRT()
    t = SelectedBoxTracker(max_s=2.0, make=lambda: fake, clock=clock)
    frame = np.zeros((H, W, 3), np.uint8)
    t.seen(frame, (100, 100, 140, 140))
    clock.t = 1.5
    t.seen(frame, (100, 100, 140, 140))     # YOLO saw it again
    clock.t = 3.0
    assert t.follow(frame) is not None


@pytest.mark.parametrize("answer", [
    (False, (100, 100, 40, 40)),             # CSRT says it lost it
    (True, (100 + int(0.08 * W) + 5, 100, 40, 40)),   # jumped too far in one frame
    (True, (100, 100, 5, 5)),                # shrank to nothing
])
def test_untrustworthy_answers_are_dropped(answer):
    fake = FakeCSRT()
    t = SelectedBoxTracker(max_s=5.0, make=lambda: fake, clock=Clock())
    frame = np.zeros((H, W, 3), np.uint8)
    t.seen(frame, (100, 100, 140, 140))
    fake.answer = answer
    assert t.follow(frame) is None and not t.active


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


# --- in the watcher ----------------------------------------------------------------------

def watcher(fake):
    """A watcher whose thread idles (no frames); _fill_in_selected is driven by hand."""
    csrt = SelectedBoxTracker(max_s=5.0, make=lambda: fake, clock=Clock())
    return TrashWatcher(lambda: (None, 0.0, 0), tracker=None, csrt=csrt)


CUP = Track(7, "cup", 0.8, (100.0, 100.0, 140.0, 140.0))
OTHER = Track(3, "can", 0.9, (250.0, 50.0, 290.0, 90.0))
frame = np.zeros((H, W, 3), np.uint8)


def test_watcher_fills_in_only_the_selected_piece_when_yolo_loses_it():
    fake = FakeCSRT()
    w = watcher(fake)
    try:
        w.select(7)
        assert w._fill_in_selected(frame, [CUP, OTHER]) == [CUP, OTHER]   # YOLO sees it: untouched
        fake.answer = (True, (104, 100, 40, 40))
        out = w._fill_in_selected(frame, [OTHER])                          # YOLO lost the cup
        assert out[-1] == Track(7, "cup", 0.8, (104.0, 100.0, 144.0, 140.0), "csrt")
    finally:
        w.close()


def test_watcher_steps_aside_when_yolo_has_it_under_a_new_id():
    fake = FakeCSRT()
    w = watcher(fake)
    try:
        w.select(7)
        w._fill_in_selected(frame, [CUP])
        renamed = Track(12, "cup", 0.6, (102.0, 100.0, 142.0, 140.0))
        assert w._fill_in_selected(frame, [renamed]) == [renamed]           # no CSRT box on top of it
    finally:
        w.close()


def test_watcher_without_a_selection_adds_nothing():
    fake = FakeCSRT()
    w = watcher(fake)
    try:
        w._fill_in_selected(frame, [CUP])
        assert w._fill_in_selected(frame, []) == []
    finally:
        w.close()
