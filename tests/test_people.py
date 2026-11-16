"""Person check: placing people, the stop guard (person box over the red depth area), SafeDog's full stop."""

import time

import numpy as np
import pytest

from robodog.camera_model import CameraModel
from robodog.vision.people import Person, PersonGuard, PersonWatcher, place

CAM = CameraModel(1280, 720, 96.0, 0.40, -8.7, 0.40, "fisheye")


def feet_at(dist_m, y_left=0.0, height_px=200):
    """A person box whose feet stand dist_m ahead of the point under the camera."""
    u, v = CAM.floor_to_pixel(CAM.forward_offset_m + dist_m, y_left)
    return (u - 40, v - height_px, u + 40, v)


def test_place_measures_the_feet_from_under_the_camera():
    assert place(feet_at(2.5), 0.9, CAM).dist == pytest.approx(2.5, abs=0.05)
    assert place(feet_at(1.8, 0.4), 0.9, CAM).dist == pytest.approx(np.hypot(1.8, 0.4), abs=0.05)


def test_feet_cut_off_has_no_distance():
    assert place((500, 100, 700, 719), 0.9, CAM).dist is None


SIZE = (1280, 720)


class FakeWatcher:
    def __init__(self, people=None, age=0.1, error=None):
        self.people, self.age, self.error = people, age, error

    def latest(self):
        return self.people, self.age, SIZE


class FakeObstacles:
    """Obstacle check result: in the path = right half of the picture's lower part, standing on
    row `foot` (bigger = closer)."""
    def __init__(self, age=0.1, error=None, red=True, foot=0.95, status="warning"):
        in_path = np.zeros((72, 128), bool)
        if red:
            in_path[40:72, 64:128] = True
        self.res = type("R", (), {"status": status, "in_path": in_path,
                                  "foot": np.full((72, 128), foot, np.float32),
                                  "mask": in_path & (foot >= 0.9)})()   # objects' red at warn_row 0.9
        self.age, self.error = age, error

    def latest(self):
        return self.res, self.age, 30.0


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


IN_RED = Person((800, 300, 1000, 700), 0.9, 1.2)       # standing in the red area
BESIDE = Person((100, 300, 300, 700), 0.9, 1.2)         # same distance, but left: no red there
FAR = Person((900, 100, 950, 350), 0.9, 4.0)            # above the red area (far away)


def guard(people, obstacles=None, clock=None, warn_row=0.9):
    return PersonGuard(FakeWatcher(people), obstacles or FakeObstacles(), warn_row=warn_row, margin=0.02,
                       overlap=0.02, clear_s=1.5, clock=clock or Clock())


def test_person_in_the_red_area_stops_everything_then_waits_after_leaving():
    clock = Clock()
    g = guard([IN_RED], clock=clock)
    assert g.stop_reason() == "PERSON in the way"
    g.watcher.people = []                          # gone
    clock.t += 1.0
    assert "waiting" in g.stop_reason()            # still within clear_s
    clock.t += 0.6
    assert g.stop_reason() is None


def test_person_beside_the_path_or_far_away_does_not_stop():
    assert guard([BESIDE]).stop_reason() is None
    assert guard([FAR]).stop_reason() is None
    assert guard([IN_RED], FakeObstacles(red=False)).stop_reason() is None   # no red at all


def test_margin_catches_a_box_just_next_to_the_red_area():
    just_left = Person((560, 300, 630, 700), 0.9, 1.2)      # ends at x=0.49; red starts at 0.5
    assert guard([just_left]).stop_reason() == "PERSON in the way"


def test_people_have_their_own_warning_line():
    between = FakeObstacles(foot=0.87)          # beyond the objects' line (0.9), before the people's (0.86)
    assert guard([IN_RED], between, warn_row=0.86).stop_reason() == "PERSON in the way"
    assert guard([IN_RED], between, warn_row=0.9).stop_reason() is None
    assert between.res.mask.sum() == 0           # ...where a box would not be red yet


def test_no_floor_visible_counts_for_people_too():
    blocked = FakeObstacles(foot=0.0, status="blocked")
    blocked.res.mask = np.ones((72, 128), bool)  # "blocked": the covered floor is the mask
    assert guard([BESIDE], blocked).stop_reason() == "PERSON in the way"


@pytest.mark.parametrize("watcher,obstacles", [
    (FakeWatcher(None), FakeObstacles()),                       # no picture yet
    (FakeWatcher([], age=3.0), FakeObstacles()),                # old picture (person check)
    (FakeWatcher([]), FakeObstacles(age=3.0)),                  # old picture (obstacle check)
    (FakeWatcher([], error="person check stopped: boom"), FakeObstacles()),
    (FakeWatcher([]), FakeObstacles(error="obstacle watcher stopped: boom")),
])
def test_guard_stops_when_not_sure(watcher, obstacles):
    assert PersonGuard(watcher, obstacles, 0.9, 0.02, 0.02, 1.5, clock=Clock()).stop_reason()


def test_watcher_places_detections_and_reports_a_crash():
    class Det:
        def detect(self, frame):
            return [(feet_at(1.5), 0.8)]
    frame = np.zeros((720, 1280, 3), np.uint8)
    w = PersonWatcher(lambda: (frame, time.monotonic(), 1), Det(), lambda w_, h_: CAM)
    try:
        deadline = time.monotonic() + 2
        while w.latest()[0] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        people, age, size = w.latest()
        assert size == (1280, 720) and len(people) == 1 and people[0].dist == pytest.approx(1.5, abs=0.05)
    finally:
        w.close()

    class Broken:
        def detect(self, frame):
            raise RuntimeError("boom")
    w = PersonWatcher(lambda: (frame, time.monotonic(), 1), Broken(), lambda w_, h_: CAM, log=lambda m: None)
    try:
        deadline = time.monotonic() + 2
        while w.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "boom" in w.error
    finally:
        w.close()


# --- SafeDog: every speed 0 ---------------------------------------------------------------

def standing_dog():
    import test_safety as ts
    from robodog.safety import STAND_SETTLE_S
    dog, robot, clock = ts.make()
    dog.stand()
    clock.advance(STAND_SETTLE_S + 0.1)
    dog.tick()
    return dog, robot, ts.moves


def test_safedog_stops_every_direction_while_a_person_is_close():
    dog, robot, moves = standing_dog()
    reason = {"r": "PERSON in the way"}
    dog.stop_guard = lambda: reason["r"]
    for cmd in [(0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.3)]:
        dog.set_velocity(*cmd)
        dog.tick()
        assert moves(robot)[-1] == (0.0, 0.0, 0.0)    # forward, back, sideways, turning: all 0
    assert dog.stopped_for == "PERSON in the way" and dog.fault is None
    reason["r"] = None
    dog.set_velocity(-0.2, 0.0, 0.3)
    dog.tick()
    assert moves(robot)[-1] == (-0.2, 0.0, 0.3) and dog.stopped_for is None


def test_safedog_treats_a_crashing_person_guard_as_stop():
    dog, robot, moves = standing_dog()
    def broken():
        raise RuntimeError("boom")
    dog.stop_guard = broken
    dog.set_velocity(0.0, 0.0, 0.3)
    dog.tick()
    assert moves(robot)[-1] == (0.0, 0.0, 0.0) and "boom" in dog.stopped_for
