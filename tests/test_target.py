"""Locking a go-to goal onto a tracked object: pure logic, no camera, detector or robot."""

import math

import pytest

from robodog.goto import GoTo, GoToLimits, wrap
from robodog.target import PoseHistory, Sighting, TargetFollower

DT = 1 / 30
LIM = GoToLimits(max_goal_m=3.0, max_vx=0.15, max_yaw=0.3, arrive_m=0.10, timeout_s=120.0)


def see(track_id, world_xy, pose, label="cup", conf=0.6, clipped=False):
    """What a perfect detector would report for an object at world_xy, seen from pose."""
    x, y, yaw = pose
    dx, dy = world_xy[0] - x, world_xy[1] - y
    floor = (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))
    return Sighting(track_id, label, conf, floor, clipped)


def make():
    g = GoTo(LIM)
    return g, TargetFollower(g, lost_after_s=1.0, reacquire_m=0.4, blend=1.0)


# --- refine_goal ---------------------------------------------------------------------

def test_refine_goal_moves_the_goal_without_restarting_the_drive():
    g = GoTo(LIM)
    g.set_goal(2.0, 0.0, (0.0, 0.0, 0.0))
    for _ in range(60):
        s = g.step((0.0, 0.0, 0.0), True, DT)
    driven, drive_time = g.commanded_m, g._drive_time
    g.refine_goal(1.5, 0.3, (0.0, 0.0, 0.0))
    assert g.goal == pytest.approx((1.5, 0.3))
    assert g.commanded_m == driven and g._drive_time == drive_time and g.state == s.state
    # the overshoot guard now counts from here: driven so far + what is still needed
    assert g.needed_m == pytest.approx(driven + math.hypot(1.5, 0.3) - 0.10)


def test_refine_goal_blends_and_ignores_finished_drives():
    g = GoTo(LIM)
    g.set_goal(2.0, 0.0, (0.0, 0.0, 0.0))
    g.refine_goal(1.0, 0.0, (0.0, 0.0, 0.0), blend=0.5)
    assert g.goal == pytest.approx((1.5, 0.0))
    g.state = "arrived"
    g.refine_goal(0.2, 0.0, (0.0, 0.0, 0.0))
    assert g.goal == pytest.approx((1.5, 0.0))
    g.clear()
    g.refine_goal(1.0, 0.0, (0.0, 0.0, 0.0))   # no goal: nothing happens
    assert g.goal is None


# --- selecting ------------------------------------------------------------------------

def test_select_sets_the_goal_on_the_object():
    g, f = make()
    pose = (1.0, 2.0, math.pi / 2)
    assert f.select(see(7, (1.0, 3.5), pose), pose, now=0.0) is None
    assert g.goal == pytest.approx((1.0, 3.5)) and f.state == "tracking" and f.track_id == 7


def test_select_refuses_cut_off_or_floorless_objects_and_too_far_ones():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    assert "cut off" in f.select(see(1, (1.0, 0.0), pose, clipped=True), pose, 0.0)
    assert f.select(Sighting(2, "cup", 0.5, None), pose, 0.0) is not None
    assert "too far" in f.select(see(3, (9.0, 0.0), pose), pose, 0.0)
    assert g.goal is None and f.track_id is None


# --- following ------------------------------------------------------------------------

def test_tracking_corrects_a_wrong_first_measurement():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    real = (2.0, 0.4)
    f.select(Sighting(5, "cup", 0.6, (2.4, 0.1)), pose, 0.0)      # first box: 40 cm too far, 30 cm off
    f.update([see(5, real, pose)], pose, now=0.1)
    assert g.goal == pytest.approx(real) and f.updates == 1


def test_goal_stays_put_when_the_object_leaves_the_view_then_goes_blind():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), pose), pose, 0.0)
    assert f.update([], pose, now=0.5) == "tracking"            # a short gap: still tracking
    assert f.update([], pose, now=1.6) == "blind"               # gone for more than lost_after_s
    assert g.goal == pytest.approx((2.0, 0.0))                  # the goal is kept for the blind finish
    assert f.update([see(5, (2.1, 0.0), pose)], pose, now=2.0) == "tracking"   # seen again: resumes
    assert g.goal == pytest.approx((2.1, 0.0))


def test_tick_goes_blind_when_the_detector_falls_silent():
    g, f = make()
    f.select(see(5, (2.0, 0.0), (0, 0, 0)), (0, 0, 0), 0.0)
    assert f.tick(0.5) == "tracking" and f.tick(1.5) == "blind"


def test_cut_off_sightings_are_not_used():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), pose), pose, 0.0)
    f.update([see(5, (3.0, 0.5), pose, clipped=True)], pose, now=0.2)   # half out of the picture: wrong foot point
    assert g.goal == pytest.approx((2.0, 0.0)) and f.updates == 0


def test_reacquires_when_the_tracker_renames_the_object():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), pose), pose, 0.0)
    f.update([see(9, (2.1, 0.05), pose), see(11, (2.0, 1.5), pose)], pose, now=0.2)
    assert f.track_id == 9 and f.reacquired == 1 and f.state == "tracking"
    assert g.goal == pytest.approx((2.1, 0.05))


def test_does_not_jump_to_a_different_object_far_from_the_goal():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), pose), pose, 0.0)
    f.update([see(9, (2.0, 0.8), pose)], pose, now=0.2)        # 80 cm away: another piece
    assert f.track_id == 5 and g.goal == pytest.approx((2.0, 0.0))


def test_reacquire_prefers_the_same_kind_of_object():
    g, f = make()
    pose = (0.0, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), pose, label="cup"), pose, 0.0)
    f.update([see(8, (2.05, 0.0), pose, label="can"), see(9, (2.2, 0.0), pose, label="cup")], pose, now=0.2)
    assert f.track_id == 9


def test_clearing_the_goal_drops_the_target():
    g, f = make()
    f.select(see(5, (2.0, 0.0), (0, 0, 0)), (0, 0, 0), 0.0)
    g.clear()
    assert f.update([see(5, (2.0, 0.0), (0, 0, 0))], (0, 0, 0), 0.1) == "idle" and f.track_id is None


def test_old_frames_are_placed_with_the_pose_the_robot_had_then():
    # The robot has driven 30 cm since the frame was taken. Placed with the old pose the
    # goal stays on the object; placed with the current pose it would land 30 cm too far.
    g, f = make()
    then, now_pose = (0.0, 0.0, 0.0), (0.3, 0.0, 0.0)
    f.select(see(5, (2.0, 0.0), then), then, 0.0)
    f.update([see(5, (2.0, 0.0), then)], then, now=0.3)
    assert g.goal == pytest.approx((2.0, 0.0))
    assert g.relative(now_pose)[0] == pytest.approx(1.7)


def test_pose_history_returns_the_pose_closest_in_time():
    h = PoseHistory(keep_s=1.0)
    assert h.at(0.0) is None
    for i in range(10):
        h.add(i * 0.1, (float(i), 0.0, 0.0))
    assert h.at(0.42)[0] == 4.0 and h.at(99.0)[0] == 9.0
    h.add(5.0, (50.0, 0.0, 0.0))                                # older entries are dropped
    assert h.at(0.0)[0] == 50.0


# --- the whole approach, simulated -----------------------------------------------------

def test_drives_to_the_real_object_despite_drifting_odometry_and_a_bad_first_click():
    """Odometry under-reports movement by 15 %. Plain go-to ends short/long of the object;
    with tracking the goal is corrected until the object leaves the view at 0.9 m."""
    def approach(track: bool) -> float:
        g = GoTo(LIM)
        f = TargetFollower(g, lost_after_s=1.0, reacquire_m=0.4, blend=0.5)
        real_obj = (2.5, 0.3)
        true = [0.0, 0.0, 0.0]            # where the robot really is
        odo = [0.0, 0.0, 0.0]             # what its odometry says
        first = Sighting(1, "cup", 0.6, (2.8, 0.1))            # first box is 30 cm off
        f.select(first, tuple(odo), 0.0)
        t = 0.0
        for i in range(int(120 / DT)):
            t += DT
            if track and i % 6 == 0:                            # detector at 5 Hz
                dx, dy = real_obj[0] - true[0], real_obj[1] - true[1]
                rel = (dx * math.cos(true[2]) + dy * math.sin(true[2]), -dx * math.sin(true[2]) + dy * math.cos(true[2]))
                visible = rel[0] > 0.9 and abs(math.atan2(rel[1], rel[0])) < math.radians(45)
                f.update([Sighting(1, "cup", 0.6, rel)] if visible else [], tuple(odo), t)
            s = g.step(tuple(odo), True, DT)
            if s.state in ("arrived", "timeout", "overshoot guard", "lost position"):
                break
            true[0] += s.vx * math.cos(true[2]) * DT; true[1] += s.vx * math.sin(true[2]) * DT
            true[2] = wrap(true[2] + s.yaw * DT)
            odo[0] += 0.85 * s.vx * math.cos(odo[2]) * DT; odo[1] += 0.85 * s.vx * math.sin(odo[2]) * DT
            odo[2] = wrap(odo[2] + s.yaw * DT)
        return math.hypot(real_obj[0] - true[0], real_obj[1] - true[1])

    blind, tracked = approach(False), approach(True)
    assert tracked < 0.30, tracked          # ends within arrive distance + the blind last stretch's drift
    assert tracked < blind - 0.15, (tracked, blind)
