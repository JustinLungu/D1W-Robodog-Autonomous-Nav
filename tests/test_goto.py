"""Go-to-goal steering, driven against a simulated robot (fixed 1/30 s steps)."""

import math

import pytest

from robodog.goto import GoTo, GoToLimits, wrap
from robodog.safety import MIN_VX, MIN_YAW

DT = 1 / 30
LIM = GoToLimits(max_goal_m=3.0, max_vx=0.15, max_yaw=0.3, arrive_m=0.10, timeout_s=60.0)


def simulate(goto, pose=(0.0, 0.0, 0.0), seconds=60.0, holding=True):
    """Integrate the commanded speeds like the robot would. Returns final pose and all steps."""
    x, y, yaw = pose
    steps = []
    for _ in range(int(seconds / DT)):
        s = goto.step((x, y, yaw), holding, DT)
        steps.append(s)
        if s.state in ("arrived", "timeout", "lost position", "overshoot guard", "no goal"):
            break
        x += s.vx * math.cos(yaw) * DT
        y += s.vx * math.sin(yaw) * DT
        yaw = wrap(yaw + s.yaw * DT)
    return (x, y, yaw), steps


@pytest.mark.parametrize("gx,gy", [(1.5, 0.0), (1.0, 0.8), (1.2, -1.0), (0.3, 2.0), (-1.0, 0.5), (-2.0, -0.2)])
def test_reaches_goals_in_every_direction(gx, gy):
    g = GoTo(LIM)
    assert g.set_goal(gx, gy, (0.0, 0.0, 0.0)) is None
    (x, y, _), steps = simulate(g)
    assert steps[-1].state == "arrived"
    assert math.hypot(x - gx, y - gy) < 0.15


def test_goal_is_fixed_in_the_world_not_relative_to_the_robot():
    g = GoTo(LIM)
    start = (2.0, 1.0, math.pi / 2)  # robot somewhere else, facing +y
    g.set_goal(1.0, 0.0, start)      # 1 m ahead of it
    assert g.goal == pytest.approx((2.0, 2.0))
    (x, y, _), steps = simulate(g, start)
    assert steps[-1].state == "arrived" and math.hypot(x - 2.0, y - 2.0) < 0.15


def test_turns_on_the_spot_first_when_goal_is_to_the_side():
    g = GoTo(LIM)
    g.set_goal(0.0, 1.5, (0.0, 0.0, 0.0))  # 90 deg left
    s = g.step((0.0, 0.0, 0.0), True, DT)
    assert s.state == "turning" and s.vx == 0.0 and s.yaw > 0


def test_speeds_within_limits_and_never_below_sdk_minimum():
    g = GoTo(LIM)
    g.set_goal(2.0, -1.5, (0.0, 0.0, 0.0))
    _, steps = simulate(g)
    for s in steps:
        assert 0.0 <= s.vx <= LIM.max_vx
        assert abs(s.yaw) <= LIM.max_yaw + 1e-9
        assert s.vx == 0.0 or s.vx >= MIN_VX
        assert s.yaw == 0.0 or abs(s.yaw) >= MIN_YAW


def test_releasing_the_key_stops_and_keeps_the_goal():
    g = GoTo(LIM)
    g.set_goal(1.5, 0.0, (0.0, 0.0, 0.0))
    s = g.step((0.0, 0.0, 0.0), False, DT)
    assert (s.vx, s.yaw, s.state) == (0.0, 0.0, "paused") and g.goal is not None


def test_too_far_refused():
    g = GoTo(LIM)
    assert "too far" in g.set_goal(3.5, 0.0, (0.0, 0.0, 0.0))
    assert g.goal is None


def test_timeout_counts_only_driving_time():
    g = GoTo(GoToLimits(3.0, 0.15, 0.3, 0.10, timeout_s=1.0))
    g.set_goal(2.5, 0.0, (0.0, 0.0, 0.0))
    for _ in range(300):                       # 10 s paused: no timeout
        g.step((0.0, 0.0, 0.0), False, DT)
    assert g.state == "paused"
    _, steps = simulate(g, seconds=5)
    assert steps[-1].state == "timeout" and (steps[-1].vx, steps[-1].yaw) == (0.0, 0.0)


def test_position_jump_stops_it():
    g = GoTo(LIM)
    g.set_goal(2.0, 0.0, (0.0, 0.0, 0.0))
    g.step((0.0, 0.0, 0.0), True, DT)
    s = g.step((1.0, 0.0, 0.0), True, DT)      # 1 m in 1/30 s: impossible
    assert s.state == "lost position" and (s.vx, s.yaw) == (0.0, 0.0)


def test_stays_stopped_after_arriving():
    g = GoTo(LIM)
    g.set_goal(0.05, 0.0, (0.0, 0.0, 0.0))
    assert g.step((0.0, 0.0, 0.0), True, DT).state == "arrived"
    assert g.step((0.0, 0.0, 0.0), True, DT).vx == 0.0


def test_no_goal_means_no_motion():
    s = GoTo(LIM).step((0.0, 0.0, 0.0), True, DT)
    assert (s.vx, s.yaw, s.state) == (0.0, 0.0, "no goal")


def test_end_to_end_with_safedog_and_simulated_robot(monkeypatch):
    """Goal set from SafeDog.pose(), speeds sent through SafeDog, positions from the fake robot."""
    import robodog.safety as safety
    from fake_robot import FakeHighLevel
    from test_safety import Clock, pinned_config

    from robodog.safety import SafeDog

    monkeypatch.setattr(safety, "STAND_SETTLE_S", 0.0)
    cfg = pinned_config()
    cfg.safety.allow_motion = True
    robot = FakeHighLevel(simulate=True, sim_dt=DT)
    clock = Clock()
    dog = SafeDog(robot, cfg, allow_motion=True, clock=clock, log=lambda m: None)
    dog.stand()
    g = GoTo(LIM)
    g.set_goal(1.2, -0.6, dog.pose())             # ahead and to the RIGHT
    for _ in range(3000):
        s = g.step(dog.pose(), True, DT)
        if s.state == "arrived":
            break
        dog.set_velocity(s.vx, 0.0, s.yaw)
        dog.tick()                                 # sends move() -> fake robot integrates one step
        clock.advance(DT)
    assert s.state == "arrived"
    assert math.hypot(robot.x - 1.2, robot.y + 0.6) < 0.15


# ---------------------------------------------------------------- stopping in front of an object
STOP = GoToLimits(max_goal_m=3.0, max_vx=0.15, max_yaw=0.3, arrive_m=0.10, timeout_s=60.0, stop_before_m=0.55)


@pytest.mark.parametrize("gx,gy", [(2.0, 0.0), (1.5, 1.0), (1.2, -1.4), (-1.5, 0.5)])
def test_stops_short_facing_the_goal(gx, gy):
    g = GoTo(STOP)
    g.set_goal(gx, gy, (0.0, 0.0, 0.0))
    (x, y, yaw), steps = simulate(g)
    assert steps[-1].state == "arrived"
    dist = math.hypot(gx - x, gy - y)
    assert 0.45 <= dist <= 0.60                                   # middle ~0.55 m short: nose in front, not on top
    bearing = math.degrees(wrap(math.atan2(gy - y, gx - x) - yaw))
    assert abs(bearing) <= 12                                     # facing the object


def test_never_drives_into_the_stop_zone_much():
    g = GoTo(STOP)
    g.set_goal(2.0, 0.0, (0.0, 0.0, 0.0))
    (x, _, _), _ = simulate(g)
    assert x <= 2.0 - 0.45


def test_goal_already_closer_than_stop_distance_only_turns_to_face_it():
    g = GoTo(STOP)
    g.set_goal(0.1, 0.4, (0.0, 0.0, 0.0))                         # 0.41 m away, 76 deg left
    (x, y, yaw), steps = simulate(g)
    assert steps[-1].state == "arrived"
    assert all(s.vx == 0.0 for s in steps)                        # never drove forward
    assert abs(math.degrees(wrap(math.atan2(0.4, 0.1) - yaw))) <= 12


def test_zero_stop_before_keeps_old_behaviour():
    g = GoTo(LIM)                                                 # stop_before_m defaults to 0
    g.set_goal(1.5, 0.0, (0.0, 0.0, 0.0))
    (x, y, _), steps = simulate(g)
    assert steps[-1].state == "arrived" and math.hypot(1.5 - x, y) < 0.15


# ---------------------------------------------------------------- overshoot guard / commanded odometry
def simulate_frozen_position(goto, seconds=60.0):
    """Like simulate(), but the robot's position estimate never changes (heading still does).
    That's the failure we saw: getPosition not following the real movement."""
    x = y = yaw = 0.0
    steps = []
    for _ in range(int(seconds / DT)):
        s = goto.step((0.0, 0.0, yaw), True, DT)        # frozen x, y
        steps.append(s)
        if s.state in ("arrived", "timeout", "lost position", "overshoot guard"):
            break
        x += s.vx * math.cos(yaw) * DT
        y += s.vx * math.sin(yaw) * DT
        yaw = wrap(yaw + s.yaw * DT)
    return (x, y, yaw), steps


def test_guard_stops_a_robot_whose_position_estimate_is_frozen():
    g = GoTo(STOP)                                       # odometry "robot"
    g.set_goal(2.5, 0.0, (0.0, 0.0, 0.0))
    (x, _, _), steps = simulate_frozen_position(g)
    assert steps[-1].state == "overshoot guard"
    needed = 2.5 - 0.55
    assert x <= needed * 1.25 + 0.40 + 0.02              # drove at most the allowed distance
    assert g.commanded_m > 0 and g.robot_m == 0.0        # the log would show the mismatch


def test_commanded_odometry_arrives_even_with_a_frozen_estimate():
    lim = GoToLimits(3.0, 0.15, 0.3, 0.10, 60.0, stop_before_m=0.55, odometry="commanded")
    for gx, gy in [(2.5, 0.0), (2.0, 1.0), (1.8, -1.2)]:
        g = GoTo(lim)
        g.set_goal(gx, gy, (0.0, 0.0, 0.0))
        (x, y, yaw), steps = simulate_frozen_position(g)
        assert steps[-1].state == "arrived"
        assert 0.45 <= math.hypot(gx - x, gy - y) <= 0.62  # stopped short, as configured


def test_guard_never_trips_on_a_normal_run():
    for gx, gy in [(2.5, 0.0), (1.5, 1.5), (0.5, -2.5), (-2.0, 0.5)]:
        g = GoTo(STOP)
        g.set_goal(gx, gy, (0.0, 0.0, 0.0))
        _, steps = simulate(g)
        assert steps[-1].state == "arrived", (gx, gy, steps[-1].state)


def test_unknown_odometry_rejected():
    with pytest.raises(ValueError):
        GoTo(GoToLimits(3.0, 0.15, 0.3, 0.10, 20.0, odometry="gps"))
