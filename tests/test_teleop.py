"""Keyboard teleop logic against the fake robot (no window, no pygame)."""

import pytest
from fake_robot import FakeHighLevel
from test_safety import pinned_config

import robodog.safety as safety
from robodog.safety import MIN_VY, SafeDog
from robodog.teleop import SPEED_LEVELS, Teleop


@pytest.fixture(autouse=True)
def no_stand_wait(monkeypatch):
    """These tests are about driving, not the stand-up pause (tested separately)."""
    monkeypatch.setattr(safety, "STAND_SETTLE_S", 0.0)


def make(battery=80, allow_motion=True):
    cfg = pinned_config()
    cfg.safety.allow_motion = allow_motion
    robot = FakeHighLevel(battery=battery)
    dog = SafeDog(robot, cfg, allow_motion=True, log=lambda m: None)
    return Teleop(dog, log=lambda m: None), dog, robot


def drive(tele, dog, keys, seconds=1.0, dt=1 / 30):
    """Hold `keys` for `seconds`, ticking SafeDog like the real loop would."""
    for _ in range(int(seconds / dt)):
        tele.set_held(keys)
        tele.update(dt)
        dog.tick()


def test_motion_keys_do_nothing_before_standing():
    tele, dog, robot = make()
    drive(tele, dog, {"forward"}, 0.3)
    assert robot.calls_named("move") == []
    assert not tele.standing


def test_hold_moves_release_stops_immediately():
    tele, dog, robot = make()
    tele.action("stand")
    assert tele.standing
    drive(tele, dog, {"forward"}, 1.0)
    vx = robot.calls_named("move")[-1][1]
    assert vx > 0
    drive(tele, dog, set(), 1 / 30)  # one frame after release
    assert robot.calls_named("move")[-1][1:] == (0.0, 0.0, 0.0)


def test_speed_ramps_up_and_respects_level():
    tele, dog, robot = make()
    tele.action("stand")
    drive(tele, dog, {"forward"}, 0.1)
    early = tele.state().cmd[0]
    drive(tele, dog, {"forward"}, 2.0)
    full = tele.state().cmd[0]
    assert 0 < early < full
    assert abs(full - dog.s.max_vx * SPEED_LEVELS[1]) < 1e-9


def test_opposite_keys_cancel_and_reverse_goes_through_zero():
    tele, dog, _ = make()
    tele.action("stand")
    drive(tele, dog, {"forward", "back"}, 0.5)
    assert tele.state().cmd == (0.0, 0.0, 0.0)
    drive(tele, dog, {"forward"}, 2.0)
    tele.set_held({"back"})
    tele.update(1 / 30)
    assert -0.1 < tele.state().cmd[0] < 0  # restarted from 0, not a jump to full reverse


def test_slow_level_never_below_sdk_minimum():
    tele, dog, _ = make()
    tele.action("stand")
    drive(tele, dog, {"strafe_left"}, 2.0)
    assert tele.state().cmd[1] >= MIN_VY  # otherwise SafeDog would snap it to 0


def test_turn_and_strafe_signs():
    tele, dog, _ = make()
    tele.action("stand")
    drive(tele, dog, {"turn_right", "strafe_left"}, 1.0)
    _, vy, yaw = tele.state().cmd
    assert vy > 0 and yaw < 0


def test_release_all_stops():
    tele, dog, _ = make()
    tele.action("stand")
    drive(tele, dog, {"forward"}, 1.0)
    tele.release_all()
    tele.update(1 / 30)
    assert tele.state().cmd == (0.0, 0.0, 0.0)


def test_lie_down_then_stand_again():
    tele, dog, robot = make()
    tele.action("stand")
    tele.action("lie_down")
    assert not tele.standing and robot.calls_named("lieDown")
    tele.action("stand")
    assert tele.standing and len(robot.calls_named("standUp")) == 2


def test_estop_latches():
    tele, dog, robot = make()
    tele.action("stand")
    tele.action("estop")
    assert robot.calls_named("passive")
    tele.action("stand")  # refused, shown as a message, no exception
    assert not tele.standing and "refused" in tele.message
    drive(tele, dog, {"forward"}, 0.3)
    assert all(m[1:] == (0.0, 0.0, 0.0) for m in robot.calls_named("move"))


def test_low_battery_refused_with_message():
    tele, _, robot = make(battery=5)
    tele.action("stand")
    assert not tele.standing and "battery" in tele.message
    assert robot.calls_named("standUp") == []


def test_speed_level_changes():
    tele, *_ = make()
    tele.action("speed_3")
    assert tele.level == 3


def test_keys_held_during_stand_up_do_nothing_then_ramp_from_zero(monkeypatch):
    monkeypatch.setattr(safety, "STAND_SETTLE_S", 0.3)
    tele, dog, robot = make()
    tele.action("stand")
    assert "standing up" in tele.message
    drive(tele, dog, {"forward"}, 0.2)               # holding W during the stand-up
    assert all(m[1:] == (0.0, 0.0, 0.0) for m in robot.calls_named("move"))
    assert tele.standing and tele.state().cmd == (0.0, 0.0, 0.0)
    import time; time.sleep(0.35)                    # transition (0.3 s, real clock) over
    drive(tele, dog, {"forward"}, 1 / 30)            # one frame
    first = tele.state().cmd[0]
    assert 0 < first < 0.1                           # starts ramping from 0, no lurch
    assert "hold W/A/S/D" in tele.message
