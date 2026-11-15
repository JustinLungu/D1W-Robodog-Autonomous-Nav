"""Safety-layer tests against the fake robot. No hardware needed: `uv run pytest`."""

import pytest

from robodog.config import CONFIG_PATH, REPO_ROOT, load_config
from fake_robot import FakeHighLevel  # tests/fake_robot.py
import robodog.safety as safety
from robodog.safety import SHAKE_HAND_S, STAND_SETTLE_S, MotionBlocked, SafeDog, SafetyError


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# Fixed values for the tests, so editing config.local.yaml never changes what the tests check.
TEST_SAFETY = dict(
    max_vx=0.3, max_vy=0.2, max_yaw=0.5, min_battery=25, rate_hz=30.0,
    cmd_timeout=0.3, connect_timeout=5.0, battery_check_period=1.0,
)


def pinned_config():
    """The real config file, with safety values pinned to TEST_SAFETY."""
    cfg = load_config(CONFIG_PATH, REPO_ROOT / "tests" / "machine_test.yaml")
    for k, v in TEST_SAFETY.items():
        setattr(cfg.safety, k, v)
    return cfg


def make(allow_motion=True, battery=80, **safety):
    cfg = pinned_config()
    cfg.safety.allow_motion = allow_motion
    for k, v in safety.items():
        setattr(cfg.safety, k, v)
    robot = FakeHighLevel(battery=battery)
    clock = Clock()
    dog = SafeDog(robot, cfg, allow_motion=allow_motion, clock=clock, log=lambda m: None)
    return dog, robot, clock


def stand(dog, clock):
    """Stand up and let the (fake) clock run past the stand-up transition."""
    dog.stand()
    clock.advance(STAND_SETTLE_S)


def moves(robot):
    return [c[1:] for c in robot.calls_named("move")]


# ---------------------------------------------------------------- motion gating
def test_motion_blocked_by_default():
    dog, robot, _ = make(allow_motion=False)
    with pytest.raises(MotionBlocked):
        dog.stand()
    dog.tick()
    assert robot.calls_named("standUp") == [] and moves(robot) == []


def test_motion_needs_both_config_and_caller():
    cfg = pinned_config()
    cfg.safety.allow_motion = False
    dog = SafeDog(FakeHighLevel(), cfg, allow_motion=True, log=lambda m: None)
    with pytest.raises(MotionBlocked):
        dog.stand()


def test_nothing_sent_before_stand():
    dog, robot, clock = make()
    for _ in range(10):
        dog.tick()
        clock.advance(0.03)
    assert moves(robot) == []


def test_velocity_requires_stand():
    dog, _, _ = make()
    with pytest.raises(SafetyError):
        dog.set_velocity(0.2)


# ---------------------------------------------------------------- shaping
def test_speeds_are_clamped():
    dog, robot, clock = make(max_vx=0.3, max_vy=0.2, max_yaw=0.5)
    stand(dog, clock)
    dog.set_velocity(5.0, -5.0, 9.0)
    dog.tick()
    assert moves(robot)[-1] == (0.3, -0.2, 0.5)


def test_values_below_sdk_minimum_become_zero():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.set_velocity(0.04, 0.09, -0.05)
    dog.tick()
    assert moves(robot)[-1] == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- dead-man / keepalive
def test_deadman_zeroes_stale_command():
    dog, robot, clock = make(cmd_timeout=0.3)
    stand(dog, clock)
    dog.set_velocity(0.2)
    dog.tick()
    assert moves(robot)[-1] == (0.2, 0.0, 0.0)
    clock.advance(0.31)
    dog.tick()
    assert moves(robot)[-1] == (0.0, 0.0, 0.0)


def test_loop_keeps_feeding_robot_while_standing():
    dog, robot, clock = make()
    stand(dog, clock)
    for _ in range(30):
        dog.tick()
        clock.advance(1 / 30)
    assert len(moves(robot)) == 30  # robot's 3 s watchdog never starves


# ---------------------------------------------------------------- battery
def test_refuses_to_stand_on_low_battery():
    dog, robot, _ = make(battery=20, min_battery=25)
    with pytest.raises(SafetyError):
        dog.stand()
    assert robot.calls_named("standUp") == []


def test_battery_drop_halts_and_lies_down():
    dog, robot, clock = make(min_battery=25)
    stand(dog, clock)
    dog.set_velocity(0.2)
    dog.tick()
    robot.battery = 20
    clock.advance(1.1)
    dog.tick()
    assert "battery" in dog.fault
    assert robot.calls_named("lieDown")
    with pytest.raises(SafetyError):
        dog.set_velocity(0.2)


# ---------------------------------------------------------------- connection / errors
def test_connection_loss_halts():
    dog, robot, clock = make()
    stand(dog, clock)
    robot.connected = False
    dog.tick()
    assert dog.fault == "connection lost"
    with pytest.raises(SafetyError):
        dog.stand()


def test_fault_latches_and_loop_goes_quiet():
    dog, robot, clock = make()
    stand(dog, clock)
    robot.connected = False
    dog.tick()
    n_calls = len(robot.calls)
    robot.connected = True  # connection comes back...
    for _ in range(10):
        dog.tick()
        clock.advance(0.1)
    assert len(robot.calls) == n_calls  # ...but nothing is sent until a new SafeDog
    assert dog.fault == "connection lost"


def test_sdk_exception_halts():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.set_velocity(0.2)

    def boom(*a):
        raise RuntimeError("socket died")

    robot.move = boom
    dog.tick()
    assert "SDK error" in dog.fault


# ---------------------------------------------------------------- e-stop
def test_estop_stops_lies_down_and_latches():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.set_velocity(0.2)
    dog.tick()
    dog.estop("test")
    dog.wait_fully_down()
    names = [c[0] for c in robot.calls]
    assert names[-3:] == ["move", "lieDown", "passive"]  # stop, crouch, then fully down
    assert robot.calls[-3] == ("move", 0.0, 0.0, 0.0)
    n = len(moves(robot))
    dog.tick()
    assert len(moves(robot)) == n  # nothing sent after e-stop
    with pytest.raises(SafetyError):
        dog.stand()
    with pytest.raises(SafetyError):
        dog.set_velocity(0.2)


def test_estop_works_even_when_motion_disabled():
    dog, robot, _ = make(allow_motion=False)
    dog.estop("test")
    assert robot.calls_named("lieDown")


@pytest.mark.parametrize("failure", ["raises", "error_code"])
def test_estop_falls_back_to_limp_if_lying_down_fails(failure):
    dog, robot, clock = make()
    stand(dog, clock)

    def bad_lie_down():
        robot._record("lieDown")
        if failure == "raises":
            raise RuntimeError("no answer")
        return 7

    robot.lieDown = bad_lie_down
    dog.estop("test")
    assert robot.calls[-1][0] == "passive"
    assert dog.fault.startswith("E-STOP")


# ---------------------------------------------------------------- handshake
def test_handshake_copies_the_vendor_demo_sequence():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.shake_hand()
    assert robot.calls[-1][0] == "shakeHand"
    n = len(robot.calls)
    for _ in range(int(SHAKE_HAND_S / 0.1) - 1):  # trick running: nothing sent at all
        dog.tick()
        clock.advance(0.1)
    assert len(robot.calls) == n and not dog.settled()
    with pytest.raises(SafetyError):
        dog.set_velocity(0.2)
    clock.advance(0.2)
    dog.tick()
    assert robot.calls[-1][0] == "standUp"  # then stands up again, like the demo
    dog.tick()
    assert robot.calls[-1][0] == "standUp" and not dog.settled()  # usual stand-up pause
    clock.advance(STAND_SETTLE_S)
    dog.tick()
    assert dog.settled() and moves(robot)[-1] == (0.0, 0.0, 0.0)
    dog.set_velocity(0.2)  # driving allowed again


def test_handshake_refused_while_standing_up_or_lying():
    dog, robot, clock = make()
    with pytest.raises(SafetyError):
        dog.shake_hand()  # lying
    dog.stand()
    with pytest.raises(SafetyError, match="standing still"):
        dog.shake_hand()  # still standing up
    assert robot.calls_named("shakeHand") == []


def test_lie_down_or_estop_cancel_the_restand():
    for stop in ("lie_down", "estop"):
        dog, robot, clock = make()
        stand(dog, clock)
        dog.shake_hand()
        getattr(dog, stop)() if stop == "lie_down" else dog.estop("test")
        clock.advance(SHAKE_HAND_S + 1)
        dog.tick()
        assert len(robot.calls_named("standUp")) == 1  # never re-stood


def test_close_lies_down_if_standing():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.close()
    assert robot.calls_named("lieDown")


def test_no_stunts_exposed():
    for name in ("jump", "frontJump", "backflip", "shakeHand", "front_jump"):
        assert not hasattr(SafeDog, name)


# ---------------------------------------------------------------- stand-up transition
def test_nothing_sent_while_standing_up():
    dog, robot, clock = make()
    dog.stand()
    for _ in range(10):
        dog.tick()
        clock.advance(0.1)  # 1 s, still inside the transition
    assert moves(robot) == []
    assert not dog.settled()
    clock.advance(STAND_SETTLE_S)
    dog.tick()
    assert moves(robot) == [(0.0, 0.0, 0.0)]  # keepalive starts once settled
    assert dog.settled()


def test_velocity_refused_while_standing_up():
    dog, robot, clock = make()
    dog.stand()
    with pytest.raises(SafetyError, match="still standing up"):
        dog.set_velocity(0.2)
    clock.advance(STAND_SETTLE_S)
    dog.set_velocity(0.2)  # fine now


def test_lie_down_during_stand_up_sends_no_move():
    dog, robot, clock = make()
    dog.stand()
    dog.lie_down()
    assert moves(robot) == []
    assert robot.calls[-1][0] == "lieDown"



# ---------------------------------------------------------------- lying down fully
def test_lie_down_crouches_then_goes_fully_down():
    dog, robot, clock = make()
    stand(dog, clock)
    dog.lie_down()
    assert [c[0] for c in robot.calls][-2:] == ["move", "lieDown"]  # stop, crouch
    dog.wait_fully_down()
    assert robot.calls[-1][0] == "passive"                          # then damping: fully down


def test_standing_up_again_cancels_fully_down(monkeypatch):
    monkeypatch.setattr(safety, "LIE_DOWN_S", 0.3)
    dog, robot, clock = make()
    stand(dog, clock)
    dog.lie_down()
    dog.stand()               # R pressed within LIE_DOWN_S
    import time; time.sleep(0.4)
    assert robot.calls_named("passive") == []
    assert robot.calls[-1][0] == "standUp"


def test_no_stop_command_to_a_crouched_robot():
    """A crouched robot rejects move() ("transition to Move failed"): don't send it."""
    dog, robot, clock = make()
    stand(dog, clock)
    dog.lie_down()
    n = len(moves(robot))
    dog.estop("test")         # SPACE after F
    dog.wait_fully_down()
    assert len(moves(robot)) == n
    assert robot.calls_named("lieDown") and robot.calls[-1][0] == "passive"


def test_error_warning_says_mode_and_time_since_last_posture_command():
    logs = []
    dog, robot, clock = make()
    dog.log = logs.append
    stand(dog, clock)                       # standUp, then 3 s
    clock.advance(0.05)
    robot.move = lambda *a: 12295           # robot refuses move()
    dog.tick()
    warning = [m for m in logs if "12295" in m][-1]
    assert "robot mode 1" in warning and "3.05 s after standUp" in warning
