"""The stand-up test script, run against the fake robot."""

import threading

import pytest

from fake_robot import FakeHighLevel
from test_safety import pinned_config

from robodog.cli.stand_test import run_stand_test
import robodog.safety as safety
from robodog.safety import SafeDog


@pytest.fixture(autouse=True)
def no_stand_wait(monkeypatch):
    """These tests are about driving, not the stand-up pause (tested separately)."""
    monkeypatch.setattr(safety, "STAND_SETTLE_S", 0.05)


def make(battery=80):
    cfg = pinned_config()
    cfg.safety.allow_motion = True
    robot = FakeHighLevel(battery=battery)
    return SafeDog(robot, cfg, allow_motion=True, log=lambda m: None), robot


def names(robot):
    return [c[0] for c in robot.calls]


def test_stands_holds_still_and_lies_down():
    dog, robot = make()
    run_stand_test(dog, hold_s=0.3, quit_event=threading.Event(), print_fn=lambda m: None)
    assert names(robot)[0] == "standUp"
    assert names(robot)[-1] == "lieDown"
    moves = robot.calls_named("move")
    assert moves and all(m[1:] == (0.0, 0.0, 0.0) for m in moves)  # never drives


def test_quit_key_ends_early():
    dog, robot = make()
    quit_event = threading.Event()
    quit_event.set()
    run_stand_test(dog, hold_s=60, quit_event=quit_event, print_fn=lambda m: None)
    assert names(robot)[-1] == "lieDown"


def test_low_battery_refuses_to_stand():
    dog, robot = make(battery=22)
    try:
        run_stand_test(dog, hold_s=0.3, quit_event=threading.Event(), print_fn=lambda m: None)
    except Exception as e:
        assert "battery" in str(e)
    finally:
        dog.close()
    assert robot.calls_named("standUp") == []
