"""Back off and turn (avoid/backoff.py): the simple default avoidance."""

import math

import numpy as np
import pytest

from robodog.avoid import BackoffAvoider, BackoffSettings, red_side
from robodog.goto import GoTo, GoToLimits, wrap

DT = 1 / 30
S = BackoffSettings()


class Res:
    def __init__(self, cols=None):
        self.mask = np.zeros((10, 100), bool)
        if cols is not None:
            self.mask[5, cols[0]:cols[1]] = True


def test_red_side():
    box = (0.4, 0.6, 0.3)                         # yellow box from 0.4 to 0.6: centre 0.5
    assert red_side(Res((52, 60)), box) == -1     # red right of the centre (the robot's right)
    assert red_side(Res((40, 49)), box) == 1      # left
    assert red_side(Res(), box) == 0              # nothing red
    assert red_side(None, box) == 0


def run(av, blocked_for, red, seconds=20.0, target=0.0):
    """Drive the avoider like go-to does: blocked for `blocked_for` s at the start. Returns the
    states it went through and the (vx, yaw) it sent, and the final heading."""
    yaw, t, states, cmds = 0.0, 0.0, [], []
    for _ in range(int(seconds / DT)):
        c = av.update(t < blocked_for, red, (0.0, 0.0, yaw), target, t)
        states.append(c.state)
        cmds.append((c.vx, c.yaw, c.active))
        yaw = wrap(yaw + c.yaw * DT)
        t += DT
    return states, cmds, yaw


def test_backs_up_turns_away_from_the_red_then_drives_on_then_hands_back():
    av = BackoffAvoider(S)
    states, cmds, yaw = run(av, blocked_for=0.1, red=-1, seconds=20.0)   # red on the right
    seq = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
    assert seq == ["backing", "turning", "forward", "clear"]      # target straight ahead: no rejoin needed
    back = [c for c, s in zip(cmds, states) if s == "backing"]
    assert all(vx < 0 and 0 < y <= 0.15 for vx, y, _ in back)    # reversing, already turning a little left
    assert len(back) * DT == pytest.approx(S.back_m / S.back_vx, abs=0.1)   # ~0.25 m at 0.15 m/s
    assert math.degrees(yaw) == pytest.approx(S.turn_deg, abs=4)            # turned LEFT, away from the red
    assert not cmds[-1][2]                                                   # go-to drives again


def test_no_red_info_turns_toward_the_target_side():
    av = BackoffAvoider(S)
    _, _, yaw = run(av, blocked_for=0.1, red=0, target=math.radians(-20))
    assert yaw < 0


def test_blocked_again_soon_keeps_the_same_side():
    av = BackoffAvoider(S)
    run(av, blocked_for=0.1, red=-1)               # first time: turned left
    _, _, yaw2 = run(av, blocked_for=0.1, red=1)   # red now says "turn right", but keep left
    assert yaw2 > 0 and av.count == 2


def test_time_spent_backing_off_counts_toward_the_goto_timeout():
    g = GoTo(GoToLimits(max_goal_m=5.0, max_vx=0.2, max_yaw=0.5, arrive_m=0.1, timeout_s=2.0))
    g.set_goal(3.0, 0.0, (0.0, 0.0, 0.0))
    for _ in range(int(2.5 / DT)):                 # a dead end: avoiding all the time
        g.spend(DT)
    assert g.step((0.0, 0.0, 0.0), True, DT).state == "timeout"



def test_rejoin_curves_back_gently_instead_of_turning_on_the_spot():
    av = BackoffAvoider(S)
    yaw, t, rejoin = 0.0, 0.0, []
    for _ in range(int(30 / DT)):
        target = wrap(0.0 - yaw)                    # the target stays straight ahead of where it started
        c = av.update(t < 0.1, -1, (0.0, 0.0, yaw), target, t)
        if c.state == "rejoin":
            rejoin.append(c)
        yaw = wrap(yaw + c.yaw * DT)
        t += DT
    assert rejoin, "never curved back"
    assert all(c.vx > 0 and abs(c.yaw) <= S.rejoin_yaw + 1e-9 for c in rejoin)   # driving AND turning gently
    assert abs(math.degrees(yaw)) <= 12                                           # facing the target again


def test_blocked_again_during_the_manoeuvre_keeps_the_side():
    av = BackoffAvoider(S)
    yaw, t = 0.0, 0.0
    sides = set()
    for i in range(int(20 / DT)):
        blocked = t < 0.1 or 6.0 < t < 6.2          # blocked again while driving beside it
        c = av.update(blocked, -1 if t < 1 else 1, (0.0, 0.0, yaw), 0.0, t)   # red flips side the 2nd time
        if c.state == "backing":
            sides.add(av.side)
        yaw = wrap(yaw + c.yaw * DT)
        t += DT
    assert sides == {1} and av.count == 2


def test_backing_off_rounds_dont_add_up_to_the_overshoot_guard():
    # Seen on the robot: drive toward a box, back off, drive toward it again... until go-to's
    # overshoot guard (driven > needed x 1.25 + 0.4) stopped it for good.
    g = GoTo(GoToLimits(max_goal_m=5.0, max_vx=0.2, max_yaw=0.5, arrive_m=0.1, timeout_s=600.0))
    g.set_goal(2.0, 0.0, (0.0, 0.0, 0.0))
    x = 0.0
    for _ in range(12):                                  # twelve rounds
        for _ in range(int(1.5 / DT)):                   # go-to drives toward it 1.5 s...
            s = g.step((x, 0.0, 0.0), True, DT)
            x += s.vx * DT
        x -= 0.3                                         # ...then backoff moves it back 0.3 m
        g.rebase_guard((x, 0.0, 0.0))
    assert g.step((x, 0.0, 0.0), True, DT).state != "overshoot guard"
