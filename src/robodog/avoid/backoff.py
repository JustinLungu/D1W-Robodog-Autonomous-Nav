"""Back off and turn: the simplest obstacle avoidance. No map, no memory, no position estimate.

Go-to drives as usual. Only when the forward block fires (something red in the yellow box, i.e.
close in the path):
1. backing:  back_m at back_vx (timed: no position estimate needed), already turning a little
             (back_yaw) toward the side it will go
2. turning:  until turn_deg away from where it faced when blocked, away from the side with more red
             (by the IMU heading, which is reliable)
3. forward:  forward_m along that new heading (timed), so the robot actually gets beside it
4. rejoin:   curve GENTLY back toward the target while driving (at most rejoin_yaw), instead of
             go-to's turn on the spot, which pointed it straight back at the obstacle (back and forth)
5. clear:    go-to takes over once facing the target (within REJOINED_DEG) or after rejoin_m
Blocked again during 3 or 4, or soon after -> again from 1, same side (no zig-zag).
Dead ends aren't detected: go-to's timeout ends them (the time spent here counts toward it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..goto import wrap

TURNED_DEG = 3.0      # turned to within this of the new heading: done turning
REJOINED_DEG = 10.0   # facing the target within this: go-to takes over again
SAME_SIDE_S = 8.0     # blocked again within this long after a manoeuvre: keep the side


@dataclass
class BackoffSettings:
    back_m: float = 0.25       # how far to back up
    back_vx: float = 0.15      # backing speed, m/s (the SDK ignores less than 0.05)
    back_yaw: float = 0.12     # turning a little while backing, rad/s (the SDK ignores less than 0.1)
    turn_deg: float = 25.0     # how far to have turned away in total
    turn_yaw: float = 0.4      # turning speed, rad/s
    forward_m: float = 1.0     # how far to drive on the new heading, to get beside the obstacle
    forward_vx: float = 0.15   # m/s
    rejoin_m: float = 1.0      # curve back toward the target for at most this far...
    rejoin_yaw: float = 0.15   # ...turning at most this fast, rad/s


@dataclass
class BackoffCmd:
    active: bool        # True = this drives (send vx, yaw); False = let go-to drive
    vx: float
    yaw: float
    state: str          # "clear" | "backing" | "turning" | "forward" | "rejoin"
    reason: str         # for the window
    vy: float = 0.0     # (never sideways; same fields as go-to's command)


def red_side(res, box: tuple[float, float, float] | None) -> int:
    """Which side of the robot's path the close obstacle (red) is mostly on: +1 left, -1 right,
    0 unknown. `box` = the yellow box (left x, right x, top row, as image fractions)."""
    mask = getattr(res, "mask", None)
    if mask is None or not mask.any():
        return 0
    cols = np.nonzero(mask)[1] / max(mask.shape[1] - 1, 1)
    centre = 0.5 if box is None else (box[0] + box[1]) / 2
    left, right = int(np.sum(cols < centre)), int(np.sum(cols >= centre))
    return 0 if left == right else (1 if left > right else -1)   # image left = robot's left


class BackoffAvoider:
    def __init__(self, settings: BackoffSettings):
        self.s = settings
        self.state = "clear"
        self.side = 0              # turning: +1 = left, -1 = right
        self.until = 0.0           # end time of the timed phase
        self.heading = 0.0         # world heading to turn to / keep
        self.last_done = -1e9      # when the last manoeuvre ended (for SAME_SIDE_S)
        self.count = 0             # manoeuvres so far (for the window)

    def update(self, blocked: bool, red: int, pose, target: float, now: float) -> BackoffCmd:
        """blocked: the forward block is on (a real obstacle); red: red_side() of the newest check;
        pose: now; target: the target's direction (rad, + = left); now: time.monotonic()."""
        yaw_now = pose[2]
        if self.state == "clear":
            if not blocked:
                return BackoffCmd(False, 0.0, 0.0, "clear", "straight" if self.count == 0 else
                                  f"straight (got around {self.count}x)")
            self._start(red, target, yaw_now, now)
        elif self.state in ("forward", "rejoin") and blocked:   # blocked again on the way: again,
            self.last_done = now                                  # on the same side
            self._start(red, target, yaw_now, now)

        where = "left" if self.side > 0 else "right"
        if self.state == "backing":
            if now < self.until:
                return BackoffCmd(True, -self.s.back_vx, self.side * self.s.back_yaw, "backing",
                                  f"backing off, turning {where}")
            self.state = "turning"

        if self.state == "turning":
            err = wrap(self.heading - yaw_now)
            if abs(err) > math.radians(TURNED_DEG):
                return BackoffCmd(True, 0.0, math.copysign(self.s.turn_yaw, err), "turning", f"turning {where}")
            self.state, self.until = "forward", now + self.s.forward_m / self.s.forward_vx

        if self.state == "forward":
            if now < self.until:
                return BackoffCmd(True, self.s.forward_vx, self._hold(yaw_now), "forward", "getting beside it")
            self.state, self.until = "rejoin", now + self.s.rejoin_m / self.s.forward_vx

        if self.state == "rejoin":
            if abs(target) > math.radians(REJOINED_DEG) and now < self.until:
                yaw = max(-self.s.rejoin_yaw, min(self.s.rejoin_yaw, 1.5 * target))
                return BackoffCmd(True, self.s.forward_vx, yaw, "rejoin", "curving back to the target")
            self.state, self.last_done = "clear", now
        return BackoffCmd(False, 0.0, 0.0, "clear", f"straight (got around {self.count}x)")

    def _hold(self, yaw_now: float) -> float:
        err = wrap(self.heading - yaw_now)
        return max(-self.s.turn_yaw, min(self.s.turn_yaw, 1.5 * err)) if abs(err) > math.radians(2) else 0.0

    def _start(self, red: int, target: float, yaw_now: float, now: float):
        """Blocked: pick the side, fix the heading to end up at, start backing up."""
        if self.side and now - self.last_done < SAME_SIDE_S:
            pass                                          # keep the side of the last manoeuvre
        elif red:
            self.side = -red                              # away from the red
        else:
            self.side = 1 if target >= 0 else -1          # no clue: toward the target's side
        self.count += 1
        self.heading = yaw_now + self.side * math.radians(self.s.turn_deg)
        self.state, self.until = "backing", now + self.s.back_m / self.s.back_vx

    def reset(self):
        self.__init__(self.s)
