"""Go to a goal point on the floor: the steering logic, with no window, SDK or camera in it.

The goal is stored in a position frame, so it stays put while the robot moves, also after it
has left the camera's view. Where the robot is comes from one of two "odometries":
- "robot":     the robot's own estimate (getPosition) + heading (IMU).
- "commanded": our own count: the forward speed we commanded x time, along the IMU heading.
               A fallback for when getPosition doesn't follow the real movement.

Each step (~30 Hz) takes the robot's pose and whether the go key is held, and returns the speed
to ask SafeDog for:
- Key not held          -> 0 (paused; the goal is kept and the timeout clock stops).
- Goal far off to the side or behind -> turn on the spot first.
- Roughly facing it     -> drive, steering toward it, slowing down near it.
- Close enough / passed -> arrived, 0. With stop_before_m > 0 the robot's MIDDLE stops that far
  short of the goal, facing it, so its nose ends just in front of the object instead of on top of it.
- Timeout, or the position estimate jumped -> stop and drop the goal.
- Overshoot guard: we've commanded more forward driving than the goal needed (+ margin) -> stop,
  whatever the position estimate says. Catches a position estimate that doesn't follow reality.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass

from .safety import MIN_VX, MIN_YAW

TURN_FIRST_DEG = 25.0   # facing further off than this: turn on the spot before driving
K_YAW = 1.5             # turn rate per radian of heading error
K_V = 0.6               # forward speed per metre left (slows down near the goal)
PASSED_M = 0.40         # within this distance and the goal is behind us -> we drove past it: arrived
FACE_DEG = 10.0         # when stopping short of a goal: turn until facing it within this
JUMP_M = 0.5            # position estimate moved more than this in one step: don't trust it
GUARD_FACTOR = 1.25     # overshoot guard: allow this much more driving than the straight line needs...
GUARD_MARGIN_M = 0.40   # ...plus this, before stopping
ODOMETRIES = ("robot", "commanded")
FINAL_STATES = ("arrived", "timeout", "lost position", "overshoot guard")


def wrap(a: float) -> float:
    """Angle in radians, wrapped to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseHistory:
    """Recent robot poses with the time they were read, to look up where the robot WAS.

    The camera picture arrives late (encoding + Wi-Fi + decoding). A click on it must be turned
    into a goal using the pose the robot had when that picture was taken, not the pose now,
    or a robot that's turning puts the goal in the wrong place.
    """

    def __init__(self, keep_s: float = 3.0):
        self.keep_s = keep_s
        self._t: deque[float] = deque()
        self._p: deque[tuple[float, float, float]] = deque()

    def add(self, t: float, pose: tuple[float, float, float]):
        self._t.append(t)
        self._p.append(pose)
        while self._t and self._t[0] < t - self.keep_s:
            self._t.popleft()
            self._p.popleft()

    def at(self, t: float) -> tuple[float, float, float] | None:
        """Pose at time t, interpolated between the two nearest samples (clamped to the oldest/newest)."""
        if not self._t:
            return None
        ts = list(self._t)
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return self._p[0]
        if i == len(ts):
            return self._p[-1]
        t0, t1 = ts[i - 1], ts[i]
        (x0, y0, a0), (x1, y1, a1) = self._p[i - 1], self._p[i]
        k = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return x0 + k * (x1 - x0), y0 + k * (y1 - y0), wrap(a0 + k * wrap(a1 - a0))


@dataclass
class GoToLimits:
    max_goal_m: float
    max_vx: float
    max_yaw: float
    arrive_m: float
    timeout_s: float
    stop_before_m: float = 0.0  # stop the robot's middle this far short of the goal (nose in front of it)
    odometry: str = "robot"     # "robot" (getPosition) or "commanded" (speed x time along the IMU heading)


@dataclass
class GoToStep:
    vx: float
    yaw: float
    state: str              # "no goal" | "paused" | "turning" | "driving" | one of FINAL_STATES
    dist: float | None      # metres to the goal (from the robot's middle)
    bearing_deg: float | None  # where the goal is, relative to where the robot faces (+ = left)


class GoTo:
    def __init__(self, limits: GoToLimits):
        if limits.odometry not in ODOMETRIES:
            raise ValueError(f"odometry must be one of {ODOMETRIES}, got {limits.odometry!r}")
        self.lim = limits
        self.goal: tuple[float, float] | None = None
        self.state = "no goal"
        self._drive_time = 0.0
        self._last_pose: tuple[float, float, float] | None = None
        self._last_vx = 0.0
        self._own_xy = (0.0, 0.0)  # "commanded" odometry: our own position count
        # Travel counters since the goal was set (for the overshoot guard and the log).
        self.commanded_m = 0.0     # forward driving we commanded (speed x time)
        self.robot_m = 0.0         # how far the robot's own estimate says it moved
        self.needed_m = 0.0        # straight-line driving the goal needed when it was set

    # ------------------------------------------------------------------ goal
    def set_goal(self, local_x: float, local_y: float, pose: tuple[float, float, float],
                 seen_pose: tuple[float, float, float] | None = None) -> str | None:
        """Set a goal given relative to the robot's middle (X forward, Y left). Returns an error, or None.

        pose: the robot's pose now. seen_pose: its pose when the camera picture was taken (the
        goal is relative to THAT); defaults to pose.
        """
        seen = seen_pose or pose
        x, y, yaw = seen
        gx = x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        gy = y + local_x * math.sin(yaw) + local_y * math.cos(yaw)
        dist = math.hypot(gx - pose[0], gy - pose[1])
        if dist > self.lim.max_goal_m:
            return f"goal {dist:.1f} m away: too far (max {self.lim.max_goal_m:.1f} m)"
        self._own_xy = (pose[0], pose[1])  # our own count ("commanded") starts at the robot's position now
        self.goal = (gx, gy)
        self.state = "paused"
        self._drive_time = 0.0
        self._last_pose = None
        self._last_vx = 0.0
        self.commanded_m = self.robot_m = 0.0
        self.needed_m = max(0.0, dist - max(self.lim.arrive_m, self.lim.stop_before_m))
        return None

    def refine_goal(self, local_x: float, local_y: float, pose: tuple[float, float, float], blend: float = 1.0):
        """Move the goal to a fresh measurement of where the target is, WITHOUT restarting the drive.

        Used while the camera still sees the target (trash tracking): each new detection
        corrects the goal, so errors in the first click and odometry drift don't add up.
        (local_x, local_y) is the target relative to the robot's middle at `pose`, which
        should be the pose the robot had when that camera frame was taken.
        blend < 1 mixes the new measurement with the old goal (smooths detector jitter).
        Does nothing without a goal or after the drive has ended.
        """
        if self.goal is None or self.state in FINAL_STATES:
            return
        x, y, yaw = self.nav_pose(pose)
        gx = x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        gy = y + local_x * math.sin(yaw) + local_y * math.cos(yaw)
        self.goal = (self.goal[0] + blend * (gx - self.goal[0]), self.goal[1] + blend * (gy - self.goal[1]))
        # Keep the overshoot guard meaningful: what we've driven so far + what's still needed now.
        dist = math.hypot(*self.relative(pose))
        self.needed_m = self.commanded_m + max(0.0, dist - max(self.lim.arrive_m, self.lim.stop_before_m))

    def clear(self):
        self.goal = None
        self.state = "no goal"

    def nav_pose(self, robot_pose: tuple[float, float, float]) -> tuple[float, float, float]:
        """The pose go-to steers by: the robot's own, or our commanded count with the robot's heading."""
        if self.lim.odometry == "commanded":
            return self._own_xy[0], self._own_xy[1], robot_pose[2]
        return robot_pose

    def relative(self, robot_pose: tuple[float, float, float]) -> tuple[float, float] | None:
        """Goal relative to the robot's middle right now (X forward, Y left), e.g. for drawing it."""
        if self.goal is None:
            return None
        x, y, yaw = self.nav_pose(robot_pose)
        dx, dy = self.goal[0] - x, self.goal[1] - y
        return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)

    # ------------------------------------------------------------------ control
    def step(self, pose: tuple[float, float, float], holding: bool, dt: float) -> GoToStep:
        if self.goal is None:
            return GoToStep(0.0, 0.0, "no goal", None, None)

        # Account for the last step's command: our own position count and the travel counters.
        moved = self._last_vx * dt
        self.commanded_m += moved
        self._own_xy = (self._own_xy[0] + moved * math.cos(pose[2]), self._own_xy[1] + moved * math.sin(pose[2]))
        if self._last_pose is not None:
            step_m = math.hypot(pose[0] - self._last_pose[0], pose[1] - self._last_pose[1])
            # A sudden jump in the robot's estimate: can't trust where the goal is (only matters if we use it).
            if step_m > JUMP_M and self.lim.odometry == "robot" and self.state not in FINAL_STATES:
                self.state = "lost position"
            self.robot_m += step_m
        self._last_pose = pose
        self._last_vx = 0.0

        if self.state in FINAL_STATES:
            return self._out(0.0, 0.0, pose)

        # Overshoot guard: independent of any position estimate.
        if self.commanded_m > self.needed_m * GUARD_FACTOR + GUARD_MARGIN_M:
            self.state = "overshoot guard"
            return self._out(0.0, 0.0, pose)

        rel = self.relative(pose)
        dist = math.hypot(*rel)
        bearing = math.atan2(rel[1], rel[0])

        stop = max(self.lim.arrive_m, self.lim.stop_before_m)
        close = dist <= stop
        facing = abs(bearing) <= math.radians(FACE_DEG)
        if (close and (facing or self.lim.stop_before_m <= self.lim.arrive_m)) or \
                (dist <= PASSED_M and abs(bearing) > math.pi / 2):
            self.state = "arrived"
            return self._out(0.0, 0.0, pose)
        if not holding:
            self.state = "paused"
            return self._out(0.0, 0.0, pose)

        self._drive_time += dt
        if self._drive_time > self.lim.timeout_s:
            self.state = "timeout"
            return self._out(0.0, 0.0, pose)

        yaw = max(-self.lim.max_yaw, min(self.lim.max_yaw, K_YAW * bearing))
        if abs(bearing) > math.radians(TURN_FIRST_DEG) or close:
            # Turn on the spot (also: already close enough, just not facing the goal yet).
            # Keep at least the SDK's minimum turn rate, or it would stall.
            self.state = "turning"
            yaw = math.copysign(max(abs(yaw), MIN_YAW * 1.2), bearing)
            return self._out(0.0, yaw, pose)

        self.state = "driving"
        vx = min(self.lim.max_vx, K_V * (dist - stop))
        vx = max(vx, MIN_VX * 1.2)        # below the SDK minimum the robot wouldn't move at all
        if abs(yaw) < MIN_YAW:
            yaw = 0.0                      # tiny corrections: the SDK ignores them anyway
        self._last_vx = vx
        return self._out(vx, yaw, pose)

    def _out(self, vx: float, yaw: float, pose) -> GoToStep:
        rel = self.relative(pose)
        dist = math.hypot(*rel)
        return GoToStep(vx, yaw, self.state, dist, math.degrees(math.atan2(rel[1], rel[0])))
