"""Lock a go-to goal onto a tracked object (e.g. a piece of trash) while the camera sees it.

Plain go-to (goto.py) measures the goal ONCE, from a click, and then drives on odometry
alone. Here the goal keeps being corrected by the detector for as long as the object is
in view, so a slightly wrong first measurement or drifting odometry no longer adds up.
When the object leaves the view (the robot is nearly on top of it), we stop correcting
and go-to finishes blind on odometry, exactly as before.

No camera, detector, window or SDK in here: the caller hands in what the tracker saw.

States
------
- "tracking": the target is visible; every detection refines the goal.
- "blind":    not seen (or only partly in view) for longer than lost_after_s. The goal is
              kept; go-to drives on odometry. If the target shows up again, tracking resumes.
- "idle":     no target selected.

Details that matter
-------------------
- A box cut off by the bottom/side of the picture has no visible foot point, so its
  distance would be wrong (too far). Such sightings are ignored, not used.
- The tracker sometimes gives the same object a new ID (after a blur, a missed frame).
  If the selected ID is gone but another object sits within reacquire_m of where the
  goal is, we adopt that one.
- A detection describes the past: the frame is ~0.3 s old when it arrives. The caller
  passes the pose the robot had THEN, so the robot's own movement in between cancels out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .goto import GoTo


@dataclass
class Sighting:
    """One tracked object in one camera frame."""
    track_id: int
    label: str
    conf: float
    floor: tuple[float, float] | None   # (X forward, Y left) from the robot's middle, None if not on the floor
    clipped: bool = False               # box cut off by the picture's edge: foot point not visible
    bearing: float | None = None        # direction of the object from the robot's middle (rad, + = left).
                                        # Also there when only the bottom is cut off; None if a side is
    near_edge: bool = False             # its bottom is close to the picture's bottom: about to leave the view


class TargetFollower:
    def __init__(self, goto: GoTo, lost_after_s: float = 1.0, reacquire_m: float = 0.4, blend: float = 0.5):
        self.goto = goto
        self.lost_after_s = lost_after_s
        self.reacquire_m = reacquire_m
        self.blend = blend
        self.track_id: int | None = None
        self.label = ""
        self.conf = 0.0
        self.state = "idle"
        self.last_seen = -float("inf")
        self.updates = 0          # how many detections have refined the goal
        self.reacquired = 0       # how many times we switched to a new track ID

    # ------------------------------------------------------------------ select / drop
    def select(self, s: Sighting, pose: tuple[float, float, float], now: float,
               pose_at_frame: tuple[float, float, float] | None = None) -> str | None:
        """Make this object the go-to goal. Returns an error text, or None.

        pose: the robot's pose now. pose_at_frame: its pose when the sighting's frame was taken.
        """
        if s.floor is None or s.clipped:
            return "can't place that object on the floor (cut off by the picture's edge or above the horizon)"
        err = self.goto.set_goal(s.floor[0], s.floor[1], pose, pose_at_frame)
        if err:
            return err
        self.track_id, self.label, self.conf = s.track_id, s.label, s.conf
        self.state, self.last_seen, self.updates, self.reacquired = "tracking", now, 0, 0
        if s.bearing is not None:   # steer by where it's seen right from the start
            self.goto.aim((pose_at_frame or pose)[2] + s.bearing)
        return None

    def drop(self):
        """Forget the target (the go-to goal itself is not touched)."""
        self.track_id, self.state = None, "idle"

    # ------------------------------------------------------------------ per detection result
    def update(self, sightings: list[Sighting], pose_at_frame: tuple[float, float, float], now: float) -> str:
        """Feed one detector result. `pose_at_frame` = the robot's pose when that frame was taken."""
        if self.track_id is None:
            return self.state
        if self.goto.goal is None:      # goal cleared elsewhere (C key, hand driving...)
            self.drop()
            return self.state

        usable = [s for s in sightings if s.floor is not None and not s.clipped]
        mine = next((s for s in usable if s.track_id == self.track_id), None)
        if mine is None:
            # The tracker may have renamed it: adopt whatever sits where the goal is.
            near = [(self._dist_to_goal(s, pose_at_frame), s) for s in usable]
            near = [(d, s) for d, s in near if d <= self.reacquire_m]
            if near:
                # Prefer the same kind of object, then the closest.
                d, mine = min(near, key=lambda e: (e[1].label != self.label, e[0]))
                self.track_id = mine.track_id
                self.reacquired += 1

        if mine is not None:
            self.goto.refine_goal(mine.floor[0], mine.floor[1], pose_at_frame, self.blend)
            self.label, self.conf = mine.label, mine.conf
            self.last_seen, self.state = now, "tracking"
            self.updates += 1
        elif now - self.last_seen > self.lost_after_s:
            self.state = "blind"

        # Centring. Its direction (the box's left-right position) is still right when only the
        # bottom is cut off, so it keeps steering a little longer than the distance can.
        aim = mine or next((s for s in sightings if s.track_id == self.track_id and s.bearing is not None), None)
        if aim is not None and aim.bearing is not None:
            # World heading of the object, as seen from where the robot was when the frame was taken.
            self.goto.aim(pose_at_frame[2] + aim.bearing)
            if aim.near_edge or aim.clipped:
                self.goto.start_align()     # about to go blind: centre it first, standing still
        return self.state

    def tick(self, now: float) -> str:
        """Call every frame, also when no detector result arrived: a silent detector means blind."""
        if self.track_id is not None and self.state == "tracking" and now - self.last_seen > self.lost_after_s:
            self.state = "blind"
        return self.state

    def _dist_to_goal(self, s: Sighting, pose) -> float:
        x, y, yaw = self.goto.nav_pose(pose)
        wx = x + s.floor[0] * math.cos(yaw) - s.floor[1] * math.sin(yaw)
        wy = y + s.floor[0] * math.sin(yaw) + s.floor[1] * math.cos(yaw)
        return math.hypot(wx - self.goto.goal[0], wy - self.goto.goal[1])
