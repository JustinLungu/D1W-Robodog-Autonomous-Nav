"""Keyboard teleop logic: which keys are held -> what speed to ask SafeDog for.

No pygame and no SDK in here, so it can be tested on its own. The window
(cli/teleop.py) only translates real key events into the calls below.

Rules
-----
- Motion keys (HOLD): the robot moves only while the key is held. Release = stop.
- Action keys (CLICK): one press = one action (stand up, lie down, speed level, e-stop).
- Releasing every key stops the robot IMMEDIATELY. Pressing a key ramps the speed up
  over a fraction of a second, so starts aren't jerky. Stops are never ramped.
- Window loses focus (you clicked elsewhere) -> all keys count as released -> stop.
- Everything still goes through SafeDog: speed caps, dead-man switch, battery, e-stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .safety import MIN_VX, MIN_VY, MIN_YAW, SafeDog, SafetyError

# Hold keys -> (axis, sign). Axis 0 = vx (forward), 1 = vy (sideways left), 2 = yaw (turn left).
MOTION = {
    "forward": (0, +1), "back": (0, -1),
    "strafe_left": (1, +1), "strafe_right": (1, -1),
    "turn_left": (2, +1), "turn_right": (2, -1),
}
ACTIONS = ("stand", "lie_down", "shake_hand", "speed_1", "speed_2", "speed_3", "estop", "quit")

# Speed levels as a fraction of the caps in config.local.yaml (max_vx / max_vy / max_yaw).
SPEED_LEVELS = {1: 0.33, 2: 0.66, 3: 1.0}

# How fast the speed builds up after a key is pressed (per second).
ACCEL = (0.6, 0.6, 1.5)  # m/s^2, m/s^2, rad/s^2


@dataclass
class TeleopState:
    """What the window shows."""
    standing: bool
    level: int
    held: frozenset
    cmd: tuple[float, float, float]
    message: str
    fault: str | None


class Teleop:
    def __init__(self, dog: SafeDog, level: int = 1, log=print):
        self.dog = dog
        self.s = dog.cfg.safety
        self.level = level
        self.log = log
        self.held: set[str] = set()
        self.standing = False
        self.quit = False
        self.message = "press R to stand up"
        self._cur = [0.0, 0.0, 0.0]  # ramped speed actually requested

    # ------------------------------------------------------------- input from the window
    def set_held(self, names):
        """The motion keys held right now (the window sends the full set every frame).

        A full set rather than press/release events, so nothing can get "stuck": if a
        release is ever missed, the next frame's set fixes it. Empty set = stop.
        """
        self.held = {n for n in names if n in MOTION}

    def release_all(self):
        """Focus lost, window hidden...: behave as if every key was let go."""
        self.held.clear()

    # ------------------------------------------------------------- click actions
    def action(self, name: str):
        """One click action: stand, lie_down, speed_1..3, estop, quit."""
        if name not in ACTIONS:
            raise ValueError(f"unknown action {name!r}")
        try:
            if name == "estop":
                self._stop_now()
                self.dog.estop("SPACE pressed")
                self.standing = False
                self.message = "E-STOP: stopped, lying down (fully down in 3 s). Restart to move again"
            elif name == "quit":
                self.quit = True
            elif name.startswith("speed_"):
                self.level = int(name[-1])
                self.message = f"speed level {self.level}"
            elif name == "stand":
                if self.standing:
                    return
                self.message = "standing up..."
                self.dog.stand()
                self.standing = True
                self.message = "standing up... (keys ignored for 3 s)"
            elif name == "shake_hand":
                if not self.standing:
                    return
                self._stop_now()
                self.dog.shake_hand()  # refused (message below) while still standing up
                self.message = "shaking hands... (keys ignored ~7 s)"
            elif name == "lie_down":
                if not self.standing:
                    return
                self._stop_now()
                self.dog.lie_down()
                self.standing = False
                self.message = "lying down (fully down in 3 s). press R to stand up"
        except SafetyError as e:  # includes MotionBlocked
            self.message = f"refused: {e}"
            self.log(self.message)

    # ------------------------------------------------------------- called every frame
    def target(self) -> tuple[float, float, float]:
        """Speed wanted from the keys held right now (before ramping)."""
        frac = SPEED_LEVELS[self.level]
        caps = (self.s.max_vx, self.s.max_vy, self.s.max_yaw)
        mins = (MIN_VX, MIN_VY, MIN_YAW)
        out = [0.0, 0.0, 0.0]
        for name in self.held:
            axis, sign = MOTION[name]
            out[axis] += sign
        # Opposite keys cancel (W+S = 0). Never go below the SDK minimum, or SafeDog
        # would snap it to 0 and the key would seem to do nothing (vy at level 1).
        return tuple(
            0.0 if d == 0 else d * min(caps[i], max(mins[i], caps[i] * frac))
            for i, d in enumerate(out)
        )

    def update(self, dt: float):
        """Ramp toward the target speed and hand it to SafeDog. Call at ~30 Hz."""
        if self.dog.fault:
            self.standing = False
            self._cur = [0.0, 0.0, 0.0]
            return
        if not self.standing:
            return
        if not self.dog.settled():
            # Still in the stand-up transition: SafeDog refuses speeds now. Keep the ramp at 0,
            # so holding W during the stand-up doesn't make it lurch the moment it's done.
            self._cur = [0.0, 0.0, 0.0]
            return
        if self.message.startswith(("standing up", "shaking hands")):
            self.message = "standing. hold W/A/S/D (or arrows) to drive"
        tgt = self.target()
        for i in range(3):
            released = tgt[i] == 0.0
            reversed_ = self._cur[i] != 0.0 and (tgt[i] > 0) != (self._cur[i] > 0)
            if released or reversed_:
                self._cur[i] = 0.0  # stop this axis at once (a reverse then ramps up from 0)
            if tgt[i] != 0.0:
                step = ACCEL[i] * dt
                mag = min(abs(tgt[i]), abs(self._cur[i]) + step)
                self._cur[i] = mag if tgt[i] > 0 else -mag
        try:
            # Sent every frame, even when 0: keeps SafeDog's dead-man switch fed.
            self.dog.set_velocity(*self._cur)
        except SafetyError as e:
            self.standing = False
            self.message = f"stopped: {e}"

    def _stop_now(self):
        self.held.clear()
        self._cur = [0.0, 0.0, 0.0]
        if self.standing:
            try:
                self.dog.set_velocity(0.0, 0.0, 0.0)
            except SafetyError:
                pass

    def state(self) -> TeleopState:
        return TeleopState(self.standing, self.level, frozenset(self.held),
                           tuple(self._cur), self.message, self.dog.fault)


class ObstacleBanner:
    """Turns the obstacle watcher's verdicts into what the teleop window shows.

    It only INFORMS. Nothing here stops the robot.

    - A warning is held for HOLD seconds after the last one, so a single missed frame
      doesn't make it flicker.
    - A verdict older than STALE seconds is not trusted: the window says so instead
      of showing "clear" for a picture that is no longer true.
    """

    HOLD = 1.0
    STALE = 1.5

    def __init__(self):
        self._last_warn = -float("inf")
        self._last_text = ""

    def update(self, res, age: float, now: float, error: str | None = None) -> tuple[str, str]:
        """(level, text). level: 'warn', 'clear', 'stale' or 'starting'."""
        if error:
            return "stale", f"obstacle check stopped: {error}"
        if res is None:
            return "starting", "obstacle check: waiting for the first picture..."
        if age > self.STALE:
            return "stale", f"obstacle check: no fresh picture ({age:.1f} s old)"
        if res.warn:
            self._last_warn = now
            what = "view blocked, something very close" if res.status == "blocked" else "collision soon"
            self._last_text = f"OBSTACLE AHEAD: {what}"
        if now - self._last_warn <= self.HOLD:
            return "warn", self._last_text
        return "clear", "path clear"
