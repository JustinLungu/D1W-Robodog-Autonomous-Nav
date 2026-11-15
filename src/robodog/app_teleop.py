"""Keyboard teleop over the vendor APP protocol (app_remote.py): no SDK, no SafeDog.

Separate from teleop.py on purpose: it's the only way to drive in the robot's real low crawl
(RL_crawl), which the SDK can't reach (the firmware forces SDK commands into RL_GaitWalk).

Same feel and limits as robodog-teleop:
- Moves only while a key is HELD; releasing every key stops the sticks at once (ramped starts).
- Speed caps from config.local.yaml: safety.max_* standing, crawl.max_* while crawling;
  speed levels 1/2/3 = 33 / 66 / 100 % of the caps.
- Sticks go to zero when the robot's state stream is stale (> STATE_STALE_S), during a
  posture change, after an e-stop (latched), and when the window loses focus.
- Posture changes (stand / crawl / stand tall) only while standing still.
"""

from __future__ import annotations

from dataclasses import dataclass

from .app_remote import AXES, FULL_SCALE, MAX_STICK, STATE_STALE_S, speeds_to_sticks
from .teleop import ACCEL, MOTION, SPEED_LEVELS

DRIVABLE = ("rl", "stand_down")    # control_mode values in which the sticks drive
STAND_SETTLE_S = 5.0               # app stand from lying took ~5 s on our robot
CRAWL_SETTLE_S = 3.0               # crawl down / stand tall took ~1.5-2.5 s
STILL_V = 0.05                     # m/s (and rad/s): "standing still" for posture changes
PROBE_STICK = 0.05                 # raw stick value sent by the probe keys (calibration)


@dataclass
class AppTeleopState:
    level: int
    held: frozenset
    cmd: tuple[float, float, float]
    sticks: list
    message: str
    estopped: bool
    crawling: bool
    busy: bool                     # a posture change is in progress


class AppTeleop:
    def __init__(self, remote, cfg, axes=AXES, full_scale=FULL_SCALE, max_stick=MAX_STICK, log=print):
        self.remote = remote
        self.cfg = cfg
        self.axes, self.full_scale, self.max_stick = axes, full_scale, max_stick
        self.log = log
        self.level = 1
        self.held: set[str] = set()
        self.probe: int | None = None         # raw stick axis being probed (0..3), or None
        self._cur = [0.0, 0.0, 0.0]
        self._sticks = [0.0, 0.0, 0.0, 0.0]
        self.busy_until = 0.0
        self.estopped = False
        self.quit = False
        self.message = "press R to stand up (app stand)"

    # ------------------------------------------------------------- state helpers
    @property
    def crawling(self) -> bool:
        return self.remote.state.control_mode == "stand_down"

    def caps(self) -> tuple[float, float, float]:
        s = self.cfg.safety
        caps = (s.max_vx, s.max_vy, s.max_yaw)
        if self.crawling:
            c = self.cfg.crawl
            caps = (min(caps[0], c.max_vx), min(caps[1], c.max_vy), min(caps[2], c.max_yaw))
        return caps

    def _link_ok(self, now) -> bool:
        return self.remote.state.age(now) <= STATE_STALE_S

    def _still(self) -> bool:
        st = self.remote.state
        measured = abs((st.v_body or (0, 0, 0))[0]) < STILL_V and abs((st.v_body or (0, 0, 0))[1]) < STILL_V \
            and abs(st.yaw_rate or 0.0) < STILL_V
        return not self.held and not any(self._cur) and self.probe is None and measured

    def _zero(self):
        self._cur = [0.0, 0.0, 0.0]
        self._sticks = [0.0, 0.0, 0.0, 0.0]
        self.remote.set_sticks(self._sticks)

    # ------------------------------------------------------------- keys
    def set_held(self, names):
        self.held = {n for n in names if n in MOTION}

    def release_all(self):
        self.held = set()
        self.probe = None

    def set_probe(self, axis: int | None):
        """Hold a probe key: send PROBE_STICK on one raw stick axis (to find out which is which)."""
        self.probe = axis

    def action(self, name: str, now: float):
        st = self.remote.state
        if name == "estop":
            self._zero()
            self.estopped = True
            self.remote.dump()
            self.message = "E-STOP: dump sent (lying down). Restart the program to move again"
            self.log(self.message)
            return
        if name == "quit":
            self.quit = True
            return
        if name.startswith("speed_"):
            self.level = int(name[-1])
            self.message = f"speed level {self.level}"
            return
        if self.estopped:
            self.message = "e-stopped: restart the program to move again"
            return
        if name == "lie_down":
            self._zero()
            self.remote.dump()
            self.busy_until = now + STAND_SETTLE_S
            self.message = "lying down (app dump). R to stand up"
            return
        if not self._link_ok(now):
            self.message = "no state from the robot: check the Wi-Fi (and that no SDK program runs)"
            return
        if now < self.busy_until:
            self.message = "wait: posture change in progress"
            return
        if name == "stand":
            if st.control_mode in DRIVABLE:
                self.message = "already standing"
                return
            self._zero()
            self.remote.stand()
            self.busy_until = now + STAND_SETTLE_S
            self.message = "standing up (app stand)... keys ignored ~5 s"
        elif name == "crawl":
            if st.control_mode not in DRIVABLE:
                self.message = "stand up first (R), then X to crawl"
                return
            if not self._still():
                self.message = "stop first: release the keys and let the robot stand still, then X"
                return
            self._zero()
            if self.crawling:
                self.remote.stand()
                self.message = "standing tall (app stand)... keys ignored ~3 s"
            else:
                self.remote.crawl()
                self.message = "crawling down (app crawl)... keys ignored ~3 s"
            self.busy_until = now + CRAWL_SETTLE_S

    # ------------------------------------------------------------- every frame
    def target(self) -> tuple[float, float, float]:
        frac = SPEED_LEVELS[self.level]
        caps = self.caps()
        out = [0.0, 0.0, 0.0]
        for name in self.held:
            axis, sign = MOTION[name]
            out[axis] += sign
        return tuple(d * caps[i] * frac for i, d in enumerate(out))

    def update(self, dt: float, now: float):
        """Ramp toward the target speed, turn it into sticks, hand them to the stream."""
        st = self.remote.state
        if self.estopped or not self._link_ok(now) or now < self.busy_until or st.control_mode not in DRIVABLE:
            self._zero()
            if not self.estopped and not self._link_ok(now):
                self.message = "no state from the robot for >1 s: sticks at zero"
            elif not self.estopped and now >= self.busy_until and self.message.startswith(("standing up", "crawling down", "standing tall")):
                self.message = f"robot mode: {st.control_mode}"
            return
        if self.message.startswith(("standing up", "crawling down", "standing tall", "robot mode", "no state")):
            self.message = ("CRAWLING (low). drive as usual, X to stand tall" if self.crawling
                            else "standing. hold W/A/S/D to drive, X to crawl")
        if self.probe is not None:
            self._cur = [0.0, 0.0, 0.0]
            self._sticks = [0.0, 0.0, 0.0, 0.0]
            self._sticks[self.probe] = PROBE_STICK
            self.remote.set_sticks(self._sticks)
            return
        tgt = self.target()
        for i in range(3):
            released = tgt[i] == 0.0
            reversed_ = self._cur[i] != 0.0 and (tgt[i] > 0) != (self._cur[i] > 0)
            if released or reversed_:
                self._cur[i] = 0.0
            step = ACCEL[i] * dt
            if tgt[i] > self._cur[i]:
                self._cur[i] = min(tgt[i], self._cur[i] + step)
            elif tgt[i] < self._cur[i]:
                self._cur[i] = max(tgt[i], self._cur[i] - step)
        self._sticks = speeds_to_sticks(*self._cur, axes=self.axes, full_scale=self.full_scale,
                                        max_stick=self.max_stick)
        self.remote.set_sticks(self._sticks)

    def state(self) -> AppTeleopState:
        return AppTeleopState(self.level, frozenset(self.held), tuple(self._cur), list(self._sticks),
                              self.message, self.estopped, self.crawling, False)
