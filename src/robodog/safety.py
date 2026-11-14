"""SafeDog: the only way our code should talk to the robot.

What it guarantees:
- Motion is blocked unless allow_motion is enabled (config AND caller). Read-only use is the default.
- Speeds are clamped to conservative caps, and values below the SDK's minimums become exactly 0.
- Dead-man switch: if your code stops calling set_velocity() for cmd_timeout seconds, speed drops to 0.
- A steady control loop keeps feeding the robot, so its 3 s watchdog never trips by accident.
- Low battery: refuses to stand, and stops and lies down if it drops below min_battery while running.
- Lost connection or an SDK error: stops, tries to lie down, and latches a fault.
- estop(): calls passive() immediately (robot goes limp) and latches. Nothing moves until a new SafeDog.
- Stunts (jump, frontJump, backflip, shakeHand) are not exposed at all.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .config import Config

MODE_PASSIVE, MODE_STAND, MODE_MOVE = 0, 1, 3

# SDK minimum magnitudes for move(): smaller non-zero values are rejected by the robot (api_zsl-1w.md).
MIN_VX, MIN_VY, MIN_YAW = 0.05, 0.1, 0.1


class SafetyError(RuntimeError):
    """The robot is in a fault state (e-stop, low battery, lost connection...)."""


class MotionBlocked(SafetyError):
    """A motion command was attempted while motion is not allowed."""


def _shape(value: float, limit: float, minimum: float) -> float:
    """Clamp to +/-limit, and snap anything below the SDK minimum to exactly 0."""
    value = max(-limit, min(limit, float(value)))
    return 0.0 if abs(value) < minimum else value


class SafeDog:
    def __init__(
        self,
        robot,
        cfg: Config,
        allow_motion: bool = False,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ):
        self.robot = robot
        self.cfg = cfg
        self.s = cfg.safety
        # Both the config file and the caller must agree before anything moves.
        self.allow_motion = bool(allow_motion and self.s.allow_motion)
        self.clock = clock
        self.log = log

        self._lock = threading.RLock()
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_time = -float("inf")
        self._active = False          # True while standing under our control (loop sends move())
        self._fault: str | None = None
        self._estopped = False
        self._last_battery_check = -float("inf")
        self.last_sent: tuple[float, float, float] | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ connection / state
    def connect(self):
        n = self.cfg.network
        self.log(f"connecting: laptop {n.laptop_ip}:{n.port} -> robot {n.robot_ip}")
        self.robot.initRobot(n.laptop_ip, n.port, n.robot_ip)
        deadline = self.clock() + self.s.connect_timeout
        while self.clock() < deadline:
            if self.robot.checkConnect():
                self.log("connected")
                return
            time.sleep(0.1)
        raise SafetyError(
            "could not connect: check target_ip on the robot matches laptop_ip, and that the robot was rebooted"
        )

    def status(self) -> dict:
        return {
            "connected": bool(self.robot.checkConnect()),
            "battery": int(self.robot.getBatteryPower()),
            "mode": int(self.robot.getCurrentCtrlmode()),
            "rpy": list(self.robot.getRPY()),
            "fault": self._fault,
            "motion_allowed": self.allow_motion,
        }

    @property
    def fault(self) -> str | None:
        return self._fault

    # ------------------------------------------------------------------ motion API
    def _require_motion(self):
        if self._fault:
            raise SafetyError(f"robot is faulted: {self._fault}")
        if not self.allow_motion:
            raise MotionBlocked(
                "motion is disabled: set safety.allow_motion: true in config.local.yaml "
                "AND pass allow_motion=True / --allow-motion"
            )

    def stand(self):
        with self._lock:
            self._require_motion()
            battery = int(self.robot.getBatteryPower())
            if battery < self.s.min_battery:
                raise SafetyError(f"battery {battery}% is below min_battery {self.s.min_battery}%; charge first")
            self._cmd = (0.0, 0.0, 0.0)
            self._check_rc("standUp", self.robot.standUp())
            self._active = True

    def lie_down(self):
        with self._lock:
            if not self.allow_motion:
                raise MotionBlocked("motion is disabled")
            if self._estopped:
                raise SafetyError("e-stopped: robot is already limp")
            self._cmd = (0.0, 0.0, 0.0)
            self._active = False
            self._check_rc("move", self.robot.move(0.0, 0.0, 0.0))
            self._check_rc("lieDown", self.robot.lieDown())

    def set_velocity(self, vx: float, vy: float = 0.0, yaw: float = 0.0):
        """Request a velocity. Must be called again within cmd_timeout or it resets to 0."""
        with self._lock:
            self._require_motion()
            if not self._active:
                raise SafetyError("call stand() before set_velocity()")
            self._cmd = (
                _shape(vx, self.s.max_vx, MIN_VX),
                _shape(vy, self.s.max_vy, MIN_VY),
                _shape(yaw, self.s.max_yaw, MIN_YAW),
            )
            self._cmd_time = self.clock()

    def stop(self):
        """Zero velocity (always allowed)."""
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)

    def estop(self, reason: str = "manual"):
        """Emergency stop: robot goes limp right away. Latched."""
        with self._lock:
            self._estopped = True
            self._fault = f"E-STOP ({reason})"
            self._active = False
            self._cmd = (0.0, 0.0, 0.0)
        self.log(f"!!! {self._fault}: calling passive()")
        try:
            self.robot.passive()
        except Exception as e:  # never let an exception stop the e-stop path
            self.log(f"passive() raised: {e!r}")

    # ------------------------------------------------------------------ control loop
    def tick(self):
        """One control-loop step. Called by the loop thread at rate_hz (or directly in tests)."""
        with self._lock:
            if self._fault:
                return  # latched: send nothing more (robot was halted or e-stopped)
            now = self.clock()
            try:
                if not self.robot.checkConnect():
                    self._halt("connection lost")
                    return

                if now - self._last_battery_check >= self.s.battery_check_period:
                    self._last_battery_check = now
                    battery = int(self.robot.getBatteryPower())
                    if self._active and battery < self.s.min_battery:
                        self._halt(f"battery {battery}% below {self.s.min_battery}%")
                        return

                if not self._active:
                    return  # lying down / not started: send nothing

                stale = now - self._cmd_time > self.s.cmd_timeout
                cmd = (0.0, 0.0, 0.0) if stale else self._cmd
                self._check_rc("move", self.robot.move(*cmd))
                self.last_sent = cmd
            except Exception as e:
                self._halt(f"SDK error: {e!r}")

    def _halt(self, reason: str):
        """Controlled stop: zero speed, try to lie down, latch a fault."""
        self._fault = reason
        was_active = self._active
        self._active = False
        self._cmd = (0.0, 0.0, 0.0)
        self.last_sent = (0.0, 0.0, 0.0)
        self.log(f"!!! halting: {reason}")
        if was_active:
            for name, call in (("move", lambda: self.robot.move(0.0, 0.0, 0.0)), ("lieDown", self.robot.lieDown)):
                try:
                    call()
                except Exception as e:
                    self.log(f"{name}() during halt raised: {e!r}")

    def _check_rc(self, name: str, rc):
        if rc not in (0, None):
            self.log(f"warning: {name}() returned error code {rc}")

    def start(self):
        if self._thread:
            return
        self._stop_event.clear()
        period = 1.0 / self.s.rate_hz

        def run():
            next_t = time.monotonic()
            while not self._stop_event.is_set():
                self.tick()
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))

        self._thread = threading.Thread(target=run, name="safedog-loop", daemon=True)
        self._thread.start()

    def close(self):
        """Stop the robot (lie down if we stood it up) and stop the loop."""
        with self._lock:
            active = self._active
        if active and not self._estopped:
            try:
                self.lie_down()
            except Exception as e:
                self.log(f"lie_down() on close raised: {e!r}")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
