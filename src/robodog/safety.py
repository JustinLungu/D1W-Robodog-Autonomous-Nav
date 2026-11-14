"""SafeDog: the only way our code should talk to the robot.

Why this exists
---------------
The vendor SDK will do whatever you tell it: full speed, backflips, keep driving after
your program crashed. It has no battery protection under SDK control. SafeDog sits
between our code and the robot and enforces the rules below.

    our script  ──►  SafeDog  ──►  robot object (real SDK from sdk.py, or FakeHighLevel from fake.py)

What it guarantees:
- Motion is blocked unless allow_motion is enabled (config AND caller). Read-only use is the default.
- Speeds are clamped to conservative caps, and values below the SDK's minimums become exactly 0.
- Dead-man switch: if your code stops calling set_velocity() for cmd_timeout seconds, speed drops to 0.
- A steady control loop keeps feeding the robot, so its 3 s watchdog never trips by accident.
- Low battery: refuses to stand, and stops and lies down if it drops below min_battery while running.
- Lost connection or an SDK error: stops, tries to lie down, and latches a fault.
- estop(): calls passive() immediately (robot goes limp) and latches. Nothing moves until a new SafeDog.
- Stunts (jump, frontJump, backflip, shakeHand) are not exposed at all.

"Latched" means the fault sticks: once set, SafeDog refuses all motion for the rest of
its life. To move again you stop the program, check what happened, and start a new one.
That way a problem can't be silently "forgotten" a moment later.

Two threads use SafeDog at the same time
----------------------------------------
1. YOUR code (main thread): calls stand(), set_velocity(), lie_down(), estop()...
   These only change SafeDog's internal state ("what we want").
2. The LOOP thread (started by start()): calls tick() 30 times a second. tick() is the
   one that actually sends move() to the robot ("what we send").
(A third, the keyboard thread in cli/keys.py, may call estop() at any moment.)

Typical use:
    with SafeDog(robot, cfg, allow_motion=True) as dog:
        dog.connect()                   # 1. open the connection
        dog.start()                     # 2. start the 30 Hz loop
        dog.stand()                     # 3. stand up; from now on the loop sends move()
        dog.set_velocity(0.2, 0, 0)     # 4. repeat at least every cmd_timeout seconds
    # leaving the `with` block calls close(): lie down + stop the loop
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .config import Config

# The SDK's move() ignores speeds that are non-zero but smaller than these
# (see agibot_D1_Edu-Ultra/docs/api_zsl-1w.md, section 1.6):
#   vx (forward) >= 0.05 m/s,  vy (sideways) >= 0.1 m/s,  yaw (turning) >= 0.1 rad/s
# We turn such values into exactly 0, so what we send is always something the robot accepts.
MIN_VX, MIN_VY, MIN_YAW = 0.05, 0.1, 0.1


class SafetyError(RuntimeError):
    """The robot is in a fault state (e-stop, low battery, lost connection...)."""


class MotionBlocked(SafetyError):
    """A motion command was attempted while motion is not allowed."""


def _shape(value: float, limit: float, minimum: float) -> float:
    """Clamp to +/-limit, and snap anything below the SDK minimum to exactly 0.

    Examples with limit=0.3, minimum=0.05:
        5.0  -> 0.3    (too fast: capped)
       -5.0  -> -0.3   (too fast backwards: capped)
        0.2  -> 0.2    (fine)
        0.04 -> 0.0    (too small for the SDK: becomes a clean stop)
    """
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
        # `robot` is anything with the vendor's method names: the real HighLevel object,
        # or FakeHighLevel in tests and dry runs. SafeDog doesn't know or care which.
        self.robot = robot
        self.cfg = cfg
        self.s = cfg.safety  # short alias, used everywhere below

        # Two keys are needed to move: the config file AND the code creating SafeDog.
        # If either one says no, every motion call raises MotionBlocked.
        self.allow_motion = bool(allow_motion and self.s.allow_motion)

        # `clock` tells the time. Normally time.monotonic (a clock that never jumps,
        # unlike wall-clock time). Tests pass a fake clock so they can "advance time"
        # instantly instead of actually waiting.
        self.clock = clock
        self.log = log

        # The lock makes sure only one thread at a time reads/changes the state below.
        # Without it, e.g. an e-stop could happen halfway through tick(), and tick()
        # would still send one last move() after the robot was told to go limp.
        # RLock (re-entrant lock) instead of a plain Lock: if the thread that already
        # holds it asks again, it's let in instead of freezing forever. No method does
        # that today, but a freeze here would also freeze the keyboard e-stop, so we
        # rule that bug out up front.
        self._lock = threading.RLock()

        # --- state, all protected by self._lock ---
        self._cmd = (0.0, 0.0, 0.0)          # latest speed the script asked for (vx, vy, yaw)
        self._cmd_time = -float("inf")       # when it asked ("-inf" = never, so it starts stale)
        self._active = False                 # True while standing under our control (loop sends move())
        self._fault: str | None = None       # reason for a latched fault, or None if all is fine
        self._estopped = False               # True after estop(): robot is limp, never touch it again
        self._last_battery_check = -float("inf")  # forces a battery check on the first tick
        self.last_sent: tuple[float, float, float] | None = None  # last move() actually sent (for display)

        # --- the loop thread ---
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()  # set() tells the loop thread to finish

    # ------------------------------------------------------------------ connection / state
    def connect(self):
        """Open the connection and wait until the robot answers (or give up after connect_timeout)."""
        n = self.cfg.network
        self.log(f"connecting: laptop {n.laptop_ip}:{n.port} -> robot {n.robot_ip}")
        # initRobot() opens the network connection. It returns immediately, so we then
        # poll checkConnect() until the robot actually answers.
        self.robot.initRobot(n.laptop_ip, n.port, n.robot_ip)
        deadline = self.clock() + self.s.connect_timeout
        while self.clock() < deadline:
            if self.robot.checkConnect():
                self.log("connected")
                return
            time.sleep(0.1)
        # The usual cause: the robot is sending its data to a different address.
        raise SafetyError(
            "could not connect: check target_ip on the robot matches laptop_ip, and that the robot was rebooted"
        )

    def status(self) -> dict:
        """Read-only snapshot of the robot. Never sends anything that could move it."""
        return {
            "connected": bool(self.robot.checkConnect()),
            "battery": int(self.robot.getBatteryPower()),      # 0-100 %
            "mode": int(self.robot.getCurrentCtrlmode()),      # vendor doc: 0 limp, 1 standing, 3 moving (unverified)
            "rpy": list(self.robot.getRPY()),                  # roll, pitch, yaw in radians
        }

    @property
    def fault(self) -> str | None:
        """Why SafeDog stopped (e.g. "connection lost"), or None if everything is fine."""
        return self._fault

    # ------------------------------------------------------------------ motion API (called by YOUR code)
    def _require_motion(self):
        """Gate in front of every motion command: no fault, and motion allowed."""
        if self._fault:
            raise SafetyError(f"robot is faulted: {self._fault}")
        if not self.allow_motion:
            raise MotionBlocked(
                "motion is disabled: set safety.allow_motion: true in config.local.yaml "
                "AND pass allow_motion=True / --allow-motion"
            )

    def stand(self):
        """Stand up. From now on the loop thread keeps sending move() to the robot."""
        with self._lock:
            self._require_motion()
            # The robot can cut power without warning on a low battery, so don't even start.
            battery = int(self.robot.getBatteryPower())
            if battery < self.s.min_battery:
                raise SafetyError(f"battery {battery}% is below min_battery {self.s.min_battery}%; charge first")
            self._cmd = (0.0, 0.0, 0.0)  # start standing still
            self._check_rc("standUp", self.robot.standUp())
            self._active = True          # switches the loop's keepalive on (see tick())

    def lie_down(self):
        """Normal, gentle stop: zero speed, then lie down. The loop stops sending move()."""
        with self._lock:
            if not self.allow_motion:
                raise MotionBlocked("motion is disabled")
            if self._estopped:
                # After an e-stop the robot is limp; don't send it anything else.
                raise SafetyError("e-stopped: robot is already limp")
            self._cmd = (0.0, 0.0, 0.0)
            self._active = False         # loop stops sending move() from the next tick
            # The vendor says lieDown() must not be called while moving, so stop first.
            self._check_rc("move", self.robot.move(0.0, 0.0, 0.0))
            self._check_rc("lieDown", self.robot.lieDown())

    def set_velocity(self, vx: float, vy: float = 0.0, yaw: float = 0.0):
        """Ask for a speed. Must be called again within cmd_timeout or it resets to 0.

        This does NOT send anything to the robot by itself. It only stores the (capped)
        wish. The loop thread picks it up on its next tick() and sends it.

        vx  = forward (+) / backward (-) in m/s
        vy  = left (+) / right (-) in m/s
        yaw = turn left (+) / right (-) in rad/s
        """
        with self._lock:
            self._require_motion()
            if not self._active:
                raise SafetyError("call stand() before set_velocity()")
            self._cmd = (
                _shape(vx, self.s.max_vx, MIN_VX),
                _shape(vy, self.s.max_vy, MIN_VY),
                _shape(yaw, self.s.max_yaw, MIN_YAW),
            )
            self._cmd_time = self.clock()  # timestamp for the dead-man switch in tick()

    def estop(self, reason: str = "manual"):
        """EMERGENCY stop: the robot goes limp right away and SafeDog never moves it again.

        Uses the SDK's passive() (damping mode): motors go soft and the robot sinks
        down where it is. Abrupt, but the fastest stop there is. For a normal stop use
        lie_down() instead. Works even when motion is disabled.
        """
        # Step 1, under the lock: mark everything as stopped, so the loop thread can't
        # send another move() after this point.
        with self._lock:
            self._estopped = True
            self._fault = f"E-STOP ({reason})"
            self._active = False
            self._cmd = (0.0, 0.0, 0.0)
        # Step 2, outside the lock: tell the robot. Outside so that a slow or stuck SDK
        # call can't block other threads that need the lock.
        self.log(f"!!! {self._fault}: calling passive()")
        try:
            self.robot.passive()
        except Exception as e:  # never let an exception stop the e-stop path
            self.log(f"passive() raised: {e!r}")

    # ------------------------------------------------------------------ control loop (LOOP thread)
    def tick(self):
        """One control-loop step. Called by the loop thread at rate_hz (or directly in tests).

        Each tick, in order:
          1. faulted?             -> do nothing (fault is latched)
          2. connection lost?     -> halt
          3. battery too low?     -> halt   (checked once per battery_check_period, not every tick)
          4. not standing?        -> send nothing
          5. otherwise            -> send move() with the latest speed, or 0 if it's stale (dead-man)
        """
        with self._lock:
            if self._fault:
                return  # latched: send nothing more (robot was halted or e-stopped)
            now = self.clock()
            try:
                if not self.robot.checkConnect():
                    self._halt("connection lost")
                    return

                # Battery check, rate-limited so we don't ask the robot 30 times a second.
                if now - self._last_battery_check >= self.s.battery_check_period:
                    self._last_battery_check = now
                    battery = int(self.robot.getBatteryPower())
                    if self._active and battery < self.s.min_battery:
                        self._halt(f"battery {battery}% below {self.s.min_battery}%")
                        return

                if not self._active:
                    return  # lying down / not started: send nothing

                # Dead-man switch: if the script hasn't called set_velocity() recently
                # (it crashed, froze, or forgot), send 0 instead of the old speed.
                stale = now - self._cmd_time > self.s.cmd_timeout
                cmd = (0.0, 0.0, 0.0) if stale else self._cmd
                # Sending every tick, even when it's (0,0,0), is also the keepalive:
                # the robot lies down by itself after 3 s without commands.
                self._check_rc("move", self.robot.move(*cmd))
                self.last_sent = cmd
            except Exception as e:
                # Any surprise from the SDK (network error, crash in the library...)
                # is treated as a fault, never ignored.
                self._halt(f"SDK error: {e!r}")

    def _halt(self, reason: str):
        """Controlled stop after a problem: zero speed, try to lie down, latch a fault.

        Gentler than estop(): the robot lies down instead of going limp.
        Called from tick(), which already holds the lock.
        """
        self._fault = reason
        was_active = self._active
        self._active = False
        self._cmd = (0.0, 0.0, 0.0)
        self.last_sent = (0.0, 0.0, 0.0)
        self.log(f"!!! halting: {reason}")
        if was_active:
            # Try both steps even if the first one fails (e.g. connection is gone).
            # If they can't reach the robot, its own 3 s watchdog lays it down anyway.
            for name, call in (("move", lambda: self.robot.move(0.0, 0.0, 0.0)), ("lieDown", self.robot.lieDown)):
                try:
                    call()
                except Exception as e:
                    self.log(f"{name}() during halt raised: {e!r}")

    def _check_rc(self, name: str, rc):
        """SDK commands return 0 for OK, anything else is an error code. Log those."""
        if rc not in (0, None):
            self.log(f"warning: {name}() returned error code {rc}")

    def start(self):
        """Start the loop thread that calls tick() rate_hz times a second."""
        if self._thread:
            return  # already running
        self._stop_event.clear()
        period = 1.0 / self.s.rate_hz  # e.g. 30 Hz -> every 0.033 s

        def run():
            # Schedule ticks on a fixed timetable (next_t += period) instead of
            # "sleep(period) after each tick", so slow ticks don't make the rate drift.
            next_t = time.monotonic()
            while not self._stop_event.is_set():
                self.tick()
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))

        # daemon=True: if the main program exits, this thread doesn't keep it alive.
        # (The robot then gets no more commands and its 3 s watchdog lays it down.)
        self._thread = threading.Thread(target=run, name="safedog-loop", daemon=True)
        self._thread.start()

    def close(self):
        """Clean shutdown: lie down if we stood the robot up, then stop the loop thread."""
        with self._lock:
            active = self._active  # read under the lock, then release it before lie_down()
        if active and not self._estopped:
            try:
                self.lie_down()
            except Exception as e:
                self.log(f"lie_down() on close raised: {e!r}")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)  # wait (max 2 s) for the loop to finish its last tick
            self._thread = None

    # `with SafeDog(...) as dog:` support: close() runs automatically when the block
    # ends, also when it ends because of an error or Ctrl+C.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False  # don't swallow the error; let it show after the robot is stopped
