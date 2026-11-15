"""SafeDog: safety wrapper between our code and the robot. All robot access goes through it."""

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

# standUp() takes a few seconds, and the vendor says not to send other commands during the
# transition. Their demo waits 3 s after standUp() before anything else, so we do the same:
# the loop sends nothing, and set_velocity() is refused, until this much time has passed.
STAND_SETTLE_S = 3.0

# shakeHand() is a ~4 s trick. The vendor demo sends nothing for 4 s and then calls standUp()
# again; we copy that exactly (then the usual STAND_SETTLE_S before motion is accepted).
SHAKE_HAND_S = 4.0

# E-stop: zero speed first, wait this long, then lieDown() (the vendor says lieDown() must not
# be switched to while moving). Their own remote's "emergency stop" (LB+RB) is a lie-down too.
ESTOP_STOP_S = 0.3

# Lying down fully takes two steps on this robot: lieDown() only crouches (mode 51); damping
# (passive(), mode 0) then lets it settle all the way down. The vendor demo's exit does exactly
# that: lieDown(), wait 3 s, passive(). Their remote also has "damping (slowly lie down)".
LIE_DOWN_S = 3.0

# Crawl = the SDK's own low crawl gait, crawl(vx, vy, yaw). The library has a state machine
# (its messages are in libmc_sdk_zsl_1w): crawl() is only accepted right after standUp()
# ("must transition to 'standUp' first"; after move() the robot counts as "running", mode 18),
# and while crawling move() is refused ("in crawl state") until cancelCrawl(). Our loop sends
# move() all the time as the keepalive, which is why a plain switch to crawl() did nothing.
# So set_crawl(True):  standUp(), send nothing for crawl.settle_s, then crawl(0, 0, 0) for
#                      crawl.change_s (the body goes down), then drive with crawl(), never move().
# set_crawl(False):    cancelCrawl(), standUp(), send nothing for STAND_SETTLE_S, then move().
# Motion is refused during both changes (settled() is False). crawl()'s minimum forward speed
# is 0.1 m/s (move(): 0.05), see the vendor docs section 1.7.
MIN_CRAWL_VX = 0.1


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
        self._settled_at = float("inf")      # when the current stand-up transition is over
        self._restand_at: float | None = None  # after a handshake: when to call standUp() again
        self._passive_timer: threading.Timer | None = None  # pending "fully down" step after lieDown()
        self._last_posture = ("start", self.clock())  # last stand/lie/handshake command + when (for warnings)
        self._crawling = False               # in the crawl gait: the loop sends crawl() instead of move()
        self._crawl_target = False           # where the current posture change is heading
        self._body_until: float | None = None  # while not None: posture change in progress (motion refused)
        self._body_phase: str | None = None  # "prep" (after standUp), "lower" (sending crawl(0,0,0)), "raise"
        self._crawl_ok = False               # crawl() accepted at least once during "lower"
        self._crawl_rc = None                # first error code crawl() returned during "lower"
        self.crawl_error: str | None = None  # why the last crawl attempt was given up (for the UI)
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
        if not n.laptop_ip:
            raise SafetyError("this laptop isn't on the robot's network: join its Wi-Fi (or set laptop_ip in config.machine.yaml)")
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
            "position": list(self.robot.getPosition()),        # x, y, z in m (robot's own estimate, drifts)
        }

    def pose(self) -> tuple[float, float, float]:
        """(x, y, yaw): the robot's own position estimate (m) and heading (rad, + = turned left).

        Read-only. Odometry: good over a few metres, drifts over longer drives.
        """
        x, y = self.robot.getPosition()[:2]
        return float(x), float(y), float(self.robot.getRPY()[2])

    def settled(self) -> bool:
        """True once a stand() has finished its STAND_SETTLE_S transition (motion commands accepted)."""
        with self._lock:
            # A body-height change counts as done only once tick() has applied it, not when its time is up.
            return self._active and self.clock() >= self._settled_at and self._body_until is None

    @property
    def crawling(self) -> bool:
        """True while the body is lowered (or being lowered): crawl speed caps apply."""
        with self._lock:
            return self._crawling or (self._body_until is not None and self._crawl_target)

    @property
    def body_moving(self) -> bool:
        """True while the body is being lowered or raised (motion is refused meanwhile)."""
        with self._lock:
            return self._body_until is not None

    def speed_caps(self) -> tuple[float, float, float]:
        """The speed limits in force right now (crawl caps while crawling, else the safety caps)."""
        with self._lock:
            caps = (self.s.max_vx, self.s.max_vy, self.s.max_yaw)
            if self._crawling or (self._body_until is not None and self._crawl_target):
                c = self.cfg.crawl
                caps = (min(caps[0], c.max_vx), min(caps[1], c.max_vy), min(caps[2], c.max_yaw))
            return caps

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
                "motion is disabled: set allow_motion: true in config.machine.yaml "
                "AND pass allow_motion=True / --allow-motion"
            )

    def stand(self):
        """Stand up. From now on the loop thread keeps sending move() to the robot."""
        with self._lock:
            self._require_motion()
            self._cancel_fully_down()  # standing up again within LIE_DOWN_S of a lie-down
            # The robot can cut power without warning on a low battery, so don't even start.
            battery = int(self.robot.getBatteryPower())
            if battery < self.s.min_battery:
                raise SafetyError(f"battery {battery}% is below min_battery {self.s.min_battery}%; charge first")
            self._cmd = (0.0, 0.0, 0.0)  # start standing still
            self._check_rc("standUp", self.robot.standUp())
            self._crawling = self._crawl_target = False   # standUp() is the normal height
            self._body_until = self._body_phase = None
            self._active = True          # switches the loop's keepalive on (see tick())
            self._settled_at = self.clock() + STAND_SETTLE_S  # ...but only after the stand-up transition

    def lie_down(self):
        """Normal, gentle stop: zero speed, crouch (lieDown), then LIE_DOWN_S later fully down
        (passive). The loop stops sending move(). stand() within LIE_DOWN_S cancels the last step.
        """
        with self._lock:
            if not self.allow_motion:
                raise MotionBlocked("motion is disabled")
            if self._estopped:
                # After an e-stop the robot is already going down; don't send it anything else.
                raise SafetyError("e-stopped: robot is already lying down")
            was_standing = self._active and self.clock() >= self._settled_at
            in_crawl = self._in_crawl_gait()
            self._cmd = (0.0, 0.0, 0.0)
            self._active = False         # loop stops sending move() from the next tick
            self._restand_at = None      # cancel a pending re-stand after a handshake
            self._crawling = self._crawl_target = False   # the next stand is tall
            self._body_until = self._body_phase = None
            # The vendor says lieDown() must not be called while moving, so stop first. Only if
            # it was actually standing: a crouched robot rejects move() ("transition to Move failed").
            if in_crawl:
                self._stop_crawl_gait()
            elif was_standing:
                self._check_rc("move", self.robot.move(0.0, 0.0, 0.0))
            self._check_rc("lieDown", self.robot.lieDown())
            self._schedule_fully_down()

    # --- second step of lying down: damping after the crouch -----------------------------
    def _schedule_fully_down(self):
        """In LIE_DOWN_S, switch to damping so the robot settles fully down (unless it stood up again)."""
        self._cancel_fully_down()
        timer = threading.Timer(LIE_DOWN_S, self._fully_down_now)
        timer.daemon = True
        self._passive_timer = timer
        timer.start()

    def _fully_down_now(self):
        with self._lock:
            if self._active:  # stood up again meanwhile: leave it standing
                return
            self._passive_timer = None
        try:
            self._check_rc("passive", self.robot.passive())
        except Exception as e:
            self.log(f"passive() raised: {e!r}")

    def _cancel_fully_down(self):
        if self._passive_timer:
            self._passive_timer.cancel()
            self._passive_timer = None

    def wait_fully_down(self):
        """Block until a pending "fully down" step has run (used by close(), so the program
        doesn't exit with the robot still crouched)."""
        timer = self._passive_timer
        if timer:
            timer.join(LIE_DOWN_S + 1.0)

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
            if self._body_until is not None:
                raise SafetyError("the body is still being lowered / raised (see settled())")
            if self.clock() < self._settled_at:
                raise SafetyError(f"still standing up: wait {STAND_SETTLE_S:.0f} s after stand() (see settled())")
            cap_vx, cap_vy, cap_yaw = self.speed_caps()   # crawl caps while crawling
            self._cmd = (
                _shape(vx, cap_vx, MIN_CRAWL_VX if self._crawling else MIN_VX),
                _shape(vy, cap_vy, MIN_VY),
                _shape(yaw, cap_yaw, MIN_YAW),
            )
            self._cmd_time = self.clock()  # timestamp for the dead-man switch in tick()

    def set_crawl(self, on: bool):
        """Switch to the crawl gait (on) or back to normal standing (off). Only while standing STILL.

        on:  standUp(), nothing for crawl.settle_s, crawl(0, 0, 0) for crawl.change_s, then the
             loop drives with crawl() under the crawl caps. If the robot refuses every crawl(),
             it goes back to standing and crawl_error says why.
        off: cancelCrawl(), standUp(), nothing for STAND_SETTLE_S, then move() as usual.
        Motion is refused during the change (settled() is False).
        """
        with self._lock:
            self._require_motion()
            if on == self.crawling and self._body_until is None:
                return
            if not self._active or self.clock() < self._settled_at:
                raise SafetyError("crawl only while standing still (wait for the current move to finish)")
            moving = any(self._cmd) or (self.last_sent is not None and any(self.last_sent))
            if moving:
                raise SafetyError("stop first: the crawl gait can only be switched while standing still")
            c = self.cfg.crawl
            now = self.clock()
            self._cmd = (0.0, 0.0, 0.0)
            z = self._body_z()
            if on:
                self.crawl_error = None
                self._check_rc("standUp", self.robot.standUp())   # crawl() needs the standUp state
                self._crawl_target = True
                self._body_phase = "prep"
                self._body_until = now + c.settle_s
                self.log(f"crawl: standUp(), then crawl(0, 0, 0) for {c.change_s:.1f} s "
                         f"(body height now {z})")
            else:
                self._stop_crawl_gait()
                self._check_rc("standUp", self.robot.standUp())
                self._crawling = self._crawl_target = False
                self._body_phase = "raise"
                self._body_until = now + STAND_SETTLE_S
                self.log(f"crawl off: cancelCrawl(), standUp() (body height while crawling {z})")
            self._settled_at = self._body_until      # no motion until the change is done

    def _body_z(self) -> str:
        """The robot's body height estimate (getPosition z), for the log: helps to measure crawl height."""
        try:
            return f"z={float(self.robot.getPosition()[2]):.3f} m"
        except Exception:
            return "z=?"

    def _in_crawl_gait(self) -> bool:
        """True when the SDK is (or may be) in its crawl state: move() would be refused."""
        return self._crawling or self._body_phase == "lower"

    def _stop_crawl_gait(self):
        """Leave the crawl state: zero speed with crawl(), then cancelCrawl(). Never raises."""
        for name, call in (("crawl", lambda: self.robot.crawl(0.0, 0.0, 0.0)), ("cancelCrawl", self.robot.cancelCrawl)):
            try:
                call()
            except Exception as e:
                self.log(f"{name}() raised: {e!r}")

    def shake_hand(self):
        """Shake hands, exactly like the vendor demo: shakeHand(), send nothing for SHAKE_HAND_S,
        then standUp() again (and the usual STAND_SETTLE_S). Only while standing still.

        settled() is False for the whole sequence, so motion is refused until it's done.
        """
        with self._lock:
            self._require_motion()
            if not self._active or self.clock() < self._settled_at:
                raise SafetyError("shake hands only while standing still (wait for the stand-up to finish)")
            if self._crawling:
                raise SafetyError("stand tall first (X), then shake hands")
            self._cmd = (0.0, 0.0, 0.0)
            self._check_rc("shakeHand", self.robot.shakeHand())
            self._settled_at = float("inf")                      # no motion until the re-stand is done
            self._restand_at = self.clock() + SHAKE_HAND_S       # tick() calls standUp() then

    def estop(self, reason: str = "manual"):
        """EMERGENCY stop: stop now, lie down, and SafeDog never moves the robot again (latched).

        Speed goes to 0 immediately (if it was standing), then after ESTOP_STOP_S the robot
        crouches (lieDown, like the vendor remote's emergency stop LB+RB) and LIE_DOWN_S later
        settles fully down (passive). If lieDown() fails, it goes straight to passive()
        (motors go soft and the robot sinks down), so the e-stop always stops the robot.
        Works even when motion is disabled. Restart the program to move again.
        """
        # Step 1, under the lock: mark everything as stopped, so the loop thread can't
        # send another move() after this point.
        with self._lock:
            was_standing = self._active and self.clock() >= self._settled_at
            in_crawl = self._in_crawl_gait()
            self._estopped = True
            self._fault = f"E-STOP ({reason})"
            self._active = False
            self._restand_at = None
            self._body_until = self._body_phase = None
            self._crawling = self._crawl_target = False
            self._cmd = (0.0, 0.0, 0.0)
            self.last_sent = (0.0, 0.0, 0.0)  # nothing moves now; don't keep showing the old speed
        # Step 2, outside the lock: tell the robot. Outside so that a slow or stuck SDK
        # call can't block other threads that need the lock.
        self.log(f"!!! {self._fault}: stopping and lying down")
        try:
            if in_crawl:      # move() is refused in the crawl state
                self._stop_crawl_gait()
                time.sleep(ESTOP_STOP_S)
            elif was_standing:  # a crouched robot rejects move()
                self.robot.move(0.0, 0.0, 0.0)
                time.sleep(ESTOP_STOP_S)  # let it stop before switching to lying down
            rc = self.robot.lieDown()
            self._last_posture = ("lieDown", self.clock())
            if rc not in (0, None):
                raise RuntimeError(f"lieDown() returned error code {rc}")
            self._schedule_fully_down()  # crouch now, fully down in LIE_DOWN_S
        except Exception as e:  # never let an exception stop the e-stop path
            self.log(f"lying down failed ({e!r}): calling passive() instead")
            try:
                self.robot.passive()
            except Exception as e2:
                self.log(f"passive() raised: {e2!r}")

    # ------------------------------------------------------------------ control loop (LOOP thread)
    def tick(self):
        """One control-loop step. Called by the loop thread at rate_hz (or directly in tests).

        Each tick, in order:
          1. faulted?             -> do nothing (fault is latched)
          2. connection lost?     -> halt
          3. battery too low?     -> halt   (checked once per battery_check_period, not every tick)
          4. not standing?        -> send nothing
          5. changing posture?    -> crawl(0, 0, 0) while entering the crawl gait, else nothing
          6. otherwise            -> send move() (crawl() while crawling) with the latest speed,
                                     or 0 if it's stale (dead-man)
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
                if self._restand_at is not None:
                    if now < self._restand_at:
                        return  # handshake trick still running: send nothing
                    self._restand_at = None
                    self._check_rc("standUp", self.robot.standUp())  # same as the vendor demo
                    self._settled_at = now + STAND_SETTLE_S
                    return
                if self._body_until is not None:
                    if self._body_phase == "lower":
                        # Entering the crawl gait (the body goes down); also the keepalive meanwhile.
                        rc = self.robot.crawl(0.0, 0.0, 0.0)
                        self.last_sent = (0.0, 0.0, 0.0)
                        if rc in (0, None):
                            self._crawl_ok = True
                        elif self._crawl_rc is None:
                            self._crawl_rc = rc
                            self._check_rc("crawl", rc)    # log the first refusal only, not 30 a second
                    if now < self._body_until:
                        return  # "prep" / "raise": like a stand-up, send nothing meanwhile
                    if self._body_phase == "prep":
                        self._body_phase = "lower"
                        self._crawl_ok, self._crawl_rc = False, None
                        self._body_until = self._settled_at = now + self.cfg.crawl.change_s
                        return
                    if self._body_phase == "lower" and not self._crawl_ok:
                        # The robot refused every crawl(): back to normal standing.
                        self.crawl_error = f"the robot refused crawl() (error code {self._crawl_rc}): standing tall again"
                        self.log(f"warning: {self.crawl_error}")
                        self._stop_crawl_gait()
                        self._check_rc("standUp", self.robot.standUp())
                        self._crawling = self._crawl_target = False
                        self._body_phase = "raise"
                        self._body_until = self._settled_at = now + STAND_SETTLE_S
                        return
                    if self._body_phase == "lower":
                        self.log(f"crawling ({self._body_z()}, robot mode {self.robot.getCurrentCtrlmode()})")
                    self._crawling = self._crawl_target
                    self._body_until = self._body_phase = None
                    self._cmd = (0.0, 0.0, 0.0)          # start the new posture standing still
                    self._settled_at = now
                if now < self._settled_at:
                    return  # still standing up: don't interrupt the transition

                # Dead-man switch: if the script hasn't called set_velocity() recently
                # (it crashed, froze, or forgot), send 0 instead of the old speed.
                stale = now - self._cmd_time > self.s.cmd_timeout
                cmd = (0.0, 0.0, 0.0) if stale else self._cmd
                # Sending every tick, even when it's (0,0,0), is also the keepalive:
                # the robot lies down by itself after 3 s without commands.
                if self._crawling:
                    self._check_rc("crawl", self.robot.crawl(*cmd))   # move() is refused while crawling
                else:
                    self._check_rc("move", self.robot.move(*cmd))
                self.last_sent = cmd
            except Exception as e:
                # Any surprise from the SDK (network error, crash in the library...)
                # is treated as a fault, never ignored.
                self._halt(f"SDK error: {e!r}")

    def _halt(self, reason: str):
        """Controlled stop after a problem: zero speed, try to lie down, latch a fault.

        Like estop(), but triggered by SafeDog itself (lost connection, low battery, SDK error).
        Called from tick(), which already holds the lock.
        """
        self._fault = reason
        was_active = self._active
        in_crawl = self._in_crawl_gait()
        self._active = False
        self._restand_at = None
        self._body_until = self._body_phase = None
        self._crawling = self._crawl_target = False
        self._cmd = (0.0, 0.0, 0.0)
        self.last_sent = (0.0, 0.0, 0.0)
        self.log(f"!!! halting: {reason}")
        if was_active:
            # Try both steps even if the first one fails (e.g. connection is gone).
            # If they can't reach the robot, its own 3 s watchdog lays it down anyway.
            if in_crawl:
                self._stop_crawl_gait()   # move() is refused in the crawl state
            for name, call in (("move", lambda: self.robot.move(0.0, 0.0, 0.0)), ("lieDown", self.robot.lieDown)):
                try:
                    call()
                except Exception as e:
                    self.log(f"{name}() during halt raised: {e!r}")
            self._schedule_fully_down()

    # Posture commands, remembered so error warnings can say "x s after standUp".
    _POSTURE = ("standUp", "lieDown", "passive", "shakeHand")

    def _check_rc(self, name: str, rc):
        """SDK commands return 0 for OK, anything else is an error code. Log those, with the
        robot's mode and how long after the last posture command it happened (to find out why)."""
        now = self.clock()
        if rc not in (0, None):
            last, at = self._last_posture
            try:
                mode = int(self.robot.getCurrentCtrlmode())
            except Exception:
                mode = "?"
            self.log(f"warning: {name}() returned error code {rc} (robot mode {mode}, {now - at:.2f} s after {last})")
        if name in self._POSTURE:
            self._last_posture = (name, now)

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
        self.wait_fully_down()  # up to LIE_DOWN_S: don't exit with the robot still crouched
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
