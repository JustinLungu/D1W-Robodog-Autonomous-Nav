"""A fake HighLevel object with the same methods as the vendor SDK, for testing without the robot.

It records every call so tests and dry runs can show exactly what would have been sent.
"""

from __future__ import annotations

import math
import threading
import time

MODE_PASSIVE, MODE_STAND, MODE_MOVE = 0, 1, 3


class FakeHighLevel:
    def __init__(self, battery: int = 80, connected: bool = True, verbose: bool = False,
                 simulate: bool = False, sim_dt: float | None = None):
        """simulate=True: the fake robot "drives": move() speeds are integrated into its position
        and heading (getPosition / getRPY). sim_dt: fixed time step per move() call (for
        deterministic tests); None = real time between move() calls (for --fake windows)."""
        self.battery = battery
        self.connected = connected
        self.mode = MODE_PASSIVE
        self.sdk_state = None        # last accepted state command (see crawl())
        self.calls: list[tuple] = []
        self.verbose = verbose
        self._lock = threading.Lock()
        self.simulate = simulate
        self.sim_dt = sim_dt
        self.x = self.y = self.yaw = 0.0
        self._vel = (0.0, 0.0, 0.0)
        self._last_move = None

    def _integrate(self):
        """Advance the simulated pose by the last commanded speed."""
        now = time.monotonic()
        dt = self.sim_dt if self.sim_dt is not None else (0.0 if self._last_move is None else min(now - self._last_move, 0.2))
        self._last_move = now
        vx, vy, w = self._vel
        self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
        self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
        self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi

    def _record(self, *call):
        with self._lock:
            self.calls.append(call)
        if self.verbose:
            name, *args = call
            print(f"  [fake robot] {name}({', '.join(f'{a:g}' if isinstance(a, float) else repr(a) for a in args)})")

    def calls_named(self, name: str) -> list[tuple]:
        with self._lock:
            return [c for c in self.calls if c[0] == name]

    # --- connection ---
    def initRobot(self, local_ip, local_port, dog_ip="192.168.234.1"):
        self._record("initRobot", local_ip, local_port, dog_ip)

    def checkConnect(self):
        return self.connected

    # --- motion (returns 0 = OK, like the real SDK) ---
    def standUp(self):
        self._record("standUp")
        self.sdk_state = "standUp"
        self.mode = MODE_STAND
        return 0

    def lieDown(self):
        self._record("lieDown")
        self.sdk_state = "lieDown"
        self.mode = MODE_PASSIVE
        self._vel = (0.0, 0.0, 0.0)
        return 0

    def passive(self):
        self._record("passive")
        self.sdk_state = "passive"
        self.mode = MODE_PASSIVE
        self._vel = (0.0, 0.0, 0.0)
        return 0

    def attitudeControl(self, roll_vel, pitch_vel, yaw_vel, height_vel):
        self._record("attitudeControl", float(roll_vel), float(pitch_vel), float(yaw_vel), float(height_vel))
        return 0

    def shakeHand(self):
        self._record("shakeHand")
        self.sdk_state = "shakeHand"
        return 0

    # The real SDK's state machine (strings in libmc_sdk_zsl_1w): crawl() only right after
    # standUp() ("must transition to 'standUp' first"; after move() the robot is "running"),
    # and while crawling move() is refused ("in crawl state") until cancelCrawl().
    SDK_REFUSED = 12295   # the code the real library returns for a refused transition

    def crawl(self, vx, vy, yaw_rate):
        self._record("crawl", float(vx), float(vy), float(yaw_rate))
        if self.sdk_state not in ("standUp", "crawl"):
            return self.SDK_REFUSED
        self.sdk_state = "crawl"
        if self.simulate:
            with self._lock:
                self._integrate()
                self._vel = (float(vx), float(vy), float(yaw_rate))
        return 0

    def cancelCrawl(self):
        self._record("cancelCrawl")
        self._vel = (0.0, 0.0, 0.0)
        if self.sdk_state == "crawl":
            self.sdk_state = "cancelCrawl"

    def move(self, vx, vy, yaw_rate):
        self._record("move", float(vx), float(vy), float(yaw_rate))
        if self.sdk_state == "crawl":
            return self.SDK_REFUSED            # "Cannot transition to 'move' state: in crawl state."
        self.sdk_state = "move"
        if self.simulate:
            with self._lock:
                self._integrate()
                self._vel = (float(vx), float(vy), float(yaw_rate)) if self.mode != MODE_PASSIVE else (0.0, 0.0, 0.0)
        if self.mode != MODE_PASSIVE:
            self.mode = MODE_MOVE if (vx or vy or yaw_rate) else MODE_STAND
        return 0

    # --- state ---
    def getBatteryPower(self):
        return self.battery

    def getCurrentCtrlmode(self):
        return self.mode

    def getRPY(self):
        return [0.0, 0.0, self.yaw]

    def getPosition(self):
        return [self.x, self.y, 0.0]
