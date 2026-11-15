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
        self.mode = MODE_STAND
        return 0

    def lieDown(self):
        self._record("lieDown")
        self.mode = MODE_PASSIVE
        self._vel = (0.0, 0.0, 0.0)
        return 0

    def passive(self):
        self._record("passive")
        self.mode = MODE_PASSIVE
        self._vel = (0.0, 0.0, 0.0)
        return 0

    def shakeHand(self):
        self._record("shakeHand")
        return 0

    def move(self, vx, vy, yaw_rate):
        self._record("move", float(vx), float(vy), float(yaw_rate))
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
