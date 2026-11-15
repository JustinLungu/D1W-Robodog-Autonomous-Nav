"""Dry run of the safety layer against a FAKE robot. No network, no robot, nothing moves.

Plays a scripted session and shows what SafeDog would send:
  stand -> drive (too fast, gets clamped) -> tiny turn (snapped to 0) -> code goes silent (dead-man)
  -> lie down.  Press SPACE at any time to try the e-stop.

    uv run python tests/dryrun.py
    uv run python tests/dryrun.py --low-battery   # battery drops mid-run -> SafeDog halts and lies down
    uv run python tests/dryrun.py --disconnect    # connection drops mid-run -> SafeDog halts
"""

from __future__ import annotations

import argparse
import threading
import time

from fake_robot import FakeHighLevel  # tests/fake_robot.py (same folder)

from robodog.cli.keys import KeyWatcher
from robodog.config import load_config
from robodog.safety import SafeDog, SafetyError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--low-battery", action="store_true", help="simulate the battery dropping mid-run")
    ap.add_argument("--disconnect", action="store_true", help="simulate losing the connection mid-run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.safety.allow_motion = True  # fake robot only: safe to enable here
    cfg.network.laptop_ip = cfg.network.laptop_ip or "127.0.0.1"  # fake robot: no robot Wi-Fi needed
    robot = FakeHighLevel(battery=80)
    dog = SafeDog(robot, cfg, allow_motion=True, log=lambda m: print(f"[safedog] {m}"))
    quit_event = threading.Event()

    def phase(title: str, seconds: float, vel=None):
        """Run for `seconds`, re-sending `vel` every 50 ms (like real code would)."""
        print(f"\n--- {title}")
        end = time.monotonic() + seconds
        last_print = 0.0
        while time.monotonic() < end and not quit_event.is_set():
            if vel is not None and not dog.fault:
                try:
                    dog.set_velocity(*vel)
                except SafetyError as e:
                    print(f"[script] refused: {e}")
                    vel = None
            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                print(f"    sent move{dog.last_sent}  mode={robot.mode}  battery={robot.battery}%  fault={dog.fault}")
            time.sleep(0.05)

    with KeyWatcher(on_estop=lambda: dog.estop("SPACE pressed"), on_quit=quit_event.set):
        dog.connect()
        dog.start()
        try:
            print(f"caps: vx<={cfg.safety.max_vx}  vy<={cfg.safety.max_vy}  yaw<={cfg.safety.max_yaw}")
            dog.stand()
            print("--- standing up (SafeDog sends nothing for 3 s)")
            while not dog.settled() and not dog.fault and not quit_event.is_set():
                time.sleep(0.1)
            phase("stand still", 1.0, (0, 0, 0))
            phase("ask for vx=1.0 m/s -> clamped", 2.0, (1.0, 0, 0))
            if args.low_battery:
                robot.battery = cfg.safety.min_battery - 5
                print(f"\n*** simulated battery drop to {robot.battery}%")
            if args.disconnect:
                robot.connected = False
                print("\n*** simulated connection loss")
            phase("ask for yaw=0.05 rad/s -> below SDK minimum, snapped to 0", 1.0, (0, 0, 0.05))
            phase("code stops updating -> dead-man zeroes speed", 1.5, None)
            if not dog.fault:
                print("\n--- lie down")
                dog.lie_down()
        except SafetyError as e:
            print(f"[script] stopped: {e}")
        finally:
            dog.close()

    moves = robot.calls_named("move")
    print(
        f"\nsummary: {len(moves)} move() calls, max vx sent = {max((m[1] for m in moves), default=0):g}, "
        f"standUp={len(robot.calls_named('standUp'))} lieDown={len(robot.calls_named('lieDown'))} "
        f"passive={len(robot.calls_named('passive'))}  final fault: {dog.fault}"
    )


if __name__ == "__main__":
    main()
