"""First motion test: stand up through SafeDog, hold still, lie down. Used for the remote takeover test.

    uv run robodog-stand-test --allow-motion              # hold standing for 60 s
    uv run robodog-stand-test --allow-motion --hold 120

Needs safety.allow_motion: true in config.local.yaml AND --allow-motion.
Keys while running: [SPACE] e-stop (robot goes limp)   [q] lie down and quit.
The robot never drives: SafeDog only sends move(0, 0, 0) to keep it standing.
"""

from __future__ import annotations

import argparse
import threading
import time

from ..config import load_config
from ..safety import SafeDog, SafetyError
from ..sdk import make_highlevel
from .keys import KeyWatcher
from .status import MODES


def run_stand_test(dog: SafeDog, hold_s: float, quit_event: threading.Event, print_fn=print):
    """Stand, print status once a second for hold_s seconds (or until quit), then lie down.

    Split out from main() so the tests can run it against the fake robot.
    """
    dog.start()  # 30 Hz loop: battery/connection checks + move(0,0,0) keepalive once standing
    dog.stand()  # raises SafetyError if battery < min_battery
    print_fn(f"standing. holding for {hold_s:.0f} s. Try the remote takeover now (LT+RT+A / L2+R2+A).")
    end = time.monotonic() + hold_s
    while time.monotonic() < end and not quit_event.is_set() and not dog.fault:
        st = dog.status()
        print_fn(
            f"  t-{end - time.monotonic():4.0f}s  battery={st['battery']}%  "
            f"mode={st['mode']} ({MODES.get(st['mode'], '?')})  sent=move{dog.last_sent}"
        )
        quit_event.wait(1.0)  # sleep 1 s, but wake up immediately on [q]
    if dog.fault:
        print_fn(f"stopped by SafeDog: {dog.fault}")
    else:
        print_fn("lying down...")
    dog.close()  # lies down (unless e-stopped) and stops the loop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--allow-motion", action="store_true", help="required: confirms you want the robot to move")
    ap.add_argument("--hold", type=float, default=60.0, help="seconds to stay standing (default 60)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not (args.allow_motion and cfg.safety.allow_motion):
        raise SystemExit("motion needs BOTH safety.allow_motion: true in config.local.yaml AND --allow-motion")

    dog = SafeDog(make_highlevel(cfg.sdk_dir), cfg, allow_motion=True)
    dog.connect()
    st = dog.status()
    print(f"battery={st['battery']}%  mode={st['mode']} ({MODES.get(st['mode'], '?')})")

    # Last chance to abort, before the keyboard switches to single-key mode.
    input("\nThe robot will STAND UP. Clear 2 m around it, remote in hand. ENTER = go, Ctrl+C = abort: ")

    quit_event = threading.Event()
    with KeyWatcher(on_estop=lambda: dog.estop("SPACE pressed"), on_quit=quit_event.set):
        try:
            run_stand_test(dog, args.hold, quit_event)
        except SafetyError as e:
            print(f"refused: {e}")
            dog.close()
        except KeyboardInterrupt:
            print("\nCtrl+C: lying down...")
            dog.close()


if __name__ == "__main__":
    main()
