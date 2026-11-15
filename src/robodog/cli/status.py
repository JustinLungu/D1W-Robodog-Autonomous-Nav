"""Read-only robot status. Never sends a motion command.

    uv run robodog-status            # print once
    uv run robodog-status --watch    # print every 0.5 s until Ctrl+C
"""

from __future__ import annotations

import argparse
import time

from ..config import load_config
from ..safety import SafeDog
from ..sdk import make_highlevel

MODES = {0: "limp/lying", 1: "standing", 3: "moving"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--watch", action="store_true", help="keep printing until Ctrl+C")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.network.laptop_ip:
        raise SystemExit("this laptop isn't on the robot's network (join its Wi-Fi), or set laptop_ip in config.machine.yaml")

    # allow_motion=False: SafeDog refuses every motion call, and nothing here calls one anyway.
    dog = SafeDog(make_highlevel(cfg.sdk_dir), cfg, allow_motion=False)
    dog.connect()
    try:
        while True:
            st = dog.status()
            low = "  <-- LOW, charge before moving" if st["battery"] < cfg.safety.min_battery else ""
            r, p, y = st["rpy"]
            print(
                f"connected={st['connected']}  battery={st['battery']}%{low}  "
                f"mode={st['mode']} ({MODES.get(st['mode'], '?')})  rpy=({r:+.2f}, {p:+.2f}, {y:+.2f})"
            )
            if not args.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
