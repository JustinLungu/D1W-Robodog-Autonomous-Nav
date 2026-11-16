"""Getting around obstacles on the way to a goal (robodog-goto --avoid).

- backoff.py  only when the forward block fires on a real obstacle: back up, turn away from the red,
              drive on a bit, then go-to again. No map, no memory, no position estimate.

The safety layers don't live here and stay in charge: a person stops everything, and SafeDog's
forward block is the last line, whatever an avoider decides.
"""

from .backoff import BackoffAvoider, BackoffCmd, BackoffSettings, red_side

METHODS = ("none", "backoff")

__all__ = ["METHODS", "BackoffAvoider", "BackoffCmd", "BackoffSettings", "red_side"]
