"""Load the vendor's wheeled-robot SDK module (mc_sdk_zsl_1w_py) from the SDK clone next to this repo."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# zsl-1w = the WHEELED D1. The SDK also contains zsl-1 (the LEGGED D1): a different
# library with different commands. Loading the wrong one can hang or misbehave, so the
# name is hard-coded here on purpose.
MODULE_NAME = "mc_sdk_zsl_1w_py"


def load_sdk_module(sdk_dir: Path):
    """Find the vendor library for this computer and import it as a Python module."""
    # Which CPU is this? The SDK ships two builds:
    #   x86_64  = normal laptop/desktop (Intel/AMD)
    #   aarch64 = ARM boards, e.g. a Jetson or the robot's own computer
    # Some systems call them amd64/arm64 instead, so normalise the name.
    arch = platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")
    lib_dir = Path(sdk_dir) / "lib" / "zsl-1w" / arch

    # Fail early with a clear message instead of a cryptic ImportError later.
    if not lib_dir.is_dir():
        raise FileNotFoundError(
            f"SDK library folder not found: {lib_dir}\n"
            "Clone https://github.com/AgibotTech/agibot_D1_Edu-Ultra next to this repo "
            "or set sdk_dir in config.machine.yaml."
        )

    # The .so file was compiled for CPython 3.10 ("cpython-310" in its name). Any other
    # Python version can't load it. `uv run` uses 3.10 because of .python-version.
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"The SDK only loads on Python 3.10, this is {platform.python_version()}. "
            "Run through `uv run`."
        )

    # Put the library folder first on Python's search path, then import it by name.
    # (Same as writing `import mc_sdk_zsl_1w_py`, but with the name in a variable.)
    sys.path.insert(0, str(lib_dir))
    return importlib.import_module(MODULE_NAME)


def make_highlevel(sdk_dir: Path):
    """Return a real HighLevel() object from the vendor SDK.

    HighLevel is the vendor's "talk to the robot" object: standUp(), move(), getBatteryPower()...
    Creating it does NOT connect yet; SafeDog.connect() calls initRobot() for that.
    Never use this object directly in our scripts: always wrap it in SafeDog.
    """
    return load_sdk_module(sdk_dir).HighLevel()
