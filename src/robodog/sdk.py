"""Load the vendor's wheeled-robot SDK module (mc_sdk_zsl_1w_py) from the SDK clone."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

MODULE_NAME = "mc_sdk_zsl_1w_py"  # zsl-1w = wheeled. Never zsl-1 (legged).


def load_sdk_module(sdk_dir: Path):
    arch = platform.machine().replace("amd64", "x86_64").replace("arm64", "aarch64")
    lib_dir = Path(sdk_dir) / "lib" / "zsl-1w" / arch
    if not lib_dir.is_dir():
        raise FileNotFoundError(
            f"SDK library folder not found: {lib_dir}\n"
            "Clone https://github.com/AgibotTech/agibot_D1_Edu-Ultra next to this repo "
            "or set sdk_dir in config.local.yaml."
        )
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"The SDK only loads on Python 3.10, this is {platform.python_version()}. "
            "Run through `uv run`."
        )
    sys.path.insert(0, str(lib_dir))
    return importlib.import_module(MODULE_NAME)


def make_highlevel(sdk_dir: Path):
    """Return a real HighLevel() object from the vendor SDK."""
    return load_sdk_module(sdk_dir).HighLevel()
