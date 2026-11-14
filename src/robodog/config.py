"""Load config.local.yaml, the one and only config file (committed to the repo).

The YAML file is the ONLY place config values live. This module just defines which
keys must exist and what type they are. A missing or misspelled key is an error, never a
silent default, because a silently ignored speed cap is the bug we least want.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.local.yaml"


class ConfigError(ValueError):
    """The config file is missing a key, has an unknown key, or has a wrong type."""


@dataclass
class NetworkConfig:
    laptop_ip: str
    robot_ip: str
    port: int


@dataclass
class SafetyConfig:
    allow_motion: bool
    max_vx: float
    max_vy: float
    max_yaw: float
    min_battery: int
    rate_hz: float
    cmd_timeout: float
    connect_timeout: float
    battery_check_period: float


@dataclass
class Config:
    network: NetworkConfig
    safety: SafetyConfig
    sdk_dir: Path


def _build(cls, raw, section: str, source: Path):
    """Create a dataclass from a YAML section, checking keys and types strictly."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: section '{section}' is missing or not a mapping")
    expected = {f.name: f.type for f in fields(cls)}
    missing = expected.keys() - raw.keys()
    unknown = raw.keys() - expected.keys()
    if missing:
        raise ConfigError(f"{source}: '{section}' is missing {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{source}: '{section}' has unknown keys {sorted(unknown)} (typo?)")

    values = {}
    for name, type_name in expected.items():
        value = raw[name]
        if type_name == "bool":
            ok = isinstance(value, bool)
        elif type_name == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif type_name == "float":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:  # str
            ok = isinstance(value, str)
        if not ok:
            raise ConfigError(f"{source}: '{section}.{name}' should be {type_name}, got {value!r}")
        values[name] = value
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML config (config.local.yaml unless another path is given, e.g. in tests)."""
    path = Path(path or CONFIG_PATH).resolve()
    raw = yaml.safe_load(path.read_text()) or {}

    unknown = raw.keys() - {"network", "safety", "sdk_dir"}
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys {sorted(unknown)} (typo?)")
    if not isinstance(raw.get("sdk_dir"), str):
        raise ConfigError(f"{path}: 'sdk_dir' is missing or not a string")

    sdk_dir = Path(raw["sdk_dir"])
    if not sdk_dir.is_absolute():
        sdk_dir = (path.parent / sdk_dir).resolve()

    return Config(
        network=_build(NetworkConfig, raw.get("network"), "network", path),
        safety=_build(SafetyConfig, raw.get("safety"), "safety", path),
        sdk_dir=sdk_dir,
    )
