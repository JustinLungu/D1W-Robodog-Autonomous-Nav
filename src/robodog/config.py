"""Load the config: shared settings + this laptop's settings.

Two files, so a `git pull` never overwrites someone's laptop setup:

- config.local.yaml   SHARED, committed. Speeds, timeouts, robot address. Edit in place.
- config.machine.yaml THIS LAPTOP ONLY, git-ignored, created automatically on first run:
                      laptop_ip ("auto" = detect), sdk_dir, allow_motion.

The YAML files are the ONLY place config values live. This module just defines which
keys must exist and what type they are. A missing or misspelled key is an error, never a
silent default, because a silently ignored speed cap is the bug we least want.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.local.yaml"
MACHINE_PATH = REPO_ROOT / "config.machine.yaml"

# Written the first time anything runs on a laptop. Each person edits their own copy.
MACHINE_TEMPLATE = """\
# Settings for THIS laptop only. Created automatically; git ignores this file,
# so pulls never overwrite it. Shared settings live in config.local.yaml.

laptop_ip: "auto"                 # "auto" = detect this laptop's address on the robot's network
sdk_dir: "../agibot_D1_Edu-Ultra" # vendor SDK clone; relative paths start from this file's folder
allow_motion: false               # true = this laptop may move the robot (still needs --allow-motion)
"""

# Keys that belong in config.machine.yaml, never in the shared file.
_MACHINE_KEYS = {"laptop_ip": "str", "sdk_dir": "str", "allow_motion": "bool"}


class ConfigError(ValueError):
    """A config file is missing a key, has an unknown key, or has a wrong type."""


@dataclass
class NetworkConfig:
    laptop_ip: str  # from config.machine.yaml; "" if "auto" found no address on the robot's network
    robot_ip: str
    port: int


@dataclass
class SafetyConfig:
    allow_motion: bool  # from config.machine.yaml
    max_vx: float
    max_vy: float
    max_yaw: float
    min_battery: int
    rate_hz: float
    cmd_timeout: float
    connect_timeout: float
    battery_check_period: float


@dataclass
class CameraConfig:
    """Front camera geometry, for turning a clicked pixel into a floor point (see camera_model.py)."""
    lens: str                # "fisheye" or "pinhole"
    hfov_deg: float
    height_m: float          # lens height above the floor, robot standing
    crawl_height_m: float    # the same in the low crawl posture (used with --posture crawl)
    pitch_deg: float         # downward tilt of the lens
    forward_offset_m: float  # lens distance in front of the robot's centre
    measured: bool           # false until height/offset are measured and pitch calibrated


@dataclass
class CrawlConfig:
    """Crawl = the SDK's low crawl gait (SafeDog.set_crawl)."""
    settle_s: float          # pause after standUp() before the first crawl()
    change_s: float          # seconds of crawl(0, 0, 0) while the body goes down
    max_vx: float            # speed caps while crawling (the safety caps still apply on top)
    max_vy: float
    max_yaw: float


@dataclass
class GoToConfig:
    """Limits for driving to a clicked goal (robodog-goto)."""
    max_goal_m: float        # refuse goals further away than this
    max_vx: float            # m/s, below safety.max_vx
    max_yaw: float           # rad/s, below safety.max_yaw
    arrive_m: float          # "arrived" when this close
    timeout_s: float         # give up after this much driving time
    stop_before_m: float     # stop the robot's middle this far short of the goal, facing it
    odometry: str            # "robot" (getPosition) or "commanded" (speed x time along the IMU heading)


@dataclass
class TrashConfig:
    """Trash detection + tracking, and following a tracked piece with go-to (vision/trash.py, target.py)."""
    model: str               # YOLO weights: a local .pt file, or "hf://<repo>/<file>" on Hugging Face
    classes: str             # comma-separated class names to keep ("" = all the model knows)
    conf: float              # confidence needed to START tracking a piece (0-1)
    imgsz: int               # detector input size in pixels (bigger = finds smaller things, slower)
    follow_vx: float         # m/s while driving to a tracked target (slow, so the tracker keeps up)
    crawl_to_target: bool    # lower the body (crawl) before following a tracked target
    camera_delay_s: float    # how old a camera frame is when it reaches the laptop
    lost_after_s: float      # not seen for this long -> drive on blind (odometry only)
    reacquire_m: float       # a new track this close to the goal is taken as the same object


@dataclass
class ObstaclesConfig:
    """Camera-based obstacle warning (vision/obstacles.py explains each value)."""
    path_left: float
    path_right: float
    vanish_x: float
    height_ratio: float
    warn_row: float


@dataclass
class Config:
    network: NetworkConfig
    safety: SafetyConfig
    camera: CameraConfig
    crawl: CrawlConfig
    goto: GoToConfig
    trash: TrashConfig
    obstacles: ObstaclesConfig
    sdk_dir: Path  # from config.machine.yaml


def _check_type(value, type_name: str) -> bool:
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


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
        if not _check_type(value, type_name):
            raise ConfigError(f"{source}: '{section}.{name}' should be {type_name}, got {value!r}")
        values[name] = float(value) if type_name == "float" else value
    return cls(**values)


def detect_laptop_ip(robot_ip: str) -> str:
    """This laptop's address on the robot's network, or "" if it isn't on it.

    Asks the OS which local address it would use to reach the robot (a UDP "connect"
    sends no packets), then checks it's in the same /24 network as the robot.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((robot_ip, 9))
            ip = s.getsockname()[0]
    except OSError:
        return ""
    same_net = ipaddress.ip_address(ip) in ipaddress.ip_network(f"{robot_ip}/24", strict=False)
    return ip if same_net else ""


def _load_machine(path: Path) -> dict:
    """Read config.machine.yaml (creating it with safe defaults if it's the default path and missing)."""
    if not path.exists():
        if path != MACHINE_PATH:
            raise ConfigError(f"{path}: not found")
        path.write_text(MACHINE_TEMPLATE)
        print(f"created {path.name} for this laptop (motion off). Edit it to change laptop settings.")
    raw = yaml.safe_load(path.read_text()) or {}
    missing = _MACHINE_KEYS.keys() - raw.keys()
    unknown = raw.keys() - _MACHINE_KEYS.keys()
    if missing:
        raise ConfigError(f"{path}: missing {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)} (typo?)")
    for name, type_name in _MACHINE_KEYS.items():
        if not _check_type(raw[name], type_name):
            raise ConfigError(f"{path}: '{name}' should be {type_name}, got {raw[name]!r}")
    return raw


def load_config(path: str | Path | None = None, machine_path: str | Path | None = None) -> Config:
    """Read the shared config.local.yaml and this laptop's config.machine.yaml."""
    path = Path(path or CONFIG_PATH).resolve()
    machine_path = Path(machine_path or MACHINE_PATH).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    machine = _load_machine(machine_path)

    unknown = raw.keys() - {"network", "safety", "camera", "crawl", "goto", "trash", "obstacles"}
    misplaced = unknown & _MACHINE_KEYS.keys()
    misplaced |= {k for k in ("laptop_ip",) if isinstance(raw.get("network"), dict) and k in raw["network"]}
    misplaced |= {k for k in ("allow_motion",) if isinstance(raw.get("safety"), dict) and k in raw["safety"]}
    if misplaced:
        raise ConfigError(f"{path}: {sorted(misplaced)} belong in {MACHINE_PATH.name} (per laptop), not the shared file")
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys {sorted(unknown)} (typo?)")

    network = dict(raw.get("network") or {})
    safety = dict(raw.get("safety") or {})
    net = _build(NetworkConfig, {**network, "laptop_ip": ""}, "network", path)
    net.laptop_ip = detect_laptop_ip(net.robot_ip) if machine["laptop_ip"] == "auto" else machine["laptop_ip"]

    sdk_dir = Path(machine["sdk_dir"])
    if not sdk_dir.is_absolute():
        sdk_dir = (machine_path.parent / sdk_dir).resolve()

    return Config(
        network=net,
        safety=_build(SafetyConfig, {**safety, "allow_motion": machine["allow_motion"]}, "safety", path),
        camera=_build(CameraConfig, raw.get("camera"), "camera", path),
        crawl=_build(CrawlConfig, raw.get("crawl"), "crawl", path),
        goto=_build(GoToConfig, raw.get("goto"), "goto", path),
        trash=_build(TrashConfig, raw.get("trash"), "trash", path),
        obstacles=_build(ObstaclesConfig, raw.get("obstacles"), "obstacles", path),
        sdk_dir=sdk_dir,
    )
