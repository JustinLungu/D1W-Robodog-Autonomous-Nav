"""Config loading must be strict: missing, misspelled or wrongly typed keys are errors."""

import pytest
import yaml

from robodog.config import CONFIG_PATH, MACHINE_TEMPLATE, REPO_ROOT, ConfigError, detect_laptop_ip, load_config

TEST_MACHINE = REPO_ROOT / "tests" / "machine_test.yaml"


def write_shared(tmp_path, mutate):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def write_machine(tmp_path, **changes):
    raw = yaml.safe_load(TEST_MACHINE.read_text())
    raw.update(changes)
    path = tmp_path / "machine.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_config_loads():
    cfg = load_config(CONFIG_PATH, TEST_MACHINE)
    assert cfg.network.port == 43988
    assert cfg.network.laptop_ip == "127.0.0.1"
    assert cfg.safety.allow_motion is False
    assert isinstance(cfg.safety.max_vx, float)


def test_shared_config_has_safe_caps():
    cfg = load_config(CONFIG_PATH, TEST_MACHINE)
    assert cfg.safety.max_vx <= 0.3 and cfg.safety.max_yaw <= 0.5


def test_missing_key_is_an_error(tmp_path):
    path = write_shared(tmp_path, lambda r: r["safety"].pop("max_vx"))
    with pytest.raises(ConfigError, match="missing.*max_vx"):
        load_config(path, TEST_MACHINE)


def test_misspelled_key_is_an_error(tmp_path):
    def typo(r):
        r["safety"]["max_vX"] = r["safety"].pop("max_vx")

    with pytest.raises(ConfigError):
        load_config(write_shared(tmp_path, typo), TEST_MACHINE)


def test_unknown_section_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(write_shared(tmp_path, lambda r: r.update(saftey={})), TEST_MACHINE)


@pytest.mark.parametrize("mutate", [
    lambda r: r["network"].update(laptop_ip="192.168.234.7"),
    lambda r: r["safety"].update(allow_motion=True),
    lambda r: r.update(sdk_dir="../sdk"),
])
def test_per_laptop_keys_rejected_in_shared_file(tmp_path, mutate):
    with pytest.raises(ConfigError, match="config.machine.yaml|machine.yaml"):
        load_config(write_shared(tmp_path, mutate), TEST_MACHINE)


def test_machine_wrong_type_is_an_error(tmp_path):
    # a string "false" would be truthy and silently allow motion
    with pytest.raises(ConfigError, match="allow_motion"):
        load_config(CONFIG_PATH, write_machine(tmp_path, allow_motion="false"))


def test_machine_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown"):
        load_config(CONFIG_PATH, write_machine(tmp_path, alow_motion=True))


def test_machine_values_are_used(tmp_path):
    cfg = load_config(CONFIG_PATH, write_machine(tmp_path, allow_motion=True, sdk_dir="sdk", laptop_ip="10.0.0.5"))
    assert cfg.safety.allow_motion is True
    assert cfg.network.laptop_ip == "10.0.0.5"
    assert cfg.sdk_dir == (tmp_path / "sdk").resolve()  # relative to the machine file's folder


def test_template_is_safe():
    raw = yaml.safe_load(MACHINE_TEMPLATE)
    assert raw == {"laptop_ip": "auto", "sdk_dir": "../agibot_D1_Edu-Ultra", "allow_motion": False}


def test_missing_non_default_machine_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(CONFIG_PATH, tmp_path / "nope.yaml")


def test_detect_laptop_ip_rejects_other_networks():
    # 127.0.0.1 is reachable via loopback, so the local address is 127.0.0.1: same /24 -> accepted
    assert detect_laptop_ip("127.0.0.1") == "127.0.0.1"
    # TEST-NET-1 is never this laptop's network
    assert detect_laptop_ip("192.0.2.1") == ""
