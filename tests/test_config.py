"""Config loading must be strict: missing, misspelled or wrongly typed keys are errors."""

import pytest
import yaml

from robodog.config import CONFIG_PATH, ConfigError, load_config


def write_variant(tmp_path, mutate):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert cfg.network.port == 43988
    assert isinstance(cfg.safety.max_vx, float)


def test_missing_key_is_an_error(tmp_path):
    path = write_variant(tmp_path, lambda r: r["safety"].pop("max_vx"))
    with pytest.raises(ConfigError, match="missing.*max_vx"):
        load_config(path)


def test_misspelled_key_is_an_error(tmp_path):
    def typo(r):
        r["safety"]["max_vX"] = r["safety"].pop("max_vx")

    with pytest.raises(ConfigError):
        load_config(write_variant(tmp_path, typo))


def test_wrong_type_is_an_error(tmp_path):
    def quoted_bool(r):
        r["safety"]["allow_motion"] = "false"  # a string, which would be truthy

    with pytest.raises(ConfigError, match="allow_motion"):
        load_config(write_variant(tmp_path, quoted_bool))


def test_unknown_section_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(write_variant(tmp_path, lambda r: r.update(saftey={})))


def test_relative_sdk_dir_resolves_from_config_folder(tmp_path):
    path = write_variant(tmp_path, lambda r: r.update(sdk_dir="sdk"))
    assert load_config(path).sdk_dir == (tmp_path / "sdk").resolve()
