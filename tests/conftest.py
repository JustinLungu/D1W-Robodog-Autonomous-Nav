import pytest

import robodog.safety as safety


@pytest.fixture(autouse=True)
def no_estop_pause(monkeypatch):
    """The e-stop waits ESTOP_STOP_S (real time) between stopping and lying down; skip it in tests."""
    monkeypatch.setattr(safety, "ESTOP_STOP_S", 0.0)


@pytest.fixture(autouse=True)
def short_lie_down(monkeypatch):
    """Real robot: crouch, wait LIE_DOWN_S (3 s), then damping. Tests use a few milliseconds."""
    monkeypatch.setattr(safety, "LIE_DOWN_S", 0.02)
