"""App-protocol teleop logic against FakeAppRemote (no network, no pygame)."""

import json
import socket
import time

import pytest
from test_safety import pinned_config

from robodog.app_link import CMD_CRAWL, CMD_DUMP, CMD_STAND
from robodog.app_remote import AXES, FULL_SCALE, AppRemote, FakeAppRemote, speeds_to_sticks
from robodog.app_teleop import CRAWL_SETTLE_S, STAND_SETTLE_S, AppTeleop


def make():
    cfg = pinned_config()
    remote = FakeAppRemote()
    return AppTeleop(remote, cfg, log=lambda m: None), remote, cfg


def run(tele, remote, keys, seconds, t0, dt=1 / 30):
    t = t0
    for _ in range(int(seconds / dt)):
        tele.set_held(keys)
        t += dt
        remote.tick(dt, t)
        tele.update(dt, t)
    return t


def standing(tele, remote):
    now = time.monotonic()
    tele.action("stand", now)
    return run(tele, remote, set(), STAND_SETTLE_S + 0.2, now)


def test_sticks_scale_clamp_and_map():
    s = speeds_to_sticks(0.2, 0.0, 0.0)
    idx, sign = AXES["vx"]
    assert s[idx] == pytest.approx(sign * 0.2 / FULL_SCALE["vx"])
    assert max(abs(v) for v in speeds_to_sticks(50, 50, 50)) <= 0.35   # never beyond MAX_STICK


def test_nothing_moves_before_standing():
    tele, remote, _ = make()
    run(tele, remote, {"forward"}, 1.0, time.monotonic())
    assert remote.sticks == [0, 0, 0, 0] and remote.commands == []


def test_stand_then_drive_within_caps_and_stop_on_release():
    tele, remote, cfg = make()
    t = standing(tele, remote)
    assert remote.commands == [CMD_STAND] and remote.state.control_mode == "rl"
    tele.action("speed_3", t)
    t = run(tele, remote, {"forward"}, 2.0, t)
    assert tele.state().cmd[0] == pytest.approx(cfg.safety.max_vx)
    assert remote.state.v_body[0] <= cfg.safety.max_vx + 1e-9   # fake moves exactly at the assumed scale
    run(tele, remote, set(), 1 / 30, t)
    assert remote.sticks == [0, 0, 0, 0]                        # release = stop at once


def test_crawl_needs_stillness_and_uses_crawl_caps():
    tele, remote, cfg = make()
    t = standing(tele, remote)
    t = run(tele, remote, {"forward"}, 0.5, t)
    tele.action("crawl", t)                                     # keys still held
    assert CMD_CRAWL not in remote.commands and "stop first" in tele.message
    t = run(tele, remote, set(), 0.2, t)
    tele.action("crawl", t)
    assert remote.commands[-1] == CMD_CRAWL and remote.state.control_mode == "stand_down"
    t = run(tele, remote, {"forward"}, CRAWL_SETTLE_S - 0.2, t)
    assert remote.sticks == [0, 0, 0, 0]                        # no driving during the change
    tele.action("speed_3", t)
    t = run(tele, remote, {"forward"}, 3.0, t)
    assert tele.state().cmd[0] == pytest.approx(min(cfg.crawl.max_vx, cfg.safety.max_vx))
    t = run(tele, remote, set(), 0.2, t)
    tele.action("crawl", t)
    assert remote.commands[-1] == CMD_STAND                     # X again: stand tall


def test_stale_link_zeroes_sticks():
    tele, remote, _ = make()
    t = standing(tele, remote)
    t = run(tele, remote, {"forward"}, 0.5, t)
    assert any(remote.sticks)
    remote.alive = False
    remote.state.t_rx = t - 2.0
    tele.update(1 / 30, t + 1 / 30)
    assert remote.sticks == [0, 0, 0, 0] and "no state" in tele.message


def test_estop_dumps_and_latches():
    tele, remote, _ = make()
    t = standing(tele, remote)
    tele.action("estop", t)
    assert remote.commands[-1] == CMD_DUMP and tele.estopped
    tele.action("stand", t + 1)
    run(tele, remote, {"forward"}, 1.0, t + 1)
    assert remote.commands.count(CMD_STAND) == 1 and remote.sticks == [0, 0, 0, 0]


def test_probe_sends_tiny_raw_stick_only_while_drivable():
    tele, remote, _ = make()
    tele.set_probe(2)
    tele.update(1 / 30, time.monotonic())
    assert remote.sticks == [0, 0, 0, 0]                        # lying: nothing
    t = standing(tele, remote)
    tele.set_probe(2)
    remote.tick(1 / 30, t)
    tele.update(1 / 30, t)
    assert remote.sticks == [0, 0, 0.05, 0]
    tele.set_probe(None)
    remote.tick(0.1, t + 0.1)
    tele.update(1 / 30, t + 0.1)
    assert remote.sticks == [0, 0, 0, 0]


def test_app_remote_streams_and_parses_on_localhost():
    """The real AppRemote against two local sockets standing in for the robot."""
    robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot.bind(("127.0.0.1", 0))
    robot.settimeout(1.0)
    port = robot.getsockname()[1]
    state_port = _free_port()
    remote = AppRemote("127.0.0.1", "127.0.0.1", port=port, state_port=state_port, log=lambda m: None)
    try:
        remote.set_sticks([0, 0.1, 0, 0])
        msgs = [json.loads(robot.recvfrom(65535)[0]) for _ in range(6)]
        assert {"heartbeat", "remote"} <= {m["type"] for m in msgs}
        assert any(m.get("joystick") == [0, 0.1, 0, 0] for m in msgs if m["type"] == "remote")
        remote.crawl()
        seen = [json.loads(robot.recvfrom(65535)[0]) for _ in range(10)]
        assert any(m.get("cmd") == CMD_CRAWL for m in seen)
        robot.sendto(json.dumps({"type": "odom_info", "position": [1, 2, 0.25], "rpy": [0, 0, 0.5],
                                 "v_body": [0.1, 0, 0], "omega_world": [0, 0, 0.2]}).encode(), ("127.0.0.1", state_port))
        robot.sendto(json.dumps({"type": "feedback", "control_mode": "stand_down"}).encode(), ("127.0.0.1", state_port))
        deadline = time.monotonic() + 1.0
        while remote.state.control_mode != "stand_down" and time.monotonic() < deadline:
            time.sleep(0.02)
        st = remote.state
        assert st.control_mode == "stand_down" and st.z == 0.25 and st.yaw == 0.5 and st.v_body[0] == 0.1
    finally:
        remote.close()
        robot.close()


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
