"""Obstacle check on synthetic depth maps where the right answer is known. No model needed.

Scene: horizon at row 0.75. Below it the floor's disparity grows linearly towards the
bottom (as measured on the real camera). An upright object has, on all its pixels,
the disparity of the floor at its foot.
"""

import numpy as np
import pytest

from robodog.teleop import ObstacleBanner
from robodog.vision.obstacles import ObstacleChecker, ObstacleConfig, ObstacleResult, draw_overlay

W, H = 384, 216
HORIZON = 0.75
CFG = ObstacleConfig(path_bottom=(0.25, 1.05), vanish_x=0.5, height_ratio=1.2, warn_row=0.88)


def scene(objects=(), floor=True):
    """objects: (x0, x1, top_row, foot_row) in image fractions."""
    v = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    d = np.clip(v - HORIZON, 0, None) if floor else np.zeros((H, W), np.float32)
    for x0, x1, top, foot in objects:
        r0, r1 = int(top * H), int(foot * H)
        d[r0:r1, int(x0 * W):int(x1 * W)] = foot - HORIZON
    return d


def check(d):
    return ObstacleChecker(CFG).check(d)


def test_empty_floor_is_clear():
    res = check(scene())
    assert res.status == "clear" and res.area == 0.0
    assert abs(res.horizon - HORIZON) < 0.02


def test_object_in_path_before_warning_line_warns():
    res = check(scene([(0.45, 0.65, 0.60, 0.95)]))
    assert res.status == "warning" and res.warn
    assert res.nearness > 1.0


def test_object_beyond_warning_line_is_clear_but_seen():
    res = check(scene([(0.45, 0.60, 0.70, 0.82)]))   # foot at 0.82, warning line at 0.88
    assert res.status == "clear"
    assert 0.3 < res.nearness < 1.0


def test_nearness_grows_as_object_comes_closer():
    far = check(scene([(0.45, 0.60, 0.70, 0.82)])).nearness
    near = check(scene([(0.45, 0.65, 0.60, 0.96)])).nearness
    assert near > far


def test_close_object_on_the_far_left_is_ignored():
    # Same distance as the warning case, but left of the robot's path (camera is off-centre).
    assert check(scene([(0.0, 0.10, 0.60, 0.95)])).status == "clear"


def test_close_object_on_the_right_inside_the_box_warns():
    # Right of centre (the camera is off-centre, so the path covers the right side),
    # but inside the yellow box's right edge.
    _, right, _ = ObstacleChecker(CFG).box(HORIZON)
    assert check(scene([(right - 0.10, right - 0.01, 0.60, 0.95)])).status == "warning"


def test_thing_above_the_robot_is_ignored():
    # A table top: as near as the warning case, but far above the horizon and not reaching down.
    assert check(scene([(0.3, 0.9, 0.10, 0.30)])).status == "clear"
    d = scene()
    d[int(0.10 * H):int(0.30 * H), int(0.3 * W):int(0.9 * W)] = 0.95 - HORIZON   # near, but high up
    assert check(d).status == "clear"


@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (37.0, 5.0), (0.01, -3.0)])
def test_verdict_does_not_depend_on_depth_scale_or_offset(scale, offset):
    # Depth Anything's output has an arbitrary scale and offset per frame.
    assert check(scene([(0.45, 0.65, 0.60, 0.95)]) * scale + offset).status == "warning"
    assert check(scene() * scale + offset).status == "clear"


def test_no_floor_visible_counts_as_blocked_warning():
    wall = np.full((H, W), 0.5, np.float32)          # one flat surface filling the view
    res = check(wall)
    assert res.status == "blocked" and res.warn


def test_floor_mostly_covered_still_warns():
    # Like the live test: a box and legs cover most of the floor, a strip remains on the left.
    res = check(scene([(0.35, 1.0, 0.30, 0.99)]))
    assert res.warn


def test_noise_on_the_floor_does_not_warn():
    rng = np.random.default_rng(0)
    d = scene() + rng.normal(0, 0.004, (H, W)).astype(np.float32)
    assert check(d).status == "clear"


def test_overlay_draws():
    img = np.zeros((H, W, 3), np.uint8)
    res = check(scene([(0.45, 0.65, 0.60, 0.95)]))
    out = draw_overlay(img, res, CFG)
    assert out.shape == img.shape and out.any() and not img.any()


# --- what teleop shows -----------------------------------------------------------

def result(status):
    return ObstacleResult(status, 1.5, 0.05, None)


def test_banner_levels():
    b = ObstacleBanner()
    assert b.update(None, float("inf"), 0.0)[0] == "starting"
    assert b.update(result("clear"), 0.2, 1.0) == ("clear", "path clear")
    level, text = b.update(result("warning"), 0.2, 2.0)
    assert level == "warn" and "collision soon" in text
    assert "blocked" in b.update(result("blocked"), 0.2, 2.1)[1]


def test_banner_holds_warning_briefly_then_clears():
    b = ObstacleBanner()
    b.update(result("warning"), 0.2, 10.0)
    assert b.update(result("clear"), 0.2, 10.5)[0] == "warn"     # one clear frame: still shown
    assert b.update(result("clear"), 0.2, 11.2)[0] == "clear"    # after the hold time: gone


def test_banner_does_not_trust_old_pictures():
    b = ObstacleBanner()
    assert b.update(result("clear"), 3.0, 5.0)[0] == "stale"     # never "clear" on an old picture
    assert b.update(result("clear"), 0.2, 5.1, error="camera stream ended")[0] == "stale"


# --- the yellow box is a hard limit --------------------------------------------------

def test_close_object_beside_the_box_is_ignored():
    # Very close (foot below the warning line) but left/right of the box's sides: the
    # robot passes it. Before the fix, nearer things got a wider window than the box.
    chk = ObstacleChecker(CFG)
    left, right, _ = chk.box(HORIZON)
    beside_right = (right + 0.01, min(1.0, right + 0.08), 0.55, 0.99)
    beside_left = (max(0.0, left - 0.08), left - 0.01, 0.55, 0.99)
    for obj in (beside_right, beside_left):
        res = chk.check(scene([obj]))
        assert res.status == "clear", obj
        assert (res.classes == 2).any()          # still shown, as "beside the path" (blue)


def test_close_object_inside_the_box_sides_warns():
    chk = ObstacleChecker(CFG)
    left, right, _ = chk.box(HORIZON)
    mid = (left + right) / 2
    assert chk.check(scene([(mid - 0.03, mid + 0.03, 0.55, 0.99)])).status == "warning"


def test_close_object_above_the_box_is_ignored():
    chk = ObstacleChecker(CFG)
    _, _, top = chk.box(HORIZON)
    left, right, _ = chk.box(HORIZON)
    d = scene()
    # near (disparity of a foot at 0.99) but entirely above the box's top edge
    d[int(0.02 * H):int((top - 0.02) * H), int(left * W) + 2:int(right * W) - 2] = 0.99 - HORIZON
    assert chk.check(d).status == "clear"


def test_box_matches_the_drawing():
    chk = ObstacleChecker(CFG)
    res = chk.check(scene([(0.55, 0.65, 0.55, 0.99)]))
    img = draw_overlay(np.zeros((H, W, 3), np.uint8), res, CFG)
    left, right, top = chk.box(res.horizon)
    yellow = (img[..., 0] < 10) & (img[..., 1] > 200) & (img[..., 2] > 240)
    row = int(CFG.warn_row * (H - 1))                     # the box's bottom edge (the legend is yellow too)
    cols = np.where(yellow[row - 1:row + 2].any(axis=0))[0]
    assert abs(cols.min() / (W - 1) - left) < 0.02 and abs(cols.max() / (W - 1) - right) < 0.02
