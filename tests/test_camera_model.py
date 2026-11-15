"""Pixel <-> floor geometry."""

import math

import pytest

from robodog.camera_model import CameraModel, fit_camera


def cam(lens="fisheye", pitch=15.0):
    return CameraModel(width=1280, height=720, hfov_deg=122.0, height_m=0.3, pitch_deg=pitch,
                       forward_offset_m=0.3, lens=lens)


def test_fisheye_matches_agibot_fov():
    c = cam()
    # 122 deg wide on 16:9 -> ~69 deg tall with an equidistant fisheye (AgiBot lists 70)
    vfov = math.degrees(2 * (c.height / 2) / c.f)
    assert 66 < vfov < 71


def test_image_centre_hits_floor_where_the_tilt_says():
    c = cam(pitch=15.0)
    X, Y = c.pixel_to_floor(c.cx, c.cy)
    assert X == pytest.approx(0.3 + 0.3 / math.tan(math.radians(15)))  # offset + h / tan(pitch)
    assert Y == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("lens", ["fisheye", "pinhole"])
@pytest.mark.parametrize("X,Y", [(1.0, 0.0), (1.5, 0.4), (2.5, -0.8), (0.9, 0.3)])
def test_floor_pixel_round_trip(lens, X, Y):
    c = cam(lens)
    uv = c.pixel_to_floor(*c.floor_to_pixel(X, Y))
    assert uv == pytest.approx((X, Y), abs=1e-6)


def test_left_of_image_is_left_of_robot():
    c = cam()
    X, Y = c.pixel_to_floor(c.cx - 200, c.cy + 150)
    assert Y > 0 and X > 0


def test_above_horizon_is_not_floor():
    c = cam()
    assert c.pixel_to_floor(c.cx, 5) is None  # top of the image: wall / sky
    h = c.horizon_v()
    assert h is not None and c.pixel_to_floor(c.cx, h - 2) is None and c.pixel_to_floor(c.cx, h + 5) is not None


def test_lower_in_the_image_is_closer():
    c = cam()
    rows = [c.pixel_to_floor(c.cx, v)[0] for v in (450, 550, 650, 719)]
    assert rows == sorted(rows, reverse=True)
    assert c.nearest_visible_floor_m() == pytest.approx(rows[-1], abs=1e-6)


def test_points_behind_the_robot_are_not_drawn():
    assert cam().floor_to_pixel(-1.0, 0.0) is None


def clicks_for(truth, objects):
    """(u, v, floor distance from the point under the lens) for objects at (distance, sideways) from the lens."""
    out = []
    for d, side in objects:
        fwd = math.sqrt(d * d - side * side)
        u, v = truth.floor_to_pixel(truth.forward_offset_m + fwd, side)
        out.append((u, v, d))
    return out


@pytest.mark.parametrize("true_pitch", [-18.0, -10.0, 5.0, 15.0])
def test_one_object_fits_the_tilt(true_pitch):
    truth = CameraModel(1280, 720, 122.0, 0.4, true_pitch, 0.4, "fisheye")
    fit = fit_camera(CameraModel(1280, 720, 122.0, 0.4, 0.0, 0.4, "fisheye"), clicks_for(truth, [(2.5, 0.0)]))
    assert fit.pitch_deg == pytest.approx(true_pitch, abs=0.05)
    assert (fit.lens, fit.hfov_deg) == ("fisheye", 122.0)  # one point: lens left alone


@pytest.mark.parametrize("lens,hfov,pitch", [("pinhole", 100.0, -12.0), ("fisheye", 105.0, -16.0), ("fisheye", 122.0, -19.0)])
def test_three_objects_fit_tilt_lens_and_field_of_view(lens, hfov, pitch):
    """Like our robot: the spec-sheet lens may be wrong, and the camera looks up."""
    truth = CameraModel(1280, 720, hfov, 0.4, pitch, 0.4, lens)
    objects = [(1.5, 0.0), (2.5, 0.2), (3.0, -0.3), (2.0, 0.9)]
    objects = [o for o in objects if truth.floor_to_pixel(truth.forward_offset_m + math.sqrt(o[0]**2 - o[1]**2), o[1])]
    assert len(objects) >= 3
    start = CameraModel(1280, 720, 122.0, 0.4, 15.0, 0.4, "fisheye")  # wrong guess, like our config was
    fit = fit_camera(start, clicks_for(truth, objects))
    assert fit.worst_error < 0.02
    assert fit.pitch_deg == pytest.approx(pitch, abs=1.0)
    assert fit.hfov_deg == pytest.approx(hfov, abs=3.0) or fit.lens != lens  # the other lens may fit equally well


def test_fitted_model_measures_new_points_correctly():
    truth = CameraModel(1280, 720, 100.0, 0.4, -12.0, 0.4, "pinhole")
    fit = fit_camera(CameraModel(1280, 720, 122.0, 0.4, 15.0, 0.4, "fisheye"),
                     clicks_for(truth, [(1.5, 0.0), (2.5, 0.0), (3.0, 0.0)]))
    model = CameraModel(1280, 720, fit.hfov_deg, 0.4, fit.pitch_deg, 0.4, fit.lens)
    X, Y = model.pixel_to_floor(*truth.floor_to_pixel(0.4 + 2.0, 0.6))  # a point it was not calibrated on
    assert math.hypot(X - 0.4, Y) == pytest.approx(math.hypot(2.0, 0.6), rel=0.05)


def test_unknown_lens_rejected():
    with pytest.raises(ValueError):
        cam(lens="tele")
