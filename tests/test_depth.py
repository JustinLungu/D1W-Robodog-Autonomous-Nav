"""Tests for the parts of vision/depth.py that don't need torch or the model."""

import numpy as np

from robodog.vision.depth import PATCH, colorize, input_shape


def test_input_shape_is_multiple_of_patch_and_keeps_aspect():
    h, w = input_shape(720, 1280, 518)
    assert h % PATCH == 0 and w % PATCH == 0
    assert abs(h - 518) < PATCH
    assert abs(w / h - 1280 / 720) < 0.05


def test_input_shape_never_zero():
    assert input_shape(10, 10, 1) == (PATCH, PATCH)


def test_colorize_shape_and_flat_input():
    depth = np.zeros((48, 64), np.float32)  # all the same: must not divide by zero
    img = colorize(depth)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
