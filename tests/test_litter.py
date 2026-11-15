"""Litter detection: the filtering and merging, which don't need the model."""

import numpy as np

from robodog.vision.litter import Detection, Filters, draw, filter_detections

W, H = 1920, 1080
F = Filters(search_from=0.5, min_bottom=0.65, max_area=0.03)


def det(label, score, x0, y0, x1, y1):
    return Detection(label, score, (x0, y0, x1, y1))


def test_drops_things_above_the_floor_line():
    on_table = det("cup", 0.9, 100, 500, 140, 600)       # bottom at 0.56 * H
    on_floor = det("piece of paper", 0.5, 100, 900, 200, 950)
    assert filter_detections([on_table, on_floor], W, H, F) == [on_floor]


def test_drops_huge_boxes():
    mattress = det("plastic bag", 0.6, 0, 700, 800, 1000)  # 0.12 of the frame
    assert filter_detections([mattress], W, H, F) == []


def test_merges_overlapping_boxes_keeping_best_score():
    a = det("plastic bag", 0.4, 100, 800, 200, 900)
    b = det("piece of paper", 0.6, 105, 805, 205, 905)    # nearly the same box
    assert filter_detections([a, b], W, H, F) == [b]


def test_merges_box_inside_another():
    big = det("cardboard box", 0.7, 100, 800, 300, 950)
    inner = det("wrapper", 0.3, 150, 850, 200, 900)       # fully inside
    assert filter_detections([inner, big], W, H, F) == [big]


def test_keeps_separate_objects():
    a = det("piece of paper", 0.5, 100, 900, 200, 950)
    b = det("piece of paper", 0.4, 600, 900, 700, 950)
    assert filter_detections([a, b], W, H, F) == [a, b]


def test_draw_returns_copy():
    img = np.zeros((H, W, 3), np.uint8)
    out = draw(img, [det("piece of paper", 0.5, 100, 900, 200, 950)], ["piece of paper"], F)
    assert out.shape == img.shape and out.any() and not img.any()
