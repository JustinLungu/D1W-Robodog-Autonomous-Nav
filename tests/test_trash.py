"""Trash tracking pieces that need no model: geometry, the model-file check, the worker, the offline tool."""

import argparse
import csv
import time

import cv2
import numpy as np
import pytest

from robodog.config import load_config
from robodog.vision import trash as T
from robodog.vision.trash import Track, TrashWatcher, camera_distance, camera_for, draw_tracks, resolve_weights, to_sighting

from test_safety import pinned_config


def cam(posture="stand", w=1920, h=1080):
    return camera_for(pinned_config().camera, w, h, posture)


# --- geometry -------------------------------------------------------------------------

def test_box_foot_point_gives_the_floor_distance():
    m = cam()
    u, v = m.floor_to_pixel(m.forward_offset_m + 1.5, 0.0)           # a point 1.5 m in front of the lens
    sg = to_sighting(Track(1, "cup", 0.6, (u - 30, v - 80, u + 30, v)), m)
    assert not sg.clipped and camera_distance(m, sg.floor) == pytest.approx(1.5, abs=0.02)


def test_same_box_is_nearer_when_crawling():
    # The same pixel means a nearer floor point from a lower camera: scaled by the height ratio.
    stand, crawl = cam("stand"), cam("crawl")
    box = Track(1, "cup", 0.6, (900, 800, 1000, 900))
    d_stand = camera_distance(stand, to_sighting(box, stand).floor)
    d_crawl = camera_distance(crawl, to_sighting(box, crawl).floor)
    cfg = pinned_config().camera
    assert d_crawl / d_stand == pytest.approx(cfg.crawl_height_m / cfg.height_m, rel=0.01)


def test_boxes_touching_the_picture_edge_are_cut_off():
    m = cam()
    assert to_sighting(Track(1, "cup", 0.6, (900, 900, 1000, 1079)), m).clipped       # bottom edge
    assert to_sighting(Track(1, "cup", 0.6, (0, 800, 60, 900)), m).clipped            # left edge
    assert to_sighting(Track(1, "cup", 0.6, (1860, 800, 1919, 900)), m).clipped       # right edge
    assert not to_sighting(Track(1, "cup", 0.6, (900, 800, 1000, 900)), m).clipped


def test_box_above_the_horizon_has_no_floor_point():
    m = cam()
    assert to_sighting(Track(1, "cup", 0.6, (900, 100, 1000, 200)), m).floor is None


def test_unknown_posture_is_an_error():
    with pytest.raises(ValueError):
        cam("sitting")


def test_draw_tracks_returns_a_marked_copy():
    m = cam(w=640, h=360)
    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_tracks(img, [Track(3, "cup", 0.61, (300, 250, 340, 300))], m, selected=3)
    assert out.shape == img.shape and out.any() and not img.any()


# --- model file -----------------------------------------------------------------------

def test_pinned_model_with_a_different_hash_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "best_model.pt"
    f.write_bytes(b"not the file that was inspected")
    spec = "hf://fabiocigaina/TACO-yolo11s/best_model.pt"
    with pytest.raises(SystemExit, match="has changed"):
        resolve_weights(spec, download=lambda repo_id, filename: str(f))


def test_pinned_model_with_the_right_hash_is_accepted(tmp_path, monkeypatch):
    f = tmp_path / "m.pt"
    f.write_bytes(b"weights")
    spec = "hf://someone/repo/m.pt"
    monkeypatch.setitem(T.PINNED_SHA256, spec, T.sha256_of(f))
    assert resolve_weights(spec, download=lambda repo_id, filename: str(f)) == f


def test_local_model_path(tmp_path):
    f = tmp_path / "mine.pt"
    f.write_bytes(b"x")
    assert resolve_weights(str(f)) == f
    with pytest.raises(SystemExit, match="not found"):
        resolve_weights(str(tmp_path / "missing.pt"))
    with pytest.raises(SystemExit, match="expected hf://"):
        resolve_weights("hf://only-a-repo")


# --- worker thread --------------------------------------------------------------------

class FakeTracker:
    """Reports one cup whose box sits wherever the frame has its bright square."""
    def __init__(self):
        self.calls = 0

    def update(self, frame):
        self.calls += 1
        ys, xs = np.nonzero(frame[..., 0] > 200)
        if len(xs) == 0:
            return []
        return [Track(1, "cup", 0.7, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))]


def test_watcher_analyses_each_new_frame_once():
    frame = np.zeros((90, 160, 3), np.uint8)
    frame[40:60, 70:90] = 255
    state = {"seq": 1}
    tr = FakeTracker()
    w = TrashWatcher(lambda: (frame, 12.5, state["seq"]), tr)
    try:
        deadline = time.monotonic() + 2
        while w.latest()[2] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        tracks, stamp, seq, _ = w.latest()
        assert seq == 1 and stamp == 12.5 and tracks[0].box == (70.0, 40.0, 89.0, 59.0)
        time.sleep(0.05)
        assert tr.calls == 1                      # same frame: not analysed again
        state["seq"] = 2
        deadline = time.monotonic() + 2
        while w.latest()[2] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert w.latest()[2] == 2 and tr.calls == 2
    finally:
        w.close()


def test_watcher_reports_a_crashing_tracker_instead_of_dying_silently():
    class Broken:
        def update(self, frame):
            raise RuntimeError("boom")
    w = TrashWatcher(lambda: (np.zeros((9, 16, 3), np.uint8), 0.0, 1), Broken(), log=lambda m: None)
    try:
        deadline = time.monotonic() + 2
        while w.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "boom" in w.error
    finally:
        w.close()


# --- offline tool on a synthetic video ------------------------------------------------

def make_video(path, n=30, size=(640, 360)):
    """A bright square that slides down the floor area (= comes closer)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for i in range(n):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        y = 250 + i * 2
        f[y:y + 30, 300:340] = 255
        w.write(f)
    w.release()


def cli_args(video, **over):
    a = dict(video=str(video), posture="stand", step=3, start=0.0, end=None, save=None, csv=None,
             headless=True, model=None, classes=None, conf=None, imgsz=None, device="auto", width=1280)
    a.update(over)
    return argparse.Namespace(**a)


def test_offline_tool_writes_distances_that_shrink_as_the_object_comes_closer(tmp_path):
    from robodog.cli.trash import run
    video, out_csv, out_mp4 = tmp_path / "v.avi", tmp_path / "d.csv", tmp_path / "a.mp4"
    make_video(video)
    summary = run(cli_args(video, csv=str(out_csv), save=str(out_mp4)), pinned_config(), tracker=FakeTracker())
    assert summary["frames"] == 10 and summary["boxes"] == 10 and list(summary["tracks"]) == [1]
    rows = list(csv.DictReader(open(out_csv)))
    d = [float(r["distance_m"]) for r in rows if r["distance_m"]]
    assert len(d) >= 8 and d[-1] < d[0] and all(b <= a + 1e-6 for a, b in zip(d, d[1:]))
    assert rows[0]["label"] == "cup" and rows[0]["track_id"] == "1" and float(rows[0]["conf"]) == pytest.approx(0.7)
    assert out_mp4.stat().st_size > 0


def test_offline_tool_posture_changes_the_distances(tmp_path):
    from robodog.cli.trash import run
    video = tmp_path / "v.avi"
    make_video(video, n=6)
    res = {}
    for posture in ("stand", "crawl"):
        out = tmp_path / f"{posture}.csv"
        run(cli_args(video, posture=posture, csv=str(out)), pinned_config(), tracker=FakeTracker())
        res[posture] = float(list(csv.DictReader(open(out)))[0]["distance_m"])
    assert res["crawl"] < res["stand"]


# --- the YOLO + ByteTrack wrapper, with a stand-in for the model -----------------------

class _Boxes:
    def __init__(self, rows, tracked=True):
        self._rows, self._tracked = rows, tracked

    class _L(list):
        def tolist(self): return list(self)

    @property
    def xyxy(self): return self._L([r[:4] for r in self._rows])
    @property
    def id(self): return self._L([r[4] for r in self._rows]) if self._tracked and self._rows else None
    @property
    def cls(self): return self._L([r[5] for r in self._rows])
    @property
    def conf(self): return self._L([r[6] for r in self._rows])


class _Result:
    def __init__(self, boxes): self.boxes = boxes


class FakeYOLO:
    """Stands in for ultralytics.YOLO: records how track() is called, returns canned boxes."""
    last = None

    def __init__(self, weights):
        self.weights, self.names, self.calls, self.next = weights, {0: "Bottle", 1: "Cup", 2: "Can"}, [], _Boxes([])
        FakeYOLO.last = self

    def track(self, frame, **kw):
        self.calls.append(kw)
        return [_Result(self.next)]


@pytest.fixture
def fake_yolo(monkeypatch, tmp_path):
    ultralytics = pytest.importorskip("ultralytics")
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    return str(w)


def test_tracker_hands_weak_detections_to_bytetrack_and_parses_tracks(fake_yolo):
    tr = T.TrashTracker(fake_yolo, conf=0.3, imgsz=640, device="cpu")
    FakeYOLO.last.next = _Boxes([(10.0, 20.0, 50.0, 80.0, 7.0, 1.0, 0.62)])
    tracks = tr.update(np.zeros((90, 160, 3), np.uint8))
    assert tracks == [Track(7, "Cup", 0.62, (10.0, 20.0, 50.0, 80.0))]
    kw = FakeYOLO.last.calls[0]
    assert kw["persist"] is True and kw["imgsz"] == 640 and kw["device"] == "cpu" and kw["classes"] is None
    assert kw["conf"] == pytest.approx(0.1)                    # weak boxes reach the tracker...
    cfg = open(kw["tracker"]).read()
    assert "tracker_type: bytetrack" in cfg and "new_track_thresh: 0.3" in cfg and "track_low_thresh: 0.1" in cfg


def test_tracker_returns_nothing_when_no_track_is_confirmed(fake_yolo):
    tr = T.TrashTracker(fake_yolo, device="cpu")
    FakeYOLO.last.next = _Boxes([(1.0, 2.0, 3.0, 4.0, None, 1.0, 0.5)], tracked=False)   # detections, but no IDs yet
    assert tr.update(np.zeros((9, 16, 3), np.uint8)) == []


def test_class_filter_for_an_ordinary_model(fake_yolo):
    tr = T.TrashTracker(fake_yolo, device="cpu", classes="cup, bottle")
    tr.update(np.zeros((9, 16, 3), np.uint8))
    assert FakeYOLO.last.calls[0]["classes"] == [1, 0]
    with pytest.raises(SystemExit, match="doesn't know"):
        T.TrashTracker(fake_yolo, device="cpu", classes="cup, sofa")


def test_bytetrack_config_never_puts_low_above_start_threshold():
    assert "track_low_thresh: 0.05" in T.bytetrack_config(0.05)
