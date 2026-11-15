"""Recorder: commands CSV + meta always; video only when ffmpeg is installed."""

import csv
import json
import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest

from robodog.recorder import COLUMNS, Recorder


def rows(rec):
    with open(rec.dir / "commands.csv") as f:
        return list(csv.DictReader(f))


def test_commands_only(tmp_path):
    rec = Recorder(tmp_path, camera=None, meta={"note": "hallway"}, log=lambda m: None)
    rec.event("stand")
    rec.row(standing=True, level=1, held={"forward", "turn_left"}, cmd=(0.1, 0.0, 0.17),
            sent=(0.1, 0.0, 0.17), battery=80, mode=3, fault=None)
    rec.row(standing=True, level=1, held=set(), cmd=(0, 0, 0), sent=None, battery=None, mode=None, fault=None)
    rec.stop()
    rec.stop()  # second stop is harmless

    r = rows(rec)
    assert list(r[0].keys()) == COLUMNS
    assert r[0]["event"] == "stand" and r[1]["event"] == ""  # events go in one row only
    assert r[0]["held"] == "forward turn_left" and r[0]["cmd_vx"] == "0.100"
    assert r[1]["sent_vx"] == "0.000" and r[1]["battery"] == ""
    meta = json.loads((rec.dir / "meta.json").read_text())
    assert meta["note"] == "hallway" and meta["camera"] is None and "duration_s" in meta
    assert not (rec.dir / "video.mkv").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_gets_zero_based_file_and_clock_offset(tmp_path):
    src = tmp_path / "src.avi"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()

    before = time.time()
    rec = Recorder(tmp_path / "rec", camera=str(src), meta={}, log=lambda m: None)
    rec._ffmpeg.wait(timeout=10)  # a file source ends by itself
    rec.stop()

    meta = json.loads((rec.dir / "meta.json").read_text())
    assert before - 1 < meta["video_start_unix"] < time.time()  # laptop clock, not 0
    assert not (rec.dir / "video_raw.mkv").exists()
    start = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0",
         str(rec.dir / "video.mkv")], capture_output=True, text=True).stdout.strip()
    assert float(start) == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_unreachable_camera_keeps_commands(tmp_path):
    rec = Recorder(tmp_path, camera=str(tmp_path / "missing.mkv"), meta={}, log=lambda m: None)
    rec._ffmpeg.wait(timeout=10)
    assert not rec.video_running
    rec.row(standing=False, level=1, held=set(), cmd=(0, 0, 0), sent=None, battery=50, mode=0, fault=None)
    rec.stop()
    assert rec.video_error and len(rows(rec)) == 1
    assert (rec.dir / "ffmpeg.log").read_text()  # the reason is in there
