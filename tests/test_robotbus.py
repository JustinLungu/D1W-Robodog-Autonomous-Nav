"""Decoding the robot's ROS 2 messages (CDR). No robot or Zenoh needed."""

import struct
from pathlib import Path

import pytest

from robodog.robotbus import CdrReader, decode_detection2d_array


class CdrWriter:
    """Tiny little-endian CDR encoder, mirroring the reader, to build test messages."""

    def __init__(self):
        self.b = bytearray()

    def _align(self, n):
        self.b += b"\0" * ((-len(self.b)) % n)

    def u32(self, v): self._align(4); self.b += struct.pack("<I", v)
    def i32(self, v): self._align(4); self.b += struct.pack("<i", v)
    def f64(self, v): self._align(8); self.b += struct.pack("<d", v)
    def string(self, s): e = s.encode() + b"\0"; self.u32(len(e)); self.b += e
    def header(self, sec, frame=""): self.i32(sec); self.u32(0); self.string(frame)
    def bytes(self): return b"\0\x01\0\0" + bytes(self.b)   # encapsulation: CDR little endian


def detection_array(boxes):
    w = CdrWriter()
    w.header(1786000000, "camera")
    w.u32(len(boxes))
    for track_id, hyps, (cx, cy, bw, bh) in boxes:
        w.header(1786000000)
        w.u32(len(hyps))
        for label, score in hyps:
            w.string(label); w.f64(score)
            for _ in range(3 + 4 + 36):
                w.f64(0.0)
        for v in (cx, cy, 0.0, bw, bh):
            w.f64(v)
        w.string(track_id)
    return w.bytes()


def test_decodes_boxes():
    data = detection_array([
        ("318", [("", 0.8)], (900, 600, 60, 170)),
        ("7", [("person", 0.4), ("car", 0.9)], (100.5, 200, 10, 20)),
    ])
    stamp, boxes = decode_detection2d_array(data)
    assert stamp == pytest.approx(1786000000)
    assert [b.track_id for b in boxes] == ["318", "7"]
    assert (boxes[0].cx, boxes[0].cy, boxes[0].w, boxes[0].h) == (900, 600, 60, 170)
    assert boxes[0].score == pytest.approx(0.8) and boxes[0].label == ""
    assert boxes[1].label == "car" and boxes[1].score == pytest.approx(0.9)   # best hypothesis wins


def test_empty_array():
    assert decode_detection2d_array(detection_array([]))[1] == []


def test_rejects_non_cdr():
    with pytest.raises(ValueError):
        CdrReader(b"\x00\x00\x00\x00junk")


def test_real_message_from_the_robot():
    # Captured from /perception/detection2d on our robot (2026-11-15): one person.
    data = (Path(__file__).parent / "data" / "detection2d_from_robot.cdr").read_bytes()
    _, boxes = decode_detection2d_array(data)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.track_id == "311" and b.label == ""
    assert 0 < b.score < 1
    assert 0 < b.cx < 1920 and 0 < b.cy < 1080 and 0 < b.w < 1920 and 0 < b.h < 1080
