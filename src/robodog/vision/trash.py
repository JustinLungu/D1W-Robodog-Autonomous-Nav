"""Trash detection + tracking: YOLO finds the pieces, ByteTrack keeps one ID per piece.

    camera frame -> YOLO (fine-tuned on TACO) -> boxes -> ByteTrack -> boxes with stable IDs
                 -> foot point of each box -> floor position (camera_model.py) -> distance

What each part is for
---------------------
- YOLO gives boxes per frame, with no memory: "a cup here, a cup there".
- ByteTrack links boxes across frames into tracks ("cup #7"), also through a few missed
  or low-confidence frames. That stable ID is what lets go-to follow ONE piece.
- The box's bottom-centre is where the object touches the floor. With the camera's height
  and tilt, that pixel becomes a floor position (X ahead, Y left) and so a distance.
  The camera is lower in the crawl posture, hence the `posture` parameter.

Model file
----------
`trash.model` in config.local.yaml: a local .pt file, or hf://<repo>/<file> (downloaded
once to ~/.cache/huggingface). YOLO .pt files are Python pickles: LOADING ONE RUNS CODE
from whoever made it. For the default model we therefore pin the SHA-256 of the file that
was inspected (it only references torch and ultralytics classes), and refuse anything else
under that name. For your own models, only use files you trust.

Needs the vision extra: `uv sync --extra vision-cpu` (or `--extra vision-gpu`).
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera_model import CameraModel
from ..target import Sighting
from .csrt import OVERLAP_IOU, SelectedBoxTracker, iou

POSTURES = ("stand", "crawl")

# SHA-256 of model files that were inspected before first use (see the module docstring).
PINNED_SHA256 = {
    "hf://fabiocigaina/TACO-yolo11s/best_model.pt": "1b9c8379aca95efa75ec1f6ac803cabd3ba531b7d4b3f459101bcfd3aaae95ef",
}


@dataclass
class Track:
    track_id: int
    label: str
    conf: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in frame pixels
    source: str = "yolo"                     # "yolo", or "csrt": YOLO lost the selected piece, CSRT's box


# ----------------------------------------------------------------------------- model file
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(spec: str, download=None) -> Path:
    """Local path of the model file for `spec` (a path, or hf://<repo>/<file>), hash-checked if pinned."""
    if spec.startswith("hf://"):
        rest = spec[len("hf://"):]
        parts = rest.split("/")
        if len(parts) < 3:
            raise SystemExit(f"trash.model {spec!r}: expected hf://<user>/<repo>/<file>")
        repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
        if download is None:
            from huggingface_hub import hf_hub_download as download
        path = Path(download(repo_id=repo, filename=filename))
    else:
        path = Path(spec).expanduser()
        if not path.is_file():
            raise SystemExit(f"trash.model: file not found: {path}")
    want = PINNED_SHA256.get(spec)
    if want is not None:
        got = sha256_of(path)
        if got != want:
            raise SystemExit(
                f"trash model {spec} has changed: SHA-256 is {got[:16]}..., expected {want[:16]}...\n"
                "Refusing to load it (a .pt file runs code when loaded). Inspect the new file, then "
                "update PINNED_SHA256 in vision/trash.py."
            )
    elif spec.startswith("hf://"):
        print(f"note: {spec} is not a pinned model: loading it runs code from its author. Only use models you trust.")
    return path


# ----------------------------------------------------------------------------- detector + tracker
LOW_CONF = 0.1   # detections down to this are handed to ByteTrack, which uses them to keep a track alive


def bytetrack_config(conf: float, buffer_frames: int = 15) -> str:
    """ByteTrack settings (as YAML) for Ultralytics.

    ByteTrack's idea: START a track only from a confident detection (>= conf), but CONTINUE
    an existing track also with weak ones (>= LOW_CONF), e.g. through a blurry frame. So the
    detector must hand over the weak boxes too, and `conf` becomes the start threshold.
    """
    return (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {conf}\n"        # first matching round: confident boxes
        f"track_low_thresh: {min(LOW_CONF, conf)}\n"   # second round: weak boxes, only for existing tracks
        f"new_track_thresh: {conf}\n"         # a new track needs this confidence
        f"track_buffer: {buffer_frames}\n"    # analysed frames a lost track is remembered for
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )


class TrashTracker:
    def __init__(self, model: str, conf: float = 0.25, imgsz: int = 896, device: str = "auto",
                 classes: str | list[str] = ""):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit(
                "Trash tracking needs the vision extra. Install it with one of:\n"
                "  uv sync --extra vision-cpu    # no NVIDIA GPU\n"
                "  uv sync --extra vision-gpu    # NVIDIA GPU"
            ) from e
        self.weights = resolve_weights(model)
        self.conf, self.imgsz = conf, imgsz
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = YOLO(str(self.weights))

        # Optional class list. Open-vocabulary models (YOLO-World, YOLOE) take it as the things
        # to look for; ordinary models only report the listed ones of the classes they know.
        wanted = [c.strip() for c in (classes.split(",") if isinstance(classes, str) else classes) if c.strip()]
        self.class_ids: list[int] | None = None
        if wanted and type(self.model).__name__ in ("YOLOWorld", "YOLOE"):
            self.model.set_classes(wanted)
        elif wanted:
            known = {str(n).lower(): i for i, n in dict(self.model.names).items()}
            missing = [c for c in wanted if c.lower() not in known]
            if missing:
                raise SystemExit(f"trash.classes: this model doesn't know {missing}. It knows: {sorted(known)}")
            self.class_ids = [known[c.lower()] for c in wanted]
        self.names: dict[int, str] = dict(self.model.names)

        f = tempfile.NamedTemporaryFile("w", suffix="_bytetrack.yaml", delete=False)
        f.write(bytetrack_config(conf))
        f.close()
        self._tracker_cfg = f.name

    def update(self, bgr: np.ndarray) -> list[Track]:
        """One frame in, the confirmed tracks out. Call with consecutive frames of ONE video."""
        r = self.model.track(bgr, persist=True, tracker=self._tracker_cfg, conf=min(LOW_CONF, self.conf),
                             imgsz=self.imgsz, device=self.device, classes=self.class_ids, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        return [
            Track(int(i), self.names.get(int(c), str(int(c))), float(p), tuple(float(v) for v in b))
            for b, i, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
        ]

    def reset(self):
        """Forget all tracks (before starting on another video)."""
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None


# ----------------------------------------------------------------------------- geometry
def camera_for(cam_cfg, width: int, height: int, posture: str = "stand",
               lens: str | None = None, hfov: float | None = None) -> CameraModel:
    """The camera model for a frame size and robot posture (the lens is lower when crawling)."""
    if posture not in POSTURES:
        raise ValueError(f"posture must be one of {POSTURES}, got {posture!r}")
    h = cam_cfg.height_m if posture == "stand" else cam_cfg.crawl_height_m
    return CameraModel(width, height, hfov or cam_cfg.hfov_deg, h, cam_cfg.pitch_deg,
                       cam_cfg.forward_offset_m, lens or cam_cfg.lens)


def to_sighting(t: Track, model: CameraModel, edge_px: float | None = None) -> Sighting:
    """Where a tracked box stands on the floor: its bottom-centre, through the camera model.

    `clipped` = the box touches the picture's bottom or side edge: the object continues outside
    the picture, so its real foot point isn't visible and the distance would be wrong.
    """
    x0, y0, x1, y1 = t.box
    edge = model.height * 0.01 if edge_px is None else edge_px
    clipped = y1 >= model.height - 1 - edge or x0 <= edge or x1 >= model.width - 1 - edge
    floor = model.pixel_to_floor((x0 + x1) / 2, min(y1, model.height - 1))
    return Sighting(t.track_id, t.label, t.conf, floor, clipped)


def camera_distance(model: CameraModel, floor: tuple[float, float]) -> float:
    """Floor distance from the point under the lens (how you'd measure with a tape)."""
    return math.hypot(floor[0] - model.forward_offset_m, floor[1])


# ----------------------------------------------------------------------------- drawing (OpenCV)
def _colour(track_id: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[(track_id * 47) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_tracks(bgr: np.ndarray, tracks: list[Track], model: CameraModel, selected: int | None = None) -> np.ndarray:
    """Boxes labelled with track ID, class, confidence and distance, on a copy of the frame."""
    img = bgr.copy()
    s = max(1.0, img.shape[0] / 720)
    for t in tracks:
        sg = to_sighting(t, model)
        x0, y0, x1, y1 = (int(v) for v in t.box)
        col = (60, 255, 60) if t.track_id == selected else _colour(t.track_id)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, max(2, int((4 if t.track_id == selected else 2) * s)))
        if sg.floor is None:
            dist = "not on floor"
        elif sg.clipped:
            dist = "cut off"
        else:
            dist = f"{camera_distance(model, sg.floor):.2f} m"
        text = f"#{t.track_id} {t.label} {t.conf:.2f}  {dist}"
        fs, th = 0.6 * s, max(1, int(2 * s))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y0 - 6 if y0 - tht - 10 > 0 else y1 + tht + 6
        cv2.rectangle(img, (x0, ty - tht - 4), (x0 + tw + 4, ty + 4), col, -1)
        cv2.putText(img, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
        if sg.floor is not None and not sg.clipped:
            cv2.circle(img, (int((x0 + x1) / 2), int(y1)), max(3, int(4 * s)), col, -1)   # the foot point used
    return img


# ----------------------------------------------------------------------------- background worker
class TrashWatcher:
    """Runs the tracker in its own thread on the newest camera frame.

    `get_frame()` must return (frame or None, arrival time (time.monotonic), sequence number),
    like LatestFrameReader.get. The caller asks for the latest result and never waits for the model.

        tracks, stamp, seq, rate = watcher.latest()   # stamp = arrival time of the analysed frame

    select(track_id) marks the piece go-to is heading for. While YOLO doesn't see that one, a CSRT
    tracker fills in its box for up to csrt_max_s (vision/csrt.py); 0 = off.
    """

    def __init__(self, get_frame, tracker, log=print, csrt_max_s: float = 0.0, csrt=None):
        self.get_frame = get_frame
        self.tracker = tracker
        self.log = log
        self._csrt = csrt or (SelectedBoxTracker(csrt_max_s) if csrt_max_s > 0 else None)
        self._selected: int | None = None
        self._sel_label, self._sel_conf = "", 0.0   # the selected piece, as YOLO last saw it
        self._lock = threading.Lock()
        self._tracks: list[Track] = []
        self._stamp = 0.0
        self._seq = 0
        self._rate = 0.0
        self.error: str | None = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="trash-watcher", daemon=True)
        self._thread.start()

    def _run(self):
        last_seq, t_prev = None, time.monotonic()
        try:
            while self._running:
                frame, stamp, seq = self.get_frame()
                if frame is None or seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                tracks = self._fill_in_selected(frame, self.tracker.update(frame))
                now = time.monotonic()
                with self._lock:
                    dt = max(now - t_prev, 1e-3)
                    self._rate = 1 / dt if self._rate == 0 else 0.8 * self._rate + 0.2 / dt
                    self._tracks, self._stamp, self._seq = tracks, stamp, self._seq + 1
                t_prev = now
        except Exception as e:   # never take the caller down; report instead
            self.error = f"trash watcher stopped: {e!r}"
            self.log(self.error)

    def select(self, track_id: int | None):
        """The piece go-to is heading for (None = none): only this one gets the CSRT fallback."""
        self._selected = track_id

    def _fill_in_selected(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        sel = self._selected
        if self._csrt is None:
            return tracks
        if sel is None:
            self._csrt.reset()
            return tracks
        mine = next((t for t in tracks if t.track_id == sel), None)
        if mine is not None:                       # YOLO sees it: CSRT restarts on YOLO's box
            self._csrt.seen(frame, mine.box)
            self._sel_label, self._sel_conf = mine.label, mine.conf
            return tracks
        box = self._csrt.follow(frame)
        if box is None:
            return tracks
        if any(iou(box, t.box) >= OVERLAP_IOU for t in tracks):
            self._csrt.reset()                     # YOLO has it under a new ID: the follower adopts that one
            return tracks
        return tracks + [Track(sel, self._sel_label, self._sel_conf, box, "csrt")]

    def latest(self) -> tuple[list[Track], float, int, float]:
        """(tracks, arrival time of the frame they belong to, result number, results per second)."""
        with self._lock:
            return list(self._tracks), self._stamp, self._seq, self._rate

    def close(self):
        self._running = False
        self._thread.join(timeout=5.0)
