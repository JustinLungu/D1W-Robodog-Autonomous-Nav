"""Litter detection with a zero-shot detector (Grounding DINO): no training needed.

You describe what to find in words ("paper. plastic bottle. can.") and the model
boxes anything matching. Good for a first look and for pre-labelling data; a small
detector trained on our own recordings (YOLO) will later be faster and more accurate.

Built for the robot's low camera:
- Only the lower part of the image (the floor) is searched. Faster, and it skips
  bottles and cups standing on tables.
- A box only counts if its bottom edge is on the floor (below min_bottom).
- Very large boxes are dropped: from this camera, litter is small. Big hits are
  usually sleeping bags, jackets, mats.
- Overlapping boxes for the same object are merged.

Needs the vision extra: `uv sync --extra vision-cpu` (or `--extra vision-gpu`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",  # ~170M params: CPU-ok (slow), the default
    "base": "IDEA-Research/grounding-dino-base",  # ~230M params: better, GPU recommended
}

DEFAULT_CLASSES = [
    # "piece of paper" finds the scraps on the floor far better than just "paper" (tested on our video).
    "piece of paper", "plastic bottle", "can", "cup", "wrapper", "cardboard box", "plastic bag",
]


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in full-frame pixels


@dataclass
class Filters:
    search_from: float = 0.5   # search only below this fraction of the image height (0 = whole image)
    min_bottom: float = 0.65   # box bottom edge must be below this fraction (= on the floor)
    max_area: float = 0.03     # drop boxes larger than this fraction of the frame
    iou_merge: float = 0.5     # merge two boxes overlapping more than this (intersection over union)
    contain_merge: float = 0.8 # ...or when this much of the smaller box lies inside the bigger one


class LitterDetector:
    def __init__(self, classes=None, model: str = "tiny", device: str = "auto",
                 threshold: float = 0.25, filters: Filters | None = None):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise SystemExit(
                "Litter detection needs the vision extra. Install it with one of:\n"
                "  uv sync --extra vision-cpu    # no NVIDIA GPU\n"
                "  uv sync --extra vision-gpu    # NVIDIA GPU"
            ) from e

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.classes = list(classes or DEFAULT_CLASSES)
        self.threshold = threshold
        self.filters = filters or Filters()

        self.model_name = MODELS.get(model, model)
        # First run downloads the weights (~700 MB for tiny) to ~/.cache/huggingface.
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

        # Grounding DINO wants lowercase phrases separated by " . "
        self.prompt = " . ".join(c.lower().strip() for c in self.classes) + " ."
        self._class_tokens = self._token_spans()

    def _token_spans(self) -> list[list[int]]:
        """Which prompt tokens belong to which class, so every box gets exactly one class name.

        (The library's own label decoding glues words together, e.g. "plastic bagper".)
        """
        enc = self.processor.tokenizer(self.prompt, return_offsets_mapping=True)
        spans, pos = [], 0
        for c in self.classes:
            start = self.prompt.index(c.lower().strip(), pos)
            end = start + len(c.strip())
            pos = end
            spans.append([i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a >= start and b <= end])
        return spans

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        torch = self._torch
        h, w = bgr.shape[:2]
        f = self.filters
        top = int(h * f.search_from)
        crop = cv2.cvtColor(bgr[top:], cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=crop, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)

        probs = out.logits[0].sigmoid()                  # (queries, prompt tokens)
        boxes = out.pred_boxes[0]                        # (queries, 4) cx, cy, w, h, relative to the crop
        ch = h - top
        dets = []
        for q in torch.nonzero(probs.max(dim=-1).values > self.threshold).flatten().tolist():
            # Class = the phrase whose tokens this box matches best.
            class_scores = [float(probs[q, span].max()) if span else 0.0 for span in self._class_tokens]
            k = int(np.argmax(class_scores))
            score = class_scores[k]
            if score <= self.threshold:
                continue
            cx, cy, bw, bh = boxes[q].tolist()
            x0, x1 = (cx - bw / 2) * w, (cx + bw / 2) * w
            y0, y1 = (cy - bh / 2) * ch + top, (cy + bh / 2) * ch + top
            box = (int(max(0, x0)), int(max(0, y0)), int(min(w - 1, x1)), int(min(h - 1, y1)))
            dets.append(Detection(self.classes[k], score, box))
        return filter_detections(dets, w, h, f)


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _overlap(a, b) -> tuple[float, float]:
    """(IoU, intersection / smaller box area)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0, 0.0
    union = _area(a) + _area(b) - inter
    return inter / union, inter / max(1, min(_area(a), _area(b)))


def filter_detections(dets: list[Detection], w: int, h: int, f: Filters) -> list[Detection]:
    """Floor / size filters, then merge duplicates (highest score wins). No torch needed."""
    keep = [
        d for d in dets
        if d.box[3] >= f.min_bottom * h and 0 < _area(d.box) <= f.max_area * w * h
    ]
    keep.sort(key=lambda d: d.score, reverse=True)
    out: list[Detection] = []
    for d in keep:
        duplicate = False
        for o in out:
            iou, inside = _overlap(d.box, o.box)
            if iou >= f.iou_merge or inside >= f.contain_merge:
                duplicate = True  # same object as a higher-scoring box already kept
                break
        if not duplicate:
            out.append(d)
    return out


# One colour per class (BGR), cycled.
_COLORS = [(60, 60, 235), (60, 200, 60), (235, 140, 40), (30, 200, 230), (200, 60, 200), (40, 140, 255),
           (180, 180, 40), (120, 120, 255)]


def draw(bgr: np.ndarray, dets: list[Detection], classes: list[str], filters: Filters | None = None) -> np.ndarray:
    """Boxes + labels on a copy of the frame. With filters: dashes mark the floor line."""
    img = bgr.copy()
    h, w = img.shape[:2]
    scale = max(1.0, h / 720)
    if filters is not None:
        y = int(filters.min_bottom * h)
        for x in range(0, w, 40):
            cv2.line(img, (x, y), (x + 20, y), (200, 200, 200), max(1, int(scale)))
    for d in dets:
        color = _COLORS[classes.index(d.label) % len(_COLORS)] if d.label in classes else (0, 0, 255)
        x0, y0, x1, y1 = d.box
        cv2.rectangle(img, (x0, y0), (x1, y1), color, max(2, int(2 * scale)))
        text = f"{d.label} {d.score:.2f}"
        fs, th = 0.6 * scale, max(1, int(scale))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y0 - 6 if y0 - tht - 10 > 0 else y1 + tht + 6
        cv2.rectangle(img, (x0, ty - tht - 4), (x0 + tw + 4, ty + 4), color, -1)
        cv2.putText(img, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return img
