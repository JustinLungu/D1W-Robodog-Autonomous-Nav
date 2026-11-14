"""Depth Anything V2: depth from a single RGB image.

What the output means
---------------------
The model returns RELATIVE inverse depth ("disparity"): bigger value = closer. There
are no units, and the scale changes from frame to frame, so "0.8" doesn't mean metres
and can't be compared between frames. It tells you what's in front of what, which is
enough to see whether it's useful for spotting obstacles. (Metric variants exist, but
they're bigger and slower; this is just a first look.)

Needs the vision extra: `uv sync --extra vision-cpu` (or `--extra vision-gpu`).
"""

from __future__ import annotations

import cv2
import numpy as np

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",   # ~25M params: the one to start with
    "base": "depth-anything/Depth-Anything-V2-Base-hf",     # ~97M params: GPU recommended
    "large": "depth-anything/Depth-Anything-V2-Large-hf",   # ~335M params: GPU only, realistically
}

# The model works on 14x14 pixel patches, so the input sides must be multiples of 14.
PATCH = 14
# ImageNet normalisation, what the model was trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def input_shape(height: int, width: int, short_side: int) -> tuple[int, int]:
    """Size (h, w) to feed the model: short side ~= short_side, aspect kept, multiples of 14.

    Smaller is faster (time grows roughly with pixel count) but loses detail.
    518 is what the model was trained at.
    """
    scale = short_side / min(height, width)

    def snap(v: float) -> int:
        return max(PATCH, int(round(v * scale / PATCH)) * PATCH)

    return snap(height), snap(width)


class DepthEstimator:
    def __init__(self, model: str = "small", device: str = "auto", short_side: int | None = None):
        # Imported here so the rest of the package works without the vision extra.
        try:
            import torch
            from transformers import AutoModelForDepthEstimation
        except ImportError as e:
            raise SystemExit(
                "Depth needs the vision extra. Install it with one of:\n"
                "  uv sync --extra vision-cpu    # no NVIDIA GPU\n"
                "  uv sync --extra vision-gpu    # NVIDIA GPU"
            ) from e

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Half precision on the GPU: about twice as fast, no visible difference here.
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # 518 is the training size. On a CPU it takes ~2 s per frame; 308 takes ~0.4 s.
        self.short_side = short_side or (518 if self.device.type == "cuda" else 308)

        name = MODELS.get(model, model)  # a key above, or any Hugging Face model id
        self.model_name = name
        # First run downloads the weights (~100 MB for small) to ~/.cache/huggingface.
        self.model = AutoModelForDepthEstimation.from_pretrained(name, dtype=self.dtype)
        self.model.to(self.device).eval()

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """BGR frame (as from OpenCV) -> float32 relative inverse depth, same h x w as the frame."""
        torch = self._torch
        h, w = bgr.shape[:2]
        in_h, in_w = input_shape(h, w, self.short_side)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, self.dtype)

        with torch.inference_mode():
            depth = self.model(pixel_values=x).predicted_depth  # (1, in_h, in_w)

        depth = depth.float().squeeze(0).cpu().numpy()
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def colorize(depth: np.ndarray) -> np.ndarray:
    """Relative depth -> BGR image for display. Bright/yellow = close, dark/purple = far.

    Scaled per frame between the 2nd and 98th percentile, so a few extreme pixels
    don't wash out the picture.
    """
    lo, hi = np.percentile(depth, (2, 98))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
