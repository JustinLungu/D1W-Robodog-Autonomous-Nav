"""Obstacle check from monocular depth (Depth Anything): "is something about to be hit?"

The problem
-----------
Depth Anything gives RELATIVE inverse depth ("disparity", bigger = closer) with an
unknown scale and offset that change every frame. And the floor right in front of the
robot is the closest thing in the picture, so "something is close" alone means nothing.

The idea
--------
1. The floor is flat. Seen from a camera fixed to the robot, the disparity of a flat
   floor is a plane over the image: it grows steadily towards the bottom. (Measured
   on our recordings: a straight line in the row, nearly the same in every column,
   despite the wide-angle lens.) Every frame we fit that plane through the floor
   pixels. Because it's fitted per frame, the model's unknown scale and offset don't
   matter, and neither does the robot pitching a bit while driving.
2. An obstacle sticks UP from the floor: at its pixels the disparity is clearly larger
   (closer) than the floor plane predicts for that spot.
3. "Close" = at least as close as the floor at the WARNING LINE (an image row,
   `warn_row`). An upright object has the same disparity as the floor where it stands,
   so this means "it stands on or before the warning line". No metres needed, only a
   one-off choice of that line: put an object at the distance you want to be warned at
   and see which row its foot is on.
4. Only the robot's PATH counts, in 3D. A pixel's disparity says which floor row it
   "stands on" (= how far away it is). At that distance the robot is a certain width
   and height in the image: wide and tall when near, small when far, shrinking
   towards the horizon. A pixel only counts if it lies inside that robot-sized window:
   - sideways: between the path's left and right edge (`path_bottom`).
   - upwards: no higher than the robot (`height_ratio`). A table top the robot fits
     under is not an obstacle; its legs are.
5. If no floor can be found at all, something is covering it, i.e. standing right in
   front of the camera. That is reported as "blocked" and counts as a warning.

All fractions are of the image size (0 = left/top, 1 = right/bottom), so they hold for
any resolution. Not a safety feature yet: it only reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ObstacleConfig:
    # Defaults = the calibration for the robot standing (README §6.9); the real values
    # come from config.local.yaml. Left and right edge of the robot's path where it
    # leaves the image at the bottom (fractions of the width; <0 or >1 = beyond the picture).
    path_bottom: tuple = (0.34, 0.64)
    vanish_x: float = 0.488     # where the robot's centre line meets the horizon
    height_ratio: float = 0.37  # (robot top above camera) / (camera above floor): higher things are passed under
    warn_row: float = 1.06      # obstacle standing on/below this image row -> warning (bigger = warn later)
    floor_from: float = 0.70    # the floor is searched below this row
    stick_out: float = 0.035    # a pixel "sticks up" if it is this many rows (fraction of height) above its foot
    min_area: float = 0.002     # fraction of the image that must be close obstacle (ignores specks)
    min_floor: float = 0.04     # fraction of the floor search area that must look like floor, else "blocked"
    work_width: int = 192       # analysis resolution (the depth map is downscaled to this)


def from_settings(o) -> ObstacleConfig:
    """Build the checker settings from the `obstacles:` section of config.local.yaml."""
    return ObstacleConfig(path_bottom=(o.path_left, o.path_right), vanish_x=o.vanish_x,
                          height_ratio=o.height_ratio, warn_row=o.warn_row)


@dataclass
class ObstacleResult:
    status: str                  # "clear", "warning", or "blocked" (no floor visible: something covers it)
    nearness: float              # closest thing in the path: 1.0 = at the warning line, 2.0 = half that distance
    area: float                  # fraction of the image covered by close obstacles in the path
    mask: np.ndarray | None      # close-obstacle pixels at work resolution (bool), for display
    reason: str = ""
    horizon: float = 0.0         # estimated horizon row (fraction), for display
    classes: np.ndarray | None = None  # per pixel, for display: see CLASSES
    # Per pixel at work resolution, for checks with their own warning distance (people.warn_row):
    in_path: np.ndarray | None = None  # sticks up from the floor, inside the robot-sized window (bool)
    foot: np.ndarray | None = None     # the image row it stands on: >= a warn_row = on or before that line

    @property
    def warn(self) -> bool:
        # "blocked" counts as a warning: the floor right in front should always be
        # visible, so if it isn't, something is standing on it.
        return self.status != "clear"


# What each pixel was judged as (ObstacleResult.classes), with its display colour (BGR).
# Only RED triggers the warning. The others show what was seen and why it was left out.
CLASSES = {
    5: ("the target go-to drives to (ignored)", (80, 200, 80)),
    4: ("in path, close: WARNS", (40, 40, 255)),
    3: ("in path, beyond the warning line", (0, 165, 255)),
    2: ("close, but beside the path", (255, 120, 40)),
    1: ("close, but above the robot", (230, 220, 60)),
}


class ObstacleChecker:
    def __init__(self, cfg: ObstacleConfig | None = None):
        self.cfg = cfg or ObstacleConfig()
        self._cache_shape = None

    def _prepare(self, h: int, w: int):
        if self._cache_shape == (h, w):
            return
        self._cache_shape = (h, w)
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        self._u, self._v = uu / (w - 1), vv / (h - 1)    # 0..1 image coordinates
        self._floor_region = self._v >= self.cfg.floor_from

    def box(self, horizon: float) -> tuple[float, float, float]:
        """The yellow box, as (left x, right x, top row) fractions: the robot-sized window at
        the warning line. Shared by the check and the drawing, so they can't disagree."""
        c = self.cfg
        tw = (c.warn_row - horizon) / (1.0 - horizon)
        return (c.vanish_x + (c.path_bottom[0] - c.vanish_x) * tw,
                c.vanish_x + (c.path_bottom[1] - c.vanish_x) * tw,
                horizon - c.height_ratio * (c.warn_row - horizon))

    def check(self, disparity: np.ndarray, ignore: tuple[float, float, float, float] | None = None) -> ObstacleResult:
        """ignore: a box (left, top, right, bottom as fractions of the image) whose pixels are never an
        obstacle: the piece go-to is driving to, which is in the path by definition."""
        c = self.cfg
        H, W = disparity.shape
        w = c.work_width
        h = max(8, round(H * w / W))
        d = cv2.resize(disparity.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        # Far background = ~0, so that ratios of d are ratios of 1/distance. Only the
        # "nearness" number uses this; the warning decision doesn't depend on it.
        lo, hi = np.percentile(d, (1, 99))
        d = (d - lo) / max(hi - lo, 1e-6)
        self._prepare(h, w)

        def blocked(reason: str) -> ObstacleResult:
            return ObstacleResult("blocked", 0.0, 1.0, self._floor_region.copy(), reason)

        # --- 1. floor candidates -------------------------------------------------
        # On the floor, disparity grows steadily towards the bottom of the image.
        # On an obstacle's upright face it stays ~constant. So floor pixels are the
        # ones whose vertical slope is close to the typical floor slope.
        gy = np.zeros_like(d)
        gy[1:-1] = (d[2:] - d[:-2]) / 2
        gy = cv2.blur(gy, (5, 3))
        slope = float(np.percentile(gy[self._floor_region], 85))  # floor-like even when the floor is the minority
        if slope <= 1e-4:
            return blocked("no floor visible")
        cand = self._floor_region & (gy > 0.5 * slope) & (gy < 2.5 * slope)
        min_px = c.min_floor * self._floor_region.sum()
        if cand.sum() < min_px:
            return blocked("no floor visible")

        # --- 2. fit the floor plane: d = a + b*v + c*u (robust: fit, drop outliers, repeat)
        A = np.stack([np.ones(int(cand.sum()), np.float32), self._v[cand], self._u[cand]], axis=1)
        y = d[cand]
        keep = np.ones(len(y), bool)
        coef = None
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            r = y - A @ coef
            mad = np.median(np.abs(r[keep])) + 1e-6
            keep = np.abs(r) < 2.5 * 1.4826 * mad
            if keep.sum() < min_px:
                return blocked("no floor visible")
        a0, b, cu = (float(x) for x in coef)
        if b <= 0:
            return blocked("floor not recognised")
        # --- 3. where does each pixel "stand"? ----------------------------------
        # foot = the floor row with the same disparity as the pixel = its distance,
        # expressed as an image row (bottom of the image = nearest floor we can see).
        foot = (d - a0 - cu * self._u) / b
        # Horizon: the row where the floor plane reaches the far-background level (0).
        horizon = float(np.clip(-(a0 + cu * c.vanish_x) / b, 0.3, c.warn_row - 0.05))
        # t: 0 at the horizon, 1 at the bottom row, >1 nearer than any visible floor.
        t = (foot - horizon) / (1.0 - horizon)

        # --- 4. the robot-sized window at that distance ---------------------------
        left = c.vanish_x + (c.path_bottom[0] - c.vanish_x) * t
        right = c.vanish_x + (c.path_bottom[1] - c.vanish_x) * t
        top = horizon - c.height_ratio * (foot - horizon)  # robot's top edge at that distance
        # The yellow box = that window at the warning line. It is a hard limit: anything
        # left of, right of or above it is ignored. (Without this, things NEARER than the
        # warning line get a wider, taller window than the box, so objects visibly beside
        # the box could still warn. Its bottom stays open: below it = nearer = must warn.)
        box_left, box_right, box_top = self.box(horizon)
        sticks_up = (t > 0) & ((foot - self._v) > c.stick_out)  # the floor itself has foot == v
        target = np.zeros(d.shape, bool)
        if ignore is not None:   # the target: in the path on purpose, not an obstacle
            u0, v0, u1, v1 = ignore
            target = (self._u >= u0) & (self._u <= u1) & (self._v >= v0) & (self._v <= v1)
            sticks_up &= ~target
        beside = (self._u < np.maximum(left, box_left)) | (self._u > np.minimum(right, box_right))
        above = self._v < np.maximum(top, box_top)
        close = foot >= c.warn_row                         # stands on or before the warning line
        in_path = sticks_up & ~beside & ~above
        obstacle = in_path & close
        area = float(obstacle.mean())

        classes = np.zeros(d.shape, np.uint8)
        classes[sticks_up & close & ~beside & above] = 1
        classes[sticks_up & close & beside] = 2
        classes[in_path & ~close] = 3
        classes[obstacle] = 4
        classes[target] = 5

        # Nearness: 1/distance relative to the warning line (distance ~ 1/(foot - horizon)).
        if in_path.mean() >= c.min_area:
            near_foot = float(np.percentile(foot[in_path], 95))
            nearness = max(0.0, (near_foot - horizon) / (c.warn_row - horizon))
        else:
            nearness = 0.0

        status = "warning" if area >= c.min_area else "clear"
        return ObstacleResult(status, nearness, area, obstacle, horizon=horizon, classes=classes,
                              in_path=in_path, foot=foot)


def draw_overlay(bgr: np.ndarray, res: ObstacleResult, cfg: ObstacleConfig,
                 people_row: float | None = None) -> np.ndarray:
    """Camera image with the path polygon, warning line, obstacle pixels and the verdict.
    people_row: also draw the people's warning line (people.warn_row) across the path, in magenta."""
    img = bgr.copy()
    H, W = img.shape[:2]
    s = max(1.0, H / 720)
    color = (80, 200, 80) if res.status == "clear" else (40, 40, 240)
    if res.classes is not None:
        cls = cv2.resize(res.classes, (W, H), interpolation=cv2.INTER_NEAREST)
        for k, (_, bgr_col) in CLASSES.items():
            m = cls == k
            if m.any():
                img[m] = (0.45 * img[m] + 0.55 * np.array(bgr_col)).astype(np.uint8)
    elif res.mask is not None and res.mask.any():   # "blocked": no per-pixel judgement, mark the area
        m = cv2.resize(res.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (0.45 * img[m] + 0.55 * np.array([40, 40, 255])).astype(np.uint8)
    lw = max(2, int(2 * s))
    if res.horizon > 0:
        # The robot's path on the floor (two lines to the horizon) and the yellow box:
        # the robot-sized window at the warning line. Only things inside the box's sides
        # and below its top can warn; below its bottom edge = nearer = warns too.
        hz = res.horizon
        def edge(xb: float, row: float) -> tuple[int, int]:
            tt = (row - hz) / (1.0 - hz)
            return int((cfg.vanish_x + (xb - cfg.vanish_x) * tt) * (W - 1)), int(row * (H - 1))
        for xb in cfg.path_bottom:
            cv2.line(img, edge(xb, 1.0), edge(xb, hz + 0.02), color, lw, cv2.LINE_AA)
        bl, br, bt = ObstacleChecker(cfg).box(hz)   # the same box the check uses
        xl, xr = int(bl * (W - 1)), int(br * (W - 1))
        yt, yw = int(bt * (H - 1)), int(cfg.warn_row * (H - 1))
        cv2.rectangle(img, (xl, yt), (xr, yw), (0, 220, 255), lw)
        if people_row is not None and people_row > hz:   # a person at or before this line stops everything
            cv2.line(img, edge(cfg.path_bottom[0], people_row), edge(cfg.path_bottom[1], people_row),
                     (230, 80, 230), lw, cv2.LINE_AA)
    text = {"warning": "OBSTACLE AHEAD", "clear": "clear", "blocked": f"BLOCKED ({res.reason})"}[res.status]
    if res.status != "blocked":
        text += f"   near {res.nearness:.2f}  area {res.area * 100:.1f}%"
    # Legend, top left.
    fs, y = 0.55 * s, int(28 * s)
    for k in sorted(CLASSES, reverse=True):
        name, bgr_col = CLASSES[k]
        cv2.rectangle(img, (int(14 * s), y - int(14 * s)), (int(34 * s), y + int(2 * s)), bgr_col, -1)
        cv2.putText(img, name, (int(42 * s), y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(4 * s), cv2.LINE_AA)
        cv2.putText(img, name, (int(42 * s), y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
        y += int(24 * s)
    cv2.putText(img, "yellow box: outside its sides/top is ignored", (int(14 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 0, 0), int(4 * s), cv2.LINE_AA)
    cv2.putText(img, "yellow box: outside its sides/top is ignored", (int(14 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 220, 255), max(1, int(s)), cv2.LINE_AA)
    if people_row is not None:
        y += int(24 * s)
        for col, th in (((0, 0, 0), int(4 * s)), ((230, 80, 230), max(1, int(s)))):
            cv2.putText(img, "magenta line: a person before it stops everything", (int(14 * s), y),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    org = (int(20 * s), int(H - 20 * s))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), int(5 * s), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, color, int(2 * s), cv2.LINE_AA)
    return img
