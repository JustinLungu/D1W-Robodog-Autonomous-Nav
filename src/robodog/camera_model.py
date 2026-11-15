"""Camera geometry: a pixel the user clicks -> a point on the floor, and back.

Frames used here
----------------
- Camera: x right, y down, z forward (along the lens axis), like OpenCV.
- Robot ("floor") frame: X forward, Y left, measured on the floor from the robot's centre.

Lens model
----------
The robot's front camera is very wide-angle. AgiBot lists 122 deg horizontal and 70 deg
vertical field of view. A normal (pinhole) model can't give both on a 16:9 image (122 deg
wide would make it ~91 deg tall), but an equidistant fisheye (angle from the centre grows
linearly with pixel distance) gives 122 x 68.6 deg, so that's the default. "pinhole" is
there for ordinary cameras (e.g. a laptop webcam). A checkerboard calibration can replace
both later.

Assumes a flat floor and the robot level (no roll), which is what we have indoors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CameraModel:
    width: int               # image size in pixels
    height: int
    hfov_deg: float          # horizontal field of view
    height_m: float          # lens height above the floor
    pitch_deg: float         # tilt of the lens axis: + = looking down, - = looking up (our robot looks up)
    forward_offset_m: float  # how far the lens is in front of the robot's centre
    lens: str = "fisheye"    # "fisheye" (equidistant) or "pinhole"

    def __post_init__(self):
        if self.lens not in ("fisheye", "pinhole"):
            raise ValueError(f"lens must be 'fisheye' or 'pinhole', got {self.lens!r}")
        half = math.radians(self.hfov_deg) / 2
        # Focal length in pixels: the image edge (width/2 from the centre) is at hfov/2.
        self.f = (self.width / 2) / (half if self.lens == "fisheye" else math.tan(half))
        self.cx, self.cy = self.width / 2, self.height / 2

    # ------------------------------------------------------------------ pixel <-> ray
    def pixel_to_ray(self, u: float, v: float) -> tuple[float, float, float]:
        """Unit direction (camera frame) of the light that lands on pixel (u, v)."""
        du, dv = u - self.cx, v - self.cy
        if self.lens == "pinhole":
            x, y, z = du / self.f, dv / self.f, 1.0
            n = math.sqrt(x * x + y * y + 1.0)
            return x / n, y / n, z / n
        r = math.hypot(du, dv)
        if r == 0:
            return 0.0, 0.0, 1.0
        theta = r / self.f  # equidistant: angle off the axis is proportional to pixel distance
        s = math.sin(theta)
        return s * du / r, s * dv / r, math.cos(theta)

    def ray_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float] | None:
        """Pixel where a camera-frame direction lands, or None if it's behind / outside the lens."""
        if self.lens == "pinhole":
            if z <= 1e-6:
                return None
            return self.cx + self.f * x / z, self.cy + self.f * y / z
        n = math.sqrt(x * x + y * y + z * z)
        theta = math.acos(max(-1.0, min(1.0, z / n)))
        if theta > math.pi / 2:  # behind the camera
            return None
        rxy = math.hypot(x, y)
        if rxy == 0:
            return self.cx, self.cy
        r = self.f * theta
        return self.cx + r * x / rxy, self.cy + r * y / rxy

    # ------------------------------------------------------------------ pixel <-> floor
    def pixel_to_floor(self, u: float, v: float) -> tuple[float, float] | None:
        """Floor point (X forward, Y left, metres from the robot's centre) seen at pixel (u, v).

        None if the pixel looks at or above the horizon (it never hits the floor).
        """
        x, y, z = self.pixel_to_ray(u, v)
        p = math.radians(self.pitch_deg)
        forward = z * math.cos(p) - y * math.sin(p)  # tilt the camera ray down by the pitch
        down = z * math.sin(p) + y * math.cos(p)
        if down <= 1e-6:
            return None
        t = self.height_m / down  # stretch the ray until it has dropped to the floor
        return self.forward_offset_m + t * forward, -t * x

    def floor_to_pixel(self, X: float, Y: float) -> tuple[float, float] | None:
        """Pixel where the floor point (X forward, Y left, from the robot's centre) appears, or None."""
        fwd, right, down = X - self.forward_offset_m, -Y, self.height_m
        p = math.radians(self.pitch_deg)
        z = fwd * math.cos(p) + down * math.sin(p)   # undo the pitch
        y = -fwd * math.sin(p) + down * math.cos(p)
        uv = self.ray_to_pixel(right, y, z)
        if uv is None:
            return None
        u, v = uv
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return uv

    def horizon_v(self) -> float | None:
        """Image row of the horizon straight ahead (rows below it see the floor), if in view."""
        uv = self.ray_to_pixel(0.0, -math.sin(math.radians(self.pitch_deg)), math.cos(math.radians(self.pitch_deg)))
        return None if uv is None else uv[1]

    def nearest_visible_floor_m(self) -> float | None:
        """Distance (from the robot's centre) of the closest floor point at the bottom-centre of the image."""
        pt = self.pixel_to_floor(self.cx, self.height - 1)
        return None if pt is None else pt[0]


@dataclass
class CameraFit:
    lens: str
    hfov_deg: float
    pitch_deg: float
    points: list[tuple[float, float | None]]  # (measured distance, distance the fitted model gives) per click

    @property
    def worst_error(self) -> float:
        """Largest relative distance error over the clicked points (0.05 = 5 %)."""
        return max((abs(m - d) / d if m is not None else 1.0) for d, m in self.points)


def _floor_dist(lens: str, f: float, cx: float, cy: float, h: float, pitch: float, u: float, v: float) -> float | None:
    """Floor distance from the point under the lens to what pixel (u, v) sees (fast, no objects)."""
    du, dv = u - cx, v - cy
    if lens == "pinhole":
        x, y, z = du / f, dv / f, 1.0
    else:
        r = math.hypot(du, dv)
        if r == 0:
            x, y, z = 0.0, 0.0, 1.0
        else:
            th = r / f
            x, y, z = math.sin(th) * du / r, math.sin(th) * dv / r, math.cos(th)
    forward = z * math.cos(pitch) - y * math.sin(pitch)
    down = z * math.sin(pitch) + y * math.cos(pitch)
    if down <= 1e-9:
        return None
    t = h / down
    return math.hypot(t * forward, t * x)


def fit_camera(model: CameraModel, clicks: list[tuple[float, float, float]],
               lenses: tuple[str, ...] = ("fisheye", "pinhole")) -> CameraFit:
    """Calibrate from objects at known distances.

    clicks: (u, v, distance) per object, where (u, v) is where the object touches the floor in
    the image and distance is measured ON THE FLOOR from the point directly under the camera lens.

    One click: only the tilt (pitch) is fitted; lens and field of view stay as in `model`.
    Two or more: tilt, field of view and lens type are fitted together, which also fixes a
    wrong lens spec. Uses the model's width/height and camera height.
    """
    cx, cy, h = model.width / 2, model.height / 2, model.height_m

    def score(lens, hfov, pitch_deg):
        half = math.radians(hfov) / 2
        f = (model.width / 2) / (half if lens == "fisheye" else math.tan(half))
        p = math.radians(pitch_deg)
        err = 0.0
        for u, v, d in clicks:
            m = _floor_dist(lens, f, cx, cy, h, p, u, v)
            err += 4.0 if m is None else ((m - d) / d) ** 2
        return err

    def frange(a, b, step):
        n = int(round((b - a) / step))
        return [a + i * step for i in range(n + 1)]

    if len(clicks) == 1:
        cands = [(model.lens, model.hfov_deg)]
    else:
        # A pinhole lens can't reach 180 deg, so its field of view range stops lower.
        cands = [(lens, hfov) for lens in lenses
                 for hfov in frange(50.0, 150.0 if lens == "fisheye" else 140.0, 1.0)]

    # Coarse grid, then refine around the best.
    best = min(((score(l, hf, pt), l, hf, pt) for l, hf in cands for pt in frange(-45.0, 45.0, 0.25)))
    _, lens, hfov, pitch = best
    hf_steps = [hfov] if len(clicks) == 1 else frange(hfov - 1.0, hfov + 1.0, 0.1)
    best = min(((score(lens, hf, pt), lens, hf, pt) for hf in hf_steps for pt in frange(pitch - 0.5, pitch + 0.5, 0.02)))
    _, lens, hfov, pitch = best

    half = math.radians(hfov) / 2
    f = (model.width / 2) / (half if lens == "fisheye" else math.tan(half))
    pts = [(d, _floor_dist(lens, f, cx, cy, h, math.radians(pitch), u, v)) for u, v, d in clicks]
    return CameraFit(lens, round(hfov, 1), round(pitch, 2), pts)
