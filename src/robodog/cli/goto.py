"""Click a spot on the floor in the robot's camera view; hold G and the robot drives there.

    uv run robodog-goto                        # robot camera, view only: click to see distances (nothing moves)
    uv run robodog-goto --allow-motion         # real robot: R stand, click a goal, HOLD G to go
    uv run robodog-goto --fake                 # fake robot + drawn floor: try everything with no robot
    uv run robodog-goto --fake --camera 0 --lens pinhole --hfov 70   # fake robot, laptop webcam

Driving to a goal needs `measured: true` in the camera section of config.local.yaml (the camera's
height / offset measured and its pitch calibrated with K). Until then you can still click to see
distances, and drive with the teleop keys.

  MOUSE   click the floor        set the goal (shown as a ring; red = refused)
  HOLD    G                      go to the goal. Let go = stop (the goal is kept)
  HOLD    W/S A/D Q/E (arrows)   drive by hand, like robodog-teleop (overrides G)
  CLICK   R stand   F lie down   C clear goal   1/2/3 speed (hand driving)
          K  calibrate: click the bases of objects at the --calib-distances, in that order
          SPACE  E-STOP (stop + lie down, latched)    Esc  lie down and quit

The window must have focus: clicking another window stops the robot.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from ..camera_model import CameraModel, fit_camera
from ..config import REPO_ROOT, load_config
from ..goto import GoTo, GoToLimits
from ..safety import SafeDog, SafetyError
from ..teleop import Teleop
from .status import MODES
from .teleop import key_maps

RATE_HZ = 30
VIEW_W = 960       # camera view width in the window
PANEL_H = 170
BG, FG, DIM = (24, 26, 30), (230, 230, 230), (120, 124, 132)
GREEN, AMBER, RED, BLUE, CYAN = (80, 200, 120), (240, 180, 60), (235, 70, 70), (90, 160, 240), (80, 210, 230)
GRID_M = (0.5, 1.0, 1.5, 2.0, 3.0)


def synthetic_floor(w=1280, h=720) -> np.ndarray:
    """A plain image for --fake without a camera: grey "wall" above, darker "floor" below."""
    img = np.full((h, w, 3), 70, np.uint8)
    img[h // 3:] = 45
    return img


def grid_polylines(model: CameraModel) -> list[tuple[float, list[tuple[float, float]]]]:
    """Distance arcs on the floor, in metres from the point UNDER THE CAMERA (how you'd measure
    with a tape), as pixel polylines."""
    out = []
    for d in GRID_M:
        pts = []
        for a in range(-70, 71, 4):
            uv = model.floor_to_pixel(model.forward_offset_m + d * math.cos(math.radians(a)),
                                      d * math.sin(math.radians(a)))
            if uv:
                pts.append(uv)
        if len(pts) >= 2:
            out.append((d, pts))
    return out


class Frames:
    """Newest camera frame, a still image, or the synthetic floor."""

    def __init__(self, source: str | None, image: str | None):
        self.reader = None
        self.still = None
        if image:
            self.still = cv2.imread(image)
            if self.still is None:
                raise SystemExit(f"can't read image {image}")
        elif source is None:
            self.still = synthetic_floor()
        else:
            from ..vision.camera import LatestFrameReader

            print(f"opening camera {source} ...")
            try:
                self.reader = LatestFrameReader(source)
            except RuntimeError as e:
                raise SystemExit(f"{e} (laptop on the robot's Wi-Fi? try `ping 192.168.234.1`)")

    def get(self) -> np.ndarray | None:
        if self.still is not None:
            return self.still
        frame, _, _ = self.reader.get()
        return frame

    def close(self):
        if self.reader:
            self.reader.close()


def run(dog: SafeDog, cfg, frames: Frames, args, fake: bool):
    import pygame as pg

    cam_cfg = cfg.camera
    drive_ok = fake or cam_cfg.measured
    limits = GoToLimits(cfg.goto.max_goal_m, min(cfg.goto.max_vx, cfg.safety.max_vx),
                        min(cfg.goto.max_yaw, cfg.safety.max_yaw), cfg.goto.arrive_m, cfg.goto.timeout_s,
                        cfg.goto.stop_before_m, cfg.goto.odometry)
    goto = GoTo(limits)
    print(f"go-to odometry: {limits.odometry}")
    tele = Teleop(dog, log=print)
    hold_keys, click_keys = key_maps()

    # Wait for the first frame: the camera model needs the image size.
    t_end = time.monotonic() + 10
    frame = frames.get()
    while frame is None and time.monotonic() < t_end:
        time.sleep(0.05)
        frame = frames.get()
    if frame is None:
        raise SystemExit("no camera frames after 10 s")
    fh, fw = frame.shape[:2]
    lens, hfov, pitch = args.lens or cam_cfg.lens, args.hfov or cam_cfg.hfov_deg, cam_cfg.pitch_deg

    def make_model():
        return CameraModel(fw, fh, hfov, cam_cfg.height_m, pitch, cam_cfg.forward_offset_m, lens)

    model = make_model()
    grid = grid_polylines(model)
    scale = VIEW_W / fw
    view_h = int(fh * scale)

    pg.init()
    pg.key.set_repeat()
    screen = pg.display.set_mode((VIEW_W, view_h + PANEL_H))
    pg.display.set_caption("robodog go-to" + (" (FAKE)" if fake else ""))
    big, mid, small = (pg.font.SysFont("dejavusans", 26, bold=True), pg.font.SysFont("dejavusans", 17),
                       pg.font.SysFont("dejavusans", 14))

    calib_dists = [float(x) for x in args.calib_distances.split(",") if x.strip()]
    calibrating = False
    calib_clicks: list[tuple[float, float, float]] = []  # (u, v, distance) while calibrating
    message = "R to stand, then click the floor to set a goal, hold G to go"
    if not drive_ok:
        message = "camera not measured yet: click to see distances (G disabled). See README §6.7"
    status, next_status = {}, 0.0
    last_click: tuple[float, float, float, float] | None = None  # u, v, X, Y
    step = goto.step((0.0, 0.0, 0.0), False, 0.0)
    clock = pg.time.Clock()
    last = time.monotonic()
    next_log = 0.0
    pose = (0.0, 0.0, 0.0)

    try:
        while not tele.quit:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    tele.quit = True
                elif ev.type == pg.KEYDOWN:
                    if ev.key in click_keys:
                        tele.action(click_keys[ev.key])
                    elif ev.key == pg.K_c:
                        goto.clear()
                        message = "goal cleared"
                    elif ev.key == pg.K_k:
                        calibrating, calib_clicks = not calibrating, []
                        message = (f"CALIBRATE: click the base of the object at {calib_dists[0]:.2f} m (1 of {len(calib_dists)})"
                                   if calibrating else "calibration cancelled")
                elif ev.type in (pg.WINDOWFOCUSLOST, pg.WINDOWHIDDEN, pg.WINDOWMINIMIZED):
                    tele.release_all()
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[1] < view_h:
                    u, v = ev.pos[0] / scale, ev.pos[1] / scale
                    if calibrating:
                        calib_clicks.append((u, v, calib_dists[len(calib_clicks)]))
                        if len(calib_clicks) < len(calib_dists):
                            n = len(calib_clicks)
                            message = (f"CALIBRATE: click the base of the object at {calib_dists[n]:.2f} m "
                                       f"({n + 1} of {len(calib_dists)})")
                            continue
                        calibrating = False
                        fit = fit_camera(model, calib_clicks)
                        lens, hfov, pitch = fit.lens, fit.hfov_deg, fit.pitch_deg
                        model = make_model()
                        grid = grid_polylines(model)
                        errs = "  ".join(f"{d:.2f}->{m:.2f}" if m is not None else f"{d:.2f}->?" for d, m in fit.points)
                        message = (f"fit: {lens}, hfov {hfov}, pitch {pitch} | {errs} "
                                   f"(worst {fit.worst_error:.0%}). Copy the terminal lines into config.local.yaml")
                        print("\ncalibration result (measured -> fitted distance, from the point under the lens):")
                        for d, m in fit.points:
                            print(f"  {d:.2f} m -> {m:.2f} m" if m is not None else f"  {d:.2f} m -> not on the floor")
                        print(f"worst error {fit.worst_error:.0%}. Put these in the camera section of config.local.yaml:")
                        print(f'  lens: "{lens}"\n  hfov_deg: {hfov}\n  pitch_deg: {pitch}\n')
                        continue
                    pt = model.pixel_to_floor(u, v)
                    if pt is None:
                        message = "that's not the floor (at or above the horizon)"
                        continue
                    last_click = (u, v, *pt)
                    from_cam = math.hypot(pt[0] - model.forward_offset_m, pt[1])
                    if not drive_ok:
                        message = (f"floor point: {from_cam:.2f} m from under the camera "
                                   f"({pt[0]:.2f} m ahead of the robot's middle, {pt[1]:+.2f} m left)")
                        continue
                    err = goto.set_goal(pt[0], pt[1], pose)
                    message = err or f"goal: {from_cam:.2f} m from under the camera. Hold G to go"

            focused = pg.key.get_focused()
            pressed = pg.key.get_pressed() if focused else None
            hand = [name for key, name in hold_keys.items() if pressed is not None and pressed[key]]
            g_held = bool(pressed is not None and pressed[pg.K_g])
            if not focused:
                tele.release_all()

            try:
                pose = dog.pose()
            except Exception as e:
                message = f"pose error: {e!r}"

            now = time.monotonic()
            dt, last = now - last, now
            going = (g_held and not hand and drive_ok and tele.standing and dog.settled() and not dog.fault)
            if hand and goto.goal is not None and goto.lim.odometry == "commanded":
                # Our own position count only knows go-to's commands, not hand driving.
                goto.clear()
                message = "goal cleared: hand driving (odometry 'commanded' can't follow it). Click again"
            step = goto.step(pose, going, dt)
            if going and now >= next_log:  # what we commanded vs what the robot's estimate says
                next_log = now + 0.5
                print(f"go-to [{step.state}] goal {step.dist:.2f} m from middle | commanded {goto.commanded_m:.2f} m, "
                      f"robot estimate {goto.robot_m:.2f} m, needed {goto.needed_m:.2f} m | pose x {pose[0]:+.2f} "
                      f"y {pose[1]:+.2f} heading {math.degrees(pose[2]):+.0f}")
            if going:
                try:
                    dog.set_velocity(step.vx, 0.0, step.yaw)
                except SafetyError as e:
                    message = f"stopped: {e}"
            else:
                tele.set_held(hand)
                tele.update(dt)  # hand driving, or zero speed (keeps the dead-man switch fed)

            if now >= next_status:
                next_status = now + 0.5
                try:
                    status = dog.status()
                except Exception:
                    status = {}

            draw(screen, (big, mid, small), frames.get(), scale, view_h, model, grid, goto, step, pose, tele,
                 status, message, calibrating, last_click, drive_ok, fake, g_held)
            clock.tick(RATE_HZ)
    finally:
        print("lying down and closing...")
        dog.close()
        pg.quit()


def draw(screen, fonts, frame, scale, view_h, model, grid, goto, step, pose, tele, status, message,
         calibrating, last_click, drive_ok, fake, g_held):
    import pygame as pg

    big, mid, small = fonts
    screen.fill(BG)
    if frame is not None:
        rgb = cv2.cvtColor(cv2.resize(frame, (int(frame.shape[1] * scale), view_h)), cv2.COLOR_BGR2RGB)
        screen.blit(pg.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))

    def sp(uv):
        return int(uv[0] * scale), int(uv[1] * scale)

    # Distance grid on the floor: if the geometry is right, these arcs match real distances.
    for d, pts in grid:
        pg.draw.lines(screen, CYAN, False, [sp(p) for p in pts], 1)
        lab = small.render(f"{d:g} m", True, CYAN)
        screen.blit(lab, (sp(pts[len(pts) // 2])[0] + 4, sp(pts[len(pts) // 2])[1] - 16))
    hv = model.horizon_v()
    if hv is not None and 0 <= hv < model.height:
        pg.draw.line(screen, DIM, (0, int(hv * scale)), (screen.get_width(), int(hv * scale)), 1)

    if last_click and not goto.goal:
        pg.draw.circle(screen, AMBER, sp(last_click[:2]), 6, 2)

    # The goal, re-projected from the robot's position estimate every frame. If it slides off
    # the real spot while driving, the geometry or the odometry is off.
    rel = goto.relative(pose)
    if rel is not None:
        uv = model.floor_to_pixel(*rel)
        col = {"arrived": GREEN, "timeout": RED, "lost position": RED}.get(step.state, AMBER)
        if uv:
            pg.draw.circle(screen, col, sp(uv), 14, 3)
            pg.draw.circle(screen, col, sp(uv), 3)
        else:
            screen.blit(mid.render("goal out of view", True, col), (12, view_h - 28))

    if calibrating:
        screen.blit(big.render("CALIBRATE: click each object's base, in order", True, AMBER), (12, 10))

    # Panel
    y = view_h + 8
    if tele.state().fault:
        title, col = "E-STOP" if "E-STOP" in tele.state().fault else "FAULT", RED
    elif not tele.standing:
        title, col = "LYING DOWN", DIM
    elif step.state in ("turning", "driving"):
        title, col = f"GOING ({step.state})", GREEN
    else:
        title, col = "STANDING", BLUE
    screen.blit(big.render(title, True, col), (12, y))
    tags = ("FAKE ROBOT  " if fake else "") + ("" if drive_ok else "CAMERA NOT MEASURED: G off")
    if tags:
        surf = mid.render(tags, True, AMBER)
        screen.blit(surf, (screen.get_width() - surf.get_width() - 12, y + 6))
    y += 36
    batt, mode = status.get("battery"), status.get("mode")
    x, yy, yaw = pose
    screen.blit(mid.render(f"battery {batt}%   mode {mode} ({MODES.get(mode, '?')})   "
                           f"pose x {x:+.2f} y {yy:+.2f} m, heading {math.degrees(yaw):+.0f} deg", True, FG), (12, y))
    y += 24
    if step.dist is not None:
        rel = goto.relative(pose)
        from_cam = math.hypot(rel[0] - model.forward_offset_m, rel[1])
        g = (f"goal {from_cam:.2f} m from camera, {step.bearing_deg:+.0f} deg   [{step.state}]   "
             f"driven: commanded {goto.commanded_m:.2f} m / robot says {goto.robot_m:.2f} m")
        if step.state == "paused" and not g_held:
            g += "   hold G to go"
    else:
        g = "no goal: click the floor"
    screen.blit(mid.render(g, True, FG), (12, y))
    y += 26
    msg = tele.state().fault or message
    screen.blit(small.render(msg[:120], True, RED if tele.state().fault else AMBER), (12, y))
    y += 22
    for h in ("CLICK floor = goal   HOLD G = go   HOLD WASD/QE = hand drive   R stand  F lie  C clear  K calibrate",
              "SPACE e-stop   Esc quit   (window must have focus)"):
        screen.blit(small.render(h, True, DIM), (12, y))
        y += 18
    if not pg.key.get_focused():
        screen.blit(mid.render("click this window to control", True, AMBER), (12, 12))
    pg.display.flip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="shared config (default: config.local.yaml)")
    ap.add_argument("--allow-motion", action="store_true", help="real robot may move (also needs allow_motion in config.machine.yaml)")
    ap.add_argument("--fake", action="store_true", help="fake robot that simulates driving (nothing real moves)")
    ap.add_argument("--camera", help="camera source: RTSP URL, video file or webcam index (default: robot camera; "
                                     "with --fake: a drawn floor)")
    ap.add_argument("--image", help="use a still image as the camera")
    ap.add_argument("--lens", choices=["fisheye", "pinhole"], help="override camera.lens (e.g. pinhole for a webcam)")
    ap.add_argument("--hfov", type=float, help="override camera.hfov_deg (e.g. 70 for a laptop webcam)")
    ap.add_argument("--calib-distances", default="1.5,2.5,3.0",
                    help="K calibration: distances of the objects you'll click, in click order, measured ON THE "
                         "FLOOR from the point under the camera lens (default 1.5,2.5,3.0). One distance fits only "
                         "the tilt; two or more also fit the lens and field of view")
    args = ap.parse_args()

    try:
        import pygame  # noqa: F401
    except ImportError:
        raise SystemExit("robodog-goto needs pygame: run `uv sync` first")

    cfg = load_config(args.config)
    if args.fake:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from fake_robot import FakeHighLevel

        cfg.safety.allow_motion = True  # fake robot only: nothing real can move
        cfg.network.laptop_ip = cfg.network.laptop_ip or "127.0.0.1"
        robot = FakeHighLevel(battery=80, simulate=True)
        allow = True
        source = args.camera
    else:
        allow = bool(args.allow_motion and cfg.safety.allow_motion)
        if not allow:
            print("VIEW ONLY: the robot won't move (needs --allow-motion and allow_motion: true in config.machine.yaml)")
        from ..sdk import make_highlevel

        robot = make_highlevel(cfg.sdk_dir)
        source = args.camera or f"rtsp://{cfg.network.robot_ip}:8554/test"

    dog = SafeDog(robot, cfg, allow_motion=allow, log=lambda m: print(f"[safedog] {m}"))
    try:
        dog.connect()
    except SafetyError as e:
        raise SystemExit(str(e))
    dog.start()

    frames = Frames(source, args.image)
    try:
        run(dog, cfg, frames, args, args.fake)
    except KeyboardInterrupt:
        print("\nCtrl+C")
        dog.close()
    finally:
        frames.close()


if __name__ == "__main__":
    main()
