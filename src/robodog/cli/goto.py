"""Click a spot on the floor in the robot's camera view; hold G and the robot drives there.

    uv run robodog-goto                        # robot camera, view only: click to see distances (nothing moves)
    uv run robodog-goto --allow-motion         # real robot: R stand, click a goal, HOLD G to go
    uv run robodog-goto --fake                 # fake robot + drawn floor: try everything with no robot
    uv run robodog-goto --fake --camera 0 --lens pinhole --hfov 70   # fake robot, laptop webcam
    uv run robodog-goto --allow-motion --trash                 # also detect + track trash: click a BOX to go to it
    uv run robodog-goto --allow-motion --trash --posture crawl # the same with the camera at crawl height

With --trash (needs the vision extra) the tracked pieces are boxed. Click a box and hold G:
the goal stays locked on that piece while the camera sees it (every detection corrects it),
and when it leaves the view the robot finishes on odometry, like a plain clicked goal.

Driving to a goal needs `measured: true` in the camera section of config.local.yaml (the camera's
height / offset measured and its pitch calibrated with K). Until then you can still click to see
distances, and drive with the teleop keys.

  MOUSE   click the floor        set the goal (shown as a ring; red = refused)
          click a trash box      (--trash) go to that piece, tracking it while it's in view
  HOLD    G                      go to the goal. Let go = stop (the goal is kept)
  HOLD    W/S A/D Q/E (arrows)   drive by hand, like robodog-teleop (overrides G)
  CLICK   R stand   F lie down   C clear goal   1/2/3 speed (hand driving)
          K  calibrate: click the bases of objects at the --calib-distances, in that order
          O  (--obstacles) show / hide the obstacle check's marks on the picture
          [ ]  camera delay -/+ 0.05 s: turn the robot (A/D) and adjust until the goal ring stays
               on the object. Put the value in camera.latency_s
          SPACE  E-STOP (stop + lie down, latched)    Esc  lie down and quit

With --people (needs --obstacles), a person in the robot's path closer than people.warn_row (the
magenta line) stops ALL motion until they've gone; go-to pauses. With --obstacles, the obstacle check runs on the same camera: while something is in the robot's
path (or the check isn't sure), FORWARD driving is blocked for go-to and the keys alike; go-to
pauses (goal kept). Backward, turning and sideways still work.

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
from ..goto import FINAL_STATES, GoTo, GoToLimits, PoseHistory
from ..target import TargetFollower
from ..safety import SafeDog, SafetyError
from ..teleop import Teleop
from .status import MODES
from .teleop import key_maps

RATE_HZ = 30
VIEW_W = 960       # camera view width in the window
PANEL_H = 235   # room for the obstacle and person lines (--obstacles, --people)
BG, FG, DIM = (24, 26, 30), (230, 230, 230), (120, 124, 132)
GREEN, AMBER, RED, BLUE, CYAN = (80, 200, 120), (240, 180, 60), (235, 70, 70), (90, 160, 240), (80, 210, 230)
MAGENTA = (230, 80, 230)   # person boxes
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


def target_box(box, width: int, height: int, margin: float) -> tuple[float, float, float, float]:
    """A tracked box in pixels -> (left, top, right, bottom) as fractions of the picture, grown by margin."""
    x0, y0, x1, y1 = box
    return (max(0.0, x0 / width - margin), max(0.0, y0 / height - margin),
            min(1.0, x1 / width + margin), min(1.0, y1 / height + margin))


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
        return self.get_stamped()[0]

    def get_stamped(self) -> tuple[np.ndarray | None, float, int]:
        """(frame, time it arrived (time.monotonic), sequence number). A still image never changes
        and doesn't lag: it counts as arriving now."""
        if self.still is not None:
            return self.still, time.monotonic(), 1
        return self.reader.get()

    def close(self):
        if self.reader:
            self.reader.close()


def run(dog: SafeDog, cfg, frames: Frames, args, fake: bool, tracker=None):
    import pygame as pg

    cam_cfg = cfg.camera
    drive_ok = fake or cam_cfg.measured
    limits = GoToLimits(cfg.goto.max_goal_m, min(cfg.goto.max_vx, cfg.safety.max_vx),
                        min(cfg.goto.max_yaw, cfg.safety.max_yaw), cfg.goto.arrive_m, cfg.goto.timeout_s,
                        cfg.goto.stop_before_m, cfg.goto.odometry)
    goto = GoTo(limits)
    print(f"go-to odometry: {limits.odometry}; posture: {args.posture}")
    base_max_vx = limits.max_vx
    # Trash tracking (--trash): a detector thread on the camera, and the goal locked onto one piece.
    watcher = None
    follower = TargetFollower(goto, cfg.trash.lost_after_s, cfg.trash.reacquire_m)
    tracks, sightings, trash_seq, trash_rate = [], [], 0, 0.0
    trash_pose = None   # robot pose when the frame of the newest detector result was taken
    if tracker is not None:
        from ..vision.trash import TrashWatcher, to_sighting

        watcher = TrashWatcher(frames.get_stamped, tracker, csrt_max_s=cfg.trash.csrt_max_s)
    # Obstacle check (--obstacles): on the same camera stream. While it isn't sure the path is
    # clear, SafeDog refuses forward driving, and go-to pauses (the goal is kept).
    obstacle_watcher, guard, ocfg = None, None, None
    show_obstacles = True   # O toggles the obstacle marks on the picture
    if args.obstacles:
        if frames.reader is None:
            raise SystemExit("--obstacles needs a live camera or a video (--camera), not --image or the drawn floor")
        from ..teleop import ForwardGuard
        from ..vision.obstacles import from_settings
        from ..vision.watcher import ObstacleWatcher

        print("loading the depth model for the obstacle check...")
        ocfg = from_settings(cfg.obstacles)
        obstacle_watcher = ObstacleWatcher(None, ocfg, camera=frames.reader)
        guard = ForwardGuard(obstacle_watcher)
        dog.forward_guard = guard.blocked
        print(f"obstacle check on ({obstacle_watcher.estimator.device}): FORWARD is blocked while something is "
              "ahead or the check isn't sure; go-to pauses meanwhile")
    # Person check (--people): on the same camera stream. A person close in front stops ALL motion
    # (SafeDog), and go-to pauses (the goal is kept) until they've gone.
    person_watcher, people = None, None
    if args.people:
        from ..vision.people import PersonDetector, PersonGuard, PersonWatcher
        from ..vision.trash import camera_for

        pc = cfg.people
        print("loading the person model...")
        detector = PersonDetector(pc.model, pc.conf, pc.imgsz)
        person_watcher = PersonWatcher(frames.get_stamped, detector,
                                       lambda w, h: camera_for(cfg.camera, w, h, args.posture))
        people = PersonGuard(person_watcher, obstacle_watcher, pc.warn_row, pc.margin, pc.overlap, pc.clear_s)
        dog.stop_guard = people.stop_reason
        print(f"person check on ({detector.device}): a person in the path closer than people.warn_row "
              f"({pc.warn_row}) stops ALL motion until that hasn't been the case for {pc.clear_s:.1f} s; go-to pauses meanwhile")
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

    # The lens is lower in the crawl posture: every distance depends on this height.
    cam_height = cam_cfg.height_m if args.posture == "stand" else cam_cfg.crawl_height_m

    def make_model():
        return CameraModel(fw, fh, hfov, cam_height, pitch, cam_cfg.forward_offset_m, lens)

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
    if watcher is not None:
        message = "R to stand, then click a trash BOX (or the floor) to set a goal, hold G to go"
    if not drive_ok:
        message = "camera not measured yet: click to see distances (G disabled). See README §6.7"
    status, next_status = {}, 0.0
    last_click: tuple[float, float, float, float] | None = None  # u, v, X, Y
    step = goto.step((0.0, 0.0, 0.0), False, 0.0)
    clock = pg.time.Clock()
    last = time.monotonic()
    next_log = 0.0
    pose = (0.0, 0.0, 0.0)
    # The picture is late: remember recent poses, so a click (and the drawn goal) uses where the
    # robot was when the picture was taken. A still image / drawn floor doesn't lag.
    history = PoseHistory()
    latency = cam_cfg.latency_s if frames.reader else 0.0
    frame, frame_t = None, time.monotonic()

    try:
        while not tele.quit:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    tele.quit = True
                elif ev.type == pg.KEYDOWN:
                    if ev.key in click_keys:
                        tele.action(click_keys[ev.key])
                    elif ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                        step_s = 0.05 if ev.key == pg.K_RIGHTBRACKET else -0.05
                        latency = round(max(0.0, latency + step_s), 2)
                        message = f"camera delay {latency:.2f} s (put it in camera.latency_s when the ring stays put)"
                        print(f"camera delay: latency_s: {latency:.2f}")
                    elif ev.key == pg.K_o and obstacle_watcher is not None:
                        show_obstacles = not show_obstacles
                        message = "obstacle marks " + ("shown" if show_obstacles else "hidden (the check still runs)")
                    elif ev.key == pg.K_c:
                        goto.clear()
                        follower.drop()
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
                    # A click inside a trash box picks that piece (the smallest box, if they overlap).
                    hit = [(t, sg) for t, sg in zip(tracks, sightings)
                           if t.box[0] <= u <= t.box[2] and t.box[1] <= v <= t.box[3]]
                    if hit:
                        t, sg = min(hit, key=lambda e: (e[0].box[2] - e[0].box[0]) * (e[0].box[3] - e[0].box[1]))
                        if not drive_ok:
                            message = f"#{t.track_id} {t.label}: camera not measured, can't go there"
                            continue
                        err = follower.select(sg, pose, time.monotonic(), trash_pose)
                        if err:
                            message = f"#{t.track_id} {t.label}: {err}"
                        else:
                            from_cam = math.hypot(sg.floor[0] - model.forward_offset_m, sg.floor[1])
                            message = f"target #{t.track_id} {t.label}, {from_cam:.2f} m from under the camera. Hold G to go"
                        continue
                    follower.drop()   # a plain floor click: an ordinary, fixed goal
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
                    err = goto.set_goal(pt[0], pt[1], pose, history.at(frame_t - latency))
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
            history.add(now, pose)
            if watcher is not None:
                new_tracks, stamp, seq, trash_rate = watcher.latest()
                if seq != trash_seq:   # a new detector result: place the boxes on the floor, correct the goal
                    trash_seq, tracks = seq, new_tracks
                    sightings = [to_sighting(t, model) for t in tracks]
                    # The frame is old by the time it's analysed: use the pose the robot had back then.
                    trash_pose = history.at(stamp - latency) or pose
                    follower.update(sightings, trash_pose, now)
                follower.tick(now)
                watcher.select(follower.track_id)   # only the selected piece gets the CSRT fallback
                if watcher.error:
                    message = watcher.error
            if obstacle_watcher is not None:
                # The selected piece is in the path on purpose: its box (+ margin) isn't an obstacle.
                # Only while it's tracked (YOLO or CSRT); everything else in the path still blocks.
                box = next((t.box for t in tracks if t.track_id == follower.track_id), None) \
                    if follower.track_id is not None else None
                obstacle_watcher.ignore(None if box is None else target_box(box, fw, fh, cfg.obstacles.target_margin))
            # Slow while going to a tracked piece, so the tracker keeps up; normal speed otherwise.
            limits.max_vx = min(base_max_vx, cfg.trash.follow_vx) if follower.track_id is not None else base_max_vx
            going = (g_held and not hand and drive_ok and tele.standing and dog.settled() and not dog.fault)
            if hand and goto.goal is not None and goto.lim.odometry == "commanded":
                # Our own position count only knows go-to's commands, not hand driving.
                goto.clear()
                message = "goal cleared: hand driving (odometry 'commanded' can't follow it). Click again"
            blocked = guard.blocked() if guard is not None else None
            person, seen_people = people.check() if people is not None else (None, [])
            if going and blocked and goto.goal is not None and step.state not in FINAL_STATES:
                message = f"paused, FORWARD BLOCKED: {blocked}. Back off / turn by hand (S, A/D, Q/E)"
            if going and person and goto.goal is not None and step.state not in FINAL_STATES:
                message = f"paused, ALL MOTION STOPPED: {person}. Continues once they've gone"
            step = goto.step(pose, going and not blocked and not person, dt)   # paused, goal kept
            if going and step.dist is not None and now >= next_log:  # (no goal: nothing to log; G does nothing)
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

            new_frame, new_t, _ = frames.get_stamped()
            if new_frame is not None:
                frame, frame_t = new_frame, new_t
            seen_pose = history.at(frame_t - latency) or pose  # where the robot was when this picture was taken
            trash = None if watcher is None else (tracks, sightings, follower, trash_rate, now)
            obstacle = guard.check() if guard is not None else None
            marks = None   # the obstacle check's newest verdict, to draw on the picture
            if obstacle_watcher is not None and show_obstacles:
                marks = (obstacle_watcher.latest()[0], ocfg, people.warn_row if people is not None else None)
            draw(screen, (big, mid, small), frame, scale, view_h, model, grid, goto, step, pose, seen_pose, tele,
                 status, message, calibrating, last_click, drive_ok, fake, g_held, latency, trash, obstacle, marks,
                 (person, seen_people) if people is not None else None)
            clock.tick(RATE_HZ)
    finally:
        print("lying down and closing...")
        dog.close()
        if watcher is not None:
            watcher.close()
        if obstacle_watcher is not None:
            obstacle_watcher.close()
        if person_watcher is not None:
            person_watcher.close()
        pg.quit()


def draw(screen, fonts, frame, scale, view_h, model, grid, goto, step, pose, seen_pose, tele, status, message,
         calibrating, last_click, drive_ok, fake, g_held, latency, trash=None, obstacle=None, marks=None,
         person=None):
    import pygame as pg

    big, mid, small = fonts
    screen.fill(BG)
    if frame is not None and marks is not None and marks[0] is not None:
        # Obstacle check: path lines, yellow box, red (blocks) / orange / blue / green (the target).
        # Its verdict belongs to a frame a few hundredths of a second older than this one.
        from ..vision.obstacles import draw_overlay
        frame = draw_overlay(frame, *marks)
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

    # People (--people): magenta boxes with their distance; "close" if they stop the robot.
    if person is not None:
        for pp in person[1]:
            x0, y0 = sp(pp.box[:2])
            x1, y1 = sp(pp.box[2:])
            pg.draw.rect(screen, MAGENTA, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)), 2)
            d = "very close" if pp.dist is None else f"{pp.dist:.1f} m"
            lab = small.render(f"person {pp.conf:.2f}  {d}", True, (0, 0, 0), MAGENTA)
            screen.blit(lab, (x0, y0 - lab.get_height() if y0 > lab.get_height() else y1))

    # Tracked trash: a box per piece with its ID, class, confidence and distance.
    if trash is not None:
        tracks, sightings, follower, _, _ = trash
        for t, sg in zip(tracks, sightings):
            mine = t.track_id == follower.track_id
            col = AMBER if t.source == "csrt" else GREEN if mine else CYAN   # amber: YOLO lost it, CSRT's box
            x0, y0 = sp(t.box[:2])
            x1, y1 = sp(t.box[2:])
            pg.draw.rect(screen, col, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)), 3 if mine else 1)
            if sg.floor is None:
                dist = "not on floor"
            elif sg.clipped:
                dist = "cut off"
            else:
                dist = f"{math.hypot(sg.floor[0] - model.forward_offset_m, sg.floor[1]):.2f} m"
            via = "  CSRT" if t.source == "csrt" else ""
            lab = small.render(f"#{t.track_id} {t.label} {t.conf:.2f}  {dist}{via}", True, (0, 0, 0), col)
            screen.blit(lab, (x0, y0 - lab.get_height() if y0 > lab.get_height() else y1))

    # The goal, re-projected every frame from where the robot was when the picture was taken (the
    # picture is late). If it slides off the real spot, the camera delay, geometry or odometry is off.
    rel = goto.relative(seen_pose)
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
    elif step.state in ("turning", "aligning", "driving"):
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
                           f"pose x {x:+.2f} y {yy:+.2f} m, heading {math.degrees(yaw):+.0f} deg   "
                           f"camera delay {latency:.2f} s", True, FG), (12, y))
    y += 24
    if step.dist is not None:
        rel = goto.relative(pose)
        from_cam = math.hypot(rel[0] - model.forward_offset_m, rel[1])
        g = (f"goal {from_cam:.2f} m from camera, {step.bearing_deg:+.0f} deg   [{step.state}]   "
             f"driven: commanded {goto.commanded_m:.2f} m / robot says {goto.robot_m:.2f} m")
        if step.state == "paused" and not g_held:
            g += "   hold G to go"
    else:
        g = "no goal: click the floor" + (" or a trash box" if trash is not None else "")
    screen.blit(mid.render(g, True, FG), (12, y))
    y += 22
    if trash is not None:
        tracks, _, follower, rate, now = trash
        if follower.track_id is None:
            tl, tcol = f"trash: {len(tracks)} tracked ({rate:.1f} checks/s). Click a box to go to it", DIM
        elif follower.state == "tracking":
            tl = (f"target #{follower.track_id} {follower.label} {follower.conf:.2f}: TRACKING, goal corrected "
                  f"{follower.updates}x ({rate:.1f} checks/s)")
            tcol = GREEN
        else:
            tl = (f"target #{follower.track_id} {follower.label}: OUT OF VIEW for {now - follower.last_seen:.1f} s, "
                  f"finishing on odometry (corrected {follower.updates}x before)")
            tcol = AMBER
        screen.blit(small.render(tl, True, tcol), (12, y))
        y += 20
    y += 4
    if person is not None:     # (why everything is stopped or None, people seen) from the person guard
        if person[0]:
            screen.blit(small.render(f"ALL MOTION STOPPED: {person[0]}", True, RED), (12, y))
        else:
            screen.blit(small.render("no person close", True, GREEN), (12, y))
        y += 20
    if obstacle is not None:   # (level, text, checks per second) from the forward guard
        level, text, fps = obstacle
        if level == "clear":
            screen.blit(small.render(f"path clear ({fps:.1f} checks/s)", True, GREEN), (12, y))
        else:
            screen.blit(small.render(f"FORWARD BLOCKED: {text}", True, RED if level == "warn" else AMBER), (12, y))
        y += 20
    msg = tele.state().fault or message
    screen.blit(small.render(msg[:120], True, RED if tele.state().fault else AMBER), (12, y))
    y += 22
    for h in ("CLICK floor/box = goal   HOLD G = go   HOLD WASD/QE = hand drive   R stand  F lie  C clear  K calibrate",
              "[ ] camera delay -/+   O obstacle marks   SPACE e-stop   Esc quit   (window must have focus)"):
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
    ap.add_argument("--trash", action="store_true",
                    help="detect + track trash (needs the vision extra): click a box to go to that piece")
    ap.add_argument("--people", action="store_true",
                    help="person check (needs --obstacles): a person in the obstacle check's red area stops "
                         "ALL motion until they've gone; go-to pauses meanwhile")
    ap.add_argument("--obstacles", action="store_true",
                    help="obstacle check on the camera (needs the vision extra): FORWARD is blocked while "
                         "something is in the robot's path or the check isn't sure; go-to pauses meanwhile")
    ap.add_argument("--posture", choices=["stand", "crawl"], default="stand",
                    help="camera height used for all distances: stand = camera.height_m, crawl = "
                         "camera.crawl_height_m. This does NOT change how the robot moves (default: stand)")
    ap.add_argument("--calib-distances", default="1.5,2.5,3.0",
                    help="K calibration: distances of the objects you'll click, in click order, measured ON THE "
                         "FLOOR from the point under the camera lens (default 1.5,2.5,3.0). One distance fits only "
                         "the tilt; two or more also fit the lens and field of view")
    args = ap.parse_args()
    if args.people and not args.obstacles:   # before anything connects to the robot
        ap.error("--people needs --obstacles: it stops for a person in the obstacle check's red area")

    try:
        import pygame  # noqa: F401
    except ImportError:
        raise SystemExit("robodog-goto needs pygame: run `uv sync` first")

    cfg = load_config(args.config)
    tracker = None
    if args.trash:   # load the model first: a missing extra or model should fail before we touch the robot
        from ..vision.trash import TrashTracker

        print("loading the trash model...")
        tracker = TrashTracker(cfg.trash.model, cfg.trash.conf, cfg.trash.imgsz, classes=cfg.trash.classes)
        print(f"trash model {tracker.weights.name} on {tracker.device}, input {tracker.imgsz}px")
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
        run(dog, cfg, frames, args, args.fake, tracker)
    except KeyboardInterrupt:
        print("\nCtrl+C")
        dog.close()
    finally:
        frames.close()


if __name__ == "__main__":
    main()
