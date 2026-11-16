"""Drive the robot with the keyboard. It moves only while a key is HELD; let go and it stops.

    uv run robodog-teleop --allow-motion    # real robot
    uv run robodog-teleop --fake            # fake robot: try the controls with no robot
    uv run robodog-teleop --allow-motion --record --note "hallway, chair legs"
                                            # also record camera video + commands (see recorder.py)
    uv run robodog-teleop --allow-motion --obstacles
                                            # obstacle check (camera + Depth Anything), with the camera
                                            # picture and what it detects in the window. While something
                                            # is ahead (or the check isn't sure), FORWARD is blocked;
                                            # backward, turning and sideways still work.
    uv run robodog-teleop --allow-motion --obstacles --people
                                            # also: a PERSON in the red (close, in the path) area
                                            # stops ALL motion until they've gone

Needs allow_motion: true in config.machine.yaml AND --allow-motion (not for --fake).

A small window opens. It must have focus (click it): keys typed into other windows
are ignored, and clicking away stops the robot.

  HOLD                                CLICK
  W / Up      forward                 R      stand up
  S / Down    backward                F      lie down
  A / Left    turn left               1 2 3  speed level (slow / medium / max cap)
  D / Right   turn right              H      shake hands (~7 s, then standing again)
                                      SPACE  E-STOP (stop + lie down, latched)
  Q / E       sideways left / right   Esc    lie down and quit (also: close the window)

Speeds stay under the caps in config.local.yaml; all SafeDog protections apply.
"""

from __future__ import annotations

import argparse
import sys
import time

from ..config import REPO_ROOT, load_config
from ..recorder import Recorder
from ..safety import SafeDog, SafetyError
from ..teleop import ForwardGuard, Teleop
from .status import MODES

RATE_HZ = 30

W, H = 560, 480
MAGENTA = (230, 80, 230)   # person boxes
VIEW_H = W * 9 // 16   # --obstacles: camera picture above the panel (16:9, window width)
BG, FG, DIM = (24, 26, 30), (230, 230, 230), (120, 124, 132)
GREEN, AMBER, RED, BLUE = (80, 200, 120), (240, 180, 60), (235, 70, 70), (90, 160, 240)


def key_maps():
    import pygame as pg

    hold = {
        pg.K_w: "forward", pg.K_UP: "forward",
        pg.K_s: "back", pg.K_DOWN: "back",
        pg.K_a: "turn_left", pg.K_LEFT: "turn_left",
        pg.K_d: "turn_right", pg.K_RIGHT: "turn_right",
        pg.K_q: "strafe_left", pg.K_e: "strafe_right",
    }
    click = {
        pg.K_r: "stand", pg.K_f: "lie_down", pg.K_h: "shake_hand",
        pg.K_1: "speed_1", pg.K_2: "speed_2", pg.K_3: "speed_3",
        pg.K_KP1: "speed_1", pg.K_KP2: "speed_2", pg.K_KP3: "speed_3",
        pg.K_SPACE: "estop", pg.K_ESCAPE: "quit",
    }
    return hold, click


def camera_view(watcher, ocfg):
    """The analysed camera frame with the obstacle marks (path lines, yellow box, red/orange...),
    scaled to the window width, as a pygame surface; None until the first check."""
    import cv2
    import pygame as pg

    from ..vision.obstacles import draw_overlay

    frame, res = watcher.view()
    if frame is None or res is None:
        return None
    img = cv2.resize(draw_overlay(frame, res, ocfg), (W, VIEW_H), interpolation=cv2.INTER_AREA)
    return pg.surfarray.make_surface(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).swapaxes(0, 1))


def draw(screen, fonts, st, status, fake, rec=None, obstacle=None, person=None):
    """The status panel (W x H). The caller shows it (pg.display.flip())."""
    import pygame as pg

    big, mid, small = fonts
    screen.fill(BG)

    if st.fault:
        title, color = ("E-STOP" if "E-STOP" in st.fault else "FAULT"), RED
    elif st.standing:
        title, color = ("MOVING" if any(st.cmd) else "STANDING"), (GREEN if any(st.cmd) else BLUE)
    else:
        title, color = "LYING DOWN", DIM
    screen.blit(big.render(title, True, color), (20, 14))
    if fake:
        screen.blit(mid.render("FAKE ROBOT", True, AMBER), (W - 150, 22))
    if rec is not None:
        secs = int(time.time() - rec.t0)
        if rec.camera and not rec.video_running:
            text, col = "REC: VIDEO STOPPED", RED   # ffmpeg died: camera lost? see ffmpeg.log
        else:
            text, col = f"\u25cf REC {secs // 60:02d}:{secs % 60:02d}" + ("" if rec.camera else " (no video)"), RED
        surf = small.render(text, True, col)
        screen.blit(surf, (W - surf.get_width() - 20, 56))

    y = 72
    batt = status.get("battery")
    mode = status.get("mode")
    line = f"battery {batt}%   mode {mode} ({MODES.get(mode, '?')})" if batt is not None else "status: -"
    screen.blit(mid.render(line, True, FG), (20, y)); y += 30
    screen.blit(mid.render(f"speed level {st.level}", True, FG), (20, y)); y += 30
    vx, vy, yaw = st.cmd
    screen.blit(mid.render(f"vx {vx:+.2f} m/s   vy {vy:+.2f} m/s   yaw {yaw:+.2f} rad/s", True, FG), (20, y)); y += 30
    held = " ".join(sorted(st.held)) or "-"
    screen.blit(mid.render(f"held: {held}", True, FG), (20, y)); y += 36

    msg = f"{st.fault}. Esc to quit, restart the program to move again" if st.fault else st.message
    screen.blit(small.render(msg[:80], True, RED if st.fault else AMBER), (20, y)); y += 34

    help_lines = [
        "HOLD  W/S fwd/back   A/D turn   Q/E sideways   (arrows too)",
        "CLICK R stand   F lie down   H handshake   1/2/3 speed   SPACE e-stop   Esc quit",
    ]
    for h in help_lines:
        screen.blit(small.render(h, True, DIM), (20, y)); y += 22

    if person is not None:     # (why everything is stopped, or None) from the person guard
        y += 8
        if person:
            pg.draw.rect(screen, RED, (12, y, W - 24, 40), border_radius=6)
            surf = mid.render(f"STOPPED: {person}", True, (255, 255, 255))
            if surf.get_width() > W - 40:   # long reason: smaller font so it fits
                surf = small.render(f"STOPPED: {person}"[:70], True, (255, 255, 255))
            screen.blit(surf, ((W - surf.get_width()) // 2, y + 8))
            y += 40
        else:
            screen.blit(small.render("no person close", True, GREEN), (20, y + 10))
            y += 26

    if obstacle is not None:   # (level, text, checks per second) from the forward guard
        level, text, fps = obstacle
        y += 8
        if level == "warn":
            pg.draw.rect(screen, RED, (12, y, W - 24, 40), border_radius=6)
            surf = mid.render("FORWARD BLOCKED: obstacle ahead", True, (255, 255, 255))
            screen.blit(surf, ((W - surf.get_width()) // 2, y + 8))
        elif level == "clear":
            screen.blit(small.render(f"path clear   ({fps:.1f} checks/s)", True, GREEN), (20, y + 10))
        else:   # not sure (starting, stale picture, check stopped): forward is blocked too
            screen.blit(small.render(f"FORWARD BLOCKED: {text}", True, AMBER), (20, y + 10))

    if not pg.key.get_focused():
        screen.blit(mid.render("click this window to control", True, AMBER), (20, H - 36))


def run(dog: SafeDog, fake: bool, rec: Recorder | None = None, watcher=None, ocfg=None, guard=None,
        people=None, camera=None):
    import pygame as pg

    pg.init()
    pg.key.set_repeat()  # no key repeat: one KEYDOWN per physical press
    # With the obstacle warning: the camera picture on top, the usual panel below it.
    view_h = VIEW_H if watcher is not None else 0
    screen = pg.display.set_mode((W, view_h + H))
    panel = screen.subsurface((0, view_h, W, H))
    pg.display.set_caption("robodog teleop" + (" (FAKE)" if fake else ""))
    fonts = (pg.font.SysFont("dejavusans", 34, bold=True), pg.font.SysFont("dejavusans", 20),
             pg.font.SysFont("dejavusans", 15))
    hold_keys, click_keys = key_maps()

    tele = Teleop(dog, log=print)
    status, next_status = {}, 0.0
    clock = pg.time.Clock()
    last = time.monotonic()
    last_fault = None
    last_level = None
    view, view_res = None, None   # camera surface, and the verdict it was drawn for
    try:
        while not tele.quit:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    tele.quit = True
                elif ev.type == pg.KEYDOWN and ev.key in click_keys:
                    tele.action(click_keys[ev.key])
                    if rec:
                        rec.event(click_keys[ev.key])
                elif ev.type in (pg.WINDOWFOCUSLOST, pg.WINDOWHIDDEN, pg.WINDOWMINIMIZED):
                    tele.release_all()
                    if rec:
                        rec.event("focus_lost")

            # Held motion keys, read fresh every frame (nothing can get stuck).
            if pg.key.get_focused():
                pressed = pg.key.get_pressed()
                tele.set_held(name for key, name in hold_keys.items() if pressed[key])
            else:
                tele.release_all()

            now = time.monotonic()
            tele.update(now - last)
            last = now

            if now >= next_status:  # don't ask the robot for its status 30 times a second
                next_status = now + 0.5
                try:
                    status = dog.status()
                except Exception as e:
                    status = {}
                    tele.message = f"status error: {e!r}"

            st = tele.state()
            if rec:
                if st.fault != last_fault:
                    rec.event(f"fault: {st.fault}")
                rec.row(standing=st.standing, level=st.level, held=st.held, cmd=st.cmd, sent=dog.last_sent,
                        battery=status.get("battery"), mode=status.get("mode"), fault=st.fault)
            last_fault = st.fault

            obstacle = None
            if watcher is not None:
                res = watcher.latest()[0]
                level, text, fps = guard.check()
                obstacle = (level, text, fps)
                if rec and level != last_level:   # mark in commands.csv when the warning starts and ends
                    near = f" near={res.nearness:.2f}" if res is not None and level == "warn" else ""
                    rec.event(f"obstacle_{level}{near}")
                last_level = level
                if res is not view_res:   # a new check: redraw the picture (not 30 times a second)
                    view, view_res = camera_view(watcher, ocfg), res

            person, seen = None, []
            if people is not None:
                person, seen = people.check()

            if view_h:
                screen.fill(BG, (0, 0, W, view_h))
                if view is not None:
                    screen.blit(view, (0, 0))
                    frame = camera.get()[0] if camera is not None else None
                    if frame is not None:   # person boxes, scaled from the frame to the view
                        kx, ky = W / frame.shape[1], VIEW_H / frame.shape[0]
                        for p in seen:
                            x0, y0, x1, y1 = p.box
                            pg.draw.rect(screen, MAGENTA, (int(x0 * kx), int(y0 * ky),
                                                           max(1, int((x1 - x0) * kx)), max(1, int((y1 - y0) * ky))), 2)
            draw(panel, fonts, st, status, fake, rec, obstacle, person)
            pg.display.flip()
            clock.tick(RATE_HZ)
    finally:
        print("lying down and closing...")
        dog.close()  # zero speed + lie down (unless e-stopped), stop the loop
        pg.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--allow-motion", action="store_true", help="required for the real robot")
    ap.add_argument("--fake", action="store_true", help="use the fake robot from tests/ (no robot, nothing moves)")
    ap.add_argument("--record", action="store_true", help="record camera video + commands to recordings/<date_time>/")
    ap.add_argument("--note", default="", help="free text saved with the recording (where, what you tested)")
    ap.add_argument("--camera", help="camera to record / watch (default: robot camera; with --fake: none). "
                                     "e.g. /dev/video0 for a webcam, or a video file for --obstacles")
    ap.add_argument("--people", action="store_true",
                    help="person check (needs --obstacles): a person in the obstacle check's red area stops "
                         "ALL motion until they've gone (settings: people: in config.local.yaml)")
    ap.add_argument("--obstacles", action="store_true",
                    help="obstacle check (needs the vision extra): FORWARD is blocked while something is in "
                         "the robot's path or the check isn't sure; backward, turning and sideways still work")
    args = ap.parse_args()
    if args.people and not args.obstacles:   # before anything connects to the robot
        ap.error("--people needs --obstacles: it stops for a person in the obstacle check's red area")

    try:
        import pygame  # noqa: F401
    except ImportError:
        raise SystemExit("teleop needs pygame: run `uv sync` first")

    cfg = load_config(args.config)
    if args.fake:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from fake_robot import FakeHighLevel

        cfg.safety.allow_motion = True  # fake robot only: nothing real can move
        cfg.network.laptop_ip = cfg.network.laptop_ip or "127.0.0.1"  # fake robot: no robot Wi-Fi needed
        robot = FakeHighLevel(battery=80)
    else:
        if not (args.allow_motion and cfg.safety.allow_motion):
            raise SystemExit("motion needs BOTH allow_motion: true in config.machine.yaml AND --allow-motion")
        from ..sdk import make_highlevel

        robot = make_highlevel(cfg.sdk_dir)

    dog = SafeDog(robot, cfg, allow_motion=True, log=lambda m: print(f"[safedog] {m}"))
    try:
        dog.connect()
    except SafetyError as e:
        raise SystemExit(str(e))
    dog.start()  # 30 Hz loop: battery/connection checks, keepalive, dead-man switch

    rec = None
    if args.record:
        camera = args.camera or (None if args.fake else f"rtsp://{cfg.network.robot_ip}:8554/test")
        s = cfg.safety
        rec = Recorder(REPO_ROOT / "recordings", camera, log=print, meta={
            "note": args.note, "fake_robot": args.fake,
            "caps": {"max_vx": s.max_vx, "max_vy": s.max_vy, "max_yaw": s.max_yaw},
        })
        print(f"recording to {rec.dir}" + ("" if camera else " (commands only, no video)"))

    watcher, ocfg, guard, people, person_watcher, camera = None, None, None, None, None, None
    if args.obstacles or args.people:
        # One camera stream, shared by both checks (two streams over Wi-Fi garble the picture).
        source = args.camera or (None if args.fake else f"rtsp://{cfg.network.robot_ip}:8554/test")
        if source is None:
            raise SystemExit("--obstacles / --people with --fake need --camera (a webcam like /dev/video0, or a video file)")
        from ..vision.camera import LatestFrameReader

        try:
            camera = LatestFrameReader(source)
        except RuntimeError as e:   # camera could not be opened
            raise SystemExit(f"camera: {e}")
    if args.obstacles:
        from ..vision.obstacles import from_settings
        from ..vision.watcher import ObstacleWatcher

        print("loading the depth model for the obstacle warning...")
        ocfg = from_settings(cfg.obstacles)
        watcher = ObstacleWatcher(None, ocfg, camera=camera)
        guard = ForwardGuard(watcher)
        dog.forward_guard = guard.blocked   # SafeDog zeroes forward speed while this says blocked
    if args.people:
        from ..vision.people import PersonDetector, PersonGuard, PersonWatcher
        from ..vision.trash import camera_for

        p = cfg.people
        print("loading the person model...")
        detector = PersonDetector(p.model, p.conf, p.imgsz)
        person_watcher = PersonWatcher(camera.get, detector, lambda w, h: camera_for(cfg.camera, w, h))
        people = PersonGuard(person_watcher, watcher, p.margin, p.overlap, p.clear_s)
        dog.stop_guard = people.stop_reason   # SafeDog sends 0 for EVERY speed while this says stop
        print(f"person check on ({detector.device}): a person in the obstacle check's red area stops ALL "
              f"motion until that hasn't been the case for {p.clear_s:.1f} s")
        print(f"obstacle check on ({watcher.estimator.device}): FORWARD is blocked while something is ahead "
              "or the check isn't sure. Backward, turning and sideways still work.")

    print("window open: click it, then R to stand up. Esc to quit.")
    try:
        run(dog, args.fake, rec, watcher, ocfg, guard, people, camera)  # lies the robot down before returning
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        if watcher:
            watcher.close()
        if person_watcher:
            person_watcher.close()
        if camera:
            camera.close()
        if rec:
            rec.stop()  # after run(), so the video also shows the robot lying down


if __name__ == "__main__":
    main()
