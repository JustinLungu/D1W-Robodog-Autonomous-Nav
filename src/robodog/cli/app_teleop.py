"""Drive the robot with the keyboard over the vendor APP protocol (like the phone app / remote).

Use this to drive in the robot's real low crawl, which the SDK can't do. It does NOT use the
SDK: don't run robodog-teleop / robodog-goto / anything SDK at the same time (the robot
then ignores these commands, and connecting the SDK while crawling makes it fold down).

    uv run robodog-app-teleop --allow-motion
    uv run robodog-app-teleop --fake             # no robot: try the controls
    uv run robodog-app-teleop --allow-motion --full-scale 0.5,1.0,1.0   # after calibrating

  HOLD                                CLICK
  W / Up      forward                 R      stand up   (app stand, ~5 s)
  S / Down    backward                X      crawl / stand tall (standing still, ~3 s)
  A / Left    turn left               F      lie down   (app dump)
  D / Right   turn right              1 2 3  speed level (33 / 66 / 100 % of the caps)
  Q / E       sideways left / right   SPACE  E-STOP (dump, latched)
                                      Esc    lie down and quit

  CALIBRATE (robot standing still, ~1 m of room in front, behind and to the sides): press C.
  It drives forward, sideways and turns, each with stick 0.2 for 1.5 s, measures the robot's own
  speed and saves the stick directions + speeds to app_teleop_calibration.json (used from then
  on). Any key aborts. Probe keys (hold): 5 6 7 8 = +0.05 on stick axis 0 / 1 / 2 / 3.

Speeds stay under the caps in config.local.yaml (safety.max_*, and crawl.max_* while
crawling). The stick scale is NOT known yet, so it assumes the largest one in the robot's
config: the robot should move at or below the speed shown, not faster. Check "measured".
"""

from __future__ import annotations

import argparse
import time

from ..app_remote import (AXES, CALIBRATION_FILE, FULL_SCALE, MAX_STICK, AppRemote, FakeAppRemote,
                          load_calibration, save_calibration)
from ..config import load_config
from .teleop import AMBER, BG, BLUE, DIM, FG, GREEN, RED

RATE_HZ = 30
W, H = 600, 470


def keys():
    import pygame as pg

    hold = {
        pg.K_w: "forward", pg.K_UP: "forward", pg.K_s: "back", pg.K_DOWN: "back",
        pg.K_a: "turn_left", pg.K_LEFT: "turn_left", pg.K_d: "turn_right", pg.K_RIGHT: "turn_right",
        pg.K_q: "strafe_left", pg.K_e: "strafe_right",
    }
    probe = {pg.K_5: 0, pg.K_6: 1, pg.K_7: 2, pg.K_8: 3}
    click = {
        pg.K_r: "stand", pg.K_x: "crawl", pg.K_f: "lie_down", pg.K_c: "calibrate",
        pg.K_1: "speed_1", pg.K_2: "speed_2", pg.K_3: "speed_3",
        pg.K_SPACE: "estop", pg.K_ESCAPE: "quit",
    }
    return hold, probe, click


def draw(screen, fonts, tele, remote, fake):
    import pygame as pg

    big, mid, small = fonts
    st, rs = tele.state(), remote.state
    screen.fill(BG)
    now = time.monotonic()
    if st.estopped:
        title, col = "E-STOP", RED
    elif rs.age(now) > 1.0:
        title, col = "NO LINK", RED
    elif rs.control_mode == "stand_down":
        title, col = ("CRAWLING" if any(st.cmd) else "CRAWL (still)"), (GREEN if any(st.cmd) else BLUE)
    elif rs.control_mode == "rl":
        title, col = ("MOVING" if any(st.cmd) else "STANDING"), (GREEN if any(st.cmd) else BLUE)
    else:
        title, col = (rs.control_mode or "waiting...").upper(), AMBER
    screen.blit(big.render(title, True, col), (20, 14))
    screen.blit(mid.render("FAKE" if fake else "APP PROTOCOL", True, AMBER), (W - 175, 22))

    def fmt(v, f="{:+.2f}"):
        return "?" if v is None else f.format(v)

    y = 70
    vb = rs.v_body or (None, None, None)
    lines = [
        (f"mode {rs.control_mode}   body z {fmt(rs.z, '{:.3f}')} m   battery {fmt(rs.battery, '{}')}", FG),
        (f"speed level {st.level}   caps {tele.caps()[0]:.2f} / {tele.caps()[1]:.2f} m/s  {tele.caps()[2]:.2f} rad/s", FG),
        (f"asked     vx {st.cmd[0]:+.2f}  vy {st.cmd[1]:+.2f}  yaw {st.cmd[2]:+.2f}", FG),
        (f"measured  vx {fmt(vb[0])}  vy {fmt(vb[1])}  yaw {fmt(rs.yaw_rate)}", FG),
        ("sticks    " + "  ".join(f"{s:+.3f}" for s in st.sticks), DIM),
    ]
    for text, c in lines:
        screen.blit(mid.render(text, True, c), (20, y)); y += 30
    screen.blit(small.render(st.message[:84], True, RED if st.estopped else AMBER), (20, y + 4)); y += 36
    full = tele.full_scale
    screen.blit(small.render(f"stick scale (full stick): vx {full['vx']:.2f}  vy {full['vy']:.2f}  yaw {full['yaw']:.2f}"
                             f"   axes {tele.axes['vx']} {tele.axes['vy']} {tele.axes['yaw']}", True, DIM), (20, y)); y += 24
    for h in ("HOLD  W/S fwd/back   A/D turn   Q/E sideways   5-8 probe stick axis 0-3",
              "CLICK R stand  X crawl/tall  F lie  C calibrate  1/2/3 speed  SPACE e-stop  Esc quit"):
        screen.blit(small.render(h, True, DIM), (20, y)); y += 22
    if not pg.key.get_focused():
        screen.blit(mid.render("click this window to control", True, AMBER), (20, H - 36))
    pg.display.flip()


def run(tele, remote, fake):
    import pygame as pg

    pg.init()
    pg.key.set_repeat()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("robodog app teleop" + (" (FAKE)" if fake else ""))
    fonts = (pg.font.SysFont("dejavusans", 34, bold=True), pg.font.SysFont("dejavusans", 19),
             pg.font.SysFont("dejavusans", 15))
    hold, probe, click = keys()
    clock = pg.time.Clock()
    last = time.monotonic()
    while not tele.quit:
        now = time.monotonic()
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                tele.quit = True
            elif ev.type == pg.KEYDOWN and ev.key in click:
                tele.action(click[ev.key], now)
            elif ev.type in (pg.WINDOWFOCUSLOST, pg.WINDOWHIDDEN, pg.WINDOWMINIMIZED):
                tele.release_all()
        if pg.key.get_focused():
            pressed = pg.key.get_pressed()
            tele.set_held(name for key, name in hold.items() if pressed[key])
            tele.set_probe(next((ax for key, ax in probe.items() if pressed[key]), None))
        else:
            tele.release_all()
        if fake:
            remote.tick(now - last)
        tele.update(now - last, now)
        last = now
        draw(screen, fonts, tele, remote, fake)
        clock.tick(RATE_HZ)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--allow-motion", action="store_true", help="required for the real robot")
    ap.add_argument("--fake", action="store_true", help="no robot: a stand-in that 'drives' on screen")
    ap.add_argument("--full-scale", help="speed at full stick for vx,vy,yaw (m/s,m/s,rad/s), e.g. 0.5,1.0,1.0. "
                                         f"Default {FULL_SCALE['vx']},{FULL_SCALE['vy']},{FULL_SCALE['yaw']} "
                                         "(the largest in the robot's config: never faster than asked)")
    ap.add_argument("--max-stick", type=float, default=MAX_STICK,
                    help=f"never deflect a stick further than this (default {MAX_STICK})")
    args = ap.parse_args()

    try:
        import pygame  # noqa: F401
    except ImportError:
        raise SystemExit("app teleop needs pygame: run `uv sync` first")

    from ..app_teleop import AppTeleop

    cfg = load_config(args.config)
    axes, full_scale = load_calibration()
    print("stick calibration:", f"from {CALIBRATION_FILE.name}" if CALIBRATION_FILE.exists() else "defaults (press C to calibrate)")
    if args.full_scale:
        vals = [float(v) for v in args.full_scale.split(",")]
        if len(vals) != 3 or min(vals) <= 0:
            raise SystemExit("--full-scale needs three positive numbers: vx,vy,yaw")
        full_scale = dict(zip(("vx", "vy", "yaw"), vals))
    max_stick = max(0.0, min(1.0, args.max_stick))

    if args.fake:
        remote = FakeAppRemote(axes, full_scale)
    else:
        if not (args.allow_motion and cfg.safety.allow_motion):
            raise SystemExit("motion needs BOTH allow_motion: true in config.machine.yaml AND --allow-motion")
        if not cfg.network.laptop_ip:
            raise SystemExit("this laptop isn't on the robot's network: join its Wi-Fi")
        print("NOTE: no SDK program (robodog-teleop / goto / status) may run at the same time.")
        remote = AppRemote(cfg.network.robot_ip, cfg.network.laptop_ip)
    tele = AppTeleop(remote, cfg, axes, full_scale, max_stick)
    if not args.fake:
        tele.on_calibrated = lambda ax, sc, note: (save_calibration(ax, sc, note=note),
                                                   print(f"saved {CALIBRATION_FILE}"))
    print(f"stick full scale {full_scale}, max stick {max_stick}. Window open: click it, R to stand up.")
    try:
        run(tele, remote, args.fake)
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        import pygame as pg
        tele.release_all()
        remote.set_sticks([0, 0, 0, 0])
        if not tele.estopped and remote.state.control_mode in ("rl", "stand_down"):
            print("lying down (app dump)...")
            remote.dump()
            time.sleep(3.0)                 # keep the stream (zero sticks) running while it lies down
        remote.close()
        pg.quit()


if __name__ == "__main__":
    main()
