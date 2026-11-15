"""Live view of the camera-based obstacle warning. Vision only: never talks to the SDK.

Use it to SEE what the check does and to tune the `obstacles:` values in config.local.yaml.

    uv run robodog-obstacles                              # robot camera
    uv run robodog-obstacles --source recordings/<run>/video.mkv   # a recording
    uv run robodog-obstacles --warn-row 0.92              # try a value without editing the config
    uv run robodog-obstacles --source <video> --save out.mp4       # write the annotated video
    uv run robodog-obstacles --headless --frames 20       # no window, print verdicts

What's drawn on the camera picture:
  two lines        the robot's path on the floor, running to the horizon
  yellow box       the robot-sized "gate" at the warning line
  red              in the path, at or before the warning line: this is what WARNS
  orange           in the path, but still beyond the warning line (seen, no warning yet)
  blue             close, but beside the path (ignored)
  light blue       close, but above the robot (ignored: it would drive under)
  text             clear / OBSTACLE AHEAD / BLOCKED, "near" (1.0 = at the warning line,
                   2.0 = half that distance) and the red area

Tuning (robot STANDING, on a clear floor):
  path_left/right  put an object next to each wheel's track; move the lines until the
                   object just outside is not red and just inside is
  warn_row         put an object at the distance you want to be warned at; raise/lower
                   the value until the yellow box's bottom edge is at its foot
  height_ratio     hold something just above the robot's top; it should NOT turn red

Keys: q / Esc = quit.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from ..config import load_config
from ..vision.camera import LatestFrameReader
from ..vision.depth import DepthEstimator, colorize
from ..vision.obstacles import ObstacleChecker, draw_overlay, from_settings

WINDOW = "robodog obstacles (q to quit)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--source", help="RTSP URL, video file, or webcam index (default: robot camera)")
    ap.add_argument("--path-left", type=float, help="override obstacles.path_left")
    ap.add_argument("--path-right", type=float, help="override obstacles.path_right")
    ap.add_argument("--vanish-x", type=float, help="override obstacles.vanish_x")
    ap.add_argument("--height-ratio", type=float, help="override obstacles.height_ratio")
    ap.add_argument("--warn-row", type=float, help="override obstacles.warn_row")
    ap.add_argument("--model", default="small", help="depth model: small / base / large (default: small)")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--size", type=int, help="depth model input short side (default: 518 on GPU, 308 on CPU)")
    ap.add_argument("--width", type=int, default=1280, help="max window width in px (default: 1280)")
    ap.add_argument("--save", help="write the annotated view to this video file (.mp4)")
    ap.add_argument("--step", type=int, default=15,
                    help="video files only: analyse every Nth frame (default 15 = 2 per second at 30 fps)")
    ap.add_argument("--headless", action="store_true", help="no window, print verdicts instead")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many analysed frames (0 = run on)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    o = cfg.obstacles
    for name in ("path_left", "path_right", "vanish_x", "height_ratio", "warn_row"):
        if getattr(args, name) is not None:
            setattr(o, name, getattr(args, name))
    ocfg = from_settings(o)
    source = args.source or f"rtsp://{cfg.network.robot_ip}:8554/test"

    print("loading depth model...")
    est = DepthEstimator(args.model, args.device, args.size)
    checker = ObstacleChecker(ocfg)
    print(f"depth {est.model_name} on {est.device}, input {est.short_side}px; opening {source} ...")

    # A live stream: always analyse the newest frame. A file: go through it in order.
    is_file = os.path.exists(source)
    cap = cv2.VideoCapture(source) if is_file else None
    try:
        cam = None if is_file else LatestFrameReader(source)
    except RuntimeError as e:
        raise SystemExit(f"{e}\nIs the laptop on the robot's Wi-Fi, and is the robot switched on? "
                         f"Check with: ping {cfg.network.robot_ip}")
    if not args.headless:   # only once the camera is open, so a failed start leaves no empty window
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    src_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
    writer, last_seq, done, n_warn, idx = None, 0, 0, 0, -1
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % args.step:
                    continue
                stamp = time.monotonic()
            else:
                frame, stamp, seq = cam.get()
                if frame is None or seq == last_seq:
                    if cam.ended:
                        print("stream ended")
                        break
                    if not args.headless and cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                        break
                    time.sleep(0.005)
                    continue
                last_seq = seq

            t0 = time.monotonic()
            depth = est.predict(frame)
            res = checker.check(depth)
            t1 = time.monotonic()
            done += 1
            n_warn += res.warn

            if args.headless and not args.save:
                when = f"{idx / src_fps:6.1f} s" if is_file else f"frame {done}"
                print(f"{when}: {res.status:8s} near {res.nearness:4.2f}  area {res.area * 100:5.2f}%  "
                      f"{(t1 - t0) * 1000:4.0f} ms  age {(t1 - stamp) * 1000:4.0f} ms  {res.reason}")
            else:
                view = np.hstack([draw_overlay(frame, res, ocfg), colorize(depth)])
                if view.shape[1] > args.width:
                    k = args.width / view.shape[1]
                    view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                if args.save:
                    if writer is None:
                        out_fps = src_fps / args.step if is_file else 3.0  # files: plays at real-time speed
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                                 (view.shape[1], view.shape[0]))
                    writer.write(view)
                if not args.headless:
                    cv2.imshow(WINDOW, view)
                    if done == 1:
                        cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                elif done % 20 == 0:
                    print(f"\r{done} frames, {n_warn} with a warning", end="", flush=True)

            if args.frames and done >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"\n{done} frames analysed, {n_warn} with a warning" + (f"; saved {args.save}" if args.save else ""))


if __name__ == "__main__":
    main()
