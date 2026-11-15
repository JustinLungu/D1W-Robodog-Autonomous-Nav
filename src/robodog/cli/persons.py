"""Show the robot's OWN person detection: its boxes drawn on the live camera picture.

The robot runs a person detector (YOLO11n on its NPU, program `detection_2d_node`) and
publishes the boxes on the ROS 2 topic /perception/detection2d. This only LISTENS to that
topic from the laptop (see robotbus.py). Nothing is sent to the robot, nothing runs on it.

    uv sync --extra robot-bus            # once (add your vision extra too, if you use one)
    uv run robodog-persons
    uv run robodog-persons --headless    # no window, print the boxes

Each box shows the robot's track number (the same person keeps it across frames) and
its confidence. The camera picture and the boxes travel separately and are both a bit
late, so boxes can trail a moving person slightly. Boxes older than --max-age are hidden.

Keys: q / Esc = quit.
"""

from __future__ import annotations

import argparse
import time
import zlib

import cv2
import numpy as np

from ..config import load_config
from ..robotbus import TopicListener, decode_detection2d_array
from ..vision.camera import LatestFrameReader

WINDOW = "robodog persons (q to quit)"
TOPIC = "/perception/detection2d"
ROBOT_IMAGE = (1920, 1080)   # the boxes are in the robot camera's full-resolution pixels


def _colour(track_id: str) -> tuple[int, int, int]:
    """A stable colour per track number, so one person keeps one colour."""
    h = zlib.crc32(track_id.encode()) % 180   # (Python's hash() changes between runs)
    hsv = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def draw(img, boxes, age: float, rate: float, max_age: float):
    H, W = img.shape[:2]
    sx, sy = W / ROBOT_IMAGE[0], H / ROBOT_IMAGE[1]
    s = max(1.0, H / 720)
    fresh = age <= max_age
    for b in boxes if fresh else []:
        x0, y0 = int((b.cx - b.w / 2) * sx), int((b.cy - b.h / 2) * sy)
        x1, y1 = int((b.cx + b.w / 2) * sx), int((b.cy + b.h / 2) * sy)
        col = _colour(b.track_id)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, max(2, int(3 * s)))
        text = f"#{b.track_id} {b.label + ' ' if b.label else ''}{b.score:.2f}"
        fs, th = 0.7 * s, max(1, int(2 * s))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y0 - 6 if y0 - tht - 10 > 0 else y1 + tht + 6
        cv2.rectangle(img, (x0, ty - tht - 4), (x0 + tw + 4, ty + 4), col, -1)
        cv2.putText(img, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    if age == float("inf"):
        status = "waiting for the robot's detections..."
    elif not fresh:
        status = f"no fresh detections ({age:.1f} s old)"
    else:
        status = f"robot detector: {len(boxes)} person(s)   {rate:.1f} Hz"
    org = (int(20 * s), int(H - 20 * s))
    cv2.putText(img, status, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), int(5 * s), cv2.LINE_AA)
    cv2.putText(img, status, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), int(2 * s), cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--source", help="camera (default: the robot's RTSP stream)")
    ap.add_argument("--max-age", type=float, default=0.5, help="hide boxes older than this many seconds (default 0.5)")
    ap.add_argument("--width", type=int, default=1280, help="max window width in px (default 1280)")
    ap.add_argument("--headless", action="store_true", help="no window, print the boxes")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run on)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ip = cfg.network.robot_ip
    print(f"listening to {TOPIC} on the robot ({ip}), receive only...")
    listener = TopicListener(ip, TOPIC, decode_detection2d_array)
    t_end = time.monotonic() + args.seconds if args.seconds else None

    try:
        if args.headless:
            last = None
            while t_end is None or time.monotonic() < t_end:
                msg, age, rate = listener.latest()
                if msg is not None and msg is not last:
                    last = msg
                    _, boxes = msg
                    desc = "  ".join(f"#{b.track_id} {b.score:.2f} at ({b.cx:.0f},{b.cy:.0f}) {b.w:.0f}x{b.h:.0f}"
                                     for b in boxes) or "-"
                    print(f"{rate:4.1f} Hz  {len(boxes)} person(s): {desc}")
                time.sleep(0.02)
            return

        source = args.source or f"rtsp://{ip}:8554/test"
        try:
            cam = LatestFrameReader(source)
        except RuntimeError as e:
            raise SystemExit(f"{e}\nIs the laptop on the robot's Wi-Fi? Check with: ping {ip}")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        shown, last_seq = 0, 0
        try:
            while t_end is None or time.monotonic() < t_end:
                frame, _, seq = cam.get()
                if frame is None or seq == last_seq:
                    if cam.ended:
                        print("camera stream ended")
                        break
                    if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                        break
                    continue
                last_seq = seq
                msg, age, rate = listener.latest()
                boxes = msg[1] if msg is not None else []
                view = draw(frame.copy(), boxes, age, rate, args.max_age)
                if view.shape[1] > args.width:
                    k = args.width / view.shape[1]
                    view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, view)
                shown += 1
                if shown == 1:
                    cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"received {listener.count} detection messages"
              + (f", {listener.errors} could not be decoded" if listener.errors else ""))
        listener.close()


if __name__ == "__main__":
    main()
