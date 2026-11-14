"""Live Depth Anything V2 view of the robot camera. Vision only: never talks to the SDK.

Shows the camera image and the estimated depth side by side.
Bright/yellow = close, dark/purple = far (relative, no units).

    uv run robodog-depth                          # robot camera (rtsp://<robot_ip>:8554/test)
    uv run robodog-depth --source 0               # laptop webcam, to try it without the robot
    uv run robodog-depth --size 252               # smaller input: faster, less detail
    uv run robodog-depth --model base             # bigger model (GPU)
    uv run robodog-depth --width 960              # smaller window (default 1280 px wide; also drag to resize)
    uv run robodog-depth --headless --frames 20   # no window, just print timings

Overlay: model FPS, inference time, and "age" = how old the frame was when its depth
was shown (time waiting for the model + inference; the camera's own network delay
comes on top and isn't measured here).

Keys in the window: q / Esc = quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..config import load_config
from ..vision.camera import LatestFrameReader
from ..vision.depth import MODELS, DepthEstimator, colorize

WINDOW = "robodog depth (q to quit)"


def _label(img: np.ndarray, lines: list[str]) -> None:
    y = 22
    for text in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.local.yaml)")
    ap.add_argument("--source", help="RTSP URL, video file, or webcam index (default: robot camera)")
    ap.add_argument("--model", default="small", help=f"{' / '.join(MODELS)} or a Hugging Face id (default: small)")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda (default: auto)")
    ap.add_argument("--size", type=int, help="model input short side in px (default: 518 on GPU, 308 on CPU)")
    ap.add_argument("--width", type=int, default=1280,
                    help="max width of the side-by-side view in px; the window can also be resized (default: 1280)")
    ap.add_argument("--headless", action="store_true", help="no window, print timings instead")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many depth frames (0 = run forever)")
    args = ap.parse_args()

    source = args.source
    if source is None:
        cfg = load_config(args.config)
        source = f"rtsp://{cfg.network.robot_ip}:8554/test"

    print(f"loading model {args.model!r}...")
    est = DepthEstimator(args.model, args.device, args.size)
    print(f"model {est.model_name} on {est.device} ({str(est.dtype).replace('torch.', '')}), input {est.short_side}px")

    if not args.headless:
        # NORMAL = resizable by dragging the edges; KEEPRATIO = the image doesn't get stretched.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    print(f"opening {source} ...")
    with LatestFrameReader(source) as cam:
        last_seq, done, fps = 0, 0, 0.0
        t_prev = time.monotonic()
        try:
            while True:
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
                t1 = time.monotonic()

                dt = t1 - t_prev
                t_prev = t1
                fps = 1.0 / dt if fps == 0.0 else 0.9 * fps + 0.1 / dt  # smoothed
                infer_ms = (t1 - t0) * 1000
                age_ms = (t1 - stamp) * 1000
                done += 1

                if args.headless:
                    print(f"frame {done}: infer {infer_ms:6.0f} ms  age {age_ms:6.0f} ms  {fps:5.1f} fps  "
                          f"{frame.shape[1]}x{frame.shape[0]}")
                else:
                    view = np.hstack([frame, colorize(depth)])
                    # Shrink before drawing text, so the labels stay readable at any camera resolution.
                    if view.shape[1] > args.width:
                        scale = args.width / view.shape[1]
                        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    half = view[:, view.shape[1] // 2:]  # a view into the depth half: draw on it in place
                    _label(half, [f"{est.device.type} {args.model} @{est.short_side}",
                                  f"{fps:4.1f} fps  infer {infer_ms:.0f} ms",
                                  f"age {age_ms:.0f} ms"])
                    cv2.imshow(WINDOW, view)
                    if done == 1:  # resizable windows open at a random default size: fit the first image
                        cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break  # window closed with the X button

                if args.frames and done >= args.frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
