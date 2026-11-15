"""Test trash detection + tracking on a recorded video (or the live camera). Nothing moves.

Each tracked piece gets a box labelled:   #ID  class  confidence  distance

    uv run robodog-trash recordings/<run>/video.mkv                    # robot was standing
    uv run robodog-trash recordings/<run>/video.mkv --posture crawl    # robot was crawling (camera lower)
    uv run robodog-trash <video> --save out.mp4 --csv out.csv          # annotated video + one row per box
    uv run robodog-trash <video> --headless --start 60 --end 88        # no window, part of the video
    uv run robodog-trash                                               # live robot camera, view only

The distance is measured on the floor from the point under the camera lens, using the
camera geometry in config.local.yaml. It depends on the posture: `--posture stand` uses
camera.height_m, `--posture crawl` uses camera.crawl_height_m. A wrong posture gives
distances that are off by the ratio of the two heights.
"cut off" = the box touches the picture's edge, so its foot isn't visible: no distance.

ByteTrack needs frames in order, so a video is processed sequentially; --step N analyses
every Nth frame (default 3 = 10 per second for a 30 fps video).

Keys: q / Esc = quit, SPACE = pause.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import cv2

from ..config import load_config
from ..vision.trash import POSTURES, TrashTracker, camera_distance, camera_for, draw_tracks, to_sighting

WINDOW = "robodog trash (q to quit)"


def run(args, cfg, tracker=None) -> dict:
    """Process the source; returns a summary (also used by the tests with a fake tracker)."""
    t = cfg.trash
    source = args.video or f"rtsp://{cfg.network.robot_ip}:8554/test"
    is_file = os.path.exists(source)
    if args.video and not is_file and "://" not in source and not source.isdigit():
        raise SystemExit(f"video not found: {source}")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    src_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
    step = max(1, args.step) if is_file else 1
    if is_file and args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * src_fps))

    if tracker is None:
        print("loading the trash model...")
        tracker = TrashTracker(args.model or t.model, args.conf or t.conf, args.imgsz or t.imgsz, args.device,
                               classes=t.classes if args.classes is None else args.classes)
        print(f"model {tracker.weights.name} on {tracker.device}, input {tracker.imgsz}px, conf >= {tracker.conf}")
    print(f"posture: {args.posture} (camera {cfg.camera.height_m if args.posture == 'stand' else cfg.camera.crawl_height_m} m high)")

    writer = csv_file = rows = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        rows = csv.writer(csv_file)
        rows.writerow(["video_t", "frame", "track_id", "label", "conf", "x0", "y0", "x1", "y1",
                       "distance_m", "ahead_m", "left_m", "cut_off"])
    model = None
    idx = int(args.start * src_fps) - 1 if is_file else -1
    done, n_boxes, ids, per_frame, paused = 0, 0, {}, [], False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if is_file and args.end and idx / src_fps > args.end:
                break
            if is_file and (idx - int(args.start * src_fps)) % step:
                continue
            if model is None:
                h, w = frame.shape[:2]
                model = camera_for(cfg.camera, w, h, args.posture)
            t0 = time.monotonic()
            tracks = tracker.update(frame)
            per_frame.append(time.monotonic() - t0)
            done += 1
            n_boxes += len(tracks)
            t_video = idx / src_fps if is_file else 0.0
            for tr in tracks:
                sg = to_sighting(tr, model)
                ok_floor = sg.floor is not None and not sg.clipped
                dist = camera_distance(model, sg.floor) if ok_floor else None
                seen = ids.setdefault(tr.track_id, {"label": tr.label, "frames": 0, "min_dist": None})
                seen["frames"] += 1
                if dist is not None and (seen["min_dist"] is None or dist < seen["min_dist"]):
                    seen["min_dist"] = dist
                if rows:
                    rows.writerow([f"{t_video:.3f}", idx, tr.track_id, tr.label, f"{tr.conf:.3f}",
                                   *(f"{v:.0f}" for v in tr.box),
                                   "" if dist is None else f"{dist:.3f}",
                                   "" if not ok_floor else f"{sg.floor[0]:.3f}", "" if not ok_floor else f"{sg.floor[1]:.3f}",
                                   int(sg.clipped)])
            if args.headless and not args.save:
                desc = "  ".join(
                    f"#{tr.track_id} {tr.label} {tr.conf:.2f} "
                    + (lambda sg: "cut-off" if sg.clipped else ("no-floor" if sg.floor is None
                                                                else f"{camera_distance(model, sg.floor):.2f}m"))(to_sighting(tr, model))
                    for tr in tracks) or "-"
                print(f"{t_video:6.1f} s  {per_frame[-1] * 1000:4.0f} ms  {desc}")
                continue

            view = draw_tracks(frame, tracks, model)
            label = f"{t_video:6.1f} s   {len(tracks)} tracked   {args.posture}"
            cv2.putText(view, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            if view.shape[1] > args.width:
                k = args.width / view.shape[1]
                view = cv2.resize(view, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            if args.save:
                if writer is None:
                    out_fps = src_fps / step if is_file else 5.0   # files: plays at real-time speed
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if not args.headless:
                if done == 1:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(WINDOW, view)
                if done == 1:
                    cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                while True:
                    key = cv2.waitKey(1 if not paused else 50) & 0xFF
                    if key == ord(" "):
                        paused = not paused
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt
                    if not paused:
                        break
            elif done % 30 == 0:
                print(f"\r{t_video:6.1f} s, {done} frames", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()

    ms = 1000 * sorted(per_frame)[len(per_frame) // 2] if per_frame else 0.0
    print(f"\n{done} frames analysed, {n_boxes} boxes, {len(ids)} tracks, {ms:.0f} ms per frame (median)")
    for tid, info in sorted(ids.items(), key=lambda e: -e[1]["frames"])[:15]:
        near = "-" if info["min_dist"] is None else f"{info['min_dist']:.2f} m"
        print(f"  #{tid:<4d} {info['label']:22s} seen in {info['frames']:4d} frames, nearest {near}")
    for path in (args.save, args.csv):
        if path:
            print(f"saved {path}")
    return {"frames": done, "boxes": n_boxes, "tracks": ids, "ms": ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file (default: the robot's live camera)")
    ap.add_argument("--config", help="shared config (default: config.local.yaml)")
    ap.add_argument("--posture", choices=POSTURES, default="stand",
                    help="how the robot was moving when the video was recorded: sets the camera height used "
                         "for distances (default: stand)")
    ap.add_argument("--step", type=int, default=3, help="video files: analyse every Nth frame (default 3)")
    ap.add_argument("--start", type=float, default=0.0, help="video files: start at this second")
    ap.add_argument("--end", type=float, help="video files: stop at this second")
    ap.add_argument("--save", help="write the annotated video to this file (.mp4)")
    ap.add_argument("--csv", help="write one row per box (time, ID, class, confidence, box, distance) to this file")
    ap.add_argument("--headless", action="store_true", help="no window: print each analysed frame instead")
    ap.add_argument("--model", help="override trash.model (a .pt file or hf://<repo>/<file>)")
    ap.add_argument("--classes", help='override trash.classes, e.g. "cup, bottle" ("" = all)')
    ap.add_argument("--conf", type=float, help="override trash.conf")
    ap.add_argument("--imgsz", type=int, help="override trash.imgsz")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--width", type=int, default=1280, help="max window / saved video width in px (default 1280)")
    args = ap.parse_args()
    run(args, load_config(args.config))


if __name__ == "__main__":
    main()
