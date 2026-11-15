"""Find litter on the floor in a recorded video and draw boxes around it. Offline, not live.

    uv run robodog-litter recordings/2026-11-15_01-02-48/video.mkv
    uv run robodog-litter <video> --fps 2 --start 20 --end 40    # denser, only 20-40 s
    uv run robodog-litter <video> --classes "paper, plastic bottle, can"
    uv run robodog-litter <video> --show                         # also watch it while it runs

Writes next to the video, in litter/:
    annotated.mp4    the analysed frames with boxes (plays at real-time speed)
    detections.csv   one row per box: video time, laptop time, class, score, box

Speed: about 5 s per frame on a laptop CPU, well under 1 s on a GPU. So on a CPU the
default is 1 frame per second of video: an 87 s video takes ~8 min. Use --start/--end
to try settings on a short piece first.

The dashed line in the output marks --min-bottom: boxes must end below it (on the floor).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from ..vision.litter import DEFAULT_CLASSES, MODELS, Filters, LitterDetector, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, e.g. recordings/<run>/video.mkv")
    ap.add_argument("--out", help="output folder (default: litter/ next to the video)")
    ap.add_argument("--fps", type=float, help="frames analysed per second of video (default: 1 on CPU, 5 on GPU)")
    ap.add_argument("--start", type=float, default=0.0, help="start at this second")
    ap.add_argument("--end", type=float, help="stop at this second (default: end of video)")
    ap.add_argument("--classes", help=f"comma-separated things to look for (default: {', '.join(DEFAULT_CLASSES)})")
    ap.add_argument("--threshold", type=float, default=0.25, help="min confidence 0-1 (default 0.25); lower = more boxes")
    ap.add_argument("--search-from", type=float, default=Filters.search_from,
                    help="only search below this fraction of the image height, 0 = all (default %(default)s)")
    ap.add_argument("--min-bottom", type=float, default=Filters.min_bottom,
                    help="box bottom must be below this fraction of the height (default %(default)s)")
    ap.add_argument("--max-area", type=float, default=Filters.max_area,
                    help="drop boxes bigger than this fraction of the frame (default %(default)s)")
    ap.add_argument("--model", default="tiny", help=f"{' / '.join(MODELS)} or a Hugging Face id (default: tiny)")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--show", action="store_true", help="show each analysed frame in a window while running")
    args = ap.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = n_frames / src_fps if n_frames else None
    end = min(args.end, duration) if (args.end and duration) else (args.end or duration)

    # Laptop time of each frame, if this came from robodog-teleop --record (see recorder.py).
    meta_path = video.parent / "meta.json"
    video_start_unix = json.loads(meta_path.read_text()).get("video_start_unix") if meta_path.exists() else None

    filters = Filters(search_from=args.search_from, min_bottom=args.min_bottom, max_area=args.max_area)
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    print(f"loading {args.model!r} (first run downloads the model)...")
    det = LitterDetector(classes, args.model, args.device, args.threshold, filters)
    fps = args.fps or (5.0 if det.device.type == "cuda" else 1.0)
    print(f"model {det.model_name} on {det.device}; looking for: {', '.join(det.classes)}")

    out_dir = Path(args.out) if args.out else video.parent / "litter"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    csv_file = open(out_dir / "detections.csv", "w", newline="")
    rows = csv.writer(csv_file)
    rows.writerow(["video_t", "t_unix", "frame", "label", "score", "x0", "y0", "x1", "y1"])

    step = max(1, round(src_fps / fps))  # analyse every step-th frame
    first = int(args.start * src_fps)
    last = int(end * src_fps) if end else None
    total = ((last - first) // step + 1) if last else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx, done, n_boxes, t_begin = first, 0, 0, time.monotonic()
    counts: dict[str, int] = {}
    try:
        while last is None or idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - first) % step:
                idx += 1
                continue
            t_video = idx / src_fps
            dets = det.detect(frame)
            img = draw(frame, dets, det.classes, filters)
            cv2.putText(img, f"{t_video:6.1f} s  {len(dets)} found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (255, 255, 255), 3, cv2.LINE_AA)

            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps / step, (w, h))
            writer.write(img)
            for d in dets:
                t_unix = f"{video_start_unix + t_video:.3f}" if video_start_unix else ""
                rows.writerow([f"{t_video:.3f}", t_unix, idx, d.label, f"{d.score:.3f}", *d.box])
                counts[d.label] = counts.get(d.label, 0) + 1
            n_boxes += len(dets)
            done += 1

            per = (time.monotonic() - t_begin) / done
            eta = f", ~{per * (total - done) / 60:.1f} min left" if total else ""
            print(f"\r{t_video:6.1f} s: {len(dets):2d} boxes   [{done}/{total or '?'} frames, {per:.1f} s/frame{eta}]   ",
                  end="", flush=True)

            if args.show:
                cv2.imshow("litter (q to stop)", cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1]))))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            idx += 1
    except KeyboardInterrupt:
        print("\nstopped early (Ctrl+C); saving what's done")
    finally:
        cap.release()
        if writer:
            writer.release()
        csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{done} frames, {n_boxes} boxes: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
    print(f"saved: {out_dir / 'annotated.mp4'}\n       {out_dir / 'detections.csv'}")


if __name__ == "__main__":
    main()
