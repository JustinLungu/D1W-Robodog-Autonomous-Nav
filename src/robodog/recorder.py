"""Record a driving session: camera video + what we commanded, on one shared clock.

One folder per run:

    recordings/2026-11-15_14-03-22/
      video.mkv      camera stream as the robot sends it (no re-encoding, no quality loss)
      commands.csv   one row per control frame (~30/s): keys, speeds, state, events
      meta.json      start time, camera URL, speed caps, note, video_start_unix

Matching video to commands
--------------------------
The CSV's `t_unix` column is the laptop clock (time.time()). ffmpeg stamps every
video frame with the same clock when it ARRIVES on the laptop. So:

    laptop time of a frame = meta["video_start_unix"] + frame time in video.mkv (seconds)

"Arrives" means the camera's own delay (encoding + Wi-Fi, often 0.2-0.5 s) is NOT
removed: a frame shows the world slightly before its timestamp.

How: ffmpeg first writes video_raw.mkv with wall-clock timestamps. On stop it's
copied (no re-encoding) into video.mkv starting at 0, which players handle normally,
and the offset goes into meta.json. If the program crashes before that, video_raw.mkv
is still there and still playable; its timestamps simply are the laptop time.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "t_unix", "t_rel", "standing", "level", "held",
    "cmd_vx", "cmd_vy", "cmd_yaw",     # what teleop asked SafeDog for
    "sent_vx", "sent_vy", "sent_yaw",  # what SafeDog last sent to the robot (after caps / dead-man)
    "battery", "mode", "fault", "event",
]


class Recorder:
    def __init__(self, root: Path, camera: str | None, meta: dict, log=print):
        """camera: RTSP URL, video file or device (/dev/video0). None = commands only, no video."""
        self.log = log
        self.dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=False)
        self.camera = camera
        self.t0 = time.time()
        self.meta = {"start": datetime.now().isoformat(timespec="seconds"), "camera": camera, **meta}
        self._ffmpeg: subprocess.Popen | None = None
        self._ff_log = None
        self._events: list[str] = []
        self.video_error: str | None = None

        self._csv_file = open(self.dir / "commands.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(COLUMNS)
        self._write_meta()

        if camera:
            self._start_video()

    # ------------------------------------------------------------------ video (ffmpeg)
    def _start_video(self):
        if shutil.which("ffmpeg") is None:
            self.video_error = "ffmpeg not installed: recording commands only"
            self.log(self.video_error)
            return
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if self.camera.startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-use_wallclock_as_timestamps", "1",  # stamp each frame with the laptop clock on arrival
            "-i", self.camera,
            "-c", "copy", "-an",                 # save the stream as-is, no audio
            "-copyts",                           # keep those clock timestamps in the file
            str(self.dir / "video_raw.mkv"),
        ]
        self._ff_log = open(self.dir / "ffmpeg.log", "w")
        # Own session: Ctrl+C in the terminal doesn't kill ffmpeg mid-write; stop() ends it cleanly.
        self._ffmpeg = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self._ff_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    @property
    def video_running(self) -> bool:
        return self._ffmpeg is not None and self._ffmpeg.poll() is None

    # ------------------------------------------------------------------ commands
    def event(self, text: str):
        """Something worth marking (a click action, focus lost, a fault). Goes in the next row."""
        self._events.append(text)

    def row(self, *, standing, level, held, cmd, sent, battery, mode, fault):
        now = time.time()
        sent = sent or (0.0, 0.0, 0.0)
        self._csv.writerow([
            f"{now:.3f}", f"{now - self.t0:.3f}", int(standing), level, " ".join(sorted(held)),
            *(f"{v:.3f}" for v in cmd), *(f"{v:.3f}" for v in sent),
            "" if battery is None else battery, "" if mode is None else mode,
            fault or "", "; ".join(self._events),
        ])
        self._events.clear()

    # ------------------------------------------------------------------ stop
    def stop(self):
        """Stop ffmpeg cleanly, make the 0-based video.mkv, close the files."""
        if self._csv_file.closed:
            return  # already stopped
        if self._ffmpeg is not None:
            if self._ffmpeg.poll() is None:
                self._ffmpeg.terminate()  # SIGTERM: ffmpeg finishes the file properly
                try:
                    self._ffmpeg.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._ffmpeg.kill()
                    self._ffmpeg.wait()
            self._ff_log.close()
            self._finish_video()
        self._csv_file.close()
        self.meta["duration_s"] = round(time.time() - self.t0, 1)
        self._write_meta()
        self.log(f"recording saved: {self.dir}")

    def _finish_video(self):
        raw, out = self.dir / "video_raw.mkv", self.dir / "video.mkv"
        if not raw.exists() or raw.stat().st_size == 0:
            self.video_error = self.video_error or "no video was recorded (camera not reachable?), see ffmpeg.log"
            self.log(self.video_error)
            raw.unlink(missing_ok=True)
            return
        start = _probe_start(raw)
        ok = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(raw), "-c", "copy", str(out)],
            capture_output=True,
        ).returncode == 0
        if ok and start is not None:
            self.meta["video_start_unix"] = start
            raw.unlink()
        else:
            self.log("could not convert the video; video_raw.mkv is kept (timestamps = laptop time)")

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")


def _probe_start(path: Path) -> float | None:
    """Timestamp of the first frame = laptop time it arrived (thanks to -copyts)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None
