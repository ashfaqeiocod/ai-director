"""download.py — get a video and convert it into the format the pipeline needs.

Used by prepare.py (Steps 2–3). Two jobs:

1. fetch(source): turn a "source" from config.yaml into a video file on disk.
   - A YouTube link is downloaded with yt-dlp into data/raw/downloads/ (and reused next time).
   - A file path is used where it is. Your own files are only READ, never moved or deleted.

2. convert(...): re-encode with ffmpeg into a clean H.264 mp4 with a constant frame rate.
   Every later step relies on "frame N" meaning the same moment in every camera.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from src.common import ROOT, ffmpeg_exe, video_info

DOWNLOADS = ROOT / "data" / "raw" / "downloads"


class InputError(Exception):
    """A problem with the input settings, explained in plain English for the user."""


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve_path(source: str) -> Path:
    """Accept absolute paths, "~/..." and paths relative to the project folder."""
    p = Path(source).expanduser()
    return p if p.is_absolute() else ROOT / p


def fetch(source: str, max_height: int) -> Path:
    """Return a local video file for `source` (download it first if it is a link)."""
    if not source:
        raise InputError("A video source is empty in config.yaml (input section).")
    if not is_url(source):
        file = resolve_path(source)
        if not file.is_file():
            raise InputError(f"Video file not found: {file}\n"
                             f"  Check the path in config.yaml (input section). Relative paths start at {ROOT}")
        return file

    # Each link gets its own file name, so changing the link never reuses the wrong video.
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    out = DOWNLOADS / f"{hashlib.sha1(source.encode()).hexdigest()[:12]}.mp4"
    if out.exists():
        print(f"[download] already downloaded: {source}")
        return out
    print(f"[download] downloading {source}")
    # Prefer mp4 video + m4a audio, at most max_height, merged into one mp4.
    fmt = (f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
           f"b[height<={max_height}][ext=mp4]/b[height<={max_height}]/b")
    result = subprocess.run(
        # yt-dlp is installed in the venv; run it with the same Python as this script.
        [sys.executable, "-m", "yt_dlp", "-f", fmt, "--merge-output-format", "mp4",
         "--ffmpeg-location", ffmpeg_exe(), "-o", str(out), "--no-cache-dir", "--no-progress",
         *js_runtime_args(), source],
    )
    if result.returncode != 0 or not out.exists():
        raise InputError(f"Could not download {source}\n"
                         "  Check the link, your internet connection, and that Node.js is installed (node --version).")
    return out


def js_runtime_args() -> list[str]:
    """YouTube now needs a JavaScript runtime to unlock video formats.

    yt-dlp only looks for deno by default; we point it at Node.js if it is installed.
    """
    if shutil.which("node"):
        return ["--js-runtimes", "node", "--remote-components", "ejs:github"]
    return []


def scaled_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Same shape, at most max_pixels pixels.

    Width is rounded to a multiple of 6, so the frame splits into 3 equal strips with even
    widths (the H.264 encoder needs even sizes). Height is rounded to an even number.
    """
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    return int(width * scale) // 6 * 6, int(height * scale) // 2 * 2


def convert(src: Path, dst: Path, start: float, frames: int, fps: float, video_filter: str,
            keep_audio: bool) -> None:
    """Cut `frames` frames starting at `start` seconds, apply the filter, write H.264 mp4.

    Using an exact frame count (not a duration) guarantees every camera has the same length.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}",                 # jump to the start point; exact because we re-encode
           "-i", str(src),
           "-vf", f"fps={fps},{video_filter}" if video_filter else f"fps={fps}",
           "-frames:v", str(frames),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if keep_audio and video_info(src)["has_audio"]:
        cmd += ["-c:a", "aac", "-t", f"{frames / fps:.3f}"]
    else:
        cmd += ["-an"]
    subprocess.run(cmd + [str(dst)], check=True)
