"""download.py — Step 2: download the source clip from YouTube.

Uses yt-dlp to fetch the video (max 1080p, mp4 with audio) into data/raw/source.mp4.
If the downloaded file has a variable frame rate, we re-encode it to a constant
frame rate: every later step relies on "frame N" meaning the same moment in all cameras.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

if __package__ in (None, ""):  # allow "python src/download.py"
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.common import ffmpeg_exe, load_config, path, video_info


def download(force: bool = False) -> None:
    cfg = load_config()
    out = path(cfg["video"]["source"])
    if out.exists() and not force:
        print(f"[download] {out.name} already exists, skipping (use --force to redo)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_name("download.mp4")
        max_h = cfg["video"]["max_height"]
        # Prefer an mp4 video + m4a audio, both at most 1080p, merged into one mp4.
        fmt = f"bv*[height<={max_h}][ext=mp4]+ba[ext=m4a]/b[height<={max_h}][ext=mp4]/b[height<={max_h}]"
        subprocess.run(
            # yt-dlp is installed in the venv; run it with the same Python as this script.
            [sys.executable, "-m", "yt_dlp", "-f", fmt, "--merge-output-format", "mp4",
             "--ffmpeg-location", ffmpeg_exe(), "-o", str(raw), "--force-overwrites",
             "--no-cache-dir", "--no-progress", *js_runtime_args(), cfg["video"]["url"]],
            check=True,
        )
        normalize(raw, out, cfg["video"]["max_height"])
        raw.unlink(missing_ok=True)

    info = video_info(out)
    print(f"[download] duration {info['duration']:.2f}s | fps {info['fps']:.3f} | "
          f"{info['width']}x{info['height']} | {info['frames']} frames | audio: {info['has_audio']}")


def js_runtime_args() -> list[str]:
    """YouTube now needs a JavaScript runtime to unlock video formats.

    yt-dlp only looks for deno by default; we point it at Node.js if it is installed.
    """
    if shutil.which("node"):
        return ["--js-runtimes", "node", "--remote-components", "ejs:github"]
    return []


def normalize(src, dst, max_height: int) -> None:
    """Re-encode to constant frame rate H.264 (and downscale if taller than max_height).

    A fixed frame rate makes "frame index" a reliable clock for all later steps.
    """
    info = video_info(src)
    fps = round(info["fps"], 3)
    vf = f"fps={fps}"
    if info["height"] > max_height:
        vf += f",scale=-2:{max_height}"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac"] if info["has_audio"] else ["-an"]
    subprocess.run(cmd + [str(dst)], check=True)


if __name__ == "__main__":
    download(force="--force" in sys.argv)
