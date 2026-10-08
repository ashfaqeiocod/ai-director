"""intake.py — turn any soccer clip (uploaded file or YouTube link) into the 2 files MVP 2 reads.

    data/clips/<name>/clip.mp4    H.264 video + AAC sound, at most 720p, 25 fps  (the browser plays this)
    data/clips/<name>/audio.wav   16 kHz mono sound                             (the audio model reads this)
    data/clips/<name>/meta.json   duration, size, fps, where it came from

Every later step reads ONLY these files (through CausalReader, never past the playhead).

Command line (the dashboard does the same thing with drag-and-drop / the URL box):
    venv/bin/python -m story.intake ~/Movies/match.mp4
    venv/bin/python -m story.intake "https://www.youtube.com/watch?v=..." --name derby
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.download import InputError, is_url, js_runtime_args, resolve_path
from src.common import video_info
from story.common import clips_dir, ffmpeg_exe, safe_name, story_config

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def clip_dir(name: str) -> Path:
    return clips_dir() / name


def list_clips() -> list[dict]:
    """Every ready clip (one that has clip.mp4, audio.wav and meta.json), newest first."""
    root = clips_dir()
    if not root.exists():
        return []
    clips = []
    for folder in root.iterdir():
        meta = folder / "meta.json"
        if meta.exists() and (folder / "clip.mp4").exists() and (folder / "audio.wav").exists():
            info = json.loads(meta.read_text())
            info["has_truth"] = (folder / "truth.csv").exists()
            clips.append(info)
    return sorted(clips, key=lambda c: c.get("created", 0), reverse=True)


def unique_name(wanted: str) -> str:
    """A folder name that isn't taken yet: "derby", "derby-2", "derby-3"..."""
    name, n = safe_name(wanted), 2
    base = name
    while clip_dir(name).exists():
        name, n = f"{base}-{n}", n + 1
    return name


def youtube_name(url: str) -> str:
    """A readable default name for a YouTube link: "yt-<video id>"."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{6,})", url)
    return f"yt-{m.group(1)}" if m else "youtube"


def download(url: str, folder: Path, max_height: int) -> Path:
    """Download a YouTube (or other yt-dlp) link into the clip folder. Same settings as MVP 1."""
    out = folder / "download.mp4"
    fmt = (f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
           f"b[height<={max_height}][ext=mp4]/b[height<={max_height}]/b")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", fmt, "--merge-output-format", "mp4",
         "--ffmpeg-location", ffmpeg_exe(), "-o", str(out), "--no-cache-dir", "--no-progress",
         "--no-playlist", *js_runtime_args(), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out.exists():
        tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        raise InputError(f"Could not download {url}: {tail[0]}\n"
                         "  Check the link, your internet connection, and that Node.js is installed.")
    return out


def normalize(src: Path, folder: Path) -> dict:
    """Write clip.mp4 + audio.wav + meta.json from any video file. Returns the meta dict."""
    cfg = story_config()["intake"]
    info = video_info(src)
    if info["duration"] > cfg["max_minutes"] * 60:
        raise InputError(f"Clip is {info['duration'] / 60:.0f} min long; the limit is {cfg['max_minutes']} min "
                         "(story.intake.max_minutes in config.yaml).")
    height = min(cfg["max_height"], info["height"]) // 2 * 2
    clip = folder / "clip.mp4"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           # -2 keeps the shape with an even width (H.264 needs even sizes).
           "-vf", f"scale=-2:{height},fps={cfg['fps']}",
           # A keyframe every second makes cutting short windows for Gemini fast and exact.
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-g", str(cfg["fps"]), "-movflags", "+faststart"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "2"] if info["has_audio"] else ["-an"]
    subprocess.run(cmd + [str(clip)], check=True)

    # The audio model needs 16 kHz mono. A clip without sound gets silence of the same length,
    # so the rest of the system never has to special-case it.
    wav = folder / "audio.wav"
    if info["has_audio"]:
        audio_cmd = ["-i", str(clip), "-vn"]
    else:
        audio_cmd = ["-f", "lavfi", "-i", f"anullsrc=r={cfg['audio_rate']}:cl=mono", "-t", f"{info['duration']:.3f}"]
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *audio_cmd,
                    "-ac", "1", "-ar", str(cfg["audio_rate"]), "-c:a", "pcm_s16le", str(wav)], check=True)

    out = video_info(clip)
    meta = {"name": folder.name, "duration": round(out["frames"] / out["fps"], 3), "fps": out["fps"],
            "width": out["width"], "height": out["height"], "frames": out["frames"],
            "has_audio": info["has_audio"], "created": time.time()}
    return meta


def intake(source: str, name: str | None = None, original_name: str | None = None) -> dict:
    """Import a file path or a URL as a new clip. Returns its meta dict.

    `original_name` is the uploaded file's name (used for the default clip name).
    """
    cfg = story_config()["intake"]
    folder_name = unique_name(name or (youtube_name(source) if is_url(source)
                                       else original_name or Path(source).name))
    folder = clip_dir(folder_name)
    folder.mkdir(parents=True)
    try:
        if is_url(source):
            src = download(source, folder, cfg["max_height"])
        else:
            src = resolve_path(source)
            if not src.is_file():
                raise InputError(f"File not found: {src}")
        meta = normalize(src, folder)
        meta["source"] = source if is_url(source) else (original_name or src.name)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)   # never leave a half-made clip behind
        raise
    finally:
        (folder / "download.mp4").unlink(missing_ok=True)   # our own download copy is no longer needed
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a clip for the Story Reader (MVP 2).")
    parser.add_argument("source", help="a video file or a YouTube link")
    parser.add_argument("--name", help="folder name under data/clips/ (default: from the file/link)")
    args = parser.parse_args()
    try:
        meta = intake(args.source, args.name)
    except InputError as error:
        print(f"[intake] {error}")
        return 1
    print(f"[intake] ready: data/clips/{meta['name']}/  {meta['duration']:.1f} s, "
          f"{meta['width']}x{meta['height']} @ {meta['fps']:g} fps, sound: {meta['has_audio']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
