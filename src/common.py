"""common.py — small helpers shared by every step.

- Finds the project folder, so paths in config.yaml work from anywhere.
- Loads config.yaml and .env.
- Points model caches INTO the project folder (the libraries would otherwise
  write to /tmp or your home folder).
- Reads basic video info (fps, size, frame count) with PyAV.
- Finds the ffmpeg program (bundled in the venv by the imageio-ffmpeg package).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# The project root is the folder that contains this "src" folder.
ROOT = Path(__file__).resolve().parent.parent

# These must be set BEFORE `inference` or `ultralytics` are imported,
# because they read them at import time.
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / "models_cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".yolo_config" / "matplotlib"))
# Roboflow's inference package tries to load optional extras we don't use; keep it quiet.
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CPUExecutionProvider]")
load_dotenv(ROOT / ".env")


def path(rel: str) -> Path:
    """Turn a path from config.yaml (relative to the project) into an absolute Path."""
    return ROOT / rel


def load_config() -> dict[str, Any]:
    """Read config.yaml into a plain dict."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def cam_path(cfg: dict, cam: int) -> Path:
    """Path of camera N's video file, e.g. data/cams/cam1.mp4."""
    return path(cfg["video"]["cams_dir"]) / f"cam{cam}.mp4"


def detections_path(cfg: dict, cam: int) -> Path:
    """Path of camera N's cached detections JSON."""
    return path(cfg["detection"]["detections_dir"]) / f"cam{cam}.json"


def ffmpeg_exe() -> str:
    """Path to the ffmpeg program.

    We use the ffmpeg binary that ships with the `imageio-ffmpeg` pip package
    (inside the venv), so nothing has to be installed system-wide.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def video_info(video: Path) -> dict[str, Any]:
    """Return fps, width, height, exact frame count, duration and audio presence.

    Uses PyAV (Python bindings for ffmpeg). It counts the real video packets,
    because the frame count written in a file's header can be wrong.
    """
    import av

    with av.open(str(video)) as container:
        v = container.streams.video[0]
        has_audio = len(container.streams.audio) > 0
        fps = float(v.average_rate)
        width, height = v.codec_context.width, v.codec_context.height
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        # One packet = one frame for normal mp4 video; demuxing is fast (no decoding).
        frames = sum(1 for p in container.demux(v) if p.size > 0)
    return {"fps": fps, "width": width, "height": height, "frames": frames,
            "duration": duration, "has_audio": has_audio}
