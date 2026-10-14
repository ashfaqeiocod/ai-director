"""common.py — small helpers shared by every Arm MVP step: the arm settings, the camera list, the folders.

All numbers live in config.yaml -> arm. All files live below data/arm/ (gitignored: real people, never commit).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.common import load_config, path


@lru_cache(maxsize=1)
def settings() -> dict:
    """The `arm:` section of config.yaml (read once per run: it is used inside per-frame loops)."""
    return load_config()["arm"]


def cam_letters() -> list[str]:
    """Camera letters in a fixed order, e.g. ['A', 'B', 'C', 'D', 'E']."""
    return list(settings()["cameras"].keys())


def raw_video(cam: str) -> Path:
    """The phone's original file."""
    s = settings()
    return path(s["raw_dir"]) / s["cameras"][cam]["file"]


def work_dir() -> Path:
    return path(settings()["work_dir"])


def synced_video(cam: str, full_res: bool = False) -> Path:
    """The camera's synced copy (common start, 30 fps). full_res: the full-size copy, if one was kept."""
    name = f"cam{cam}_full.mp4" if full_res else f"cam{cam}.mp4"
    return work_dir() / "synced" / name


def sync_file() -> Path:
    return work_dir() / "synced" / "sync.json"


def cache_dir() -> Path:
    return work_dir() / "cache"


def out_dir() -> Path:
    return work_dir() / "out"
