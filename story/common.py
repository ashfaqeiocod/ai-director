"""common.py — small helpers shared by the story package.

Re-uses MVP 1's helpers (project folder, config.yaml, .env, ffmpeg) so both phases read
the same config file and secrets.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.common import ROOT, ffmpeg_exe, load_config, path  # noqa: F401  (re-exported)


def story_config() -> dict[str, Any]:
    """The `story:` section of config.yaml."""
    return load_config()["story"]


def gemini_key() -> str:
    """The Gemini API key from .env ("" if missing or still the placeholder)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return "" if key in ("", "your_key_here") else key


def gemini_keys() -> list[str]:
    """Every Gemini key in .env: GEMINI_API_KEY, then GEMINI_API_KEY1, GEMINI_API_KEY2 ... (each Google account
    has its own free quota, so a second key doubles the requests per minute and per day)."""
    names = sorted((k for k in os.environ if re.fullmatch(r"GEMINI_API_KEY\d*", k)), key=lambda k: (len(k), k))
    keys = [os.environ[k].strip() for k in names]
    return list(dict.fromkeys(k for k in keys if k and k != "your_key_here"))


def safe_name(text: str) -> str:
    """Turn any file name or title into a short, safe folder name (letters, digits, - and _)."""
    if Path(text).suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
        text = Path(text).stem
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    return (name or "clip")[:60].lower()


def clips_dir() -> Path:
    return path(story_config()["clips_dir"])


def outputs_dir() -> Path:
    return path(story_config()["outputs_dir"])
