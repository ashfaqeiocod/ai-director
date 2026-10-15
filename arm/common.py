"""common.py — small helpers shared by every Arm MVP step: the arm settings, the camera list, the folders.

All numbers live in config.yaml -> arm. All files live below data/arm/ (gitignored: real people, never commit).
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

from src.common import load_config, path


def merge(base: dict, extra: dict) -> dict:
    """`base` with `extra` laid on top (nested dicts are merged key by key, other values replaced)."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def match_name() -> str:
    """Which recorded match to work on: ARM_MATCH=match2 venv/bin/python -m arm.main (default: the first match)."""
    return os.environ.get("ARM_MATCH", "").strip()


@lru_cache(maxsize=1)
def settings() -> dict:
    """The `arm:` section of config.yaml (read once per run: it is used inside per-frame loops).

    Another match (env ARM_MATCH) = the same settings with its own block from `arm_matches:` laid on top
    (its own videos, folders and shirt words). Subprocesses inherit the env var, so every step sees the same match.
    """
    config = load_config()
    name = match_name()
    if not name:
        return config["arm"]
    if name not in config.get("arm_matches", {}):
        raise SystemExit(f"ARM_MATCH={name}: no such block under arm_matches in config.yaml")
    return merge(config["arm"], config["arm_matches"][name])


def athlete_word(who: str) -> str:
    """How an athlete is called on screen: 'black' -> 'BLACK', 'tan' -> 'TAN' (or 'BLUE' in a match where the
    second athlete wears blue: the code still calls him "tan" inside, only the words shown change)."""
    return settings().get("athlete_words", {}).get(who, who.upper())


def for_this_match(prompt: str) -> str:
    """A Gemini prompt with this match's shirt words (`prompt_words` in the match block, e.g. TAN -> BLUE).

    The first match uses the prompt unchanged (its cached answers are keyed by the exact prompt text)."""
    words = settings().get("prompt_words", {})
    if not words:
        return prompt
    for old, new in words.items():
        prompt = prompt.replace(old, new)
    return prompt + ("\nNote: the answer fields keep the names black / tan: 'tan' always means the athlete in the "
                     "blue shirt.")


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
