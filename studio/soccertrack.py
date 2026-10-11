"""soccertrack.py — get SoccerTrack v2 footage + annotations, and turn them into small, fast files.

SoccerTrack v2 (CC BY 4.0, https://huggingface.co/datasets/atomscott/soccertrack-v2): 10 real university
matches filmed by one fixed 4K panoramic camera (4096x1080, 25 fps), one video per half, with:
  - gsr/   players per frame: pixel box in the video + position on the pitch (metres)   -> oracle players
  - bas/   ball actions (PASS, SHOT, GOAL, ...) with the exact frame                       -> SCORING ONLY
  - ball/  ball position per frame, rebuilt from the bas events (knows the future!)      -> SCORING ONLY

Usage:
  venv/bin/python -m studio.soccertrack --match 128057 --half 2              # download (resumable)
  venv/bin/python -m studio.soccertrack --match 128057 --half 2 --convert    # gsr json -> players.npz

The Hugging Face token comes from HF_TOKEN in .env. It is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

from src.common import ROOT, load_config

BASE_URL = "https://huggingface.co/datasets/atomscott/soccertrack-v2/resolve/main"
HALF_NAMES = {1: ("1st", "1st_half"), 2: ("2nd", "2nd_half")}
ROLES = ["player", "goalkeeper", "referee", "other"]       # stored as numbers 0..3 in players.npz
TEAMS = ["left", "right", "none"]                           # 0..2


def data_dir() -> Path:
    return ROOT / load_config()["studio"]["data_dir"]


def half_files(match: str, half: int) -> dict[str, str]:
    """Remote path -> local file name for everything one half needs."""
    short, long = HALF_NAMES[half]
    return {
        f"videos/{match}/{match}_panorama_{long}.mp4": f"videos/{match}_{short}.mp4",
        f"gsr/{match}/{match}_{short}.json": f"gsr/{match}_{short}.json",
        f"ball/{match}_{short}_ball.npz": f"ball/{match}_{short}_ball.npz",
        f"bas/{match}/{match}_12_class_events.json": f"bas/{match}.json",
        f"raw/{match}/{match}_homography.npy": f"raw/{match}_homography.npy",
        f"raw/{match}/{match}_camera_intrinsics.npz": f"raw/{match}_camera_intrinsics.npz",
    }


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token == "your_token_here":
        sys.exit("HF_TOKEN is missing in .env (huggingface.co -> Settings -> Access Tokens).")
    return token


def download_file(remote: str, local: Path, token: str) -> None:
    """Download one file; if a partial file exists, continue where it stopped (HTTP Range)."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/{remote}"
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))
    have = local.stat().st_size if local.exists() else 0
    if total and have == total:
        print(f"  ok      {local.name} ({total / 1e6:.0f} MB)")
        return
    if have > total > 0:
        have = 0                       # bigger than the remote file: start again
    local.parent.mkdir(parents=True, exist_ok=True)
    if have:
        headers["Range"] = f"bytes={have}-"
    print(f"  get     {local.name} ({total / 1e6:.0f} MB{', resuming' if have else ''})", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local, "ab" if have else "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    if total and local.stat().st_size != total:
        raise RuntimeError(f"{local.name}: got {local.stat().st_size} of {total} bytes; run again to resume")


def download_half(match: str, half: int) -> None:
    token = _token()
    for remote, name in half_files(match, half).items():
        if name.endswith(".json") and name.startswith("gsr/") and (data_dir() / name.replace(".json", "_players.npz")).exists():
            print(f"  ok      {Path(name).name} (already converted to _players.npz; the 1-3 GB json is not needed)")
            continue
        download_file(remote, data_dir() / name, token)


# ----------------------------------------------------------------------------- gsr -> players.npz
def iter_json_array(path: Path, key: str, chunk_size: int = 16 << 20):
    """Yield the items of the top-level array `key` one by one, without loading the whole file.

    The gsr files are 1-2 GB of JSON (they also hold pitch-line labels for every frame); json.load
    would need several GB of RAM. We read 16 MB at a time and decode one item at a time.
    """
    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    with open(path, encoding="utf-8") as f:
        buf = ""
        while marker not in buf:                     # find the array
            more = f.read(chunk_size)
            if not more:
                raise ValueError(f"no '{key}' array in {path}")
            buf = buf[-len(marker):] + more
        buf = buf[buf.index(marker) + len(marker):]
        while "[" not in buf:                        # the "[" may be in the next chunk
            more = f.read(chunk_size)
            if not more:
                raise ValueError(f"'{key}' is not an array in {path}")
            buf += more
        buf = buf[buf.index("[") + 1:]
        pos = 0
        while True:
            # Skip spaces and commas between items.
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(chunk_size)
                if not more:
                    return
                buf, pos = more, 0
            if buf[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk_size)            # the item continues in the next chunk
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield item
            pos = end
            if pos > chunk_size:                     # drop what we've used, keep memory small
                buf, pos = buf[pos:], 0


def convert_gsr(match: str, half: int) -> Path:
    """gsr/<match>_<half>.json -> gsr/<match>_<half>_players.npz (one row per person per frame)."""
    short = HALF_NAMES[half][0]
    src = data_dir() / f"gsr/{match}_{short}.json"
    out = data_dir() / f"gsr/{match}_{short}_players.npz"
    cols: dict[str, list] = {k: [] for k in
                             ("frame", "track", "role", "team", "jersey", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y")}
    for n, a in enumerate(iter_json_array(src, "annotations")):
        if a.get("supercategory") != "object" or "bbox_image" not in a:
            continue                                  # pitch-line labels etc.
        attr, box, pitch = a.get("attributes") or {}, a["bbox_image"], a.get("bbox_pitch") or {}
        role = attr.get("role") or "other"
        team = attr.get("team") or "none"
        jersey = str(attr.get("jersey") or "")
        cols["frame"].append(int(a["image_id"]) % 1_000_000)    # image ids end in a 6-digit frame number (1-based)
        cols["track"].append(int(a.get("track_id", -1)))
        cols["role"].append(ROLES.index(role) if role in ROLES else 3)
        cols["team"].append(TEAMS.index(team) if team in TEAMS else 2)
        cols["jersey"].append(int(jersey) if jersey.isdigit() else -1)
        cols["x1"].append(box["x"])
        cols["y1"].append(box["y"])
        cols["x2"].append(box["x"] + box["w"])
        cols["y2"].append(box["y"] + box["h"])
        cols["pitch_x"].append(pitch.get("x_bottom_middle", np.nan))
        cols["pitch_y"].append(pitch.get("y_bottom_middle", np.nan))
        if n % 500_000 == 0 and n:
            print(f"  ... {n:,} annotations read", flush=True)
    arrays = {k: np.asarray(v, dtype=np.float32 if k.startswith(("x", "y", "pitch")) else np.int32)
              for k, v in cols.items()}
    np.savez_compressed(out, **arrays)
    print(f"  wrote {out.relative_to(ROOT)}: {len(arrays['frame']):,} boxes, "
          f"frames {arrays['frame'].min()}-{arrays['frame'].max()}")
    return out


def load_players(match: str, half: int) -> dict[str, np.ndarray]:
    short = HALF_NAMES[half][0]
    z = np.load(data_dir() / f"gsr/{match}_{short}_players.npz")
    return {k: z[k] for k in z.files}


def load_events(match: str, half: int) -> list[dict]:
    """Ground-truth ball actions of one half: [{frame, time_s, label, team}]. SCORING ONLY."""
    data = json.loads((data_dir() / f"bas/{match}.json").read_text())
    fps = float(data.get("fps", 25.0))
    return [{"frame": int(a["frame"]), "time_s": int(a["frame"]) / fps, "label": a["label"], "team": a.get("team")}
            for a in data["actions"] if a["gameTime"].startswith(f"{half} ")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", required=True)
    parser.add_argument("--half", type=int, choices=[1, 2], required=True)
    parser.add_argument("--convert", action="store_true", help="only convert the downloaded gsr json")
    args = parser.parse_args()
    if not args.convert:
        download_half(args.match, args.half)
    short = HALF_NAMES[args.half][0]
    if args.convert or not (data_dir() / f"gsr/{args.match}_{short}_players.npz").exists():
        convert_gsr(args.match, args.half)
        # The json is 1-3 GB and everything we need is now in the small npz: free the disk.
        (data_dir() / f"gsr/{args.match}_{short}.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
