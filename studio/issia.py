"""issia.py — 6 REAL fixed broadcast-style cameras of one match (ISSIA-CNR Soccer Dataset), ready for the studio.

The ISSIA-CNR Soccer Dataset (D'Orazio et al., "A semi-automatic system for ground truth generation of soccer
video sequences", AVSS 2009; free for research use) filmed an Italian Serie A match with 6 synchronised,
FIXED full-HD cameras placed around the stadium, 2 per area of the pitch:

  cam1 + cam2   the RIGHT goal area (two angles)
  cam3 + cam4   the MIDDLE of the pitch (two angles)
  cam5 + cam6   the LEFT goal area (two angles)

The files are named "Film Role-0 ID-<cam> T-<take> ....avi" (DivX, 25 fps, ~2 min, no sound track).
Browsers can't play DivX AVI, so we convert to H.264 MP4 (the dashboard plays the camera videos directly).

Usage:
  venv/bin/python -m studio.issia                 # finds the AVIs in the repo root (or data/issia/raw), converts
  venv/bin/python -m studio.issia --force         # convert again

Output: data/issia/T<take>/cam1.mp4 ... cam6.mp4 (1920x1080, 25 fps, same frame count) + meta.json
The original AVIs are MOVED (not copied) to data/issia/raw/ (gitignored, like all footage).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import ROOT, ffmpeg_exe, video_info

ISSIA_DIR = ROOT / "data" / "issia"
RAW_DIR = ISSIA_DIR / "raw"
NAME = re.compile(r"Film Role-0 ID-(\d) T-(\d+) .*\.avi$")
CREDIT = "ISSIA-CNR Soccer Dataset (D'Orazio et al., AVSS 2009), research use only"


def find_raw() -> dict[int, dict[int, Path]]:
    """{take: {cam: path}}. AVIs left in the repo root are moved into data/issia/raw/ first."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in ROOT.glob("Film Role-0 ID-* T-*.avi"):
        print(f"  moving {path.name} -> data/issia/raw/")
        shutil.move(str(path), RAW_DIR / path.name)
    takes: dict[int, dict[int, Path]] = {}
    for path in sorted(RAW_DIR.glob("*.avi")):
        m = NAME.match(path.name)
        if m:
            takes.setdefault(int(m.group(2)), {})[int(m.group(1))] = path
    return takes


def convert(src: Path, dst: Path) -> None:
    """DivX AVI -> H.264 MP4. Two of the cameras are 1920x1088 (codec padding): crop every camera to 1080 rows."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "crop=1920:1080:0:0", "-an", "-r", "25",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-g", "25",                           # a keyframe every second: the page seeks fast and stays in sync
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def prepare(force: bool = False) -> list[dict]:
    takes = find_raw()
    if not takes:
        sys.exit("No 'Film Role-0 ID-<cam> T-<take> ....avi' files in the repo root or data/issia/raw/")
    out = []
    for take, cams in sorted(takes.items()):
        folder = ISSIA_DIR / f"T{take}"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Take {take}: cameras {sorted(cams)}")
        jobs = [(cams[c], folder / f"cam{c}.mp4") for c in sorted(cams) if force or not (folder / f"cam{c}.mp4").exists()]
        with ThreadPoolExecutor(max_workers=3) as pool:          # 3 at a time: the M1 has 8 GB
            list(pool.map(lambda job: convert(*job), jobs))
        infos = {c: video_info(folder / f"cam{c}.mp4") for c in sorted(cams)}
        frames = {c: i["frames"] for c, i in infos.items()}
        print(f"  frames per camera: {frames}")
        meta = {"dataset": "ISSIA-CNR Soccer", "take": take, "fps": 25, "frames": min(frames.values()),
                "length_s": round(min(frames.values()) / 25, 2), "cameras": {f"cam{c}": cams[c].name for c in cams},
                "audio": False, "credit": CREDIT}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        out.append(meta)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="convert again even if the MP4s exist")
    args = parser.parse_args()
    for meta in prepare(args.force):
        print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
