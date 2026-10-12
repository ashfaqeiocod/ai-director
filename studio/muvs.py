"""muvs.py — get REAL multi-camera football from the MUVS dataset and make it ready for the studio.

MUVS (Zenodo 10.5281/zenodo.20708683, CC BY 4.0, UFAM + Motorola): fans filmed live matches with 2-3 phones
at the same time from different seats. We use one MLS match: Chicago Fire 0-1 Charlotte FC (2024-05-15,
goal at 60'). Each "period" is a stretch where the phones overlap (2-6 min), 4K ~60 fps, NO sound track.
The dataset also says which camera a human picked for every 3 s fragment -> we can score our cuts against it.

Usage:
  venv/bin/python -m studio.muvs --period 19              # download (resumable) + convert
  venv/bin/python -m studio.muvs --period 19 --keep-4k    # same, but keep the big 4K originals

Output: data/muvs/P<period>/cam1.mp4 ... (1080p, 25 fps, same start + same length), human_cuts.csv, meta.json
The label CSVs (data/muvs/meta/) come from the dataset's Zenodo zip.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import requests

from src.common import ROOT, ffmpeg_exe, video_info

EVENT = "chicagoFire_vs_charlotteFC_20240515"
BASE_URL = f"https://storage.googleapis.com/dataset-ugv-sports/public/RAW_DATA/soccer/{EVENT}"
MUVS_DIR = ROOT / "data" / "muvs"
META_DIR = MUVS_DIR / "meta"
FRAGMENT_S = 3.0          # MUVS labels one camera per 3-second fragment
OUT_FPS = 25              # same as SoccerTrack, so the rest of the studio works unchanged
OUT_HEIGHT = 1080         # hardware rule: max working resolution 1080p


def labels(period: int) -> list[dict]:
    """The human's camera pick per fragment, plus each camera's time offset (seconds)."""
    path = META_DIR / f"{EVENT}_ANNOTATIONS.csv"
    rows = [r for r in csv.DictReader(open(path, newline="")) if int(r["period"]) == period]
    if not rows:
        sys.exit(f"No labels for period {period} in {path}")
    return sorted(rows, key=lambda r: int(r["fragment_number"]))


def download(url: str, dst: Path) -> None:
    """Download one file, resuming if a partial copy exists. Skips files that are already complete."""
    size = int(requests.head(url, timeout=30).headers.get("Content-Length", 0))
    have = dst.stat().st_size if dst.exists() else 0
    if size and have == size:
        print(f"  {dst.name}: already downloaded")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  {dst.name}: {dst.stat().st_size / 1e9:.2f} GB")


def convert(src: Path, dst: Path, start: float, length: float) -> None:
    """4K variable-fps phone video -> 1080p, constant 25 fps, cut to the shared start and length."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-vf", f"scale=-2:{OUT_HEIGHT},fps={OUT_FPS}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def prepare(period: int, keep_4k: bool) -> dict:
    rows = labels(period)
    offsets: dict[str, float] = json.loads(rows[0]["offsets"])   # e.g. {"cam_1": 0.0, "cam_2": 0.07, ...}
    cams = sorted(offsets)                                        # cameras active in this period
    raw_dir = MUVS_DIR / "raw" / f"P{period}"
    out_dir = MUVS_DIR / f"P{period}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Period {period}: cameras {cams}, offsets {offsets}")
    for cam in cams:
        download(f"{BASE_URL}/{cam}/VIDEOS/P{period}/video.mp4", raw_dir / f"{cam}.mp4")

    # Same moment in every file: skip each camera's offset, then keep only the length all cameras share.
    durations = {cam: video_info(raw_dir / f"{cam}.mp4")["duration"] for cam in cams}
    length = min(durations[c] - offsets[c] for c in cams)
    length = min(length, len(rows) * FRAGMENT_S)                  # don't go past the labelled fragments
    print(f"  shared length {length:.1f} s (raw durations {durations})")

    cam_files = {}
    for i, cam in enumerate(cams, start=1):
        dst = out_dir / f"cam{i}.mp4"
        print(f"  converting {cam} -> {dst.name} ...")
        convert(raw_dir / f"{cam}.mp4", dst, offsets[cam], length)
        cam_files[f"cam{i}"] = cam

    # Human cuts as a simple timeline: start, end, our camera name.
    to_ours = {muvs: ours for ours, muvs in cam_files.items()}
    with open(out_dir / "human_cuts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "camera"])
        for r in rows:
            start = (int(r["fragment_number"]) - 1) * FRAGMENT_S
            if start < length:
                w.writerow([start, min(start + FRAGMENT_S, length), to_ours.get(r["camera_selected"], r["camera_selected"])])

    meta = {"event": EVENT, "period": period, "fps": OUT_FPS, "length_s": round(length, 2),
            "cameras": cam_files, "offsets_s": offsets, "audio": False,
            "credit": "MUVS dataset (Pessoa, Alencar, de Freitas, UFAM), CC BY 4.0"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if not keep_4k:
        for cam in cams:
            (raw_dir / f"{cam}.mp4").unlink()
        print("  deleted the 4K originals (use --keep-4k to keep them)")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", type=int, required=True, help="MUVS period, e.g. 19 or 15 (3 cameras)")
    parser.add_argument("--keep-4k", action="store_true", help="keep the downloaded 4K files")
    args = parser.parse_args()
    meta = prepare(args.period, args.keep_4k)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
