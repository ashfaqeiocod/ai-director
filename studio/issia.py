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
  venv/bin/python -m studio.issia --calibrate     # after detection: pitch calibration of every camera (+ check pictures)

Steps for a new take (each skips what already exists):
  1. python -m studio.issia                                  convert + story mosaic
  2. detection (outside Claude Code's sandbox: CoreML):     python -c "from pathlib import Path; from studio.intensity
     import detect; detect(Path('data/issia/T2'))"
  3. python -m studio.issia --calibrate                      calibration.json (needs the detections)

Output: data/issia/T<take>/cam1.mp4 ... cam6.mp4 (1920x1080, 25 fps, same frame count), story_mosaic.mp4, meta.json,
calibration.json, calibration_check/camN.jpg. The original AVIs are MOVED to data/issia/raw/ (gitignored).

Calibration: config.yaml -> studio.issia.calibration has pitch landmarks clicked on each picture (box corners,
goal area, centre circle...). From them we get each camera's pitch -> picture mapping (a homography), like
SoccerTrack's pitch_to_image.npy. Stand B cameras film from the opposite side, so their landmarks can't tell
on their own which goal / touchline they see (the lines are symmetric). We try the possible turns and keep
the one where that camera and its stand-A partner put the same detected players at the same pitch spots.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, ffmpeg_exe, load_config, video_info

ISSIA_DIR = ROOT / "data" / "issia"
RAW_DIR = ISSIA_DIR / "raw"
NAME = re.compile(r"Film Role-0 ID-(\d) T-(\d+) .*\.avi$")
CREDIT = "ISSIA-CNR Soccer Dataset (D'Orazio et al., AVSS 2009), research use only"
MOSAIC_CAMS = (5, 3, 1)       # stand A, left to right: left goal area, midfield, right goal area
# How a stand-B camera's picture is turned compared with stand A: multiply the pitch x / y by these signs.
TURNS = {"same": (1, 1), "rotated": (-1, -1), "mirror_y": (1, -1), "mirror_x": (-1, 1)}
PARTNER = {2: 1, 4: 3, 6: 5}  # stand-B camera -> the stand-A camera that films the same area


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


def make_mosaic(folder: Path, force: bool = False) -> None:
    """story_mosaic.mp4: three stand-A cameras side by side (whole pitch, 1920x360) for the story layer.

    MVP 2's story layer (Gemini + the Colab specialist) watches ONE video. No single ISSIA camera sees the
    whole pitch, so it watches this strip: left goal area | midfield | right goal area."""
    dst = folder / "story_mosaic.mp4"
    if dst.exists() and not force:
        return
    inputs = []
    for c in MOSAIC_CAMS:
        inputs += ["-i", str(folder / f"cam{c}.mp4")]
    graph = ";".join(f"[{k}:v]scale=640:360[v{k}]" for k in range(3)) + ";[v0][v1][v2]hstack=inputs=3[out]"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-g", "25", str(dst)]
    print(f"  story mosaic -> {dst.name}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- calibration
def local_homography(points: list[list[float]]) -> np.ndarray:
    """Pitch metres -> picture pixels, from the clicked landmarks (least squares over all of them)."""
    pts = np.asarray(points, np.float64)
    H, _ = cv2.findHomography(pts[:, :2], pts[:, 2:], 0)
    return H


def turned(H: np.ndarray, turn: str) -> np.ndarray:
    """The camera's real mapping when its picture is turned: real (x, y) -> landmark coords (sx*x, sy*y) -> picture."""
    sx, sy = TURNS[turn]
    return H @ np.diag([sx, sy, 1.0])


def to_pitch(H: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    if not pts:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(np.asarray([pts], np.float64), np.linalg.inv(H))[0]


def feet_points(det: dict) -> list[tuple[float, float]]:
    return [((p["box"][0] + p["box"][2]) / 2, p["box"][3]) for p in det.get("players", [])]


def on_pitch(p, margin: float = 2.0) -> bool:
    return abs(p[0]) <= 52.5 + margin and abs(p[1]) <= 34 + margin


def agreement(H_a: np.ndarray, H_b: np.ndarray, det_a: dict, det_b: dict) -> float:
    """Average distance (m) from each player camera B puts on the pitch to the nearest one camera A puts there.
    Only frames where both see at least 3 players. Lower = the two cameras agree."""
    dists = []
    for frame, da in det_a["frames"].items():
        db = det_b["frames"].get(frame)
        if not db:
            continue
        pa = [p for p in to_pitch(H_a, feet_points(da)) if on_pitch(p)]
        pb = [p for p in to_pitch(H_b, feet_points(db)) if on_pitch(p)]
        if len(pa) < 3 or len(pb) < 3:
            continue
        a = np.asarray(pa)
        for p in pb:
            dists.append(float(np.min(np.hypot(*(a - p).T))))
    return float(np.median(dists)) if dists else math.inf


def calibrate(folder: Path) -> dict:
    cfg = load_config()["studio"]["issia"]["calibration"]
    dets = {c: json.loads((folder / "detections" / f"cam{c}.json").read_text()) for c in range(1, 7)}
    local = {c: local_homography(cfg[f"cam{c}"]) for c in range(1, 7)}
    out = {}
    for c in (1, 3, 5):
        err = np.mean([math.dist(cv2.perspectiveTransform(np.array([[p[:2]]], np.float64), local[c])[0, 0], p[2:])
                       for p in cfg[f"cam{c}"]])
        out[c] = {"turn": "same", "landmark_error_px": round(float(err), 1)}
    for b, a in PARTNER.items():
        scores = {turn: agreement(local[a], turned(local[b], turn), dets[a], dets[b]) for turn in TURNS}
        best = min(scores, key=scores.get)
        print(f"  cam{b} vs cam{a}: median player distance per turn (m) "
              + ", ".join(f"{k} {v:.1f}" for k, v in scores.items()) + f" -> {best}")
        out[b] = {"turn": best, "agreement_m": {k: round(v, 2) for k, v in scores.items()}}
    result = {}
    for c in range(1, 7):
        H = turned(local[c], out[c]["turn"])
        result[f"cam{c}"] = {**out[c], "pitch_to_image": H.tolist()}
    (folder / "calibration.json").write_text(json.dumps(result, indent=2))
    draw_checks(folder, {c: np.asarray(result[f"cam{c}"]["pitch_to_image"]) for c in range(1, 7)})
    return result


def pitch_lines() -> list[np.ndarray]:
    """The pitch markings as short polylines in metres (for the check pictures)."""
    lines = [[(-52.5, -34), (52.5, -34)], [(52.5, -34), (52.5, 34)], [(52.5, 34), (-52.5, 34)],
             [(-52.5, 34), (-52.5, -34)], [(0, -34), (0, 34)]]
    for s in (-1, 1):
        x0 = s * 52.5
        lines.append([(x0, -20.16), (x0 - s * 16.5, -20.16), (x0 - s * 16.5, 20.16), (x0, 20.16)])
        lines.append([(x0, -9.16), (x0 - s * 5.5, -9.16), (x0 - s * 5.5, 9.16), (x0, 9.16)])
    lines.append([(9.15 * math.cos(a), 9.15 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)])
    out = []
    for line in lines:
        pts = []
        for (x1, y1), (x2, y2) in zip(line, line[1:]):
            pts += [(x1 + (x2 - x1) * k / 20, y1 + (y2 - y1) * k / 20) for k in range(21)]
        out.append(np.asarray(pts, np.float64))
    return out


def draw_checks(folder: Path, homographies: dict[int, np.ndarray], frame: int = 250) -> None:
    """calibration_check/camN.jpg: the pitch lines drawn back onto each picture. They should sit on the real lines."""
    out = folder / "calibration_check"
    out.mkdir(exist_ok=True)
    for c, H in homographies.items():
        cap = cv2.VideoCapture(str(folder / f"cam{c}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        for line in pitch_lines():
            pts = cv2.perspectiveTransform(line[None], H)[0]
            # keep points in front of the camera and near the picture
            pts = [p for p in pts if -2000 < p[0] < 4000 and -2000 < p[1] < 3000]
            for p, q in zip(pts, pts[1:]):
                cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 0, 255), 2)
        cv2.imwrite(str(out / f"cam{c}.jpg"), img)
    print(f"  check pictures: {out}/camN.jpg (red = calibrated pitch lines)")


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
        make_mosaic(folder, force)
        meta = {"dataset": "ISSIA-CNR Soccer", "take": take, "fps": 25, "frames": min(frames.values()),
                "length_s": round(min(frames.values()) / 25, 2), "cameras": {f"cam{c}": cams[c].name for c in cams},
                "audio": False, "credit": CREDIT}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        out.append(meta)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="convert again even if the MP4s exist")
    parser.add_argument("--calibrate", action="store_true", help="pitch calibration (needs the detections)")
    parser.add_argument("--take", type=int, default=2)
    args = parser.parse_args()
    if args.calibrate:
        result = calibrate(ISSIA_DIR / f"T{args.take}")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pitch_to_image"} for k, v in result.items()}, indent=2))
        return 0
    for meta in prepare(args.force):
        print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
