"""alfheim.py — a FULL professional half with 3 real fixed cameras (Alfheim dataset), ready for the studio.

The Alfheim dataset (Pettersen et al., "Soccer video and player position dataset", ACM MMSys 2014; Simula Research
Laboratory, non-commercial research use only) filmed Tromsø IL's home games with 3 synchronised FIXED cameras that
stand together at the halfway line, high in the main stand, each looking at one third of the pitch:

  cam1   the LEFT goal third        (dataset camera 0)
  cam2   the MIDDLE third            (dataset camera 1)
  cam3   the RIGHT goal third       (dataset camera 2)

ISSIA gave us 2 minutes; this is 45+ minutes of one half, a match the director was never tuned on.
Each camera comes as 3-second H.264 pieces (1280x960, 30 fps) named "<number>_<date> <time>.h264"; all three
cameras share the same piece numbers and timestamps, so piece N of every camera is the same 3 seconds.

We only use the VIDEO. The dataset's player-sensor data (ZXY) is not used: our own detector, pitch map and team
colours do that work, exactly as they would at a club with just cameras.

Usage:
  venv/bin/python -m studio.alfheim                    # download (~2.8 GB, resumable), join, colour fix (~30 min)
  venv/bin/python -m studio.alfheim --minutes 10       # only the first 10 minutes (a quicker test)
  detection (own Terminal, ~4-5 h for a full half):
     venv/bin/python -c "from pathlib import Path; from studio.intensity import detect; detect(Path('data/alfheim/2013-11-03_H1'))"
  venv/bin/python -m studio.alfheim --calibrate        # calibration.json + check pictures (needs nothing else)

Output: data/alfheim/<match>_H<half>/cam1-3.mp4 (1280x960, 30 fps, colour-fixed), cam1-3_480.mp4 (640x480 light
copies), story_mosaic.mp4 (the 3 cameras side by side for the story layer), meta.json, calibration.json.
Settings: config.yaml -> studio.alfheim.

Calibration: landmarks read off each picture give a first pitch -> picture mapping; it is then fine-tuned so the drawn
pitch lines sit on the white lines of the real picture (median of frames across the half = no players). The cameras
look very obliquely at the near side: a landmark-only mapping put the near touchline ~3 m (100+ px) off.
The lenses need no correction: long white lines are straight within 3 px in the raw pictures.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, ffmpeg_exe, load_config, video_info
from studio.issia import draw_checks, local_homography, pitch_lines

CREDIT = "Alfheim dataset (Pettersen et al., ACM MMSys 2014, Simula), non-commercial research use only"
URL = "https://datasets.simula.no/downloads/alfheim/{match}/{half}/{cam}.tgz"
HALVES = {1: "First%20Half", 2: "Second%20Half"}
FPS = 30


def settings() -> dict:
    return load_config()["studio"]["alfheim"]


def folder_for(match: str, half: int) -> Path:
    return ROOT / settings()["dir"] / f"{match}_H{half}"


def download(match: str, half: int) -> Path:
    """The 3 camera archives (~0.9 GB each). curl resumes a broken download (-C -). The site's certificate chain
    is incomplete (Python and curl can't verify it), hence -k; the files are plain video, nothing is run."""
    raw = ROOT / settings()["dir"] / "raw" / f"{match}_H{half}"
    raw.mkdir(parents=True, exist_ok=True)

    def one(cam: int) -> None:
        if (raw / str(cam)).is_dir():
            return                                            # already unpacked
        dst = raw / f"cam{cam}.tgz"
        url = URL.format(match=match, half=HALVES[half], cam=cam)
        print(f"  downloading camera {cam} -> {dst.relative_to(ROOT)}", flush=True)
        # (curl reports an error when asked to resume a file that is already complete: the unpack below decides)
        subprocess.run(["curl", "-sk", "--retry", "5", "-C", "-", "-o", str(dst), url])
        with tarfile.open(dst) as tar:                        # unpacks into raw/<cam>/
            tar.extractall(raw, filter="data")
        dst.unlink()

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, range(3)))
    return raw


def pieces(raw: Path, minutes: float | None) -> dict[int, list[Path]]:
    """{camera: [pieces in order]}, only piece numbers that ALL 3 cameras have (keeps them in sync)."""
    per_cam = {cam: {p.name.split("_")[0]: p for p in (raw / str(cam)).glob("*.h264")} for cam in range(3)}
    common = sorted(set.intersection(*(set(d) for d in per_cam.values())))
    missing = sum(len(d) for d in per_cam.values()) - 3 * len(common)
    if missing:
        print(f"  {missing} pieces exist for only some cameras: left out so the cameras stay in sync")
    if minutes:
        common = common[:round(minutes * 60 / 3)]
    return {cam: [per_cam[cam][n] for n in common] for cam in range(3)}


def colour_filter() -> str:
    """ffmpeg filter for the colour fix. The cameras give a strong green cast at night (white lines come out
    green-white): small per-channel gains plus a little brightening. Tested on 3 x 30 s: the detector found
    slightly more balls with it (38% vs 34%), the same players."""
    c = settings()["colour"]
    expr = lambda gain: f"clip(pow(val/255*{gain}\\,{c['gamma']})*255\\,0\\,255)"
    return f"lutrgb=r='{expr(c['red'])}':g='{expr(c['green'])}':b='{expr(c['blue'])}'"


def join(files: list[Path], dst: Path, small: Path) -> None:
    """3-second pieces -> one MP4 (+ the 640x480 light copy). Raw H.264 pieces can simply be glued together
    byte by byte; ffmpeg then reads them as one stream at 30 fps."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "h264", "-framerate", str(FPS), "-i", "pipe:0",
           "-filter_complex", f"[0:v]{colour_filter()},split=2[a][b];[b]scale=640:480[s]",
           "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-g", str(FPS), "-movflags", "+faststart", str(dst),
           "-map", "[s]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-g", str(FPS), "-movflags", "+faststart", str(small)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k, f in enumerate(files):
        proc.stdin.write(f.read_bytes())
        if k % 200 == 0:
            print(f"  {dst.name}: piece {k}/{len(files)}", flush=True)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"ffmpeg failed on {dst}")


def make_mosaic(folder: Path) -> None:
    """story_mosaic.mp4: the 3 cameras side by side (whole pitch, 1920x480) for the story layer (Gemini)."""
    inputs = []
    for c in (1, 2, 3):
        inputs += ["-i", str(folder / f"cam{c}_480.mp4")]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *inputs, "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[out]",
           "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-g", str(FPS), str(folder / "story_mosaic.mp4")]
    print("  story mosaic -> story_mosaic.mp4", flush=True)
    subprocess.run(cmd, check=True)


def prepare(match: str, half: int, minutes: float | None, force: bool = False) -> dict:
    folder = folder_for(match, half)
    folder.mkdir(parents=True, exist_ok=True)
    if force or not all((folder / f"cam{c}_480.mp4").exists() for c in (1, 2, 3)):
        raw = download(match, half)
        parts = pieces(raw, minutes)
        print(f"  {len(parts[0])} pieces per camera = {len(parts[0]) * 3 / 60:.1f} min", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:              # one ffmpeg per camera
            list(pool.map(lambda cam: join(parts[cam], folder / f"cam{cam + 1}.mp4", folder / f"cam{cam + 1}_480.mp4"),
                          range(3)))
    frames = {c: video_info(folder / f"cam{c}.mp4")["frames"] for c in (1, 2, 3)}
    print(f"  frames per camera: {frames}", flush=True)
    if len(set(frames.values())) > 1:
        print("  WARNING: the cameras have different frame counts; the shortest one sets the length")
    if force or not (folder / "story_mosaic.mp4").exists():
        make_mosaic(folder)
    n = min(frames.values())
    meta = {"dataset": "Alfheim", "match": match, "half": half, "teams": settings()["matches"].get(match, ""),
            "fps": FPS, "frames": n, "length_s": round(n / FPS, 2), "audio": False, "credit": CREDIT}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def background(video: Path, samples: int = 25) -> np.ndarray:
    """Median of frames spread over the whole video: players move, the pitch lines stay -> an empty pitch."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / samples))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8)


def line_distance(image: np.ndarray) -> np.ndarray:
    """For every pixel: how far (px) to the nearest white pitch line (thin bright structures = top-hat filter)."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    lines = (tophat > 40).astype(np.uint8)
    return cv2.distanceTransform(1 - lines, cv2.DIST_L2, 5).astype(np.float32)


def dense(line: np.ndarray, step_m: float = 0.25) -> np.ndarray:
    """Points every 25 cm along a pitch line, so every metre of line counts the same (a long touchline used to
    get as few points as a short goal-area line and was ignored)."""
    out = []
    for a, b in zip(line, line[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step_m))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def hint_pixels(dist: np.ndarray, hint: list) -> np.ndarray:
    """White-line pixels along a hint [axis, value, x1, y1, x2, y2]: the pixels within 12 px of the rough segment
    (its middle 90%), at most 80 of them, spread along it."""
    a, b = np.array(hint[2:4], float), np.array(hint[4:6], float)
    ys, xs = np.nonzero(dist == 0)
    pts = np.c_[xs, ys].astype(float)
    t = b - a
    length = np.linalg.norm(t)
    t /= length
    along, across = (pts - a) @ t, (pts - a) @ np.array([-t[1], t[0]])
    keep = (np.abs(across) < 12) & (along > 0.05 * length) & (along < 0.95 * length)
    pts, along = pts[keep], along[keep]
    return pts[np.argsort(along)][:: max(1, len(pts) // 80)]


def fine_tune(H0: np.ndarray, marks: np.ndarray, dist: np.ndarray, hints: list) -> tuple[np.ndarray, float]:
    """Adjust the mapping so the drawn pitch lines sit on the real white lines.

    1. landmarks + line hints: "these white pixels are the near touchline (y = 34)". The cameras look very
       obliquely at the near side, so a few metres of error there is 100+ px; a general search got stuck.
    2. then every pitch line in the picture is pulled onto the nearest white line (coarse to fine).
    Returns (H, median distance in px from the drawn lines to the real ones)."""
    from scipy.optimize import least_squares        # only needed here

    h, w = dist.shape
    model = np.vstack([dense(line) for line in pitch_lines()])
    p0 = H0.flatten()[:8] / H0[2, 2]

    def project(p, pts):
        return cv2.perspectiveTransform(pts[None], np.append(p, 1).reshape(3, 3))[0]

    q = project(p0, model)
    model = model[(q[:, 0] > 2) & (q[:, 0] < w - 3) & (q[:, 1] > 2) & (q[:, 1] < h - 3)]   # lines in the picture

    def line_gap(p):
        q = project(p, model)
        x, y = np.clip(q[:, 0], 0, w - 1).astype(np.float32), np.clip(q[:, 1], 0, h - 1).astype(np.float32)
        return cv2.remap(dist, x[None], y[None], cv2.INTER_LINEAR)[0]

    hinted = [(0 if h[0] == "x" else 1, float(h[1]), hint_pixels(dist, h)) for h in hints]

    def hint_gap(p):
        """Pixel distance from each hinted white pixel to where its pitch line is drawn."""
        H = np.append(p, 1).reshape(3, 3)
        out = []
        for axis, value, px in hinted:
            on_pitch = cv2.perspectiveTransform(px[None], np.linalg.inv(H))[0]
            on_pitch[:, axis] = value                     # the nearest point ON the line (along the other axis)
            out.append(np.hypot(*(cv2.perspectiveTransform(on_pitch[None], H)[0] - px).T))
        return np.concatenate(out) if out else np.zeros(0)

    marks_gap = lambda p: (project(p, marks[:, :2]) - marks[:, 2:]).ravel() * 0.3
    p = least_squares(lambda p: np.concatenate([hint_gap(p), marks_gap(p)]), p0, loss="soft_l1", f_scale=5,
                      x_scale=np.abs(p0) + 1e-6).x
    for clip, scale in ((30, 3), (10, 1.5)):
        residual = lambda p: np.concatenate([np.minimum(line_gap(p), clip), hint_gap(p), marks_gap(p)])
        p = least_squares(residual, p, loss="soft_l1", f_scale=scale, x_scale=np.abs(p0) + 1e-6).x
    return np.append(p, 1).reshape(3, 3), float(np.median(line_gap(p)))


def calibrate(folder: Path) -> dict:
    """Pitch metres -> picture pixels for every camera picture. All 3 cameras stand on the same
    side, so there is no 'turn' to find as with ISSIA's stand B."""
    s = settings()
    result = {}
    for c in (1, 2, 3):
        marks = np.asarray(s["calibration"][f"cam{c}"], np.float64)
        H0 = local_homography(marks.tolist())
        dist = line_distance(background(folder / f"cam{c}.mp4"))
        H, gap = fine_tune(H0, marks, dist, s["line_hints"].get(f"cam{c}", []))
        err = float(np.mean(np.hypot(*(cv2.perspectiveTransform(marks[None, :, :2], H)[0] - marks[:, 2:]).T)))
        result[f"cam{c}"] = {"turn": "same", "landmark_error_px": round(err, 1), "line_distance_px": round(gap, 2),
                             "pitch_to_image": H.tolist()}
        print(f"  cam{c}: pitch lines {gap:.1f} px from the real lines (median), landmarks {err:.1f} px")
    (folder / "calibration.json").write_text(json.dumps(result, indent=2))
    draw_checks(folder, {c: np.asarray(result[f"cam{c}"]["pitch_to_image"]) for c in (1, 2, 3)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", default=None, help="match date, e.g. 2013-11-03 (default: config)")
    parser.add_argument("--half", type=int, default=None, help="1 or 2 (default: config)")
    parser.add_argument("--minutes", type=float, default=None, help="use only the first N minutes")
    parser.add_argument("--force", action="store_true", help="join and convert again")
    parser.add_argument("--calibrate", action="store_true", help="pitch calibration + check pictures")
    args = parser.parse_args()
    s = settings()
    match, half = args.match or s["match"], args.half or s["half"]
    if args.calibrate:
        calibrate(folder_for(match, half))
        return 0
    print(json.dumps(prepare(match, half, args.minutes, args.force), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
