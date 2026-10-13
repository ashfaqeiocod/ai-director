"""intensity.py — "how intense is play right now?" from PICTURES ONLY (for footage with no sound, e.g. MUVS).

The story layer's fast layer gets 40% of its intensity from crowd sound. Fans' phone footage (MUVS) has no
sound track, so here intensity comes from what the cameras see, once per second, using only the past:

  pressure  players bunched around the goalkeeper (= near a goal), more if the ball is there too   0..1
  speed     how fast the fastest players run, in body heights per second, with the PHONE'S OWN
            MOVEMENT removed (a fan panning the camera is not a counter-attack)                     0..1
  STOP      a trigger: fast play that suddenly stops (a whistle, a foul, a corner, a goal)

Each camera is scored on its own; the match value is the best camera's (the one that sees the action).
Intensity = weighted mix (config studio.vision_intensity.weights), averaged over the last few seconds, 0..100.

Usage:
  venv/bin/python -m studio.intensity --folder data/muvs/P19 --expect 75-110
     (--expect only prints a check: is intensity higher inside that window? It never changes the result.)

Detection uses MVP 1's football model (CoreML). Inside Claude Code's sandbox it can't load: run it outside.
Cache: <folder>/detections/camN.json (resumes). Output: <folder>/intensity.csv and intensity.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, load_config


def settings() -> dict:
    return load_config()["studio"]["vision_intensity"]


def cam_videos(folder: Path) -> list[Path]:
    videos = sorted(folder.glob("cam[0-9].mp4"))
    if not videos:
        raise SystemExit(f"no cam1.mp4, cam2.mp4 ... in {folder}")
    return videos


# --------------------------------------------------------------------------- step 1: detect + camera movement
def camera_shift(prev: np.ndarray, cur: np.ndarray, window: np.ndarray) -> tuple[float, float]:
    """How far the picture's content moved between two small grayscale frames (pixels of the small frame).

    Phase correlation finds the one shift that best lines the two pictures up. Most of a phone picture is
    pitch and stands, so this is the CAMERA's movement (players are small and move in all directions).
    """
    (dx, dy), _ = cv2.phaseCorrelate(prev, cur, window)
    return float(dx), float(dy)


def detect(folder: Path, allow_fallback: bool = False) -> dict[int, dict]:
    """Run the football detector every few frames on every camera; cache players, ball and camera shift."""
    cfg = load_config()
    s = settings()
    videos = cam_videos(folder)
    out_dir = folder / "detections"
    out_dir.mkdir(exist_ok=True)
    caches = {}
    for video in videos:
        path = out_dir / f"{video.stem}.json"
        caches[video.stem] = json.loads(path.read_text()) if path.exists() else {"frames": {}}
    fps = cv2.VideoCapture(str(videos[0])).get(cv2.CAP_PROP_FPS)
    total = int(cv2.VideoCapture(str(videos[0])).get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / s["detect_fps"]))
    wanted = list(range(0, total, step))
    if all(len(c["frames"]) >= len(wanted) for c in caches.values()):
        print(f"[intensity] detections cached for {len(videos)} cams")
        return {int(k[3:]): v for k, v in caches.items()}

    from src.detect import Detector                     # slow import (model libraries)
    detector = Detector(cfg)
    if detector.backend != "roboflow" and not allow_fallback:
        raise SystemExit("[intensity] the football model did not load (sandbox?). Run outside the sandbox, "
                         "or pass --allow-fallback (generic YOLO: far fewer balls and no goalkeeper class).")
    pool = ThreadPoolExecutor(max_workers=len(videos))
    print(f"[intensity] detecting {len(wanted)} frames x {len(videos)} cams (every {step}th frame) with "
          f"{detector.describe()}", flush=True)
    start = time.time()

    def one_camera(video: Path) -> None:
        cache = caches[video.stem]
        cache.update({"fps": fps, "step": step})
        cap = cv2.VideoCapture(str(video))
        window, prev = None, None
        for i in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            for _ in range(step - 1):
                cap.grab()                               # skip the frames in between (faster than seeking)
            h, w = frame.shape[:2]
            small = cv2.cvtColor(cv2.resize(frame, (s["shift_width"], round(h * s["shift_width"] / w))),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
            if window is None:
                window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            dx, dy = camera_shift(prev, small, window) if prev is not None else (0.0, 0.0)
            prev = small
            if str(i) in cache["frames"]:
                continue
            result = detector.detect(frame)
            scale = w / s["shift_width"]                 # small-frame pixels -> full-frame pixels
            result["shift"] = [round(dx * scale, 1), round(dy * scale, 1)]
            cache["frames"][str(i)] = result
            if len(cache["frames"]) % 300 == 0:          # checkpoint (~1 min): a long run (Alfheim: hours) resumes
                tmp = out_dir / f"{video.stem}.json.tmp"     # write + rename: a kill mid-write never corrupts it
                tmp.write_text(json.dumps(cache))
                tmp.replace(out_dir / f"{video.stem}.json")

    try:
        list(pool.map(one_camera, videos))
    finally:
        for video in videos:
            (out_dir / f"{video.stem}.json").write_text(json.dumps(caches[video.stem]))
        pool.shutdown()
    print(f"[intensity] detection done in {(time.time() - start) / 60:.1f} min", flush=True)
    return {int(k[3:]): v for k, v in caches.items()}


# --------------------------------------------------------------------------- step 2: per-camera signals
def feet(box: list[float]) -> tuple[float, float]:
    """Bottom-centre of a box: where the player stands."""
    return (box[0] + box[2]) / 2, box[3]


def on_pitch(hsv: np.ndarray, box: list[float]) -> bool:
    """True when the ground just under the feet is pitch green. Fans, stewards and the bench are not on grass."""
    s = settings()
    h, w = hsv.shape[:2]
    x, y = feet(box)
    half = max(3, int((box[2] - box[0]) / 4))
    patch = hsv[min(h - 1, int(y) + 2):min(h, int(y) + 10), max(0, int(x) - half):min(w, int(x) + half)]
    if patch.size == 0:
        return False
    green = cv2.inRange(patch, np.array(s["pitch_hsv_low"]), np.array(s["pitch_hsv_high"]))
    return float(np.mean(green > 0)) >= s["pitch_min_share"]


def camera_signals(cache: dict, video: Path) -> list[dict]:
    """For each detected frame (in time order): t, pressure 0..1, speed 0..1. Uses only that frame and earlier."""
    import supervision as sv

    s = settings()
    fps, step = cache["fps"], cache["step"]
    tracker = sv.ByteTrack(frame_rate=max(1, round(fps / step)))
    last_pos: dict[int, tuple[float, float]] = {}
    cap = cv2.VideoCapture(str(video))
    out = []
    for i in sorted(int(k) for k in cache["frames"]):
        det = cache["frames"][str(i)]
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < i:
            cap.grab()
        ok, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if ok else None
        players = [p for p in det["players"] if hsv is not None and on_pitch(hsv, p["box"])]
        heights = [p["box"][3] - p["box"][1] for p in players]
        body = float(np.median(heights)) if heights else 0.0

        # Pressure: players around the goalkeeper (the keeper only stands near a goal).
        pressure = 0.0
        keepers = [p for p in players if p["cls"] == "goalkeeper"]
        if keepers and body > 0:
            kx, ky = feet(max(keepers, key=lambda p: p["conf"])["box"])
            radius = s["pressure_radius_heights"] * body
            near = sum(1 for p in players if p["cls"] != "goalkeeper"
                       and np.hypot(feet(p["box"])[0] - kx, feet(p["box"])[1] - ky) <= radius)
            pressure = min(1.0, near / s["pressure_full"])
            if det.get("ball"):
                bx, by = feet(det["ball"]["box"])
                if np.hypot(bx - kx, by - ky) <= radius:
                    pressure = min(1.0, pressure + s["ball_near_keeper_bonus"])

        # Speed: track players, then subtract the camera's own movement from how far each one moved.
        speed = 0.0
        if players:
            dets = sv.Detections(xyxy=np.array([p["box"] for p in players], dtype=float),
                                 confidence=np.array([p["conf"] for p in players], dtype=float),
                                 class_id=np.zeros(len(players), dtype=int))
            tracked = tracker.update_with_detections(dets)
            dx, dy = det.get("shift", [0.0, 0.0])
            dt = step / fps
            speeds = []
            positions = {}
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                x, y = feet(list(box))
                positions[int(tid)] = (x, y)
                if int(tid) in last_pos:
                    px, py = last_pos[int(tid)]
                    moved = np.hypot((x - px) - dx, (y - py) - dy)       # true movement on screen
                    h = max(1.0, box[3] - box[1])
                    speeds.append(min(2.0, moved / h / dt / s["sprint_heights_s"]))   # cap ID-switch jumps
            last_pos = positions
            if speeds:
                fastest = sorted(speeds, reverse=True)[:s["fast_players"]]
                speed = min(1.0, float(np.mean(fastest)))
        out.append({"t": i / fps, "pressure": pressure, "speed": speed})
    return out


# --------------------------------------------------------------------------- step 3: one value per second
def per_second(signals: dict[int, list[dict]]) -> list[dict]:
    """Best camera per second, then the weighted mix, smoothed over the last seconds, plus STOP triggers."""
    s = settings()
    w = s["weights"]
    seconds: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for cam, rows in signals.items():
        for r in rows:
            seconds[int(np.ceil(r["t"])) or 1][cam].append(r)      # second k holds samples in (k-1, k]
    history: deque[float] = deque(maxlen=s["smooth_s"])
    speeds: deque[float] = deque(maxlen=s["stop_lookback_s"])
    rows = []
    for k in sorted(seconds):
        cams = seconds[k]
        squash = np.max if s["pressure_per_second"] == "max" else np.mean
        pressure = {c: float(squash([r["pressure"] for r in v])) for c, v in cams.items()}
        speed = {c: float(np.mean([r["speed"] for r in v])) for c, v in cams.items()}
        p, sp = max(pressure.values()), max(speed.values())
        raw = (w["pressure"] * p + w["speed"] * sp) / (w["pressure"] + w["speed"])
        history.append(raw)
        trigger = ""
        if speeds and max(speeds) >= s["stop_fast"] and sp <= s["stop_slow"]:
            trigger = "STOP"
        speeds.append(sp)
        rows.append({"t": k, "pressure": round(p, 3), "speed": round(sp, 3),
                     "intensity": round(100 * float(np.mean(history))),
                     "best_cam": f"cam{max(pressure, key=lambda c: pressure[c] + speed[c])}",
                     **{f"cam{c}_pressure": round(v, 3) for c, v in sorted(pressure.items())},
                     **{f"cam{c}_speed": round(v, 3) for c, v in sorted(speed.items())},
                     "trigger": trigger})
    return rows


def save(folder: Path, rows: list[dict], expect: tuple[float, float] | None) -> None:
    with open(folder / "intensity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    if expect:
        ax.axvspan(*expect, color="#f2c94c", alpha=0.25, label="moment seen by a human (check only)")
    ax.plot(t, [r["intensity"] for r in rows], color="#c0392b", lw=2.2, label="intensity (0-100)")
    ax.plot(t, [100 * r["pressure"] for r in rows], color="#2d6cdf", lw=1, alpha=0.8, label="pressure near goal x100")
    ax.plot(t, [100 * r["speed"] for r in rows], color="#27ae60", lw=1, alpha=0.8, label="player speed x100")
    for r in rows:
        if r["trigger"]:
            ax.axvline(r["t"], color="#555", ls=":", lw=1)
    ax.set_xlabel("seconds")
    ax.set_ylim(0, 105)
    ax.set_title(f"Vision-only intensity: {folder.name} (dotted lines = STOP triggers)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "intensity.png", dpi=110)
    plt.close(fig)


def report(rows: list[dict], expect: tuple[float, float] | None) -> None:
    peak = max(rows, key=lambda r: r["intensity"])
    stops = [r["t"] for r in rows if r["trigger"]]
    print(f"[intensity] {len(rows)} s | peak {peak['intensity']} at {peak['t']} s ({peak['best_cam']}) | "
          f"STOP triggers at {stops}")
    if expect:
        inside = [r["intensity"] for r in rows if expect[0] <= r["t"] <= expect[1]]
        outside = [r["intensity"] for r in rows if not expect[0] <= r["t"] <= expect[1]]
        print(f"[intensity] check: mean inside {expect[0]:.0f}-{expect[1]:.0f} s = {np.mean(inside):.0f}, "
              f"outside = {np.mean(outside):.0f}, peak inside: {expect[0] <= peak['t'] <= expect[1]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="folder with cam1.mp4, cam2.mp4 ... (e.g. data/muvs/P19)")
    parser.add_argument("--expect", help="check only: a window like 75-110 where a human saw the big moment")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    folder = (ROOT / args.folder) if not Path(args.folder).is_absolute() else Path(args.folder)
    expect = tuple(float(x) for x in args.expect.split("-")) if args.expect else None
    caches = detect(folder, args.allow_fallback)
    rows = per_second({cam: camera_signals(cache, folder / f"cam{cam}.mp4") for cam, cache in caches.items()})
    save(folder, rows, expect)
    report(rows, expect)
    print(f"[intensity] wrote {folder / 'intensity.csv'} and intensity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
