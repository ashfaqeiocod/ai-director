"""detect.py — MVP 3 Step 3a: MVP 1's ball + player detector on cams 1-3, cached to disk.

  venv/bin/python -m studio.detect --segment dev

Why only cams 1-3 and only `studio.detect_fps` frames per second: on the M1 one detection takes ~0.3 s
per camera, so every frame of 3 cams would take ~3x longer than the video. 5 fps is enough for a director
that decides about once a second. Cams 4 (wide) and 5 (close-up) are crops of the same panorama, so the
ball and players found in cams 1-3 are mapped onto them (each camera's crop rectangle is known).

Cache: data/studio/<segment>/detections/camN.json  {step, frames: {"<frame index>": {players, ball}}}
It is saved every few hundred frames and resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.common import load_config
from studio.cameras import cam_video, group_center, load_cams
from studio.prepare import load_segment, work_dir
from studio.video import to_source

DETECT_CAMS = (1, 2, 3)


def det_path(segment: str, cam: int) -> Path:
    return work_dir(segment) / "detections" / f"cam{cam}.json"


def detect_step(fps: float) -> int:
    """Run the detector every N-th frame."""
    return max(1, round(fps / load_config()["studio"]["detect_fps"]))


def load_dets(segment: str, cam: int) -> dict:
    path = det_path(segment, cam)
    if not path.exists():
        raise SystemExit(f"no detections for cam{cam}: run  venv/bin/python -m studio.detect --segment {segment}")
    data = json.loads(path.read_text())
    data["frames"] = {int(k): v for k, v in data["frames"].items()}
    return data


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def run(segment: str, allow_fallback: bool = False) -> None:
    from src.detect import Detector                 # slow import (model libraries)

    meta = load_segment(segment)
    step = detect_step(meta["fps"])
    cfg = load_config()
    caches = {}
    for c in DETECT_CAMS:
        path = det_path(segment, c)
        caches[c] = json.loads(path.read_text()) if path.exists() else {"cam": c, "step": step, "frames": {}}
        if caches[c]["step"] != step:
            caches[c] = {"cam": c, "step": step, "frames": {}}     # detect_fps changed: start again
    wanted = list(range(0, meta["frames"], step))
    todo = [i for i in wanted if any(str(i) not in caches[c]["frames"] for c in DETECT_CAMS)]
    if not todo:
        print(f"[detect] {segment}: all {len(wanted)} frames x {len(DETECT_CAMS)} cams already cached")
        report(segment)
        return
    detector = Detector(cfg)
    if detector.backend != "roboflow" and not allow_fallback:
        # This happened: CoreML could not write its compiled model (sandboxed temp folder), the football
        # model silently fell back to generic YOLO, and the ball rate dropped from ~16% to ~2%.
        raise SystemExit("[detect] the football model did not load (see the error above) and the generic YOLO "
                         "fallback finds almost no balls on this footage. Fix the model, or pass --allow-fallback.")
    for c in DETECT_CAMS:
        if caches[c].get("setup") not in (None, detector.describe()):
            print(f"[detect] cam{c}: cache was made with '{caches[c]['setup']}', starting again")
            caches[c] = {"cam": c, "step": step, "frames": {}}
        caches[c]["setup"] = detector.describe()
    todo = [i for i in wanted if any(str(i) not in caches[c]["frames"] for c in DETECT_CAMS)]
    caps = {c: cv2.VideoCapture(str(cam_video(segment, c))) for c in DETECT_CAMS}
    pool = ThreadPoolExecutor(max_workers=len(DETECT_CAMS))
    print(f"[detect] {segment}: {len(todo)} frames to do (every {step}th frame), cams {DETECT_CAMS}", flush=True)
    start, done = time.time(), 0
    try:
        for i in todo:
            images = []
            for c in DETECT_CAMS:
                if caps[c].get(cv2.CAP_PROP_POS_FRAMES) != i:      # only after a resume gap
                    caps[c].set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, image = caps[c].read()
                if not ok:
                    raise RuntimeError(f"cam{c}: cannot read frame {i}")
                images.append(image)
                # Skip the frames in between by reading (grab) instead of seeking: faster for H.264.
                for _ in range(step - 1):
                    caps[c].grab()
            results = list(pool.map(detector.detect, images))
            for c, r in zip(DETECT_CAMS, results):
                caches[c]["frames"][str(i)] = r
            done += 1
            if done % 100 == 0 or done == len(todo):
                for c in DETECT_CAMS:
                    _save(det_path(segment, c), caches[c])
                rate = done / (time.time() - start)
                left = (len(todo) - done) / max(rate, 1e-6)
                print(f"[detect] {done}/{len(todo)} frames, {rate:.2f} frames/s, ~{left / 60:.0f} min left", flush=True)
    finally:
        for c in DETECT_CAMS:
            _save(det_path(segment, c), caches[c])
        pool.shutdown()
    report(segment)


def report(segment: str) -> dict:
    thr = load_config()["detection"]["ball_confidence"]
    dets = {c: load_dets(segment, c)["frames"] for c in DETECT_CAMS}
    frames = sorted(set.intersection(*(set(d) for d in dets.values())))
    rates = {f"cam{c}": np.mean([_ball(dets[c][i], thr) is not None for i in frames]) for c in DETECT_CAMS}
    rates["any"] = np.mean([any(_ball(dets[c][i], thr) is not None for c in DETECT_CAMS) for i in frames])
    print(f"[detect] {segment}: {len(frames)} frames; ball rate " + " | ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    return rates


def _ball(det: dict, thr: float) -> dict | None:
    b = det.get("ball")
    return b if b and b["conf"] >= thr else None


class PanoramaDetections:
    """Detections of cams 1-3 mapped to panorama pixels, always the latest ones at or before frame i.

    "At or before" keeps it causal: frame i never sees a detection from a later frame.
    """

    def __init__(self, segment: str):
        self.cams = load_cams(segment)
        self.size = self.cams["size"]
        cfg = load_config()
        self.thr = cfg["detection"]["ball_confidence"]
        self.dets = {c: load_dets(segment, c) for c in DETECT_CAMS}
        self.step = self.dets[DETECT_CAMS[0]]["step"]
        parked = cfg["studio"]["parked_ball"]
        fps = load_segment(segment)["fps"]
        self.parked_window = max(1, round(parked["window_seconds"] * fps / self.step))  # in detection steps
        self.parked_radius = parked["radius_px"]
        self.parked_share = parked["min_share"]
        self.parked_history = parked["min_history"]

    def _balls(self, frame: int) -> list[tuple[float, float, float]]:
        """Every ball the 3 cams saw at this detection frame, as (x, y, conf) in panorama pixels."""
        balls = []
        for c in DETECT_CAMS:
            det = self.dets[c]["frames"].get(frame)
            b = _ball(det, self.thr) if det else None
            if b:
                x1, y1, x2, y2 = b["box"]
                rect = self.cams["cams"][c - 1]["rect"]
                balls.append((*to_source(rect, *self.size, ((x1 + x2) / 2, (y1 + y2) / 2)), b["conf"]))
        return balls

    def _parked(self, x: float, y: float, frame: int) -> bool:
        """True if a ball sat at (x, y) in most of the PAST detections: a spare ball, not the match ball.

        Only earlier frames are read, so this stays causal. A match ball that stops for a set piece
        is dropped too after a few seconds; the director then follows the players, which is fine.
        """
        past = [frame - k * self.step for k in range(1, self.parked_window + 1)]
        past = [f for f in past if f >= 0]
        if len(past) < self.parked_history:          # not enough history yet (start of the segment)
            return False
        r2 = self.parked_radius ** 2
        near = sum(any((bx - x) ** 2 + (by - y) ** 2 <= r2 for bx, by, _ in self._balls(f)) for f in past)
        return near >= self.parked_share * len(past)

    def at(self, i: int) -> dict:
        """{ball: (x, y, conf) | None, players: [(x, y), ...]} in panorama pixels, from frame <= i."""
        frame = i - i % self.step
        balls = [b for b in self._balls(frame) if not self._parked(b[0], b[1], frame)]
        players = []
        for c in DETECT_CAMS:
            det = self.dets[c]["frames"].get(frame)
            if det is None:
                continue
            rect = self.cams["cams"][c - 1]["rect"]
            for p in det["players"]:
                x1, y1, x2, y2 = p["box"]
                players.append(to_source(rect, *self.size, ((x1 + x2) / 2, (y1 + y2) / 2)))
        return {"ball": max(balls, key=lambda b: b[2]) if balls else None, "players": players}


class DetectorAim:
    """Aim for the close-up from MVP 1 detections: the ball if any camera sees it, else the main group."""

    def __init__(self, segment: str, meta: dict):
        self.pano = PanoramaDetections(segment)
        self.radius_px = load_config()["studio"]["cameras"]["close"]["group_radius_px"]

    def point(self, i: int) -> tuple[float, float] | None:
        d = self.pano.at(i)
        if d["ball"]:
            return d["ball"][0], d["ball"][1]
        if d["players"]:
            pts = np.asarray(d["players"], dtype=float)
            return group_center(pts, pts, self.radius_px)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--report", action="store_true", help="only print the ball rates")
    parser.add_argument("--allow-fallback", action="store_true", help="accept the generic YOLO fallback detector")
    args = parser.parse_args()
    report(args.segment) if args.report else run(args.segment, args.allow_fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
