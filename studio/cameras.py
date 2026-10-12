"""cameras.py — MVP 3 Step 2: turn one 4K panorama into 5 "cameras", as if a crew had filmed the match.

  venv/bin/python -m studio.cameras --segment dev              # cams 1-4 + close-up aimed by the oracle
  venv/bin/python -m studio.cameras --segment dev --only 5 --aim detector   # close-up only (after Step 3)
  venv/bin/python -m studio.cameras --segment dev --only 6 7               # the two goal cameras

  cam1 LEFT, cam2 CENTRE, cam3 RIGHT   fixed crops of the panorama ("medium" shots), like MVP 1's 3 cams
  cam4 WIDE                            the whole pitch
  cam6 GOAL LEFT, cam7 GOAL RIGHT      fixed sharp views on each goal, only for the AI goal watcher (not aired)
  cam5 CLOSE                           a virtual camera operator: a zoomed crop of the 4K picture that follows
                                       the action. It only knows the current frame (aim) and the past (smoothing),
                                       never the future, exactly like a real operator.

The close-up needs to know where the action is. Two "aim" sources:
  oracle    the annotated player positions of THIS frame (dataset gsr). Labelled "oracle" everywhere.
  detector  MVP 1 detections of cams 1-3 for THIS frame (ball if seen, else the main group of players).

Output: data/studio/<segment>/cams/camN.mp4 (all 1280x720, same fps and frame count) + cams.json
(each camera's type and its crop rectangle in source pixels; the close-up's rectangle per frame is in
cam5_<aim>_rects.npy). Scoring needs the rectangles to know what each camera could see.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from src.common import load_config
from studio.prepare import annotation_frame, load_segment, work_dir
from studio.soccertrack import load_players
from studio.video import VideoWriter, crop_to

FIXED_CAMS = (1, 2, 3, 4)
CLOSE_CAM = 5
GOAL_CAMS = (6, 7)      # one fixed view on each goal: the AI's "eyes" for goals (studio.goalwatch), never on air


def cams_cfg() -> dict:
    return load_config()["studio"]["cameras"]


def cams_dir(segment: str) -> Path:
    return work_dir(segment) / "cams"


def cam_video(segment: str, cam: int, aim: str = "oracle") -> Path:
    return cams_dir(segment) / (f"cam{cam}_{aim}.mp4" if cam == CLOSE_CAM else f"cam{cam}.mp4")


# ----------------------------------------------------------------------------- aiming the close-up
class OracleAim:
    """Where the action is, from the annotated players of ONE frame (no past, no future).

    The action is where most players are close together (measured on the pitch, in metres, so
    perspective doesn't matter): find the player with the most neighbours within `radius_m`, and
    aim at the middle of that group, in video pixels.
    """

    def __init__(self, segment_meta: dict, radius_m: float):
        p = load_players(segment_meta["match"], segment_meta["half"])
        order = np.argsort(p["frame"], kind="stable")
        self.p = {k: v[order] for k, v in p.items()}
        self.meta = segment_meta
        self.radius = radius_m

    def point(self, i: int) -> tuple[float, float] | None:
        frame_no = annotation_frame(self.meta, i)
        lo, hi = np.searchsorted(self.p["frame"], [frame_no, frame_no + 1])
        if hi - lo == 0:
            return None
        sl = slice(lo, hi)
        pitch = np.stack([self.p["pitch_x"][sl], self.p["pitch_y"][sl]], axis=1)
        image = np.stack([(self.p["x1"][sl] + self.p["x2"][sl]) / 2, (self.p["y1"][sl] + self.p["y2"][sl]) / 2], axis=1)
        return group_center(pitch, image, self.radius)


def group_center(positions: np.ndarray, image_points: np.ndarray, radius: float) -> tuple[float, float]:
    """Middle (in image_points) of the biggest group of points within `radius` (in positions units)."""
    ok = np.all(np.isfinite(positions), axis=1)
    if not ok.any():
        return tuple(np.mean(image_points, axis=0))
    positions, image_points = positions[ok], image_points[ok]
    dist = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    near = dist <= radius
    best = int(np.argmax(near.sum(axis=1)))
    x, y = image_points[near[best]].mean(axis=0)
    return float(x), float(y)


class VirtualOperator:
    """Turns a (jumpy) aim point per frame into a smooth camera move, using only the past.

    Like a real operator: ignore small movements (dead zone), follow bigger ones gently (smoothing),
    and never pan faster than a real head can (max speed). If the aim is unknown, hold still.
    """

    def __init__(self, cfg: dict, src_w: int, src_h: int):
        self.w = cfg["zoom_width"]
        self.h = self.w * 9 / 16
        self.src_w, self.src_h = src_w, src_h
        self.smoothing = cfg["smoothing"]
        self.max_speed = cfg["max_speed_px"]
        self.dead_zone = cfg["dead_zone_px"]
        self.center: list[float] | None = None

    def update(self, aim: tuple[float, float] | None) -> tuple[float, float, float, float]:
        if self.center is None:
            self.center = list(aim) if aim else [self.src_w / 2, self.src_h / 2]
        elif aim is not None:
            for axis in (0, 1):
                gap = aim[axis] - self.center[axis]
                if abs(gap) <= self.dead_zone:
                    continue
                step = self.smoothing * (gap - math.copysign(self.dead_zone, gap))
                self.center[axis] += max(-self.max_speed, min(self.max_speed, step))
        # Keep the whole frame inside the picture.
        self.center[0] = min(max(self.center[0], self.w / 2), self.src_w - self.w / 2)
        self.center[1] = min(max(self.center[1], self.h / 2), self.src_h - self.h / 2)
        return (self.center[0] - self.w / 2, self.center[1] - self.h / 2, self.w, self.h)


# ----------------------------------------------------------------------------- bad-feed test
def degrade(image: np.ndarray, cam: int, t: float, test: dict | None, rng: np.random.Generator) -> np.ndarray:
    """Make one camera blurry/shaky for a while (config studio.segments.<name>.degrade) to test rejection."""
    if not test or test["cam"] != cam or not (test["start_s"] <= t < test["end_s"]):
        return image
    if test.get("blur", 0):
        k = int(test["blur"]) | 1
        image = cv2.GaussianBlur(image, (k, k), 0)
    if test.get("shake_px", 0):
        dx, dy = rng.normal(0, test["shake_px"], 2)
        image = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]),
                               borderMode=cv2.BORDER_REFLECT)
    return image


# ----------------------------------------------------------------------------- rendering
def render(segment: str, only: list[int] | None = None, aim: str = "oracle") -> None:
    meta = load_segment(segment)
    cfg = cams_cfg()
    seg_cfg = load_config()["studio"]["segments"][segment]
    width, height = cfg["size"]
    fps, total = meta["fps"], meta["frames"]
    todo = only or [*FIXED_CAMS, CLOSE_CAM]
    out = cams_dir(segment)
    writers = {c: VideoWriter(cam_video(segment, c, aim), width, height, fps) for c in todo}
    rects = {c: tuple(cfg["fixed"][f"cam{c}"]["rect"]) for c in todo if c in FIXED_CAMS}
    rects |= {c: tuple(cfg["goal"][f"cam{c}"]["rect"]) for c in todo if c in GOAL_CAMS}
    operator = aimer = None
    close_rects = []
    if CLOSE_CAM in todo:
        operator = VirtualOperator(cfg["close"], meta["width"], meta["height"])
        aimer = make_aim(aim, segment, meta)
    rng = np.random.default_rng(0)
    start = time.time()
    try:
        source = sv.get_video_frames_generator(str(work_dir(segment) / "source.mp4"))
        for i, image in enumerate(source):
            t = i / fps
            for c in todo:
                if c == CLOSE_CAM:
                    rect = operator.update(aimer.point(i))
                    close_rects.append(rect)
                else:
                    rect = rects[c]
                writers[c].write(degrade(crop_to(image, rect, width, height), c, t, seg_cfg.get("degrade"), rng))
            if i % 500 == 0:
                rate = (i + 1) / max(1e-6, time.time() - start)
                print(f"[cameras] frame {i}/{total} ({rate:.0f} fps)", flush=True)
    except BaseException:
        for w in writers.values():
            w.abort()
        raise
    for w in writers.values():
        w.close()
    if close_rects:
        np.save(out / f"cam5_{aim}_rects.npy", np.asarray(close_rects, dtype=np.float32))
    write_cams_json(segment, meta)
    print(f"[cameras] done: {len(todo)} cams, {i + 1} frames in {time.time() - start:.0f} s")


def make_aim(aim: str, segment: str, meta: dict):
    if aim == "oracle":
        return OracleAim(meta, cams_cfg()["close"]["group_radius_m"])
    if aim == "detector":
        from studio.detect import DetectorAim   # Step 3

        return DetectorAim(segment, meta)
    raise SystemExit(f"unknown aim '{aim}' (oracle | detector)")


def write_cams_json(segment: str, meta: dict) -> None:
    cfg = cams_cfg()
    cams = []
    for c in FIXED_CAMS:
        f = cfg["fixed"][f"cam{c}"]
        cams.append({"cam": c, "name": f["name"], "shot_type": f["shot_type"], "rect": f["rect"]})
    cams.append({"cam": CLOSE_CAM, "name": cfg["close"]["name"], "shot_type": cfg["close"]["shot_type"],
                 "rect": None, "rects_file": "cam5_<aim>_rects.npy"})
    # Goal cameras are listed apart: the director never airs them, only the goal watcher looks at them.
    goal = [{"cam": c, "name": cfg["goal"][f"cam{c}"]["name"], "side": cfg["goal"][f"cam{c}"]["side"],
             "rect": cfg["goal"][f"cam{c}"]["rect"]} for c in GOAL_CAMS if cam_video(segment, c).exists()]
    info = {"segment": segment, "size": cfg["size"], "fps": meta["fps"], "frames": meta["frames"],
            "source_size": [meta["width"], meta["height"]], "cams": cams, "goal_cams": goal}
    (cams_dir(segment) / "cams.json").write_text(json.dumps(info, indent=2))


def load_cams(segment: str) -> dict:
    return json.loads((cams_dir(segment) / "cams.json").read_text())


def cam_rect(cams: dict, cam: int, i: int, close_rects: np.ndarray | None) -> tuple[float, float, float, float]:
    """The crop rectangle (source pixels) camera `cam` showed at frame i."""
    if cam == CLOSE_CAM:
        return tuple(float(v) for v in close_rects[min(i, len(close_rects) - 1)])
    return tuple(cams["cams"][cam - 1]["rect"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--only", type=int, nargs="*", help="camera numbers to render (default: all)")
    parser.add_argument("--aim", default="oracle", choices=["oracle", "detector"], help="what aims the close-up")
    args = parser.parse_args()
    render(args.segment, args.only, args.aim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
