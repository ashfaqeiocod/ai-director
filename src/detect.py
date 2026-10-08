"""detect.py — Step 4: find the ball and the players in every frame of every camera.

Each camera is processed on its own, as if it were a separate real camera.
Results are saved to outputs/detections/camN.json, so later steps (director, render,
dashboard replay) can reuse them without running the slow model again.

The `Detector` class is the ONE place that runs a model. The engine's live mode
uses the same class, so live mode and the cached JSON always match.

Run it:  python src/detect.py            (skips cams that already have a JSON)
         python src/detect.py --force    (redo all)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

if __package__ in (None, ""):  # allow "python src/detect.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ROOT, cam_path, detections_path, load_config, video_info

import numpy as np
import supervision as sv

warnings.filterwarnings("ignore")  # the model libraries print many harmless warnings

PLAYER_CLASSES = {"player", "goalkeeper", "referee"}


class Detector:
    """Loads the model(s) once and turns one image into {players: [...], ball: {...} | None}."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["detection"]
        self.backend = "roboflow"
        try:
            self._load_roboflow()
        except Exception as error:  # bad key, no network, model removed...
            print(f"[detect] Roboflow model failed ({type(error).__name__}: {error}); using YOLO fallback")
            self._load_yolo()
        print(f"[detect] backend: {self.describe()}")

    # ---------- model loading ----------
    def _load_roboflow(self) -> None:
        from inference import get_model  # imported here: slow import, only needed for this backend

        key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not key or key == "PASTE_KEY_HERE":
            raise RuntimeError("ROBOFLOW_API_KEY missing in .env")
        self.model = get_model(self.cfg["model_id"], api_key=key)
        self.ball_model = None
        if self.cfg["use_ball_model"]:
            self.ball_model = get_model(self.cfg["ball_model_id"], api_key=key)
        self.slicer = None
        if self.cfg["use_slicer"]:
            # Tiling: cut the frame into small tiles, detect on each, merge. The ball looks bigger.
            tile = self.cfg["slicer_tile"]
            self.slicer = sv.InferenceSlicer(callback=self._ball_on_tile, slice_wh=tile, overlap_wh=tile // 5)

    def _load_yolo(self) -> None:
        from ultralytics import YOLO

        self.backend = "yolo"
        weights = ROOT / "models_cache" / self.cfg["fallback_model"]
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        self.device = self.cfg["device"]
        try:
            self.model.predict(np.zeros((64, 64, 3), np.uint8), device=self.device, verbose=False)
        except Exception:
            print("[detect] MPS failed, using CPU")
            self.device = "cpu"

    def describe(self) -> str:
        if self.backend == "yolo":
            return f"yolo {self.cfg['fallback_model']} on {self.device}"
        parts = [self.cfg["model_id"]]
        if self.ball_model is not None:
            parts.append(f"ball: {self.cfg['ball_model_id']}")
        if self.slicer is not None:
            parts.append(f"slicer {self.cfg['slicer_tile']}px")
        return " + ".join(parts)

    # ---------- detection ----------
    def detect(self, frame: np.ndarray) -> dict:
        """Detect players and the single best ball in one BGR frame."""
        # The camera strip is tall and narrow (640x1080). The model expects a wide broadcast
        # image and would squash the strip, so we place it in the middle of a black 16:9
        # canvas first. The model still only sees THIS camera's pixels.
        frame, offset = pad_to_canvas(frame, *self.cfg["model_canvas"])
        dets = self._run_main(frame)
        names = dets.data.get("class_name", np.array([]))

        players = []
        for box, conf, name in zip(dets.xyxy, dets.confidence, names):
            if name in PLAYER_CLASSES and conf >= self.cfg["player_confidence"]:
                players.append({"box": _round(box, offset), "cls": str(name), "conf": round(float(conf), 3)})

        if self.backend == "roboflow" and (self.ball_model is not None or self.slicer is not None):
            ball_dets = self.slicer(frame) if self.slicer is not None else self._ball_on_tile(frame)
            balls = [(b, c) for b, c in zip(ball_dets.xyxy, ball_dets.confidence)]
        else:
            balls = [(b, c) for b, c, n in zip(dets.xyxy, dets.confidence, names) if n == "ball"]

        # There is only one ball: keep the most confident one above the threshold.
        balls = [(b, c) for b, c in balls if c >= self.cfg["ball_confidence"]]
        ball = None
        if balls:
            box, conf = max(balls, key=lambda bc: bc[1])
            ball = {"box": _round(box, offset), "conf": round(float(conf), 3)}
        return {"players": players, "ball": ball}

    def _run_main(self, frame: np.ndarray) -> sv.Detections:
        low = min(self.cfg["ball_confidence"], self.cfg["player_confidence"])
        if self.backend == "roboflow":
            result = self.model.infer(frame, confidence=low)[0]
            return sv.Detections.from_inference(result)
        # YOLO (COCO classes): person=0 -> "player", sports ball=32 -> "ball"
        result = self.model.predict(frame, device=self.device, conf=low, classes=[0, 32], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(result)
        dets.data["class_name"] = np.array(["ball" if c == 32 else "player" for c in dets.class_id])
        return dets

    def _ball_on_tile(self, image: np.ndarray) -> sv.Detections:
        """Run the ball model (or main model) on one image/tile and keep only balls."""
        model = self.ball_model or self.model
        dets = sv.Detections.from_inference(model.infer(image, confidence=self.cfg["ball_confidence"])[0])
        if len(dets) == 0:
            return dets
        return dets[dets.data["class_name"] == "ball"]


def pad_to_canvas(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int]:
    """Center the frame horizontally on a black width x height canvas. Returns (canvas, x offset)."""
    h, w = frame.shape[:2]
    if w >= width or h != height:
        return frame, 0
    x = (width - w) // 2
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    canvas[:, x:x + w] = frame
    return canvas, x


def _round(box, x_offset: int = 0) -> list[float]:
    """Box back in camera coordinates (undo the canvas offset), rounded for a small JSON."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [round(x1 - x_offset, 1), round(y1, 1), round(x2 - x_offset, 1), round(y2, 1)]


# ---------- cache files ----------
def save_detections(out: Path, cam: int, info: dict, setup: str, frames: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"cam": cam, "fps": info["fps"], "width": info["width"], "height": info["height"],
            "setup": setup, "frames": frames}
    out.write_text(json.dumps(data))


def load_detections(path: Path) -> dict:
    return json.loads(path.read_text())


def ball_rate(frames: list[dict]) -> float:
    """Share of frames (0..1) that have a ball."""
    return sum(f["ball"] is not None for f in frames) / max(1, len(frames))


def any_cam_ball_rate(all_frames: list[list[dict]]) -> float:
    """Share of frames where AT LEAST ONE camera sees the ball."""
    n = min(len(f) for f in all_frames)
    return sum(any(cam[i]["ball"] is not None for cam in all_frames) for i in range(n)) / max(1, n)


def detect_camera(detector: Detector, cfg: dict, cam: int) -> tuple[list[dict], float]:
    """Run the detector on every frame of one camera. Returns frames and ms per frame."""
    video = cam_path(cfg, cam)
    info = video_info(video)
    frames: list[dict] = []
    start = time.time()
    # Frame by frame: the video is never fully loaded into memory.
    for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
        frames.append(detector.detect(frame))
        if i % 100 == 0:
            print(f"[detect] cam{cam} frame {i}/{info['frames']}", flush=True)
    ms = (time.time() - start) * 1000 / max(1, len(frames))
    save_detections(detections_path(cfg, cam), cam, info, detector.describe(), frames)
    return frames, ms


def run(force: bool = False) -> dict:
    cfg = load_config()
    n = cfg["video"]["num_cams"]
    todo = [c for c in range(1, n + 1) if force or not detections_path(cfg, c).exists()]
    if todo:
        detector = Detector(cfg)
        for cam in todo:
            frames, ms = detect_camera(detector, cfg, cam)
            print(f"[detect] cam{cam}: ball rate {ball_rate(frames):.0%} | {ms:.0f} ms/frame")
    else:
        print("[detect] detections already cached, skipping (use --force to redo)")
    return report(cfg)


def report(cfg: dict) -> dict:
    """Print and return the ball rate per camera and for 'any camera'."""
    n = cfg["video"]["num_cams"]
    all_frames = [load_detections(detections_path(cfg, c))["frames"] for c in range(1, n + 1)]
    rates = {f"cam{c + 1}": ball_rate(f) for c, f in enumerate(all_frames)}
    rates["any"] = any_cam_ball_rate(all_frames)
    print("[detect] ball rate: " + " | ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    return rates


if __name__ == "__main__":
    run(force="--force" in sys.argv)
