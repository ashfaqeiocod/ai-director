"""perceive.py — Arm MVP Step 3: what every camera SEES, 10 times a second, cached to disk.

For each synced camera, on every 3rd frame (10 per second):
  people  YOLO26s-pose: a box plus 17 body points per person (nose, eyes, ears, shoulders, elbows, wrists,
          hips, knees, ankles), each with a confidence. The wrists and elbows tell us where the grip is and
          how the arms are bent.
  shirt   the average colour of each person's chest. The two athletes wear a black and a tan shirt, so the
          colour tells WHO is who in every camera, even when they swap sides (cameras A and D look from
          opposite sides of the table).
  face    MediaPipe Face Landmarker on a crop around each person's head (its detector only finds faces that
          fill the picture, so we crop first). It gives 52 "blendshapes" (0..1 muscle movements such as
          mouthSmile, browDown, eyeSquint, jawOpen) = the expression, plus how far the head is turned (yaw).
  quality how sharp the picture is (blur) and how much the phone moved since the last look (shake).
Nothing here decides anything; arm/match.py and arm/shots.py read this cache. Every value of frame n comes
only from frame n (and the one before for shake), so a live system could compute it as it happens.

Usage:
  venv/bin/python -m arm.perceive            # all cameras (skips cameras already cached)
  venv/bin/python -m arm.perceive --cam A --force
Output: data/arm/cache/cam{A..E}.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np

import src.common  # noqa: F401  (sets the model cache folders before ultralytics is imported)
from arm.common import cache_dir, cam_letters, settings, synced_video
from src.common import path

# COCO body point numbers used below.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12

# The blendshapes we keep (of 52): the ones that show effort, joy and surprise.
KEEP_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "eyeSquintLeft", "eyeSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "mouthPressLeft", "mouthPressRight", "mouthStretchLeft", "mouthStretchRight", "mouthFrownLeft",
    "mouthFrownRight", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
    "mouthRollLower", "mouthShrugUpper",
]


class Perceiver:
    """Loads the two models once and turns one frame into people + faces."""

    def __init__(self, cfg: dict):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from ultralytics import YOLO

        self.cfg = cfg
        self.pose = YOLO(str(path(cfg["pose_model"])))
        self.device = "mps"
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path(cfg["face_model"]))),
            num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE, min_face_detection_confidence=cfg["face_min_conf"])
        self.faces = vision.FaceLandmarker.create_from_options(options)

    def people(self, frame: np.ndarray) -> list[dict]:
        try:
            result = self.pose.predict(frame, imgsz=self.cfg["pose_imgsz"], device=self.device,
                                       conf=self.cfg["person_conf"], verbose=False)[0]
        except Exception:      # MPS problem on this Mac: fall back to the CPU, once and for all
            self.device = "cpu"
            result = self.pose.predict(frame, imgsz=self.cfg["pose_imgsz"], device=self.device,
                                       conf=self.cfg["person_conf"], verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        found = []
        for box, conf, kp in zip(boxes, confs, points):
            person = {"box": [round(float(v), 1) for v in box], "conf": round(float(conf), 3),
                      "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in kp],
                      "shirt": shirt_colour(frame, kp, box)}
            person["face"] = self.face(frame, kp, box)
            found.append(person)
        return found

    def face(self, frame: np.ndarray, kp: np.ndarray, box: np.ndarray) -> dict | None:
        """Expression of this person's face (None if the face can't be seen, e.g. from behind)."""
        import mediapipe as mp

        rect = head_rect(kp, box)
        if rect is None:
            return None
        # Two tries: the head square, then a tighter one (the face detector wants a face that fills the crop).
        for zoom in (1.0, 0.65):
            x, y, size = rect
            cx, cy = x + size / 2, y + size / 2
            size = int(size * zoom)
            x, y = int(cx - size / 2), int(cy - size / 2)
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + size), min(H, y + size)
            if x2 - x1 < 16 or y2 - y1 < 16:
                return None
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256),
                              interpolation=cv2.INTER_AREA if size > 256 else cv2.INTER_LINEAR)
            result = self.faces.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if result.face_blendshapes:
                break
        if not result.face_blendshapes:
            return None
        shapes = {b.category_name: round(float(b.score), 3) for b in result.face_blendshapes[0]
                  if b.category_name in KEEP_BLENDSHAPES}
        yaw = 0.0
        if result.facial_transformation_matrixes:
            m = np.array(result.facial_transformation_matrixes[0])
            yaw = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
        landmarks = result.face_landmarks[0]
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        sx, sy = (x2 - x1) / 256, (y2 - y1) / 256
        face_box = [x1 + min(xs) * 256 * sx, y1 + min(ys) * 256 * sy, x1 + max(xs) * 256 * sx, y1 + max(ys) * 256 * sy]
        return {"box": [round(v, 1) for v in face_box], "yaw": round(yaw, 1), "bs": shapes}


def head_rect(kp: np.ndarray, box: np.ndarray) -> tuple[int, int, int] | None:
    """A square around the head (x, y, size in pixels), from the face points of the body pose.

    Size: the largest of ~2.6x ear-to-ear, ~5x eye-to-eye, ~3x nose-to-ear (a face in profile has both ears
    and eyes on top of each other) and 0.8x shoulder width, but at least 18% of the person's box height.
    Needs the nose or both eyes, i.e. a face turned at least partly to the camera.
    """
    head = kp[:5]
    seen = head[head[:, 2] > 0.4]
    if kp[NOSE, 2] < 0.4 and (kp[L_EYE, 2] < 0.4 or kp[R_EYE, 2] < 0.4):
        return None
    if len(seen) < 2:
        return None
    cx, cy = float(seen[:, 0].mean()), float(seen[:, 1].mean())
    sizes = [0.18 * (box[3] - box[1])]
    if kp[L_EAR, 2] > 0.4 and kp[R_EAR, 2] > 0.4:
        sizes.append(2.6 * abs(kp[L_EAR, 0] - kp[R_EAR, 0]))
    if kp[L_EYE, 2] > 0.4 and kp[R_EYE, 2] > 0.4:
        sizes.append(5.0 * abs(kp[L_EYE, 0] - kp[R_EYE, 0]))
    if kp[NOSE, 2] > 0.4:
        for ear in (L_EAR, R_EAR):
            if kp[ear, 2] > 0.4:
                sizes.append(3.0 * float(np.hypot(*(kp[NOSE, :2] - kp[ear, :2]))))
    if kp[L_SHOULDER, 2] > 0.4 and kp[R_SHOULDER, 2] > 0.4:
        sizes.append(0.8 * abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0]))
    size = float(np.clip(max(sizes), 48, 0.6 * (box[3] - box[1]) + 48))
    if kp[NOSE, 2] > 0.4:      # the nose is the face's centre; the ears sit at the back of the head
        cx, cy = float(kp[NOSE, 0]), float(kp[NOSE, 1]) - 0.08 * size
    return int(cx - size / 2), int(cy - size * 0.5), int(size)


def shirt_colour(frame: np.ndarray, kp: np.ndarray, box: np.ndarray) -> list[float] | None:
    """Mean HSV colour of the chest: the patch between the shoulders, a bit below them."""
    if kp[L_SHOULDER, 2] < 0.4 or kp[R_SHOULDER, 2] < 0.4:
        return None
    (lx, ly), (rx, ry) = kp[L_SHOULDER, :2], kp[R_SHOULDER, :2]
    width = abs(lx - rx)
    if width < 10:
        return None
    x1, x2 = int(min(lx, rx) + 0.2 * width), int(max(lx, rx) - 0.2 * width)
    y1 = int(max(ly, ry) + 0.15 * width)
    y2 = int(y1 + 0.5 * width)
    H, W = frame.shape[:2]
    patch = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    return [round(float(v), 1) for v in np.median(hsv, axis=0)]


def picture_quality(gray: np.ndarray, prev: np.ndarray | None) -> tuple[float, float]:
    """(sharpness, shake). Sharpness = variance of the Laplacian (edges) on a 480-wide copy: low = blurry.
    Shake = how many pixels (of 480) the picture moved since the previous look (phase correlation)."""
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if prev is None:
        return sharp, 0.0
    (dx, dy), _ = cv2.phaseCorrelate(prev.astype(np.float32), gray.astype(np.float32))
    return sharp, float(math.hypot(dx, dy))


def perceive_camera(cam: str, perceiver: Perceiver, cfg: dict) -> dict:
    """Run the models over one synced camera video and return the cache dict."""
    cap = cv2.VideoCapture(str(synced_video(cam)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / cfg["analysis_fps"]))
    frames, prev, started = [], None, time.time()
    for n in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if n % step:
            continue
        small = cv2.cvtColor(cv2.resize(frame, (480, int(480 * H / W)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp, shake = picture_quality(small, prev)
        prev = small
        frames.append({"n": n, "t": round(n / fps, 3), "sharp": round(sharp, 1), "shake": round(shake, 2),
                       "people": perceiver.people(frame)})
        if len(frames) % 50 == 0:
            print(f"  cam {cam}: {n}/{total} frames, {len(frames) / (time.time() - started):.1f} looks/s")
    cap.release()
    return {"cam": cam, "fps": fps, "step": step, "width": W, "height": H, "frames": frames}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", help="only this camera letter")
    parser.add_argument("--force", action="store_true", help="redo cameras that are already cached")
    args = parser.parse_args()
    cfg = settings()["perceive"]
    cams = [args.cam] if args.cam else cam_letters()
    perceiver = None
    for cam in cams:
        out = cache_dir() / f"cam{cam}.json"
        if out.exists() and not args.force:
            print(f"cam {cam}: cached ({out.name})")
            continue
        perceiver = perceiver or Perceiver(cfg)
        started = time.time()
        result = perceive_camera(cam, perceiver, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, separators=(",", ":")))
        faces = sum(1 for f in result["frames"] for p in f["people"] if p["face"])
        people = sum(len(f["people"]) for f in result["frames"])
        print(f"cam {cam}: {len(result['frames'])} looks, {people / max(1, len(result['frames'])):.1f} people/look, "
              f"{faces / max(1, people):.0%} with a readable face, {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
