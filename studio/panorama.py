"""panorama.py — the 3 real Alfheim cameras stitched into ONE wide panorama (MVP 4 Step 1).

Why: in MVP 4 every TV camera is a "virtual" one cut out of real pixels (MAIN, CLOSE, WIDE ...). A robotic camera
must be able to pan smoothly from the left goal to the right goal, so it needs one continuous picture of the whole
pitch, not three separate ones with jumps between them.

How (in plain words):
  1. The 3 Alfheim cameras stand together on one platform at the halfway line. From their calibrations
     (calibration.json: pitch metres -> picture pixels, one 3x3 "homography" per camera) we fit ONE shared
     position plus each camera's focal length and direction (`fit_rig`). Result: the rig is ~12 m high,
     ~8 m behind the near touchline. This is our MAIN camera position for the rest of MVP 4.
  2. The panorama is a CYLINDER around that point: a panorama column = a compass direction (angle), a row = how
     far down you look (also an angle; an "equirectangular" cylinder). A flat (planar) panorama would stretch the
     far ends enormously (the rig sees ~170 degrees); this keeps them in proportion and keeps vertical things
     (players, posts) upright. (Rows as angles, not heights: the grass right under the rig would otherwise be
     stretched ~3x and the panorama would be twice as tall for nothing.)
  3. Lining the cameras up. Because the 3 cameras share one spot, ONE 3x3 homography maps cam2's picture onto
     cam1's (and one onto cam3's) for everything at any distance: grass, players and stands. We measure these two
     homographies from the pictures themselves (`match_cameras`: SIFT feature matches in the overlaps, over 40
     moments of the half). Measured: they fit the pitch AND the stands to ~1 px. The plan's first idea,
     `H_N @ inv(H_2)` from the calibrations, was ~15-25 px off in the overlaps (each calibration is best in the
     middle of its own picture and weakest at the edges, which is exactly where the seams are).
  4. Where two cameras overlap we pick the camera that sees the spot closest to its own picture centre and blend
     the two linearly over a short band (a "feather") so the seam doesn't show.
  5. Pitch metres <-> panorama pixels: in each camera's part of the panorama we use THAT camera's calibration (its
     best estimate for what it sees), blended with the same weights over the seams. So pitch lines drawn from the
     calibration sit on the real white lines everywhere, also at the far goals.
     The 3 MVP 3 calibrations were each fitted alone; at the seams (bare grass, few lines) neighbours disagreed by
     ~40 px. `refine_calibrations` fits all 3 again TOGETHER: each still sits on its own white lines, and in the
     seam bands it must agree with cam2's (carried over by the picture alignment). The refined calibrations live
     in panorama.json only; calibration.json (MVP 3) is unchanged.
All this geometry is computed once (cv2.remap lookup tables); each frame is then just 3 lookups + a blend.

Coordinates: pitch metres, origin at the centre spot, x to the right (-52.5 .. 52.5), y towards the NEAR
touchline (-34 .. 34). For heights we use a third axis z that points DOWN (so x, y, z is a right-handed frame);
`height_m` in panorama.json is the usual "metres above the grass".

Usage:
  venv/bin/python -m studio.panorama --check           # panorama.json + panorama_check.jpg + seam numbers (~1 min)
  venv/bin/python -m studio.panorama --check --refit   # build panorama.json again (after a config change)
  venv/bin/python -m studio.panorama --minutes 2       # quick test video: panorama_2min.mp4
  venv/bin/python -m studio.panorama                   # the whole half -> panorama.mp4 (~1 h; own Terminal)

Output (in data/alfheim/<match>_H<half>/): panorama.mp4 (same fps and frame count as the cameras), panorama.json
(the geometry: use `Panorama.load(folder)` to place pitch points in panorama pixels and back), panorama_check.jpg.
Settings: config.yaml -> studio.stadium.panorama.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.common import ROOT, load_config
from studio.alfheim import CREDIT, background, dense, folder_for, hint_pixels, line_distance
from studio.issia import pitch_lines
from studio.video import VideoWriter

CAMS = (1, 2, 3)
SRC_W, SRC_H = 1280, 960


def settings() -> dict:
    return load_config()["studio"]["stadium"]["panorama"]


def load_homographies(folder: Path) -> dict[int, np.ndarray]:
    cal = json.loads((folder / "calibration.json").read_text())
    return {c: np.array(cal[f"cam{c}"]["pitch_to_image"], np.float64) for c in CAMS}


def read_frames(folder: Path, start: int, count: int) -> Iterator[list[np.ndarray]]:
    """The 3 cameras together, one frame at a time (never the whole video in memory)."""
    caps = [cv2.VideoCapture(str(folder / f"cam{c}.mp4")) for c in CAMS]
    try:
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(count):
            frames = []
            for cap in caps:
                ok, f = cap.read()
                if not ok:
                    return
                frames.append(f)
            yield frames
    finally:
        for cap in caps:
            cap.release()


# ---------------------------------------------------------------------------------------------------------------
# 1. The rig: one shared position, a focal length and a direction per camera
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Rig:
    centre: np.ndarray                 # (x, y, z) metres, z points DOWN (so z < 0 = above the grass)
    focal: dict[int, float]            # pixels
    rotation: dict[int, np.ndarray]    # 3x3, world direction -> camera direction (x right, y down, z forward)
    error_px: float                    # median distance between this model and the calibrations

    def camera_matrix(self, cam: int) -> np.ndarray:
        """K @ R: a world direction -> that camera's pixel (homogeneous)."""
        f = self.focal[cam]
        K = np.array([[f, 0, SRC_W / 2], [0, f, SRC_H / 2], [0, 0, 1.0]])
        return K @ self.rotation[cam]


def _pose_guess(H: np.ndarray, f: float) -> np.ndarray:
    """A first guess of a camera's direction from its homography (H = K [r1 r2 t]), as a rotation vector."""
    K = np.array([[f, 0, SRC_W / 2], [0, f, SRC_H / 2], [0, 0, 1.0]])
    A = np.linalg.inv(K) @ H
    A = A / ((np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2)
    if A[2, 2] < 0:                                     # the pitch must be in FRONT of the camera
        A = -A
    R = np.column_stack([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])])
    U, _, Vt = np.linalg.svd(R)                         # nearest true rotation
    return cv2.Rodrigues(U @ Vt)[0].ravel()


def fit_rig(H: dict[int, np.ndarray]) -> Rig:
    """Least squares: one shared position + per camera focal length and direction, so that pitch points land
    where each camera's calibration puts them. (The 3 cameras stand together, so sharing a position is real.)"""
    gx, gy = np.meshgrid(np.linspace(-55, 55, 45), np.linspace(-36, 36, 30))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    targets = []
    for c in CAMS:
        p = cv2.perspectiveTransform(grid[None], H[c])[0]
        w = (H[c] @ np.column_stack([grid, np.ones(len(grid))]).T)[2]
        ok = (p[:, 0] > 0) & (p[:, 0] < SRC_W) & (p[:, 1] > 0) & (p[:, 1] < SRC_H) & (w > 0)
        targets.append((np.column_stack([grid[ok], np.zeros(ok.sum())]), p[ok]))

    def residuals(x: np.ndarray) -> np.ndarray:
        out = []
        for i, (P, p) in enumerate(targets):
            f, R = x[3 + i], cv2.Rodrigues(x[6 + 3 * i: 9 + 3 * i])[0]
            q = (R @ (P - x[:3]).T).T
            out += [f * q[:, 0] / q[:, 2] + SRC_W / 2 - p[:, 0], f * q[:, 1] / q[:, 2] + SRC_H / 2 - p[:, 1]]
        return np.concatenate(out)

    f0 = 800.0                                          # the fit lands on the same answer from 500 .. 1100
    x0 = np.concatenate([[0, 40, -12], [f0] * 3, *[_pose_guess(H[c], f0) for c in CAMS]])
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=3)   # soft_l1: a few far-off points don't rule
    x = sol.x
    return Rig(centre=x[:3], focal={c: float(x[3 + i]) for i, c in enumerate(CAMS)},
               rotation={c: cv2.Rodrigues(x[6 + 3 * i: 9 + 3 * i])[0] for i, c in enumerate(CAMS)},
               error_px=float(np.median(np.abs(residuals(x)))))


# ---------------------------------------------------------------------------------------------------------------
# 2. Lining the cameras up from the pictures (cam2 pixel -> cam1 / cam3 pixel)
# ---------------------------------------------------------------------------------------------------------------
def match_cameras(folder: Path, H: dict[int, np.ndarray], moments: int, total: int) -> dict:
    """SIFT feature matches between cam2 and each neighbour at `moments` times spread over the half (players,
    lines, ad boards and stands all help), then one robust (RANSAC) homography per neighbour.
    The calibrations give the first guess: a match more than 60 px away from it is a wrong match."""
    sift = cv2.SIFT_create(3000)
    guess = {c: H[c] @ np.linalg.inv(H[2]) for c in (1, 3)}
    pairs: dict[int, tuple[list, list]] = {1: ([], []), 3: ([], [])}
    for frame in np.linspace(total * 0.01, total * 0.99, moments).astype(int):
        frames = next(read_frames(folder, int(frame), 1))
        feats = {c: sift.detectAndCompute(cv2.cvtColor(frames[c - 1], cv2.COLOR_BGR2GRAY), None) for c in CAMS}
        for c in (1, 3):
            (k2, d2), (kc, dc) = feats[2], feats[c]
            if d2 is None or dc is None:
                continue
            knn = cv2.BFMatcher().knnMatch(d2, dc, k=2)
            good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
            if not good:
                continue
            a = np.float32([k2[m.queryIdx].pt for m in good])
            b = np.float32([kc[m.trainIdx].pt for m in good])
            near = np.linalg.norm(cv2.perspectiveTransform(a[None], guess[c])[0] - b, axis=1) < 60
            pairs[c][0].append(a[near])
            pairs[c][1].append(b[near])
    out = {}
    for c in (1, 3):
        a, b = np.vstack(pairs[c][0]), np.vstack(pairs[c][1])
        G, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
        inliers = inliers.ravel() > 0
        err = np.linalg.norm(cv2.perspectiveTransform(a[inliers][None], G)[0] - b[inliers], axis=1)
        out[str(c)] = {"cam2_to_cam": G.tolist(), "matches": int(len(a)), "inliers": int(inliers.sum()),
                  "median_error_px": round(float(np.median(err)), 2)}
    return out


def refine_calibrations(folder: Path, H: dict[int, np.ndarray], pano: "Panorama", weight: float) -> tuple[dict, dict]:
    """All 3 calibrations fitted again together (same method as studio.alfheim.fine_tune, plus one extra rule):
      - each camera's pitch lines are pulled onto its white lines (median picture of the half = no players),
        with its landmarks and line hints from config.yaml (studio.alfheim) as anchors;
      - in the seam bands, cam1 and cam3 must put pitch points where cam2 does, carried over by the measured
        picture alignment (`weight` = how strongly, per pixel of disagreement).
    Returns (refined calibrations, numbers before/after)."""
    alf = load_config()["studio"]["alfheim"]
    G = {2: np.eye(3), **{c: np.array(pano.data["camera_match"][str(c)]["cam2_to_cam"]) for c in (1, 3)}}
    dist = {c: line_distance(background(folder / f"cam{c}.mp4")) for c in CAMS}
    model = np.vstack([dense(line) for line in pitch_lines()])
    lines, marks, hints = {}, {}, {}
    for c in CAMS:
        q = cv2.perspectiveTransform(model[None], H[c])[0]
        lines[c] = model[(q[:, 0] > 2) & (q[:, 0] < SRC_W - 3) & (q[:, 1] > 2) & (q[:, 1] < SRC_H - 3)]
        marks[c] = np.asarray(alf["calibration"][f"cam{c}"], np.float64)
        hints[c] = [(0 if h[0] == "x" else 1, float(h[1]), hint_pixels(dist[c], h))
                    for h in alf["line_hints"].get(f"cam{c}", [])]
    # pitch points in the seam bands (a grid every metre; the band is taken a bit wider than the blend)
    gx, gy = np.meshgrid(np.linspace(-54, 54, 109), np.linspace(-35, 35, 71))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    rays = pano._pitch_via(2, grid)
    edges = {c: pano.cam_pixels(c, rays)[1] for c in CAMS}
    wide = Panorama({**pano.data, "feather_px": 2 * pano.feather}).blend_weights(edges)
    shared = {c: grid[(wide[c] > 0) & (wide[2] > 0) & (edges[c] > 10) & (edges[2] > 10)] for c in (1, 3)}

    def line_gap(Hc: np.ndarray, c: int) -> np.ndarray:
        q = cv2.perspectiveTransform(lines[c][None], Hc)[0]
        x = np.clip(q[:, 0], 0, SRC_W - 1).astype(np.float32)
        y = np.clip(q[:, 1], 0, SRC_H - 1).astype(np.float32)
        return cv2.remap(dist[c], x[None], y[None], cv2.INTER_LINEAR)[0]

    def hint_gap(Hc: np.ndarray, c: int) -> np.ndarray:
        out = []
        for axis, value, px in hints[c]:
            on_pitch = cv2.perspectiveTransform(px[None], np.linalg.inv(Hc))[0]
            on_pitch[:, axis] = value                   # the nearest point ON the hinted line
            out.append(np.hypot(*(cv2.perspectiveTransform(on_pitch[None], Hc)[0] - px).T))
        return np.concatenate(out) if out else np.zeros(0)

    def marks_gap(Hc: np.ndarray, c: int) -> np.ndarray:
        return (cv2.perspectiveTransform(marks[c][None, :, :2], Hc)[0] - marks[c][:, 2:]).ravel() * 0.3

    def disagree(Hs: dict, c: int) -> np.ndarray:
        """cam c's own calibration vs cam2's carried over by the picture alignment, in cam c pixels."""
        return (cv2.perspectiveTransform(shared[c][None], Hs[c])[0]
                - cv2.perspectiveTransform(shared[c][None], G[c] @ Hs[2])[0]).ravel()

    def unpack(x: np.ndarray) -> dict:
        return {c: np.append(x[8 * i:8 * i + 8], 1).reshape(3, 3) for i, c in enumerate(CAMS)}

    def numbers(Hs: dict) -> dict:
        out = {}
        for c in CAMS:
            gap = line_gap(Hs[c], c)
            out[f"cam{c}_lines_median_px"] = round(float(np.median(gap)), 2)
            out[f"cam{c}_lines_p90_px"] = round(float(np.percentile(gap, 90)), 1)
        for c in (1, 3):
            out[f"cam{c}_vs_cam2_seam_median_px"] = round(float(np.median(np.hypot(*disagree(Hs, c).reshape(-1, 2).T))), 1)
        return out

    x = np.concatenate([(H[c] / H[c][2, 2]).flatten()[:8] for c in CAMS])
    scale = np.abs(x) + 1e-6
    before = numbers(unpack(x))
    for clip, f_scale in ((30, 3), (10, 1.5)):        # coarse to fine, like fine_tune
        def residual(x: np.ndarray) -> np.ndarray:
            Hs = unpack(x)
            own = [np.concatenate([np.minimum(line_gap(Hs[c], c), clip), hint_gap(Hs[c], c), marks_gap(Hs[c], c)])
                   for c in CAMS]
            return np.concatenate(own + [weight * disagree(Hs, c) for c in (1, 3)])
        x = least_squares(residual, x, loss="soft_l1", f_scale=f_scale, x_scale=scale).x
    refined = unpack(x)
    return refined, {"before": before, "after": numbers(refined)}


# ---------------------------------------------------------------------------------------------------------------
# 3. The panorama geometry (saved as panorama.json; everything else in MVP 4 uses this class)
# ---------------------------------------------------------------------------------------------------------------
class Panorama:
    """Cylinder panorama around the rig. Converts between pitch metres, 3D points, rays and panorama pixels."""

    def __init__(self, data: dict):
        self.data = data
        self.width, self.height = data["size"]
        self.focal = data["focal_px"]                   # panorama pixels per radian
        self.u0, self.v0 = data["origin_px"]            # the pixel of the reference direction (towards centre spot)
        self.centre = np.array(data["centre_m"], np.float64)     # z down
        self.to_pano = np.array(data["world_to_pano"])  # rotation: world direction -> panorama frame
        self.feather = data["feather_px"]
        self.ray_to_cam = {int(c): np.array(m) for c, m in data["ray_to_cam"].items()}     # ray -> camera pixel
        self.pitch_to_cam = {int(c): np.array(m) for c, m in data["pitch_to_cam"].items()}  # the calibrations

    @classmethod
    def load(cls, folder: Path) -> "Panorama":
        return cls(json.loads((folder / "panorama.json").read_text()))

    # rays (world directions, z down) <-> panorama pixels
    def rays_to_pano(self, d: np.ndarray) -> np.ndarray:
        p = (self.to_pano @ np.asarray(d, np.float64).reshape(-1, 3).T).T      # x right, y down, z forward
        angle = np.arctan2(p[:, 0], p[:, 2])
        rise = np.arctan2(p[:, 1], np.hypot(p[:, 0], p[:, 2]))           # angle below the horizon
        return np.column_stack([self.u0 + self.focal * angle, self.v0 + self.focal * rise])

    def pano_to_rays(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        angle = (uv[:, 0] - self.u0) / self.focal
        rise = (uv[:, 1] - self.v0) / self.focal
        p = np.column_stack([np.sin(angle) * np.cos(rise), np.sin(rise), np.cos(angle) * np.cos(rise)])
        return (self.to_pano.T @ p.T).T

    # which camera sees a ray, and where
    def cam_pixels(self, cam: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rays -> (pixels in that camera, 'edge' = how far inside its picture: negative outside, -inf behind)."""
        q = (self.ray_to_cam[cam] @ d.T).T
        front = q[:, 2] > 1e-6
        z = np.where(front, q[:, 2], 1.0)
        x, y = q[:, 0] / z, q[:, 1] / z
        edge = np.minimum.reduce([x, SRC_W - 1 - x, y, SRC_H - 1 - y])
        return np.column_stack([x, y]), np.where(front, edge, -np.inf)

    def blend_weights(self, edges: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Per camera 0..1 (summing to 1): the camera that sees a spot most centrally wins; over a band of
        2 x feather pixels around the seam the two neighbours fade into each other."""
        weights = {}
        with np.errstate(invalid="ignore"):             # -inf minus -inf (nobody sees it): weight 0
            for c in CAMS:
                best_other = np.maximum.reduce([edges[o] for o in CAMS if o != c])
                w = np.clip((edges[c] - best_other) / (2 * self.feather) + 0.5, 0, 1)
                weights[c] = np.where(np.isfinite(edges[c]) & np.isfinite(w), w, 0.0)
        total = np.maximum(sum(weights.values()), 1e-9)
        return {c: w / total for c, w in weights.items()}

    # pitch (ground) points <-> panorama pixels: each camera's own calibration, blended over the seams
    def _pitch_via(self, cam: int, xy: np.ndarray) -> np.ndarray:
        q = self.pitch_to_cam[cam] @ np.column_stack([xy, np.ones(len(xy))]).T
        d = np.linalg.solve(self.ray_to_cam[cam], q).T
        return d * np.sign(d[:, 2:3] + 1e-12)          # a ground point is BELOW the rig: z (down) > 0

    def pitch_to_pano(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        rays = {c: self._pitch_via(c, xy) for c in CAMS}
        weights = self.blend_weights({c: self.cam_pixels(c, rays[2])[1] for c in CAMS})
        d = sum(weights[c][:, None] * rays[c] / np.linalg.norm(rays[c], axis=1, keepdims=True) for c in CAMS)
        return self.rays_to_pano(d)

    def pano_to_pitch(self, uv: np.ndarray) -> np.ndarray:
        """Panorama pixels -> pitch metres; NaN for pixels at or above the horizon (they never hit the grass)."""
        d = self.pano_to_rays(uv)
        return self.rays_to_pitch(d)

    def rays_to_pitch(self, d: np.ndarray) -> np.ndarray:
        weights = self.blend_weights({c: self.cam_pixels(c, d)[1] for c in CAMS})
        out = np.zeros((len(d), 2))
        for c in CAMS:
            P = np.linalg.solve(self.pitch_to_cam[c], self.ray_to_cam[c] @ d.T).T
            with np.errstate(invalid="ignore", divide="ignore"):
                out += weights[c][:, None] * np.nan_to_num(P[:, :2] / P[:, 2:3])
        out[d[:, 2] <= 1e-6] = np.nan
        return out

    def world_to_pano(self, xyh: np.ndarray) -> np.ndarray:
        """3D points (x, y, height above the grass in metres) -> panorama pixels (straight rays from the rig)."""
        P = np.asarray(xyh, np.float64).reshape(-1, 3) * np.array([1, 1, -1.0])
        return self.rays_to_pano(P - self.centre)

    def cam_to_pano(self, cam: int, pts: np.ndarray) -> np.ndarray:
        """Pixels of one real camera -> panorama pixels (any point: grass, players' heads, stands)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        d = np.linalg.solve(self.ray_to_cam[cam], np.column_stack([pts, np.ones(len(pts))]).T).T
        d *= np.sign((self.ray_to_cam[cam] @ d.T)[2])[:, None]   # the ray the camera looks ALONG, not behind
        return self.rays_to_pano(d)


def build_geometry(folder: Path, cfg: dict, total_frames: int) -> Panorama:
    """Fit the rig, line the cameras up, choose the cylinder (direction, scale, crop): the panorama geometry."""
    H = load_homographies(folder)
    rig = fit_rig(H)
    matches = match_cameras(folder, H, cfg["match_moments"], total_frames)
    focal = cfg["focal_scale"] * float(np.mean(list(rig.focal.values())))
    # Cylinder axis = straight up; the reference direction = horizontally towards the centre spot.
    forward = np.array([-rig.centre[0], -rig.centre[1], 0.0])
    forward /= np.linalg.norm(forward)
    down = np.array([0, 0, 1.0])
    to_pano = np.vstack([np.cross(down, forward), down, forward])
    A2 = rig.camera_matrix(2)                           # cam2 is the reference: world ray -> cam2 pixel
    ray_to_cam = {"2": A2.tolist()}
    for c in (1, 3):
        M = np.array(matches[str(c)]["cam2_to_cam"]) @ A2
        look = rig.rotation[c].T @ np.array([0, 0, 1.0])            # where camera c points
        ray_to_cam[str(c)] = (M * np.sign((M @ look)[2])).tolist()  # so "in front" = positive depth
    data = {"centre_m": rig.centre.tolist(), "height_m": float(-rig.centre[2]), "rig_focal_px": rig.focal,
            "rig_error_px": rig.error_px, "focal_px": focal, "origin_px": [0.0, 0.0], "size": [1, 1],
            "feather_px": cfg["feather_px"], "world_to_pano": to_pano.tolist(), "ray_to_cam": ray_to_cam,
            "pitch_to_cam": {str(c): H[c].tolist() for c in CAMS}, "camera_match": matches,
            "rig_rotation": {str(c): rig.rotation[c].tolist() for c in CAMS}, "credit": CREDIT}
    refined, numbers = refine_calibrations(folder, H, Panorama(data), cfg["seam_agreement"])
    data["pitch_to_cam"] = {str(c): refined[c].tolist() for c in CAMS}
    data["calibration_refit"] = numbers
    pano = Panorama(data)
    # Crop: the pitch plus a margin of grass, and some stand above the far touchline.
    m, s = cfg["margin_m"], cfg["stand_m"]
    edge = np.concatenate([np.column_stack([np.linspace(-52.5 - m, 52.5 + m, 400), np.full(400, y)])
                           for y in (-34 - m, 34 + m)]
                          + [np.column_stack([np.full(200, x), np.linspace(-34 - m, 34 + m, 200)])
                             for x in (-52.5 - m, 52.5 + m)])
    stand = np.column_stack([np.linspace(-52.5 - m, 52.5 + m, 200), np.full(200, -34 - m), np.full(200, s)])
    pts = np.vstack([pano.pitch_to_pano(edge), pano.world_to_pano(stand)])
    u_min, v_min = np.floor(pts.min(axis=0))
    u_max, v_max = np.ceil(pts.max(axis=0))
    w, h = int(u_max - u_min), int(v_max - v_min)
    data["origin_px"] = [float(-u_min), float(-v_min)]
    data["size"] = [w + w % 2, h + h % 2]               # H.264 wants even sizes
    return Panorama(data)


# ---------------------------------------------------------------------------------------------------------------
# 4. Lookup tables + blending weights (once), then stitching each frame
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class CamWarp:
    cam: int
    roi: tuple[int, int, int, int]     # x, y, w, h in the panorama: the part this camera fills
    map1: np.ndarray                   # cv2.remap tables (fast fixed-point form) for the roi
    map2: np.ndarray


SEAM_COLS = 1024                       # seam pixels are packed into a compact picture this many pixels wide


def _fixed_maps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float source coordinates -> OpenCV's fast fixed-point remap tables."""
    return cv2.convertMaps(np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y, np.float32), cv2.CV_16SC2)


class Stitcher:
    """The lookup tables, built once. Per frame the 3 camera pictures are stacked into one tall picture
    (cam1 on top, cam3 at the bottom) so that ONE remap fills every pixel a single camera owns; the seam pixels
    (packed into a small picture) get two remaps, one per neighbour, blended with their weights."""

    def __init__(self, pano: Panorama):
        self.pano = pano
        W, H = pano.width, pano.height
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        rays = pano.pano_to_rays(np.column_stack([uu.ravel(), vv.ravel()]))
        del uu, vv
        pixels, edges = {}, {}
        for c in CAMS:
            pixels[c], edges[c] = pano.cam_pixels(c, rays)
            pixels[c] = pixels[c].astype(np.float32)
        del rays
        weights = pano.blend_weights(edges)
        # For the picture a camera may only paint where it really has pixels: drop the rest, share out again.
        for c in CAMS:
            weights[c] = np.where(edges[c] >= 1, weights[c], 0.0)      # >= 1 px inside: no black border mixed in
        total = sum(weights.values())
        self.coverage = (total > 0).reshape(H, W)
        weights = {c: (w / np.maximum(total, 1e-9)).astype(np.float32) for c, w in weights.items()}
        del edges, total
        self.weights = {c: w.reshape(H, W) for c, w in weights.items()}

        # 1. pixels one camera owns -> one remap into the stacked picture (row offset = camera's slot x 960)
        wmat = np.stack([weights[c] for c in CAMS])                  # (3, H*W)
        best = wmat.argmax(axis=0)
        solo = wmat.max(axis=0) >= 1.0
        sx, sy = np.full(H * W, -1, np.float32), np.full(H * W, -1, np.float32)
        for k, c in enumerate(CAMS):
            own = solo & (best == k)
            sx[own] = pixels[c][own, 0]
            sy[own] = pixels[c][own, 1] + k * SRC_H
        self.map1, self.map2 = _fixed_maps(sx.reshape(H, W), sy.reshape(H, W))
        del sx, sy

        # 2. seam pixels: the two cameras with the largest weights ("a" and "b"), packed into rows of SEAM_COLS
        self.seam_index = np.flatnonzero(self.coverage.ravel() & ~solo)
        n = len(self.seam_index)
        rows = -(-n // SEAM_COLS)
        order = np.argsort(-wmat[:, self.seam_index], axis=0)        # best camera first
        self.seam_maps, seam_w = [], []
        for slot in (0, 1):
            k = order[slot]
            px = np.stack([pixels[c][self.seam_index] for c in CAMS])[k, np.arange(n)]
            w = wmat[k, self.seam_index]
            x, y = np.full(rows * SEAM_COLS, -1, np.float32), np.full(rows * SEAM_COLS, -1, np.float32)
            x[:n], y[:n] = px[:, 0], px[:, 1] + k * SRC_H
            self.seam_maps.append(_fixed_maps(x.reshape(rows, SEAM_COLS), y.reshape(rows, SEAM_COLS)))
            pad = np.zeros(rows * SEAM_COLS, np.float32)
            pad[:n] = w
            seam_w.append(pad.reshape(rows, SEAM_COLS))
        self.seam_weights = seam_w                                   # (a rare third camera is left out)

        # 3. per camera, for the checks: that camera alone laid on the panorama (its roi)
        self.warps: list[CamWarp] = []
        for c in CAMS:
            ys, xs = np.nonzero(self.weights[c] > 0)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            px = pixels[c].reshape(H, W, 2)[y0:y1, x0:x1]
            m1, m2 = _fixed_maps(px[..., 0], px[..., 1])
            self.warps.append(CamWarp(c, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), m1, m2))

    def warp(self, cam_index: int, frame: np.ndarray) -> np.ndarray:
        """One camera's frame laid onto the panorama (its roi only)."""
        wp = self.warps[cam_index]
        return cv2.remap(frame, wp.map1, wp.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def stitch(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames = [cam1, cam2, cam3] (BGR, 1280x960) -> one panorama frame (BGR)."""
        stacked = np.vstack(frames)
        out = cv2.remap(stacked, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a, b = (cv2.remap(stacked, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                for m1, m2 in self.seam_maps)
        seam = cv2.blendLinear(a, b, self.seam_weights[0], self.seam_weights[1])
        out.reshape(-1, 3)[self.seam_index] = seam.reshape(-1, 3)[:len(self.seam_index)]
        return out


# ---------------------------------------------------------------------------------------------------------------
# 5. Checks: pitch lines across the seams, and players seen by two cameras in the seam bands
# ---------------------------------------------------------------------------------------------------------------
def dense_lines(step_m: float = 0.5) -> list[np.ndarray]:
    """The pitch markings as polylines with a point every ~0.5 m (close to the rig, 5 m is a big angle)."""
    out = []
    for line in pitch_lines():
        pieces = [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step_m) + 1)) for a, b in zip(line, line[1:])]
        out.append(np.vstack(pieces))
    return out


def draw_lines(image: np.ndarray, pano: Panorama, colour=(0, 0, 255), thickness: int = 2) -> None:
    for line in dense_lines():
        q = np.round(pano.pitch_to_pano(line)).astype(np.int32)
        cv2.polylines(image, [q.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)


def seam_band(st: Stitcher, a: int, b: int) -> np.ndarray:
    """Bool mask (panorama size): pixels blended from cameras a and b."""
    return (st.weights[a] > 0) & (st.weights[b] > 0)


def line_jumps(st: Stitcher, frames: list[np.ndarray], a: int, b: int, patch: int = 48) -> list[float]:
    """How far apart camera a and camera b put the white pitch lines inside their seam band (pixels).
    Each camera is laid on the panorama on its own; small patches around pitch-line points are compared with
    phase correlation (it finds the shift between two pictures)."""
    W, H = st.pano.width, st.pano.height
    full = []
    for c in (a, b):
        k = CAMS.index(c)
        x, y, w, h = st.warps[k].roi
        img = np.zeros((H, W), np.float32)
        img[y:y + h, x:x + w] = cv2.cvtColor(st.warp(k, frames[k]), cv2.COLOR_BGR2GRAY)
        full.append(img)
    band = seam_band(st, a, b)
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    jumps, r = [], patch // 2
    for line in dense_lines(1.0):
        for u, v in np.round(st.pano.pitch_to_pano(line)).astype(int):
            if not (r <= u < W - r and r <= v < H - r) or not band[v, u]:
                continue
            pa, pb = (img[v - r:v + r, u - r:u + r] for img in full)
            if pa.min() == 0 or pb.min() == 0:          # patch reaches outside one camera's picture
                continue
            (dx, dy), response = cv2.phaseCorrelate(pa, pb, win)
            if response > 0.15:                         # a clear match (a line in the patch, not just grass)
                jumps.append(math.hypot(dx, dy))
    return jumps


def player_matches(folder: Path, st: Stitcher, a: int, b: int, samples: int = 300) -> dict:
    """Players (feet) both cameras see inside their seam band, laid on the panorama: counts + distance apart.
    (Detector boxes wobble by a few pixels per camera, so this is a rough check, not a precise one.)"""
    dets = {c: json.loads((folder / "detections" / f"cam{c}.json").read_text())["frames"] for c in (a, b)}
    keys = sorted(set(dets[a]) & set(dets[b]), key=int)
    keys = keys[:: max(1, len(keys) // samples)]
    band = seam_band(st, a, b)
    counts, dists = {a: 0, b: 0}, []
    for k in keys:
        feet = {}
        for c in (a, b):
            pts = [((p["box"][0] + p["box"][2]) / 2, p["box"][3]) for p in dets[c][k].get("players", [])]
            uv = st.pano.cam_to_pano(c, np.array(pts)) if pts else np.zeros((0, 2))
            ok = [(0 <= u < st.pano.width and 0 <= v < st.pano.height and band[int(v), int(u)]) for u, v in uv]
            feet[c] = uv[np.array(ok, bool)] if len(uv) else uv
            counts[c] += len(feet[c])
        for p in feet[a]:                               # nearest partner from the other camera
            if len(feet[b]):
                d = np.linalg.norm(feet[b] - p, axis=1).min()
                if d < 60:
                    dists.append(float(d))
    return {"frames": len(keys), f"players_cam{a}": counts[a], f"players_cam{b}": counts[b],
            "matched": len(dists), "median_px": round(float(np.median(dists)), 1) if dists else None}


def calibration_gap(st: Stitcher, a: int, b: int) -> dict:
    """Where camera a's and camera b's calibrations put the same panorama pixel on the pitch, inside their seam
    band (metres; the far half and the near half separately: right under the rig 1 m is 100+ pixels)."""
    ys, xs = np.nonzero(seam_band(st, a, b))
    d = st.pano.pano_to_rays(np.column_stack([xs, ys])[::20].astype(np.float64))
    P = {}
    for c in (a, b):
        q = np.linalg.solve(st.pano.pitch_to_cam[c], st.pano.ray_to_cam[c] @ d.T).T
        P[c] = q[:, :2] / q[:, 2:3]
    ok = (d[:, 2] > 1e-6) & (np.abs(P[a][:, 0]) < 55) & (np.abs(P[a][:, 1]) < 36)
    gap, y = np.linalg.norm(P[a] - P[b], axis=1)[ok], P[a][ok, 1]
    out = {}
    for name, part in (("far_half", y < 0), ("near_half", y >= 0)):
        if part.any():
            out[f"{name}_median_m"] = round(float(np.median(gap[part])), 2)
            out[f"{name}_p90_m"] = round(float(np.percentile(gap[part], 90)), 2)
    return out


def check(folder: Path, pano: Panorama, st: Stitcher, frame: int = 250) -> dict:
    """panorama_check.jpg (pitch lines drawn from the calibration, seam bands in yellow) + the seam numbers."""
    frames = next(read_frames(folder, frame, 1))
    image = st.stitch(frames)
    report = {"size": [pano.width, pano.height], "rig_height_m": round(pano.data["height_m"], 2),
              "rig_xy_m": [round(v, 2) for v in pano.data["centre_m"][:2]],
              "rig_fit_error_px": round(pano.data["rig_error_px"], 2),
              "camera_match": {f"cam2->cam{c}": {k: v for k, v in m.items() if k != "cam2_to_cam"}
                               for c, m in pano.data["camera_match"].items()},
              "calibration_refit": pano.data["calibration_refit"], "seams": {}}
    for a, b in ((1, 2), (2, 3)):
        jumps = line_jumps(st, frames, a, b)
        report["seams"][f"cam{a}|cam{b}"] = {
            "line_points": len(jumps), "line_jump_median_px": round(float(np.median(jumps)), 2) if jumps else None,
            "line_jump_max_px": round(float(max(jumps)), 2) if jumps else None,
            "players": player_matches(folder, st, a, b), "calibration_gap": calibration_gap(st, a, b)}
        edge = cv2.morphologyEx(seam_band(st, a, b).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        image[edge > 0] = (0, 220, 255)
    draw_lines(image, pano)
    cv2.putText(image, CREDIT, (20, pano.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(folder / "panorama_check.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return report


# ---------------------------------------------------------------------------------------------------------------
# 6. The video
# ---------------------------------------------------------------------------------------------------------------
def render(folder: Path, st: Stitcher, out: Path, start: int, count: int, fps: float, cfg: dict) -> int:
    """Stitch `count` frames from `start` into `out`. A small reader thread decodes the next frames while this one
    stitches (OpenCV lets other threads run while it works); ffmpeg encodes in its own process."""
    frames_q: queue.Queue = queue.Queue(maxsize=4)

    def reader() -> None:
        for frames in read_frames(folder, start, count):
            frames_q.put(frames)
        frames_q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    writer = VideoWriter(out, st.pano.width, st.pano.height, fps, crf=cfg["crf"], threads=cfg["threads"])
    done, t0 = 0, time.time()
    try:
        while (frames := frames_q.get()) is not None:
            writer.write(st.stitch(frames))
            done += 1
            if done % 300 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{count} frames  {rate:.1f} fps  ~{(count - done) / rate / 60:.0f} min left", flush=True)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return done


def parse_time(text: str) -> float:
    """"23:40" or "1420" -> seconds."""
    parts = [float(p) for p in text.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


def main() -> None:
    alf = load_config()["studio"]["alfheim"]
    ap = argparse.ArgumentParser(description="Stitch the 3 Alfheim cameras into one panorama (MVP 4 Step 1).")
    ap.add_argument("--match", default=alf["match"])
    ap.add_argument("--half", type=int, default=alf["half"])
    ap.add_argument("--check", action="store_true", help="only panorama.json + panorama_check.jpg + seam numbers")
    ap.add_argument("--minutes", type=float, help="quick test: only this many minutes -> panorama_<N>min.mp4")
    ap.add_argument("--start", default="0:00", help="with --minutes: where to begin (MM:SS)")
    ap.add_argument("--refit", action="store_true",
                    help="build panorama.json again (then panorama.mp4 must be made again too: they belong together)")
    args = ap.parse_args()

    cfg = settings()
    folder = folder_for(args.match, args.half)
    meta = json.loads((folder / "meta.json").read_text())
    fps, total = meta["fps"], meta["frames"]
    geometry = folder / "panorama.json"
    if geometry.exists() and not args.refit:          # the video's pixels are tied to this geometry: keep it
        pano = Panorama.load(folder)
    else:
        pano = build_geometry(folder, cfg, total)
        geometry.write_text(json.dumps(pano.data, indent=1))
        if (folder / "panorama.mp4").exists():
            print("  NOTE: panorama.json changed -> make panorama.mp4 again (run without --check)")
    print(f"Rig: {pano.data['height_m']:.1f} m high at x={pano.centre[0]:.1f}, y={pano.centre[1]:.1f} m "
          f"(fit error {pano.data['rig_error_px']:.1f} px). Panorama {pano.width}x{pano.height}.")
    st = Stitcher(pano)
    report = check(folder, pano, st)
    print(json.dumps(report, indent=1))
    print(f"  check picture: {(folder / 'panorama_check.jpg').relative_to(ROOT)}")
    if args.check:
        return
    if args.minutes:
        start = round(parse_time(args.start) * fps)
        count = min(round(args.minutes * 60 * fps), total - start)
        out = folder / f"panorama_{args.minutes:g}min.mp4"
    else:
        start, count, out = 0, total, folder / "panorama.mp4"
    t0 = time.time()
    done = render(folder, st, out, start, count, fps, cfg)
    print(f"Wrote {out.relative_to(ROOT)}: {done} frames in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
