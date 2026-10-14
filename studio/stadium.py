"""stadium.py — the virtual stadium's camera set (MVP 4 Step 2): camera models, robotic heads, cards and real feeds.

Why: MVP 3 had 3 fixed cameras. A TV crew has robotic / manned cameras that pan and zoom. Here every camera of the
T2 set (docs/MVP4_STADIUM_PLAN.md section 5) is a small MATH MODEL, like a real camera:
  - where it stands (metres), where it looks (a point on the pitch) and how wide the shot is (`field_width_m`:
    metres across the picture at the point it looks at); together that is a pinhole camera with a level horizon;
  - robotic ones (kind "ptz") get a `PTZHead`: the director says `aim(x, y)` and `zoom(width)`, and the head moves
    there like a motor would: it ignores tiny moves (dead zone), moves smoothly, and never faster than its limits.

The director never looks at pixels: for each camera we build its CARD (studio.cards.CamCard: is the ball in view and
where, players in view, shot type...) by PROJECTING the pitch map's ball and players through that camera's model.
So the director can also judge the 3D behind-goal cameras (Step 4), which have no real pixels.

Pictures:
  panorama cameras (MAIN, CLOSE, WIDE)  cut out of panorama.mp4 (studio.panorama). MAIN and CLOSE use a true
                                        perspective view: every output pixel's ray -> its panorama pixel
  source cameras (BOX L, BOX R)         the real Alfheim cam1 / cam3 pictures, as they are (16:9 crop)
  3d cameras (GOAL L, GOAL R)           drawn in the browser (Step 4); here only their model and card

Coordinates: pitch metres (origin at the centre spot, x to the right, y towards the near touchline). Heights are
"metres above the grass"; inside the maths z points DOWN (same as studio.panorama).

Usage (needs panorama.mp4, see studio.panorama):
  venv/bin/python -m studio.stadium --segment alfheim-2013-11-03-H1 --start 23:40 --seconds 300 --aim follow
Writes outputs/stadium/<segment>/<run>/cam_main.mp4, cam_close.mp4, cam_wide.mp4, cam_boxl.mp4, cam_boxr.mp4,
cameras.jsonl (every camera's state per frame) and summary.json (the Step 2 checks).
Settings: config.yaml -> studio.stadium.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, load_config
from studio.cards import BallInfo, CamCard, Quality
from studio.panorama import Panorama, parse_time
from studio.pitchmap import MapState
from studio.video import VideoWriter, crop_to, to_camera

DOWN = np.array([0.0, 0.0, 1.0])          # world "down" (z points down)


def settings() -> dict:
    return load_config()["studio"]["stadium"]


# ---------------------------------------------------------------------------------------------------------------
# 1. A camera: its fixed facts, and a pinhole view (where it looks + how wide)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Camera:
    id: str
    name: str
    kind: str                                  # "ptz" (robotic) | "fixed"
    built_from: str                            # "panorama" | "source" | "3d"
    position_m: tuple[float, float, float]     # x, y, height above the grass
    look_at_m: tuple[float, float] = (0.0, 0.0)
    field_width_m: float = 40.0
    min_field_width_m: float = 5.0
    max_field_width_m: float = 120.0
    max_pan_deg_s: float = 0.0                 # motor limits (robotic heads only)
    max_zoom_ratio_s: float = 1.0              # the shot may get at most this many times wider/tighter per second
    dead_zone_deg: float = 0.0
    smoothing_s: float = 0.5                   # how quickly the head closes the gap to its target (time constant)
    source_cam: int | None = None              # built_from "source": which real Alfheim camera
    crop: tuple[float, float, float, float] | None = None   # built_from "source": the 16:9 part of its picture

    @property
    def centre(self) -> np.ndarray:
        """Position in the maths' frame (z down)."""
        x, y, h = self.position_m
        return np.array([x, y, -h], np.float64)


def make_cameras(cfg: dict, pano: Panorama) -> list[Camera]:
    """The camera list from config.yaml (studio.stadium.cameras); position "rig" = the fitted Alfheim platform."""
    rig = (float(pano.centre[0]), float(pano.centre[1]), float(-pano.centre[2]))
    fields = set(Camera.__dataclass_fields__)
    out = []
    for cam_id, c in cfg["cameras"].items():
        position = rig if c["position"] == "rig" else tuple(float(v) for v in c["position"])
        look = c.get("look_at", [0, 0])
        extra = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k in fields}
        out.append(Camera(id=cam_id, position_m=position, look_at_m=(float(look[0]), float(look[1])), **extra))
    return out


@dataclass
class View:
    """Where a camera looks right now. pan: compass angle (0 = straight across to the far touchline, + = to the
    right), tilt: angle below the horizon, hfov: horizontal field of view. All degrees."""
    pan: float
    tilt: float
    hfov: float

    def forward(self) -> np.ndarray:
        p, t = math.radians(self.pan), math.radians(self.tilt)
        return np.array([math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)])


def view_towards(centre: np.ndarray, x: float, y: float, field_width_m: float) -> View:
    """The view of a camera at `centre` looking at pitch point (x, y) with a shot `field_width_m` wide there."""
    d = np.array([x, y, 0.0]) - centre
    dist = float(np.linalg.norm(d))
    pan = math.degrees(math.atan2(d[0], -d[1]))
    tilt = math.degrees(math.asin(d[2] / dist))
    hfov = math.degrees(2 * math.atan(field_width_m / 2 / dist))
    return View(pan, tilt, hfov)


def look_at_of(centre: np.ndarray, view: View) -> tuple[float, float] | None:
    """The pitch point in the middle of the picture (None if the camera looks at or above the horizon)."""
    f = view.forward()
    if f[2] <= 1e-6:
        return None
    s = -centre[2] / f[2]
    return float(centre[0] + s * f[0]), float(centre[1] + s * f[1])


def field_width_of(centre: np.ndarray, view: View) -> float:
    look = look_at_of(centre, view)
    if look is None:
        return float("inf")
    dist = float(np.linalg.norm(np.array([look[0], look[1], 0.0]) - centre))
    return 2 * dist * math.tan(math.radians(view.hfov) / 2)


class Pinhole:
    """A level-horizon pinhole camera: world ray <-> picture pixel (width x height)."""

    def __init__(self, view: View, width: int, height: int):
        f = view.forward()
        right = np.cross(DOWN, f)                     # level horizon: "right" is always horizontal
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        self.R = np.vstack([right, down, f])          # world direction -> camera (x right, y down, z forward)
        self.focal = (width / 2) / math.tan(math.radians(view.hfov) / 2)
        self.K = np.array([[self.focal, 0, width / 2], [0, self.focal, height / 2], [0, 0, 1.0]])
        self.width, self.height = width, height

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World rays (N, 3) -> pixels (N, 2) and whether each is in the picture (in front and inside)."""
        q = (self.K @ self.R @ np.asarray(rays, np.float64).reshape(-1, 3).T).T
        front = q[:, 2] > 1e-9
        z = np.where(front, q[:, 2], 1.0)
        px = np.column_stack([q[:, 0] / z, q[:, 1] / z])
        inside = front & (px[:, 0] >= 0) & (px[:, 0] < self.width) & (px[:, 1] >= 0) & (px[:, 1] < self.height)
        return px, inside

    def rays(self, px: np.ndarray) -> np.ndarray:
        """Picture pixels (N, 2) -> world rays (N, 3)."""
        p = np.column_stack([np.asarray(px, np.float64).reshape(-1, 2), np.ones(len(px))])
        return (self.R.T @ np.linalg.solve(self.K, p.T)).T


# ---------------------------------------------------------------------------------------------------------------
# 2. The robotic head
# ---------------------------------------------------------------------------------------------------------------
class PTZHead:
    """A robotic pan/tilt/zoom head. The director sets targets (aim / zoom); `step(dt)` moves towards them.

    Same behaviour as MVP 3's VirtualOperator, in degrees instead of pixels:
      - dead zone: a target closer than `dead_zone_deg` is ignored (no nervous little moves);
      - smoothing: each step closes part of the gap (time constant `smoothing_s`), so moves start and end softly;
      - motor limits: pan and tilt never faster than `max_pan_deg_s`; zoom never more than `max_zoom_ratio_s`
        times wider/tighter per second.
    Only the current target and the past are used (causal)."""

    def __init__(self, cam: Camera):
        self.cam = cam
        self.centre = cam.centre
        self.view = view_towards(self.centre, *cam.look_at_m, self._clamp(cam.field_width_m))
        self.target_look = cam.look_at_m
        self.target_width = self._clamp(cam.field_width_m)
        self.speed_deg_s = 0.0                     # how fast the head turned in the last step

    def _clamp(self, width: float) -> float:
        return min(max(width, self.cam.min_field_width_m), self.cam.max_field_width_m)

    def aim(self, x_m: float, y_m: float) -> None:
        self.target_look = (float(x_m), float(y_m))

    def zoom(self, field_width_m: float) -> None:
        self.target_width = self._clamp(float(field_width_m))

    def step(self, dt: float) -> View:
        cam = self.cam
        target = view_towards(self.centre, *self.target_look, self.target_width)
        alpha = 1 - math.exp(-dt / max(cam.smoothing_s, 1e-6))
        limit = cam.max_pan_deg_s * dt
        moves = []
        for axis in ("pan", "tilt"):
            gap = getattr(target, axis) - getattr(self.view, axis)
            if abs(gap) <= cam.dead_zone_deg:
                moves.append(0.0)
                continue
            moves.append(alpha * (gap - math.copysign(cam.dead_zone_deg, gap)))
        size = math.hypot(*moves)
        if size > limit:                           # the motor's top speed (for the combined move)
            moves = [m * limit / size for m in moves]
        pan, tilt = self.view.pan + moves[0], self.view.tilt + moves[1]
        # zoom works in ratios ("twice as wide"), not metres
        gap = math.log(target.hfov / self.view.hfov)
        zoom = max(-math.log(cam.max_zoom_ratio_s) * dt, min(math.log(cam.max_zoom_ratio_s) * dt, alpha * gap))
        self.view = View(pan, tilt, self.view.hfov * math.exp(zoom))
        self.speed_deg_s = math.hypot(*moves) / dt if dt > 0 else 0.0
        return self.view

    def state(self) -> dict:
        look = look_at_of(self.centre, self.view)
        return {"look_at": [round(look[0], 2), round(look[1], 2)] if look else None,
                "field_width_m": round(field_width_of(self.centre, self.view), 2),
                "pan": round(self.view.pan, 3), "tilt": round(self.view.tilt, 3), "hfov": round(self.view.hfov, 3),
                "speed_deg_s": round(self.speed_deg_s, 2)}


# ---------------------------------------------------------------------------------------------------------------
# 3. The stadium: all cameras, their projections, cards and pictures
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Shot:
    """What one camera shows at one moment: a pinhole view, or a fixed rectangle of a real picture."""
    cam: Camera
    pinhole: Pinhole | None = None             # panorama PTZ cameras and 3d cameras
    rect: tuple[float, float, float, float] | None = None   # WIDE (panorama pixels) and BOX L/R (source pixels)
    state: dict = field(default_factory=dict)


class Stadium:
    """Holds the camera models and robotic heads. Per frame: `step(dt)` moves the heads, `shots()` gives each
    camera's current view, `cards(...)` what each camera sees, `picture(...)` its real picture."""

    def __init__(self, cfg: dict, pano: Panorama):
        self.cfg = cfg
        self.pano = pano
        self.size = tuple(cfg["output_size"])
        self.cameras = make_cameras(cfg, pano)
        self.heads = {c.id: PTZHead(c) for c in self.cameras if c.kind == "ptz"}
        self.fixed_views = {c.id: view_towards(c.centre, *c.look_at_m, c.field_width_m)
                            for c in self.cameras if c.kind == "fixed" and c.built_from == "3d"}
        self._maps: dict[str, tuple] = {}

    def camera(self, cam_id: str) -> Camera:
        return next(c for c in self.cameras if c.id == cam_id)

    def step(self, dt: float) -> None:
        for head in self.heads.values():
            head.step(dt)

    def shots(self) -> dict[str, Shot]:
        out = {}
        w, h = self.size
        for c in self.cameras:
            if c.id in self.heads:
                head = self.heads[c.id]
                out[c.id] = Shot(c, pinhole=Pinhole(head.view, w, h), state=head.state())
            elif c.built_from == "3d":
                v = self.fixed_views[c.id]
                out[c.id] = Shot(c, pinhole=Pinhole(v, w, h), state={
                    "look_at": list(c.look_at_m), "field_width_m": c.field_width_m,
                    "pan": round(v.pan, 3), "tilt": round(v.tilt, 3), "hfov": round(v.hfov, 3)})
            elif c.built_from == "source":
                out[c.id] = Shot(c, rect=tuple(c.crop), state=self._fixed_state(c))
            else:                                   # WIDE: the whole panorama
                out[c.id] = Shot(c, rect=(0, 0, self.pano.width, self.pano.height), state=self._fixed_state(c))
        return out

    def _fixed_state(self, c: Camera) -> dict:
        look = list(c.look_at_m)
        if c.built_from == "source":                # the pitch point in the middle of its picture
            x, y, w, h = c.crop
            H = self.pano.pitch_to_cam[c.source_cam]
            mid = cv2.perspectiveTransform(np.array([[[x + w / 2, y + h / 2]]], np.float64), np.linalg.inv(H))[0, 0]
            look = [round(float(mid[0]), 2), round(float(mid[1]), 2)]
            # its field of view from its real lens (the rig fit in panorama.json), across the 16:9 crop
            hfov = math.degrees(2 * math.atan(w / 2 / self.pano.data["rig_focal_px"][str(c.source_cam)]))
            dist = math.dist((look[0], look[1], 0.0), c.position_m)
            return {"look_at": look, "hfov": round(hfov, 2),
                    "field_width_m": round(2 * dist * math.tan(math.radians(hfov) / 2), 1)}
        return {"look_at": look, "field_width_m": round(self.shot_width(c), 1)}

    def shot_width(self, c: Camera) -> float:
        """Field width of a fixed real camera: how many metres across its picture is at its centre."""
        if c.built_from == "source":
            x, y, w, h = c.crop
            H = self.pano.pitch_to_cam[c.source_cam]
            pts = np.array([[x, y + h / 2], [x + w, y + h / 2]], np.float64)
            on_pitch = cv2.perspectiveTransform(pts[None], np.linalg.inv(H))[0]
            return float(np.linalg.norm(on_pitch[1] - on_pitch[0]))
        return 105.0 + 2 * load_config()["studio"]["stadium"]["panorama"]["margin_m"]

    # ------------------------------------------------------------------ projections
    def project(self, shot: Shot, points_m: np.ndarray, heights: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Pitch points (N, 2) [+ heights in metres] -> pixels in this camera's 1280x720 picture + in view.

        Real-pixel cameras go through the panorama / the real camera's calibration, so a projected point lands
        exactly where that thing is in the PICTURE; 3d cameras use straight rays from their position."""
        pts = np.asarray(points_m, np.float64).reshape(-1, 2)
        hts = np.zeros(len(pts)) if heights is None else np.asarray(heights, np.float64)
        w, h = self.size
        c = shot.cam
        if len(pts) == 0:
            return np.zeros((0, 2)), np.zeros(0, bool)
        if c.built_from == "3d":
            rays = np.column_stack([pts, -hts]) - c.centre
            return shot.pinhole.project(rays)
        if c.built_from == "source":
            src = cv2.perspectiveTransform(pts[None], self.pano.pitch_to_cam[c.source_cam])[0]
            px = np.array([to_camera(shot.rect, w, h, tuple(p)) for p in src])
            inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            return px, inside
        uv = self._pano_points(pts, hts)
        if shot.pinhole is not None:
            return shot.pinhole.project(self.pano.pano_to_rays(uv))
        px = np.array([to_camera(shot.rect, w, h, tuple(p)) for p in uv])
        inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        return px, inside

    def _pano_points(self, pts: np.ndarray, hts: np.ndarray) -> np.ndarray:
        """Panorama pixels of pitch points; a point above the grass is lifted by the ray difference."""
        uv = self.pano.pitch_to_pano(pts)
        if np.any(hts > 0):
            lift = self.pano.world_to_pano(np.column_stack([pts, hts])) - self.pano.world_to_pano(
                np.column_stack([pts, np.zeros(len(pts))]))
            uv = uv + lift
        return uv

    # ------------------------------------------------------------------ cards (what each camera sees)
    def cards(self, t: float, shots: dict[str, Shot], pitch: MapState | None,
              group: tuple[float, float] | None) -> list[CamCard]:
        """One CamCard per camera, from the pitch map projected through each camera's model."""
        w, h = self.size
        types = self.cfg["shot_types"]
        out = []
        ball = pitch.ball if pitch is not None and pitch.ball_status != "lost" else None
        status = {"seen": "detected", "remembered": "predicted"}.get(pitch.ball_status if pitch else "", "lost")
        players = np.array(pitch.players, np.float64).reshape(-1, 2) if pitch is not None and pitch.players else \
            np.zeros((0, 2))
        action = ball if ball is not None else group
        for k, c in enumerate(self.cameras):
            shot = shots[c.id]
            info = None
            if ball is not None:
                px, ok = self.project(shot, np.array([ball]))
                if ok[0]:
                    info = BallInfo(round(px[0, 0] / w, 3), round(px[0, 1] / h, 3), status)
            n_players = int(self.project(shot, players)[1].sum()) if len(players) else 0
            act = None
            if action is not None:
                px, ok = self.project(shot, np.array([action]))
                if ok[0]:
                    act = (round(px[0, 0] / w, 3), round(px[0, 1] / h, 3))
            width = shot.state.get("field_width_m", 0)
            kind = "wide" if width > types["wide_above_m"] else "close_up" if width < types["close_below_m"] else "medium"
            out.append(CamCard(cam=k + 1, name=c.name, time=round(t, 2), logged_at=round(t, 2), shot_type=kind,
                               ball=info, players=n_players, play_speed=0.0, quality=Quality(),
                               source="detector", action_in_view=act is not None, action=act,
                               zone=pitch.zone if pitch else "unknown",
                               map_intensity=pitch.intensity if pitch else 0.0,
                               moving_deg_s=round(shot.state.get("speed_deg_s", 0.0), 2)))
        return out

    # ------------------------------------------------------------------ pictures (real-pixel cameras)
    def picture(self, shot: Shot, pano_frame: np.ndarray | None, sources: dict[int, np.ndarray]) -> np.ndarray | None:
        """This camera's 1280x720 picture right now (None for 3d cameras: the browser draws them)."""
        w, h = self.size
        c = shot.cam
        if c.built_from == "source":
            return crop_to(sources[c.source_cam], shot.rect, w, h)
        if c.built_from != "panorama" or pano_frame is None:
            return None
        if shot.pinhole is None:                    # WIDE: the whole panorama, scaled down (letterboxed)
            return letterbox(pano_frame, w, h)
        map1, map2 = self.view_maps(shot.pinhole)
        image = cv2.remap(pano_frame, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # How many output pixels one SOURCE camera pixel became. The videos store colour at half the resolution
        # of brightness; magnified, thin white lines get rainbow edges. Softening only the colour hides that.
        source_focal = self.pano.focal / self.cfg["panorama"]["focal_scale"]
        return soften_colour(image, self.cfg["colour_soften"] * shot.pinhole.focal / source_focal)

    def view_maps(self, pin: Pinhole, cell: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Lookup tables "output pixel -> panorama pixel" for a perspective view. Computed on a coarse grid (one
        point per 8x8 pixels: the mapping is smooth) and enlarged; ~1 ms instead of ~40 ms per frame."""
        w, h = self.size
        gw, gh = w // cell, h // cell
        # cv2.resize puts coarse sample k at output pixel (k + 0.5) * cell - 0.5: sample exactly there
        u = (np.arange(gw) + 0.5) * cell - 0.5
        v = (np.arange(gh) + 0.5) * cell - 0.5
        uu, vv = np.meshgrid(u, v)
        uv = self.pano.rays_to_pano(pin.rays(np.column_stack([uu.ravel(), vv.ravel()])))
        mx = cv2.resize(uv[:, 0].reshape(gh, gw).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(uv[:, 1].reshape(gh, gw).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.convertMaps(mx, my, cv2.CV_16SC2)


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The whole picture scaled into width x height, black bars where the shapes differ (no copy first)."""
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = round(image.shape[1] * scale), round(image.shape[0] * scale)
    out = np.zeros((height, width, 3), image.dtype)
    x, y = (width - w) // 2, (height - h) // 2
    out[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return out


def soften_colour(image: np.ndarray, factor: float) -> np.ndarray:
    """Blur only the colour (not the brightness) of a picture by about `factor` pixels."""
    if factor < 1.5:
        return image
    h, w = image.shape[:2]
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    small = cv2.resize(ycc[..., 1:], (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    ycc[..., 1:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------------------------------------------
# 4. Step 2's run: follow the play with MAIN and CLOSE, write the 5 real feeds
# ---------------------------------------------------------------------------------------------------------------
class NoStory:
    """An empty story feed: Step 2 runs the pitch map only (the director and story come in Step 5)."""

    def known_events(self, t: float) -> list[dict]:
        return []

    def known_updates(self, t: float) -> list[dict]:
        return []

    def latest_phase(self, t: float, max_age: float) -> None:
        return None


def follow_aim(pitch: MapState | None, group: tuple[float, float] | None) -> tuple[float, float] | None:
    """Step 2's simple aim: the map's ball (seen, or remembered for a moment), else the main group of players."""
    if pitch is not None and pitch.ball is not None and pitch.ball_status != "lost":
        return pitch.ball
    return group


class Prefetch:
    """Reads the next frames of several videos in a background thread (OpenCV decodes while the main thread
    draws and writes). get() -> {key: frame} or None at the end of any video."""

    def __init__(self, readers: dict, ahead: int = 3):
        import queue
        import threading

        self.readers = readers
        self.queue: queue.Queue = queue.Queue(maxsize=ahead)
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stop:
            frames = {k: r.read() for k, r in self.readers.items()}
            if any(f is None for f in frames.values()):
                self.queue.put(None)
                return
            self.queue.put(frames)

    def get(self) -> dict | None:
        return self.queue.get()

    def close(self) -> None:
        self.stop = True
        try:
            while True:
                self.queue.get_nowait()             # unblock the reader thread
        except Exception:
            pass


class FrameReader:
    """Frames of one video from a start frame on, one at a time."""

    def __init__(self, path: Path, start: int):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


# ---------------------------------------------------------------------------------------------------------------
# 5. Step 5: the camera operators + the per-frame brain of a stadium run
# ---------------------------------------------------------------------------------------------------------------
class TVOperators:
    """Aims and zooms the robotic cameras like their operators would (the director only says what is on air):
      MAIN   the ball, with room ahead of it in the direction of play; 40 m wide in open play, 28 m in an attack
             in a box, 60 m for a kick-off / goal kick; zooms out by itself if the ball gets near its edge
      CLOSE  the player nearest the ball (or the ball / the main group), 12 m wide
    Uses only the pitch map's present and past (causal)."""

    def __init__(self, cfg: dict, stadium: "Stadium"):
        self.cfg = cfg
        self.stadium = stadium
        self.last_ball: tuple[float, tuple[float, float]] | None = None    # (t, position) of the last sighting
        self.velocity = np.zeros(2)                                       # ball, m/s (smoothed)

    def update(self, pitch: MapState | None, group: tuple[float, float] | None, state: str) -> None:
        o = self.cfg
        ball = pitch.ball if pitch is not None and pitch.ball_status != "lost" else None
        if ball is not None and pitch.ball_status == "seen":
            if self.last_ball is not None and pitch.t > self.last_ball[0]:
                v = (np.array(ball) - self.last_ball[1]) / (pitch.t - self.last_ball[0])
                self.velocity = 0.5 * self.velocity + 0.5 * v
            self.last_ball = (pitch.t, np.array(ball))
        main, close = self.stadium.heads["main"], self.stadium.heads["close"]
        target = ball if ball is not None else group
        if target is not None:
            lead = self.velocity * o["lead_s"] if ball is not None else np.zeros(2)
            size = float(np.linalg.norm(lead))
            if size > o["lead_max_m"]:
                lead *= o["lead_max_m"] / size
            main.aim(target[0] + lead[0], target[1] + lead[1])
        width = {"ATTACK": o["main_attack_m"], "RESTART": o["main_restart_m"]}.get(state, o["main_open_m"])
        look = look_at_of(main.centre, main.view)
        if ball is not None and look is not None:          # the ball near the edge of the shot: widen to keep it
            now = field_width_of(main.centre, main.view)
            off = math.dist(look, ball)
            if off > o["keep_margin"] * now:
                width = max(width, off / o["keep_margin"])
        main.zoom(width)
        # CLOSE: the player on the ball
        aim = target
        if ball is not None and pitch.players:
            pts = np.array(pitch.players, np.float64)
            d = np.linalg.norm(pts - ball, axis=1)
            if d.min() <= o["close_player_m"]:
                aim = tuple(pts[d.argmin()])
        if aim is not None:
            close.aim(*aim)
        close.zoom(o["close_m"])


class StadiumPipeline:
    """The per-frame brain of a stadium run (live or offline): MVP 3's calibrated pipeline keeps the pitch map
    (detections of the 3 real cameras -> ball + players in metres, kick-off check, teams); the stadium gives every
    camera's card; the operators aim MAIN / CLOSE; DirectorV2 in tv mode picks what is on air.
    aim "follow" = Step 2 (MAIN and CLOSE simply follow the ball, no director)."""

    def __init__(self, src, stadium: "Stadium", aim: str, feed=None):
        from studio.director import DirectorV2

        cfg = settings()
        self.src, self.stadium, self.aim = src, stadium, aim
        self.base = src.pipeline(feed or NoStory())
        self.director = DirectorV2({**src.cfg["director"], **cfg["director"]}, src.fps, len(stadium.cameras))
        self.ops = TVOperators(cfg["operators"], stadium)
        self.started = False                            # heads framed on the play yet?
        self.fresh = True
        self.cards: list = []
        self.began = False                              # director told where the run starts?

    @property
    def pitch(self) -> MapState | None:
        return self.base.pitch_state

    def detections(self, i: int, t: float, detector: dict, images: dict | None = None) -> list[dict]:
        self.fresh = True                               # new map data: the cards are built again next frame
        events = self.base.detections(i, t, detector=detector, images=images)
        if self.aim == "follow":
            target = follow_aim(self.pitch, self.base.group)
            if target is not None:
                for head in self.stadium.heads.values():
                    head.aim(*target)
        else:
            self.ops.update(self.pitch, self.base.group, self.director.state)
        if not self.started and any(h.target_look != h.cam.look_at_m for h in self.stadium.heads.values()):
            for head in self.stadium.heads.values():   # start already framing the play (a crew would)
                head.view = view_towards(head.centre, *head.target_look, head.target_width)
            self.started = True
        return events

    def frame(self, i: int, t: float, dt: float):
        """Move the heads one frame; returns (decision or None, cards, shots)."""
        self.stadium.step(dt)
        shots = self.stadium.shots()
        if self.fresh or not self.cards:              # what each camera sees: when the map has news (5x per second)
            self.cards = self.stadium.cards(t, shots, self.pitch, self.base.group)
            self.fresh = False
        for card, cam in zip(self.cards, self.stadium.cameras):   # every frame: how fast the robotic heads turn
            card.moving_deg_s = round(shots[cam.id].state.get("speed_deg_s", 0.0), 2)
        cards = self.cards
        decision = None
        if self.aim == "director":
            if not self.began:                          # the first shot starts where the run starts
                self.director.start_at(i)
                self.began = True
            decision = self.director.decide(i, t, cards, [], [], None, pitch=self.pitch)
        return decision, cards, shots


# ---------------------------------------------------------------------------------------------------------------
# 6. The 3D cameras' videos (recorded by the browser) + the program picture
# ---------------------------------------------------------------------------------------------------------------
def record_3d(segment: str, start_s: float, seconds: float, run_id: str, server: str) -> bool:
    """Have headless Chrome render + record GOAL L / GOAL R for this stretch into the run folder (the dashboard
    server must be running: it serves the 3D page and turns the recordings into cam_goall.mp4 / cam_goalr.mp4)."""
    import shutil
    import subprocess
    import tempfile
    import urllib.request

    cfg = settings()
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    try:
        urllib.request.urlopen(f"{server}/studio/stadium/setup/{segment}", timeout=10).read()
    except Exception as e:  # noqa: BLE001
        print(f"[stadium] 3D cameras skipped: no dashboard at {server} ({e}). Start it: python -m dashboard.server",
              flush=True)
        return False
    chrome = cfg["program"]["chrome"]
    if not Path(chrome).exists():
        print(f"[stadium] 3D cameras skipped: Chrome not found at {chrome}", flush=True)
        return False
    when = f"{int(start_s // 60)}:{start_s % 60:04.1f}"
    url = f"{server}/static/stadium.html?segment={segment}&record={when}&secs={seconds:g}&run={run_id}"
    profile = tempfile.mkdtemp(prefix="stadium-chrome-")
    print(f"[stadium] recording the 3D cameras in headless Chrome ({seconds:g} s of match)...", flush=True)
    proc = subprocess.Popen([chrome, "--headless=new", *cfg["program"]["chrome_flags"], "--window-size=1600,1200",
                             f"--user-data-dir={profile}", "--remote-debugging-port=0", url],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while time.time() - t0 < 120 + seconds * 6:
            if all((out / f"cam_{c}.mp4").exists() for c in ("goall", "goalr")):
                break
            time.sleep(3)
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    ok = all((out / f"cam_{c}.mp4").exists() for c in ("goall", "goalr"))
    print(f"[stadium] 3D cameras {'recorded' if ok else 'NOT recorded (timeout)'} in {time.time() - t0:.0f} s",
          flush=True)
    return ok


def placeholder(name: str, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 24, np.uint8)
    cv2.putText(img, f"{name}: 3D camera not recorded", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    return img


def program_picture(image: np.ndarray, cam: Camera, t: float, score: list[int], cfg: dict, credit: str,
                    label: bool = True) -> np.ndarray:
    """The broadcast picture: a TV scorebug (teams, score, match clock), the camera's name, the dataset credit."""
    out = image.copy()
    if cfg["program"]["scorebug"]:
        a, b = cfg["program"]["teams_short"]
        text = f"{a}  {score[0]}-{score[1]}  {b}"
        clock = f"{int(t // 60):02d}:{int(t % 60):02d}"
        cv2.rectangle(out, (24, 22), (330, 62), (20, 20, 28), -1)
        cv2.rectangle(out, (330, 22), (430, 62), (40, 40, 160), -1)
        cv2.putText(out, text, (36, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, clock, (342, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        name = cam.name + ("  (3D)" if cam.built_from == "3d" else "")
        cv2.putText(out, name, (26, out.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(out, credit, (out.shape[1] - 10 - 7 * len(credit), out.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8,
                (170, 170, 170), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 7. A run: all feeds (+ program, director log with --aim director)
# ---------------------------------------------------------------------------------------------------------------
def run(segment: str, start_s: float, seconds: float, aim: str, panorama: Path | None = None,
        panorama_start_s: float = 0.0, record3d: bool = True, server: str | None = None) -> Path:
    """aim "follow": Step 2 (the 5 real feeds, MAIN/CLOSE follow the ball). aim "director": Step 5 (+ the 3D
    feeds, the TV director, program.mp4, director_log.csv).
    panorama / panorama_start_s: a short test clip made with `studio.panorama --minutes N --start MM:SS`
    instead of the whole half's panorama.mp4 (which starts at 0:00)."""
    from studio.sources import get_source

    if aim not in ("follow", "director"):
        raise SystemExit("--aim follow | director")
    cfg = settings()
    src = get_source(segment)
    pano = Panorama.load(src.folder)
    pano_video = panorama or src.folder / "panorama.mp4"
    if not pano_video.exists():
        raise SystemExit(f"{pano_video} is missing: run `venv/bin/python -m studio.panorama` first")
    stadium = Stadium(cfg, pano)
    fps = src.fps
    start = round(start_s * fps)
    count = min(round(seconds * fps), src.frames - start)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{aim}"
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[stadium] {segment} {int(start_s // 60)}:{int(start_s % 60):02d} +{seconds:.0f}s -> {out.relative_to(ROOT)}",
          flush=True)
    if aim == "director" and record3d:
        record_3d(segment, start_s, count / fps, run_id, server or cfg["program"]["server"])

    pipe = StadiumPipeline(src, stadium, aim)
    w, h = stadium.size
    real = [c for c in stadium.cameras if c.built_from in ("panorama", "source")]
    writers = {c.id: VideoWriter(out / f"cam_{c.id}.mp4", w, h, fps) for c in real}
    readers = {"pano": FrameReader(pano_video, start - round(panorama_start_s * fps))}
    for c in real:
        if c.built_from == "source":
            readers[c.source_cam] = FrameReader(src.video(c.source_cam), start)
    if aim == "director":
        for c in stadium.cameras:
            if c.built_from == "3d" and (out / f"cam_{c.id}.mp4").exists():
                readers[c.id] = FrameReader(out / f"cam_{c.id}.mp4", 0)
        writers["program"] = VideoWriter(out / "program.mp4", w, h, fps)
        dlog = open(out / "director_log.csv", "w")
        dlog.write("frame,time,cam,cam_name,state,reason\n")
    prefetch = Prefetch(readers)
    seen = in_main = 0
    fastest = {cid: 0.0 for cid in stadium.heads}
    airtime: dict[str, int] = {}
    cuts: list[dict] = []
    log = open(out / "cameras.jsonl", "w")
    credit = "Alfheim dataset (Simula), non-commercial research"
    t0 = time.time()
    n = -1
    try:
        for n in range(count):
            i = start + n
            t = i / fps
            if i % src.step == 0:
                pipe.detections(i, t, detector=src.detection_at(i))
            decision, cards, shots = pipe.frame(i, t, 1 / fps)
            for cid, head in stadium.heads.items():
                fastest[cid] = max(fastest[cid], head.speed_deg_s)
            pitch = pipe.pitch
            if i % src.step == 0 and pitch is not None and pitch.ball_status == "seen":
                seen += 1
                in_main += int(stadium.project(shots["main"], np.array([pitch.ball]))[1][0])
            frames = prefetch.get()
            if frames is None:
                print("[stadium] a video ended early", flush=True)
                break
            sources = {k: f for k, f in frames.items() if isinstance(k, int)}
            pictures = {}
            for c in real:
                pictures[c.id] = stadium.picture(shots[c.id], frames["pano"], sources)
                writers[c.id].write(pictures[c.id])
            entry = {"i": i, "t": round(t, 3), "cams": {cid: s.state for cid, s in shots.items()}}
            if decision is not None:
                cam = stadium.cameras[decision.cam - 1]
                picture = pictures.get(cam.id)
                if picture is None:
                    picture = frames.get(cam.id) if cam.id in frames else placeholder(cam.name, w, h)
                writers["program"].write(program_picture(picture, cam, t, getattr(pipe.base, "score", [0, 0]),
                                                         cfg, credit))
                airtime[cam.id] = airtime.get(cam.id, 0) + 1
                cuts += [e for e in decision.events if e["type"] == "CUT"]
                reason = decision.reason.replace('"', "'")
                dlog.write(f'{i},{t:.3f},{decision.cam},{cam.name},{decision.state},"{reason}"\n')
                entry["on_air"] = cam.id
            log.write(json.dumps(entry) + "\n")
            if n % 300 == 0 and n:
                print(f"  {n}/{count} frames  {n / (time.time() - t0):.1f} fps", flush=True)
    except BaseException:
        for wr in writers.values():
            wr.abort()
        raise
    finally:
        log.close()
        prefetch.close()
        if aim == "director":
            dlog.close()
    for wr in writers.values():
        wr.close()
    summary = {"segment": segment, "start_s": start_s, "seconds": seconds, "aim": aim, "frames": n + 1,
               "ball_seen_steps": seen, "ball_in_main_pct": round(100 * in_main / seen, 1) if seen else None,
               "fastest_pan_deg_s": {k: round(v, 2) for k, v in fastest.items()},
               "pan_limit_deg_s": {k: stadium.camera(k).max_pan_deg_s for k in stadium.heads},
               "minutes": round((time.time() - t0) / 60, 1), "credit": src.meta.get("credit", "")}
    if aim == "director":
        summary.update(director_summary(airtime, cuts, n + 1, fps, cfg))
        summary["feeds_3d"] = [c.id for c in stadium.cameras if c.built_from == "3d" and c.id in readers]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return out


def director_summary(airtime: dict[str, int], cuts: list[dict], frames: int, fps: float, cfg: dict) -> dict:
    """Airtime per camera, number of cuts, shot lengths (the last shot is still running: not counted as short)."""
    shots = [c["shot_seconds"] for c in cuts]
    return {"airtime_pct": {k: round(100 * v / frames, 1) for k, v in sorted(airtime.items(), key=lambda x: -x[1])},
            "cuts": len(cuts), "avg_shot_s": round(frames / fps / (len(cuts) + 1), 2),
            "shots_under_min": sum(1 for s in shots if s < cfg["director"]["min_shot_seconds"] - 1e-6),
            "shortest_shot_s": round(min(shots), 2) if shots else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="The virtual stadium's camera feeds (MVP 4 Steps 2 and 5).")
    ap.add_argument("--segment", default="alfheim-2013-11-03-H1")
    ap.add_argument("--start", default="23:40", help="MM:SS into the half")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--aim", default="follow", help="follow (Step 2) | director (Step 5: TV director + program)")
    ap.add_argument("--panorama", type=Path, help="a short test panorama clip instead of panorama.mp4")
    ap.add_argument("--panorama-start", default="0:00", help="where that clip starts in the half (MM:SS)")
    ap.add_argument("--no-3d", action="store_true", help="don't record the 3D goal cameras (program shows a card)")
    ap.add_argument("--server", help="dashboard address for the 3D page (default: config studio.stadium.program.server)")
    args = ap.parse_args()
    run(args.segment, parse_time(args.start), args.seconds, args.aim, args.panorama, parse_time(args.panorama_start),
        not args.no_3d, args.server)


if __name__ == "__main__":
    main()
