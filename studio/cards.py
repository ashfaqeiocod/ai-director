"""cards.py — MVP 3 Step 3b: the "camera card", a live description of every camera about once a second.

This is the plan's central idea (docs/AI_LIVE_DIRECTOR_PLAN.md section 3): with a good card per camera,
choosing the shot is just rules. A card says what the shot is, where the ball is, how fast play moves,
how many players are in view, what the story layer last reported, and whether the picture is usable.

Three helpers fill it, all causal (frame i only uses frames <= i):
  World         ball + players + "action point" in panorama pixels (all 5 cams are views of one panorama)
  QualityMeter  blur + shake per camera, measured on the camera's own pictures
  StoryFeed     story events/phases from the MVP 2 story run, each usable only from its logged_at
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.ball_tracker import DETECTED, LOST, PREDICTED, BallState, BallTracker
from studio.cameras import group_center


@dataclass
class BallInfo:
    x: float                       # 0..1 across this camera's picture
    y: float
    status: str                    # detected / predicted


@dataclass
class Quality:
    usable: bool = True
    blur: float = 0.0              # 0 sharp .. 1 very blurry (compared with this camera's own normal)
    shake: float = 0.0             # 0 steady .. 1 shaking a lot


@dataclass
class CamCard:
    cam: int
    name: str
    time: float                    # clip time this card describes
    logged_at: float               # clip time it was ready (same here: everything in it is from <= time)
    shot_type: str                 # wide / medium / close_up ...
    ball: BallInfo | None
    players: int
    play_speed: float              # 0..1
    quality: Quality
    event: dict | None = None      # latest story event known at this time
    phase: str | None = None       # latest story phase known at this time
    source: str = "detector"       # detector | oracle
    action_in_view: bool = False   # is the action point inside this camera's view?
    action: tuple[float, float] | None = None   # where the action point is in this camera (0..1), if in view
    zone: str = "unknown"          # where the ball is on the pitch map (left_box, middle, ...), shared by all cams
    map_intensity: float = 0.0     # 0..1 from the pitch map (ball near goal, fast ball, crowded box)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- the world (panorama)
@dataclass
class WorldState:
    ball: tuple[float, float] | None = None       # panorama pixels
    ball_status: str = LOST
    players: list[tuple[float, float]] = field(default_factory=list)
    action: tuple[float, float] | None = None     # where play is: the ball, else the main group of players
    play_speed: float = 0.0


class World:
    """Ball, players and the action point in panorama pixels, updated whenever new detections arrive."""

    def __init__(self, cfg: dict, source_size: tuple[int, int], fps: float, step: int):
        bt = cfg["ball_tracker"]
        self.tracker = BallTracker(bt["buffer_size"], bt["max_jump_px"], bt["max_predict_updates"], source_size)
        self.radius_px = cfg["cameras"]["close"]["group_radius_px"]
        self.fast_px_s = cfg["director"]["fast_play_px_s"]
        self.smooth = cfg["director"]["action_smoothing"]
        self.fps, self.step = fps, step
        self.history: deque[tuple[float, tuple[float, float]]] = deque()   # (time, smoothed action point)
        self.state = WorldState()

    def update(self, t: float, ball: tuple[float, float] | None, players: list[tuple[float, float]],
               group: tuple[float, float] | None = None, ball_status: str | None = None) -> WorldState:
        """New detections for time t (ball center or None, player centers). `group` overrides the players' group.

        ball_status: if given, the ball already comes tracked (from the pitch map: "detected" or "predicted")
        and MVP 1's pixel tracker is skipped."""
        if ball_status is not None:
            b = BallState(ball_status if ball is not None else LOST, ball)
        else:
            box = None if ball is None else [ball[0] - 1, ball[1] - 1, ball[0] + 1, ball[1] + 1]
            b = self.tracker.update(box)
        if group is None and players:
            pts = np.asarray(players, dtype=float)
            group = group_center(pts, pts, self.radius_px)
        target = b.center if b.status in (DETECTED, PREDICTED) else group
        action = self.state.action
        if target is not None:
            # Smooth with the past only, so a group centre that hops between players doesn't look like speed.
            action = target if action is None else tuple(a + self.smooth * (g - a) for a, g in zip(action, target))
        if action is not None:
            self.history.append((t, action))
        while self.history and t - self.history[0][0] > 1.0:
            self.history.popleft()
        speed = 0.0
        if len(self.history) >= 2:
            (t0, p0), (t1, p1) = self.history[0], self.history[-1]
            speed = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(t1 - t0, 1e-6))
        self.state = WorldState(b.center, b.status, players, action, min(1.0, speed / self.fast_px_s))
        return self.state


# ----------------------------------------------------------------------------- feed quality
class QualityMeter:
    """Blur and shake of ONE camera, from its own pictures (past only).

    Blur: sharpness = variance of the Laplacian (edges) of a small grey picture, compared with the median
    of this camera's own recent sharpness (grass alone is less sharp than a crowd: each camera is its own
    reference). Shake: how far the picture jumps between two frames 1/25 s apart (phase correlation),
    minus any planned camera move (the close-up's operator pans on purpose).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sharpness: deque[float] = deque(maxlen=cfg["history"])
        self.prev: np.ndarray | None = None
        self.shake_samples: deque[float] = deque(maxlen=cfg["shake_samples"])
        self.recent_moves: deque = deque(maxlen=cfg.get("pan_window", 12))   # jitter mode: the recent smooth move
        self.quality = Quality()
        self.bad_since: float | None = None

    def update(self, t: float, image: np.ndarray, planned_move: tuple[float, float] = (0.0, 0.0),
               measure: bool = True, frames_since: int = 1) -> Quality:
        """frames_since: frames since the previous picture (live mode skips frames when it is late)."""
        width = self.cfg["width"]
        small = cv2.cvtColor(cv2.resize(image, (width, width * image.shape[0] // image.shape[1])), cv2.COLOR_BGR2GRAY)
        small = small.astype(np.float32)
        if self.prev is not None and frames_since == 1:
            # Only back-to-back frames: across skipped frames (live mode when late) the picture moved for
            # several frames, which would look like shake.
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, small)
            scale = width / image.shape[1]
            # In pixels of a 1280-wide picture, whatever the camera's size (phones are 1920 wide), so one
            # shake limit means the same on every camera. (SoccerTrack cameras are 1280 wide: unchanged.)
            unit = 1280 / image.shape[1]
            move = ((dx - planned_move[0] * scale) / scale * unit, (dy - planned_move[1] * scale) / scale * unit)
            if self.cfg.get("shake_mode") == "jitter":
                # Hand-held cameras pan smoothly all the time: that is filming, not shaking. Shake = how far
                # this frame's move differs from the recent smooth move (past frames only).
                if self.recent_moves:
                    mx, my = np.mean(self.recent_moves, axis=0)
                    jump = float(np.hypot(move[0] - mx, move[1] - my))
                else:
                    jump = 0.0
                self.recent_moves.append(move)
            else:
                jump = float(np.hypot(*move))       # fixed cameras: any jump is shake
            self.shake_samples.append(jump)
        self.prev = small
        if not measure:
            return self.quality
        sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
        reference = float(np.median(self.sharpness)) if len(self.sharpness) >= self.cfg["min_history"] else None
        blur = 0.0 if not reference else max(0.0, min(1.0, 1 - sharp / reference))
        shake = min(1.0, float(np.mean(self.shake_samples)) / self.cfg["shake_max_px"]) if self.shake_samples else 0.0
        bad = blur >= self.cfg["blur_bad"] or shake >= 1.0
        if not bad:
            self.sharpness.append(sharp)      # only good pictures teach the meter what "normal" is
        # Recover only after the picture has been fine for a moment (no flicker in and out).
        if bad:
            self.bad_since = t
        usable = self.bad_since is None or t - self.bad_since >= self.cfg["recover_s"]
        self.quality = Quality(usable, round(blur, 2), round(shake, 2))
        return self.quality


# ----------------------------------------------------------------------------- story events
class StoryFeed:
    """The story run's events and phases, each one usable only from the moment it was known (causal).

    events.jsonl gets a new line every time a fused event changes (e.g. a second look raises its
    confidence), and `logged_at` keeps the FIRST report. So each line's own moment is taken from its
    wall-clock stamp: playhead = (wall_time - wall_start) x speed, never earlier than logged_at.
    """

    def __init__(self, run_folder: Path | None, wall_start: float = 0.0, speed: float = 1.0):
        self.events: list[dict] = []
        self.updates: list[dict] = []
        if run_folder is None:
            return
        for line in _lines(run_folder / "events.jsonl"):
            line["known_at"] = max(line["logged_at"], (line["wall_time"] - wall_start) * speed) if wall_start \
                else line["logged_at"]
            self.events.append(line)
        self.events.sort(key=lambda e: e["known_at"])
        self.updates = sorted(_lines(run_folder / "story.jsonl"), key=lambda u: u["logged_at"])

    def add_goal_looks(self, looks: list[dict], cfg: dict) -> None:
        """Add the goal watcher's answers (studio.goalwatch): goal / shot events and "looks" for corroboration.
        Each is known from its logged_at (when the answer would have arrived live)."""
        last_goal: dict | None = None
        for n, lk in enumerate(sorted(looks, key=lambda x: x["logged_at"])):
            src = {"sources": ["goalcam"], "logged_at": lk["logged_at"], "known_at": lk["logged_at"]}
            if lk["goal"] and lk["goal_confidence"] >= cfg["min_goal_confidence"]:
                same = last_goal and lk["goal_time"] - last_goal["event_time"] <= cfg["merge_s"]
                event = {"id": last_goal["id"] if same else f"gw{n}", "type": "goal",
                         "event_time": last_goal["event_time"] if same else lk["goal_time"],
                         "confidence": max(lk["goal_confidence"], last_goal["confidence"] if same else 0), **src}
                last_goal = event
                self.events.append(event)
            if lk["shot"]:
                self.events.append({"id": f"gs{n}", "type": "shot", "event_time": lk["shot_time"], "confidence": 0.7, **src})
            if lk["goal"] or lk["celebration"]:
                self.updates.append({"window_end": lk["t"], "logged_at": lk["logged_at"], "window_start": lk["t"] - 6,
                                     "phase": "goal" if lk["goal"] else "celebration", "sentiment": "excited",
                                     "intensity": 90, "source": "goalcam",
                                     "events": [{"type": "goal"}] if lk["goal"] else [],
                                     "narration": f"Goal camera ({lk['side']}): {lk['note']}"})
        self.events.sort(key=lambda e: e["known_at"])
        self.updates.sort(key=lambda u: u["logged_at"])

    def known_events(self, t: float) -> list[dict]:
        """Every event as it was known at time t (the newest version of each id)."""
        out: dict[str, dict] = {}
        for e in self.events:
            if e["known_at"] > t:
                break
            out[e["id"]] = e
        return [e for e in out.values() if not e.get("rejected")]

    def known_updates(self, t: float) -> list[dict]:
        return [u for u in self.updates if u["logged_at"] <= t]

    def latest_phase(self, t: float, max_age_s: float) -> str | None:
        known = self.known_updates(t)
        if known and t - known[-1]["logged_at"] <= max_age_s:
            return known[-1]["phase"]
        return None


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
