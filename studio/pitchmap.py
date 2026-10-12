"""pitchmap.py — the pitch map: ball and players on a 105 x 68 m pitch, live, from the cameras only.

Why: the director and the story layer think in camera pictures; a human director thinks about the PITCH
("the ball is on the left wing, heading into the box"). The map gives the system that view:
  - the ball's position, speed and trail (last seconds), remembered for a moment when the detector loses it
    (measured on dev: half of the "no ball on air" moments came < 2 s after the last ball sighting);
  - zones and triggers: ball enters / leaves a penalty box, goes out over a line, crosses the goal line
    between the posts (a goal candidate);
  - a map intensity (ball near goal, fast ball, players in the box) for the fast layer, with no AI quota.

Calibration: the fixed camera's pitch -> picture mapping (pitch_to_image.npy, made by studio.prepare) is a
one-time SETUP measurement, like calibrating a real camera when it is installed; it is not per-frame truth.
Everything here uses only the current and past frames (causal).

Pitch coordinates: metres, origin at the centre spot, x to the right (-52.5 .. 52.5), y towards the near
touchline (-34 .. 34). Penalty box: 16.5 m deep, 40.32 m wide. Goal: 7.32 m wide.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

HALF_X, HALF_Y = 52.5, 34.0
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16
GOAL_HALF_W = 3.66


def zone_of(x: float, y: float) -> str:
    """Where a point is: out / left_box / right_box / left_third / middle / right_third."""
    if abs(x) > HALF_X or abs(y) > HALF_Y:
        return "out"
    if abs(y) <= BOX_HALF_W and x <= -HALF_X + BOX_DEPTH:
        return "left_box"
    if abs(y) <= BOX_HALF_W and x >= HALF_X - BOX_DEPTH:
        return "right_box"
    if x < -HALF_X / 3:
        return "left_third"
    if x > HALF_X / 3:
        return "right_third"
    return "middle"


class Projection:
    """Panorama pixels <-> pitch metres (the fixed camera's calibration)."""

    def __init__(self, pitch_to_image: np.ndarray):
        self.to_image = pitch_to_image
        self.to_pitch = np.linalg.inv(pitch_to_image)

    def pitch(self, px: float, py: float) -> tuple[float, float]:
        x, y = cv2.perspectiveTransform(np.array([[[px, py]]], np.float32), self.to_pitch)[0, 0]
        return float(x), float(y)

    def image(self, x: float, y: float) -> tuple[float, float]:
        px, py = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.to_image)[0, 0]
        return float(px), float(py)

    def coverage(self, rect: tuple[float, float, float, float], n: int = 7) -> list[list[float]]:
        """The part of the pitch a camera rectangle sees, as a polygon in metres (clipped to near the pitch)."""
        x, y, w, h = rect
        pts = []
        for gx in np.linspace(x, x + w, n):
            for gy in np.linspace(y, y + h, n):
                mx, my = self.pitch(gx, gy)
                if abs(mx) <= HALF_X + 6 and abs(my) <= HALF_Y + 6:
                    pts.append([mx, my])
        if len(pts) < 3:
            return []
        hull = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
        # Only the part ON the pitch is interesting (and the lens makes the outer corners unreliable).
        pitch = np.array([[-HALF_X, -HALF_Y], [HALF_X, -HALF_Y], [HALF_X, HALF_Y], [-HALF_X, HALF_Y]], np.float32)
        area, clipped = cv2.intersectConvexConvex(hull.astype(np.float32), pitch)
        if area <= 0 or clipped is None:
            return []
        return [[round(float(a), 1), round(float(b), 1)] for a, b in clipped.reshape(-1, 2)]


@dataclass
class MapState:
    t: float
    ball: tuple[float, float] | None = None       # metres
    ball_status: str = "lost"                     # seen / remembered / lost
    ball_age_s: float = 0.0                       # seconds since the detector last saw it
    speed_ms: float = 0.0                         # ball speed, metres per second
    zone: str = "unknown"
    players: list[tuple[float, float]] = field(default_factory=list)
    players_in_box: int = 0
    intensity: float = 0.0                        # 0..1, from the map only
    events: list[dict] = field(default_factory=list)   # BOX_ENTRY / BOX_EXIT / BALL_OUT / GOAL_LINE

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else [round(v[0], 1), round(v[1], 1)]   # noqa: E731
        return {"t": round(self.t, 2), "ball": r(self.ball), "status": self.ball_status,
                "age": round(self.ball_age_s, 1), "speed": round(self.speed_ms, 1), "zone": self.zone,
                "players": [r(p) for p in self.players], "in_box": self.players_in_box,
                "intensity": round(self.intensity, 2), "events": self.events}


class PitchMap:
    """Tracks the ball on the pitch (with speed), remembers it briefly, and raises zone triggers."""

    def __init__(self, cfg: dict, projection: Projection):
        self.cfg = cfg
        self.proj = projection
        self.ball: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)                # m/s
        self.last_seen_t: float | None = None
        self.last_t: float | None = None
        self.zone = "unknown"
        self.trail: deque = deque()

    def update(self, t: float, ball_px: tuple[float, float] | None,
               feet_px: list[tuple[float, float]]) -> MapState:
        cfg = self.cfg
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        players = [p for p in (self.proj.pitch(*f) for f in feet_px)
                   if abs(p[0]) <= HALF_X + 3 and abs(p[1]) <= HALF_Y + 3]   # people on (or next to) the pitch
        seen = None
        if ball_px is not None:
            seen = self.proj.pitch(*ball_px)
            if abs(seen[0]) > HALF_X + 8 or abs(seen[1]) > HALF_Y + 8:
                seen = None                       # far outside the pitch: not the match ball
        if seen is not None and self.ball is not None and self.last_seen_t is not None:
            gap = max(t - self.last_seen_t, 1e-3)
            if math.dist(seen, self.ball) > cfg["max_ball_speed_ms"] * gap + 2.0:
                seen = None                       # impossible jump: a false detection
        status = "lost"
        if seen is not None:
            if self.ball is not None and self.last_seen_t is not None:
                gap = max(t - self.last_seen_t, 1e-3)
                v = ((seen[0] - self.ball[0]) / gap, (seen[1] - self.ball[1]) / gap)
                a = cfg["velocity_smoothing"]
                self.velocity = (a * v[0] + (1 - a) * self.velocity[0], a * v[1] + (1 - a) * self.velocity[1])
            self.ball, self.last_seen_t, status = seen, t, "seen"
        elif self.ball is not None and self.last_seen_t is not None and t - self.last_seen_t <= cfg["remember_s"]:
            # Not seen now: keep it where it was going, slowing down (a rolling ball slows, players stop it).
            damp = cfg["remember_damping"] ** dt
            self.velocity = (self.velocity[0] * damp, self.velocity[1] * damp)
            self.ball = (self.ball[0] + self.velocity[0] * dt, self.ball[1] + self.velocity[1] * dt)
            status = "remembered"
        else:
            self.ball, self.velocity, self.last_seen_t = None, (0.0, 0.0), None
        age = 0.0 if self.last_seen_t is None else t - self.last_seen_t

        events = []
        zone = zone_of(*self.ball) if self.ball else "unknown"
        if self.ball and status == "seen":
            events = self._zone_events(self.zone, zone, self.ball)
            self.zone = zone
        speed = math.hypot(*self.velocity)
        in_box = sum(1 for p in players if zone_of(*p) in ("left_box", "right_box"))
        state = MapState(t, self.ball, status, age, speed, zone, players, in_box, 0.0, events)
        state.intensity = self.intensity(state)
        return state

    def _zone_events(self, before: str, after: str, ball: tuple[float, float]) -> list[dict]:
        events = []
        boxes = ("left_box", "right_box")
        if after in boxes and before not in boxes:
            events.append({"type": "BOX_ENTRY", "zone": after})
        if before in boxes and after not in boxes and after != "out":
            events.append({"type": "BOX_EXIT", "zone": before})
        if after == "out" and before not in ("out", "unknown"):
            over_goal_line = abs(ball[0]) > HALF_X
            if over_goal_line and abs(ball[1]) <= GOAL_HALF_W + self.cfg["goal_line_margin_m"]:
                # Crossed the goal line between (or right at) the posts: a goal candidate, not a goal yet.
                events.append({"type": "GOAL_LINE", "side": "left" if ball[0] < 0 else "right"})
            else:
                events.append({"type": "BALL_OUT", "line": "goal line" if over_goal_line else "touchline"})
        return events

    def intensity(self, s: MapState) -> float:
        """0..1 from the map: how close the ball is to a goal, how fast it moves, how crowded the boxes are."""
        w = self.cfg["intensity_weights"]
        near_goal = 0.0
        if s.ball and s.ball_status != "lost":
            near_goal = max(0.0, min(1.0, (abs(s.ball[0]) - 0.0) / HALF_X)) ** 2
        fast = min(1.0, s.speed_ms / self.cfg["fast_ball_ms"])
        crowd = min(1.0, s.players_in_box / self.cfg["box_crowd"])
        return w["near_goal"] * near_goal + w["speed"] * fast + w["box_crowd"] * crowd
