"""multicam.py — the per-frame brain for several REAL fixed cameras that are calibrated to the pitch (ISSIA).

SoccerTrack has ONE panorama (all cameras are crops of it); MUVS has phones with no calibration. ISSIA is the
real broadcast set-up: 6 fixed cameras around the stadium, each with its own pitch calibration (a homography,
made once by `python -m studio.issia --calibrate`). So:

  1. 5x per second every camera's detections (players' feet, the ball) are put ON THE PITCH in metres;
  2. one pitch map for the whole match: players from all cameras (the same player seen twice = one dot),
     the ball from the camera that sees it (closest to where the ball was), remembered briefly when lost,
     zone triggers (box entry, goal line...), kick-off line-up, map intensity (studio.pitchmap, unchanged);
  3. each camera's card asks the map: "is the ball in MY picture, and where?" (pitch -> picture);
  4. DirectorV2 picks the camera (MVP 1 scoring on every camera: ball seen +100, centred up to +30, players).

Same interface as Pipeline / IndependentPipeline (detections / frame / confirmed / pitch_state), so studio.live
uses it unchanged. Everything uses only the current frame and the past (causal).
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from studio.cameras import group_center
from studio.cards import BallInfo, CamCard, QualityMeter
from studio.director import DirectorV2
from studio.pipeline import StepResult, event_summary
from studio.pitchmap import HALF_X, HALF_Y, PitchMap, Projection
from studio.teams import TeamClassifier


def to_pitch(H_inv: np.ndarray, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not pts:
        return []
    out = cv2.perspectiveTransform(np.asarray([pts], np.float64), H_inv)[0]
    return [(float(x), float(y)) for x, y in out]


def to_picture(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float] | None:
    """Pitch metres -> picture pixels, or None if the point is behind the camera."""
    x, y, w = H @ np.array([point[0], point[1], 1.0])
    if w <= 1e-9:
        return None
    return float(x / w), float(y / w)


def feet(det: dict) -> list[tuple[float, float]]:
    """Bottom-centre of each player box: where the player stands (that point is on the pitch plane)."""
    return [((p["box"][0] + p["box"][2]) / 2, p["box"][3]) for p in det.get("players", [])]


def merge_points(per_cam: dict[int, list[tuple]], radius: float) -> list[tuple]:
    """All cameras' players on one pitch: a point closer than `radius` to a point from ANOTHER camera is the same
    player (averaged); players of the same camera are never merged (they are different people).
    Points are (x, y) or (x, y, team); with teams, the merged player keeps the team most cameras gave it."""
    merged: list[list] = []                       # [x, y, count, cams, team votes]
    for cam, pts in per_cam.items():
        for point in pts:
            x, y = point[0], point[1]
            team = point[2] if len(point) > 2 else None
            best, best_d = None, radius
            for m in merged:
                d = math.hypot(m[0] - x, m[1] - y)
                if d <= best_d and cam not in m[3]:
                    best, best_d = m, d
            if best is None:
                best = [x, y, 1, {cam}, []]
                merged.append(best)
            else:
                n = best[2]
                best[0], best[1], best[2] = (best[0] * n + x) / (n + 1), (best[1] * n + y) / (n + 1), n + 1
                best[3].add(cam)
            if team is not None:
                best[4].append(team)
    if not any(len(p) > 2 for pts in per_cam.values() for p in pts):
        return [(m[0], m[1]) for m in merged]
    return [(m[0], m[1], max(set(m[4]), key=m[4].count) if m[4] else None) for m in merged]


class CalibratedPipeline:
    def __init__(self, cfg: dict, cams: dict, fps: float, feed, step: int, homographies: dict[int, np.ndarray],
                 settings: dict):
        self.cfg, self.cams, self.feed, self.step, self.fps = cfg, cams, feed, step, fps
        self.H = homographies                                   # cam -> pitch->picture
        self.H_inv = {c: np.linalg.inv(H) for c, H in homographies.items()}
        self.thr = settings["ball_tracker"]["min_confidence"]
        self.merge_m = settings["merge_players_m"]
        # Parked ball: a "ball" that sits at the same pitch spot in most past detections is a painted spot
        # (ISSIA cam1 saw the penalty spot as a ball in 149 of 600 frames) or a spare ball, not the match ball.
        parked = cfg["parked_ball"]
        self.parked_window = max(1, round(parked["window_seconds"] * fps / step))       # in detection steps
        self.parked_radius = settings["parked_radius_m"]
        self.parked_share, self.parked_history = parked["min_share"], parked["min_history"]
        self.ball_history = {c["cam"]: deque(maxlen=self.parked_window) for c in cams["cams"]}
        # The map works in metres directly: its "picture" IS the pitch (identity projection).
        self.pitch = PitchMap(cfg["map"], Projection(np.eye(3)))
        self.meters = [QualityMeter(cfg["quality"]) for _ in cams["cams"]]
        self.director = DirectorV2(cfg["director"], fps, len(cams["cams"]))
        self.confirmed: dict[str, dict] = {}
        self.pitch_state = None
        self.group: tuple[float, float] | None = None         # main group of players (metres)
        self.players: dict[int, int] = {}                     # cam -> players it sees on the pitch
        self.seen_ball_cam: int | None = None                 # the camera whose detection placed the ball
        self.cards: list[CamCard] = []
        self.known_events, self.updates, self.phase, self.story_line = [], [], None, ""
        self.last_i: int | None = None
        self.teams = TeamClassifier(cfg["teams"])
        self.possession = [0, 0]                              # detection steps each team had the ball
        self.score = [0, 0]
        self.last_ball_team: int | None = None

    # ------------------------------------------------------------------ detections (5x per second)
    def detections(self, i: int, t: float, detector: dict | None = None, oracle=None,
                   images: dict | None = None) -> list[dict]:
        """detector = {cam: {"players": [...], "ball": {...} | None}} for this detection frame.
        images = {cam: picture} of the same moment (for the shirt colours, optional)."""
        per_cam, balls = {}, []
        for c, det in (detector or {}).items():
            img = (images or {}).get(c)
            teams = self.teams.observe(img, det.get("players", [])) if img is not None and img.size > 1000 else \
                [None] * len(det.get("players", []))
            spots = to_pitch(self.H_inv[c], feet(det))
            pts = [(x, y, team) for (x, y), team in zip(spots, teams) if abs(x) <= HALF_X + 3 and abs(y) <= HALF_Y + 3]
            per_cam[c] = pts
            self.players[c] = len(pts)
            b = det.get("ball")
            spot = None
            if b and b["conf"] >= self.thr:
                x1, y1, x2, y2 = b["box"]
                spot = to_pitch(self.H_inv[c], [((x1 + x2) / 2, y2)])[0]   # where the ball touches the ground
                if not self._parked(c, spot):
                    balls.append((spot, b["conf"], c))
            self.ball_history[c].append(spot)                               # past only: checked before adding
        ball = self._pick_ball(balls)
        merged = merge_points(per_cam, self.merge_m)
        players = [(p[0], p[1]) for p in merged]
        self.pitch_state = self.pitch.update(t, ball, players)
        self._possession(merged)
        if players:
            pts = np.asarray(players)
            self.group = group_center(pts, pts, self.cfg["goalwatch"]["group_radius_m"])
        else:
            self.group = None
        events = [{"type": e["type"],
                   "message": f"{e['type'].lower().replace('_', ' ')} ({e.get('zone') or e.get('side') or e.get('line') or ''})"}
                  for e in self.pitch_state.events]
        if any(e["type"] == "KICKOFF_LINEUP" for e in self.pitch_state.events):
            events += self._confirm_goals(t)
        return events

    def _parked(self, cam: int, spot: tuple[float, float]) -> bool:
        """True if this camera saw a ball at (about) this spot in most of its recent detections."""
        past = [p for p in self.ball_history[cam] if p is not None]
        if len(past) < self.parked_history:
            return False
        same = sum(1 for p in past if math.dist(p, spot) <= self.parked_radius)
        return same / len(self.ball_history[cam]) >= self.parked_share

    def _possession(self, merged: list[tuple]) -> None:
        """The team of the player nearest the ball (within possession_m) has the ball. Needs the shirt colours."""
        st = self.pitch_state
        if not self.teams.ready:
            return
        st.teams = [p[2] if len(p) > 2 else None for p in merged]
        if st.ball is None or st.ball_status == "lost":
            return
        near = [(math.dist(p[:2], st.ball), p[2]) for p in merged if len(p) > 2 and p[2] is not None]
        if near:
            d, team = min(near)
            if d <= self.cfg["teams"]["possession_m"]:
                st.ball_team = team
                self.possession[team] += 1
                self.last_ball_team = team

    def team_info(self) -> dict | None:
        """For the scoreboard: team names + shirt colours, possession %, score."""
        if not self.teams.ready:
            return None
        total = max(1, sum(self.possession))
        return {"teams": self.teams.teams, "possession": [round(100 * p / total) for p in self.possession],
                "score": self.score}

    def _pick_ball(self, balls: list) -> tuple[float, float] | None:
        """Several cameras may see "a ball": prefer the one nearest to where the map's ball is, else the surest."""
        if not balls:
            self.seen_ball_cam = None
            return None
        last = self.pitch_state.ball if self.pitch_state is not None else None
        if last is not None:
            spot, _, cam = min(balls, key=lambda b: math.dist(b[0], last))
        else:
            spot, _, cam = max(balls, key=lambda b: b[1])
        self.seen_ball_cam = cam
        return spot

    def _confirm_goals(self, t: float) -> list[dict]:
        """A kick-off line-up confirms a goal the goal camera reported in the last minute (past only)."""
        window = self.cfg["map"]["goal_confirm_window_s"]
        out = []
        for g in self.feed.known_events(t):
            if (g["type"] == "goal" and "goalcam" in g["sources"] and g["id"] not in self.confirmed
                    and 0 <= t - g["event_time"] <= window):
                self.confirmed[g["id"]] = {"goal_time": g["event_time"], "confirmed_at": round(t, 2)}
                # Who scored: the team that attacked that goal = the team with the ball when it was last seen
                scorer = self.last_ball_team
                if scorer is not None:
                    self.score[scorer] += 1
                out.append({"type": "GOAL_CONFIRMED", "goal_time": g["event_time"], "team": scorer,
                            "message": f"goal at {g['event_time']:.0f}s confirmed by the kick-off line-up"})
        return out

    # ------------------------------------------------------------------ pitch -> one camera's picture
    def in_picture(self, cam: int, point: tuple[float, float] | None) -> tuple[float, float] | None:
        """Where a pitch point appears in this camera, as 0..1 of the picture, or None if it is not in view."""
        if point is None:
            return None
        p = to_picture(self.H[cam], point)
        w, h = self.cams["size"]
        if p is None or not (0 <= p[0] < w and 0 <= p[1] < h):
            return None
        return round(p[0] / w, 3), round(p[1] / h, 3)

    # ------------------------------------------------------------------ every frame
    def frame(self, i: int, t: float, images: list, events: list[dict] | None = None) -> StepResult:
        gap = i - self.last_i if self.last_i is not None else 1
        self.last_i = i
        quality = [self.meters[k].update(t, img, (0.0, 0.0), measure=i % self.step == 0, frames_since=gap)
                   for k, img in enumerate(images)]
        if i % self.step == 0 or not self.cards:
            self.known_events = self.feed.known_events(t)
            self.updates = self.feed.known_updates(t)
            self.phase = self.feed.latest_phase(t, self.cfg["director"]["reaction_max_s"])
            self.story_line = self.updates[-1]["narration"] if self.updates else ""
            headline = [e for e in self.known_events if e["type"] not in ("none", "whistle")]
            latest = max(headline, key=lambda e: e["known_at"]) if headline else None
            st = self.pitch_state
            status = {"seen": "detected", "remembered": "predicted"}.get(st.ball_status, "lost") if st else "lost"
            speed = min(1.0, st.speed_ms / self.cfg["map"]["fast_ball_ms"]) if st else 0.0
            cards = []
            for k, cam in enumerate(self.cams["cams"]):
                c = cam["cam"]
                spot = self.in_picture(c, st.ball if st and status != "lost" else None)
                ball = BallInfo(spot[0], spot[1], status) if spot else None
                action = self.in_picture(c, self.group)
                cards.append(CamCard(cam=c, name=cam["name"], time=round(t, 2), logged_at=round(t, 2),
                                     shot_type=cam["shot_type"], ball=ball, players=self.players.get(c, 0),
                                     play_speed=round(speed, 2), quality=quality[k], event=event_summary(latest),
                                     phase=self.phase, source="detector", action_in_view=action is not None,
                                     action=action, zone=st.zone if st else "unknown",
                                     map_intensity=st.intensity if st else 0.0))
            self.cards = cards
        decision = self.director.decide(i, t, self.cards, self.known_events, self.updates, self.phase)
        return StepResult(decision, self.cards, [None] * len(self.cams["cams"]), None, events or [], self.story_line)
