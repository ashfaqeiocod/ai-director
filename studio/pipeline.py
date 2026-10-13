"""pipeline.py — the per-frame "brain" of MVP 3, shared by the offline run (studio.run) and live mode (studio.live).

For every frame the Pipeline takes the camera pictures (and, 5 times a second, new detections) and returns the
director's decision. On the way it keeps:
  World         ball + players + action point in panorama pixels
  PitchMap      ball + players on the pitch, zone triggers, kick-off line-up
  QualityMeter  blur / shake per camera
  cards         one CamCard per camera
  goal checks   a goal the goal camera reported, confirmed when a kick-off line-up follows
  DirectorV2    the camera choice

Everything uses only the current frame and the past (causal). The story comes in through `feed` (anything with
known_events / known_updates / latest_phase: the recorded StoryFeed offline, a LiveFeed in live mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from studio.cameras import CLOSE_CAM, cam_rect
from studio.cards import BallInfo, CamCard, QualityMeter, World
from studio.director import Decision, DirectorV2
from studio.pitchmap import MapState, PitchMap, Projection
from studio.video import to_camera


def inside(point, rect) -> bool:
    x, y, w, h = rect
    return point is not None and x <= point[0] < x + w and y <= point[1] < y + h


def build_cards(t: float, cams: dict, rects: list, world, quality: list, phase, event, source: str,
                pitch: MapState | None = None) -> list[CamCard]:
    w, h = cams["size"]
    cards = []
    for k, cam in enumerate(cams["cams"]):
        rect = rects[k]
        ball = None
        if world.ball is not None and inside(world.ball, rect):
            x, y = to_camera(rect, w, h, world.ball)
            ball = BallInfo(round(x / w, 3), round(y / h, 3), world.ball_status)
        players = sum(1 for p in world.players if inside(p, rect))
        action = None
        if inside(world.action, rect):
            ax, ay = to_camera(rect, w, h, world.action)
            action = (round(ax / w, 3), round(ay / h, 3))
        cards.append(CamCard(cam=cam["cam"], name=cam["name"], time=round(t, 2), logged_at=round(t, 2),
                             shot_type=cam["shot_type"], ball=ball, players=players,
                             play_speed=round(world.play_speed, 2), quality=quality[k], event=event, phase=phase,
                             source=source, action_in_view=action is not None, action=action,
                             zone=pitch.zone if pitch else "unknown",
                             map_intensity=pitch.intensity if pitch else 0.0))
    return cards


def event_summary(e: dict | None) -> dict | None:
    if not e:
        return None
    return {"type": e["type"], "event_time": e["event_time"], "known_at": round(e["known_at"], 2),
            "confidence": round(e["confidence"], 2), "sources": e["sources"]}


@dataclass
class StepResult:
    decision: Decision
    cards: list[CamCard]
    rects: list
    pitch: MapState | None = None                # new pitch-map state (detection steps only)
    events: list[dict] = field(default_factory=list)   # map triggers + GOAL_CONFIRMED, for logs and the page
    story_line: str = ""


class Pipeline:
    def __init__(self, cfg: dict, cams: dict, meta: dict, close_rects: np.ndarray, pitch_to_image: np.ndarray,
                 feed, source: str, step: int):
        self.cfg, self.cams, self.feed, self.source, self.step = cfg, cams, feed, source, step
        self.fps = meta["fps"]
        self.close_rects = close_rects
        self.world = World(cfg, (meta["width"], meta["height"]), self.fps, step)
        self.projection = Projection(pitch_to_image)
        self.pitch = PitchMap(cfg["map"], self.projection)
        self.meters = [QualityMeter(cfg["quality"]) for _ in cams["cams"]]
        self.director = DirectorV2(cfg["director"], self.fps, len(cams["cams"]))
        self.confirmed: dict[str, dict] = {}     # goal id -> when the kick-off line-up confirmed it
        self.cards: list[CamCard] = []
        self.pitch_state: MapState | None = None
        self.known_events: list[dict] = []
        self.updates: list[dict] = []
        self.phase = None
        self.story_line = ""
        self.last_i: int | None = None
        self.fresh = False                            # detections arrived since the last frame()

    def rects(self, i: int) -> list:
        return [cam_rect(self.cams, c["cam"], i, self.close_rects) for c in self.cams["cams"]]

    # ------------------------------------------------------------------ detections (5x per second)
    def detections(self, i: int, t: float, detector: dict | None = None, oracle: tuple | None = None) -> list[dict]:
        """New detections for frame i. detector = PanoramaDetections.at(i); oracle = OraclePlayers.at(i).
        Returns the events this caused (map triggers, GOAL_CONFIRMED)."""
        self.fresh = True
        cfg = self.cfg
        if oracle is not None:
            players, group, feet = oracle
            self.pitch_state = self.pitch.update(t, None, feet)
            self.world.update(t, None, players, group)
        else:
            d = detector
            self.pitch_state = self.pitch.update(t, d["ball"][:2] if d["ball"] else None, d["feet"])
            if cfg["map"]["drive_ball"]:
                # The map's ball (seen, or remembered for a moment) is THE ball for every camera.
                st = self.pitch_state
                ball_px = self.projection.image(*st.ball) if st.ball else None
                status = {"seen": "detected", "remembered": "predicted"}.get(st.ball_status, "lost")
                self.world.update(t, ball_px, d["players"], ball_status=status)
            else:
                self.world.update(t, d["ball"][:2] if d["ball"] else None, d["players"])
        events = [{"type": e["type"],
                   "message": f"{e['type'].lower().replace('_', ' ')} ({e.get('zone') or e.get('side') or e.get('line') or ''})"}
                  for e in self.pitch_state.events]
        if any(e["type"] == "KICKOFF_LINEUP" for e in self.pitch_state.events):
            events += self._confirm_goals(t)
        return events

    def _confirm_goals(self, t: float) -> list[dict]:
        """A kick-off line-up confirms a goal the goal camera reported in the last minute (past only)."""
        window = self.cfg["map"]["goal_confirm_window_s"]
        out = []
        for g in self.feed.known_events(t):
            if (g["type"] == "goal" and "goalcam" in g["sources"] and g["id"] not in self.confirmed
                    and 0 <= t - g["event_time"] <= window):
                self.confirmed[g["id"]] = {"goal_time": g["event_time"], "confirmed_at": round(t, 2)}
                out.append({"type": "GOAL_CONFIRMED", "goal_time": g["event_time"],
                            "message": f"goal at {g['event_time']:.0f}s confirmed by the kick-off line-up"})
        return out

    # ------------------------------------------------------------------ every frame
    def frame(self, i: int, t: float, images: list, events: list[dict] | None = None) -> StepResult:
        """Quality, cards (on detection steps) and the director's decision for frame i."""
        # New detections since the last frame (live mode may have skipped their own frame) or a detection frame.
        fresh, self.fresh = self.fresh or i % self.step == 0, False
        cfg, cams = self.cfg, self.cams
        w = cams["size"][0]
        rects = self.rects(i)
        gap = i - self.last_i if self.last_i is not None else 1   # live mode skips frames when it is late
        self.last_i = i
        quality = []
        for k, image in enumerate(images):
            move = (0.0, 0.0)
            if cams["cams"][k]["cam"] == CLOSE_CAM and i > 0:
                prev = cam_rect(cams, CLOSE_CAM, i - 1, self.close_rects)
                scale = w / rects[k][2]
                move = ((rects[k][0] - prev[0]) * -scale, (rects[k][1] - prev[1]) * -scale)
            quality.append(self.meters[k].update(t, image, move, measure=fresh, frames_since=gap))
        if fresh or not self.cards:
            self.known_events = self.feed.known_events(t)
            self.updates = self.feed.known_updates(t)
            self.phase = self.feed.latest_phase(t, cfg["director"]["reaction_max_s"])
            self.story_line = self.updates[-1]["narration"] if self.updates else ""
            headline = [e for e in self.known_events if e["type"] not in ("none", "whistle")]
            latest = max(headline, key=lambda e: e["known_at"]) if headline else None
            self.cards = build_cards(t, cams, rects, self.world.state, quality, self.phase,
                                     event_summary(latest), self.source, self.pitch_state)
        decision = self.director.decide(i, t, self.cards, self.known_events, self.updates, self.phase)
        return StepResult(decision, self.cards, rects, self.pitch_state, events or [], self.story_line)


class IndependentPipeline:
    """The per-frame brain for footage where each camera is its own, real camera (MUVS fans' phones).

    There is no shared panorama and no pitch calibration, so no pitch map, goal cameras or kick-off check:
      - each camera keeps MVP 1's BallTracker on its OWN detections (pixels of that camera);
      - the action point of a camera = the ball if it sees it, else its main group of players;
      - cards, quality (hand-held "jitter" mode) and DirectorV2 are the same as for SoccerTrack.
    Same interface as Pipeline (detections / frame / confirmed / pitch_state) so live mode can use either."""

    def __init__(self, cfg: dict, cams: dict, fps: float, feed, step: int, ball_cfg: dict, quality_cfg: dict):
        from src.ball_tracker import BallTracker

        self.cfg, self.cams, self.feed, self.step, self.fps = cfg, cams, feed, step, fps
        w, h = cams["size"]
        self.trackers = {c["cam"]: BallTracker(ball_cfg["buffer_size"], ball_cfg["max_jump_px"],
                                               ball_cfg["max_predict_updates"], (w, h)) for c in cams["cams"]}
        self.meters = [QualityMeter(quality_cfg) for _ in cams["cams"]]
        self.director = DirectorV2(cfg["director"], fps, len(cams["cams"]))
        self.confirmed: dict[str, dict] = {}
        self.pitch_state = None
        self.balls: dict[int, object] = {}            # cam -> BallState
        self.groups: dict[int, tuple | None] = {}     # cam -> main group of players (pixels)
        self.players: dict[int, int] = {}
        self.cards: list[CamCard] = []
        self.known_events, self.updates, self.phase, self.story_line = [], [], None, ""
        self.radius = cfg["cameras"]["close"]["group_radius_px"]
        self.thr = ball_cfg["min_confidence"]
        self.last_i: int | None = None
        self.fresh = False                            # detections arrived since the last frame()

    def detections(self, i: int, t: float, detector: dict | None = None, oracle=None) -> list[dict]:
        """detector = {cam: {"players": [...], "ball": {...} | None}} for this detection frame."""
        from studio.cameras import group_center

        self.fresh = True

        for c, det in (detector or {}).items():
            b = det.get("ball")
            box = b["box"] if b and b["conf"] >= self.thr else None
            self.balls[c] = self.trackers[c].update(box)
            feet = [((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2) for p in det.get("players", [])]
            self.players[c] = len(feet)
            pts = np.asarray(feet, dtype=float)
            self.groups[c] = group_center(pts, pts, self.radius) if len(feet) else None
        return []

    def frame(self, i: int, t: float, images: list, events: list[dict] | None = None) -> StepResult:
        from studio.cards import BallInfo

        # New detections since the last frame (live mode may have skipped their own frame) or a detection frame.
        fresh, self.fresh = self.fresh or i % self.step == 0, False
        cams = self.cams
        w, h = cams["size"]
        gap = i - self.last_i if self.last_i is not None else 1
        self.last_i = i
        quality = [self.meters[k].update(t, img, (0.0, 0.0), measure=fresh, frames_since=gap)
                   for k, img in enumerate(images)]
        if fresh or not self.cards:
            self.known_events = self.feed.known_events(t)
            self.updates = self.feed.known_updates(t)
            self.phase = self.feed.latest_phase(t, self.cfg["director"]["reaction_max_s"])
            self.story_line = self.updates[-1]["narration"] if self.updates else ""
            headline = [e for e in self.known_events if e["type"] not in ("none", "whistle")]
            latest = max(headline, key=lambda e: e["known_at"]) if headline else None
            cards = []
            for k, cam in enumerate(cams["cams"]):
                c = cam["cam"]
                b = self.balls.get(c)
                ball = BallInfo(round(b.center[0] / w, 3), round(b.center[1] / h, 3), b.status) \
                    if b is not None and b.center is not None else None
                g = self.groups.get(c)
                action = (round(g[0] / w, 3), round(g[1] / h, 3)) if g is not None else None
                cards.append(CamCard(cam=c, name=cam["name"], time=round(t, 2), logged_at=round(t, 2),
                                     shot_type=cam["shot_type"], ball=ball, players=self.players.get(c, 0),
                                     play_speed=0.0, quality=quality[k], event=event_summary(latest), phase=self.phase,
                                     source="detector", action_in_view=action is not None, action=action))
            self.cards = cards
        decision = self.director.decide(i, t, self.cards, self.known_events, self.updates, self.phase)
        return StepResult(decision, self.cards, [None] * len(cams["cams"]), None, events or [], self.story_line)
