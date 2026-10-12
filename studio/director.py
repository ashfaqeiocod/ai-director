"""director.py — MVP 3 Step 5: director v2, a small state machine over the camera cards.

States (plan section 6):
  IN_PLAY    follow the ball (or the main group of players) between LEFT / CENTRE / RIGHT with MVP 1's
             scoring. Only `on_air_cams` can be chosen: WIDE (the whole panorama) is too small to watch.
  REACTION   the story layer reported a goal (corroborated) or a celebration / foul / injury: show the
             CLOSE-up for a while, then go back to IN_PLAY.
Always:
  - never air a camera whose picture is unusable (blurry / shaking); if the live one goes bad, leave it at
    once (the only cut allowed to break the minimum shot length);
  - hold every shot at least `min_shot_seconds`; another camera must win for `hysteresis_seconds` first.

Every decision carries a plain-English reason (director_log.csv).
All numbers: config.yaml -> studio.director.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.ball_tracker import BallState
from src.director import score_camera
from studio.cards import CamCard

IN_PLAY, REACTION = "IN_PLAY", "REACTION"
FOLLOW_CAMS = (1, 2, 3)
WIDE, CLOSE = 4, 5


@dataclass
class Decision:
    cam: int
    state: str
    scores: list[float]
    reason: str
    events: list[dict] = field(default_factory=list)


class DirectorV2:
    def __init__(self, cfg: dict, fps: float, num_cams: int = 5):
        self.cfg = cfg
        self.fps = fps
        self.num_cams = num_cams
        self.min_shot = round(cfg["min_shot_seconds"] * fps)
        self.hysteresis = max(1, round(cfg["hysteresis_seconds"] * fps))
        self.current = cfg["start_cam"]
        self.state = IN_PLAY
        self.shot_start = 0
        self.challenger: int | None = None
        self.challenger_frames = 0
        self.reaction_start: float | None = None
        self.reaction_reason = ""
        self.handled: set[str] = set()            # story events already reacted to

    # ------------------------------------------------------------------ scores
    def scores(self, cards: list[CamCard]) -> list[float]:
        """One score per camera for IN_PLAY (unusable cameras get -1)."""
        c = self.cfg
        out = []
        for card in cards:
            if not card.quality.usable or card.cam not in c["on_air_cams"]:
                out.append(-1.0)             # never chosen
                continue
            if card.cam != self.current and card.quality.shake >= c.get("avoid_shake", 1.01):
                out.append(-1.0)             # already shaking hard: don't cut TO it (staying on it is allowed)
                continue
            bonus = c.get("cam_bonus", {}).get(card.cam, 0)   # a preferred viewing angle (set per venue)
            if card.cam in FOLLOW_CAMS:
                # MVP 1 scoring, on this camera's own ball status and players.
                if card.ball:
                    ball = BallState(card.ball.status, (card.ball.x * 1000, card.ball.y * 1000))
                elif card.action and c["action_as_ball"]:
                    # No ball seen (the detector finds it in only ~16% of frames; the oracle has none): the
                    # action point (main group of players) stands in, scored like a predicted ball, so a camera
                    # centred on the play keeps the shot instead of falling back to WIDE every few seconds.
                    ball = BallState("predicted", (card.action[0] * 1000, card.action[1] * 1000))
                else:
                    ball = BallState("lost", None)
                out.append(score_camera(ball, card.players, 1000, c["mvp1"]) + bonus)
            elif card.cam == WIDE:
                score = c["wide_base"] + (c["wide_fast_bonus"] if card.play_speed >= c["fast_play"] else 0)
                out.append(float(score))
            else:                                   # CLOSE
                slow = card.play_speed <= c["slow_play"] and card.action_in_view
                # The ball is in a penalty box (pitch map) and the close-up shows it: the danger moment.
                box = card.zone in ("left_box", "right_box") and card.ball is not None \
                    and card.map_intensity >= c["box_min_intensity"]
                out.append(float(c["close_box_score"] if box else c["close_slow_score"] if slow else 0))
        return out

    # ------------------------------------------------------------------ decide
    def decide(self, i: int, t: float, cards: list[CamCard], story_events: list[dict],
               story_updates: list[dict], phase: str | None) -> Decision:
        scores = self.scores(cards)
        # "usable" = a picture we may air: sharp, steady, and one of the on-air cameras.
        usable = [card.quality.usable and card.cam in self.cfg["on_air_cams"] for card in cards]
        events: list[dict] = []
        can_cut = i - self.shot_start >= self.min_shot

        # 1. A bad picture on air: leave it now, whatever the shot length.
        if not usable[self.current - 1]:
            best = max((k for k in range(self.num_cams) if usable[k]), key=lambda k: scores[k], default=None)
            if best is not None:
                reason = self._cut(best + 1, i, f"cam{self.current} picture unusable (blur/shake)", events)
                return Decision(self.current, self.state, scores, reason, events)

        # 2. Story: start a REACTION?
        trigger = self._reaction_trigger(t, story_events, story_updates, phase)
        if trigger and self.state == IN_PLAY:
            self.state, self.reaction_start, self.reaction_reason = REACTION, t, trigger
            events.append({"type": "REACTION", "message": f"reaction: {trigger}"})

        if self.state == REACTION:
            return self._react(i, t, scores, usable, can_cut, phase, events)
        return self._follow(i, scores, can_cut, events)

    def _reaction_trigger(self, t: float, story_events: list[dict], updates: list[dict], phase: str | None) -> str | None:
        c = self.cfg
        for e in story_events:
            if e["id"] in self.handled or t - e["event_time"] > c["reaction_max_age_s"]:
                continue
            if e["type"] == "goal" and self.goal_is_corroborated(e, story_events, updates):
                self.handled.add(e["id"])
                return f"goal at {e['event_time']:.0f}s ({e['confidence']:.2f}, {'+'.join(e['sources'])})"
            if e["type"] in c["reaction_events"] and e["confidence"] >= c["reaction_min_confidence"]:
                self.handled.add(e["id"])
                return f"{e['type']} at {e['event_time']:.0f}s"
        if phase in c["reaction_phases"] and self.state == IN_PLAY:
            key = f"phase-{phase}-{int(t // c['reaction_max_s'])}"
            if key not in self.handled:
                self.handled.add(key)
                return f"phase {phase}"
        return None

    def goal_is_corroborated(self, goal: dict, events: list[dict], updates: list[dict]) -> bool:
        """No scoreboard on this footage, so one AI look saying "goal" is not enough (MVP 2 saw invented goals).

        Count independent looks near the goal: story updates with phase goal/celebration or a goal event,
        plus any specialist goal/shot. Need `goal_min_looks` of them and enough confidence.
        """
        c = self.cfg
        if goal["confidence"] < c["goal_min_confidence"]:
            return False
        near = lambda x: abs(x - goal["event_time"]) <= c["goal_corroborate_s"]   # noqa: E731
        looks = sum(1 for u in updates if near(u["window_end"])
                    and (u["phase"] in ("goal", "celebration") or any(e["type"] == "goal" for e in u["events"])))
        # A shot seen just before (by the specialist, or by the goal camera AI) is independent evidence too.
        looks += sum(1 for e in events if near(e["event_time"]) and (
            ("specialist" in e["sources"] and e["type"] in ("goal", "shot"))
            or ("goalcam" in e["sources"] and e["type"] == "shot")))
        return looks >= c["goal_min_looks"]

    def _react(self, i: int, t: float, scores: list[float], usable: list[bool], can_cut: bool,
               phase: str | None, events: list[dict]) -> Decision:
        c = self.cfg
        held = t - self.reaction_start
        play_again = phase in c["play_phases"] and held >= c["reaction_hold_s"]
        if held >= c["reaction_max_s"] or play_again:
            self.state = IN_PLAY
            events.append({"type": "REACTION_END", "message": "back to play" if play_again else "reaction time over"})
            return self._follow(i, scores, can_cut, events)
        # The close-up; if it is unusable, the best on-air camera instead.
        has_close = self.num_cams >= CLOSE and usable[CLOSE - 1]   # footage without a close-up: best camera
        target = CLOSE if has_close else max(range(self.num_cams), key=lambda k: scores[k]) + 1
        if self.current == target:
            return Decision(self.current, self.state, scores, f"reaction: holding cam{target} ({self.reaction_reason})", events)
        if not can_cut:
            return Decision(self.current, self.state, scores, "reaction: waiting (min shot length)", events)
        reason = self._cut(target, i, f"reaction to {self.reaction_reason}", events)
        return Decision(self.current, self.state, scores, reason, events)

    def _follow(self, i: int, scores: list[float], can_cut: bool, events: list[dict]) -> Decision:
        best = max(range(self.num_cams), key=lambda k: scores[k]) + 1
        if best == self.current or scores[best - 1] <= scores[self.current - 1]:
            self.challenger, self.challenger_frames = None, 0
            return Decision(self.current, self.state, scores, "live cam has the best score", events)
        if best == self.challenger:
            self.challenger_frames += 1
        else:
            self.challenger, self.challenger_frames = best, 1
        if self.challenger_frames < self.hysteresis:
            return Decision(self.current, self.state, scores,
                            f"cam{best} better, waiting (hysteresis {self.challenger_frames}/{self.hysteresis})", events)
        if not can_cut:
            return Decision(self.current, self.state, scores, f"cam{best} better, waiting (min shot length)", events)
        reason = self._cut(best, i, f"cam{best} scored higher for {self.cfg['hysteresis_seconds']}s", events)
        return Decision(self.current, self.state, scores, reason, events)

    def _cut(self, cam: int, i: int, reason: str, events: list[dict]) -> str:
        events.append({"type": "CUT", "message": f"cam{self.current} -> cam{cam} ({reason})", "from": self.current,
                       "to": cam, "shot_seconds": round((i - self.shot_start) / self.fps, 2)})
        self.current, self.shot_start = cam, i
        self.challenger, self.challenger_frames = None, 0
        return "CUT: " + reason
