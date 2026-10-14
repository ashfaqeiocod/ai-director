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

MVP 4 adds a "tv" mode (config `mode: tv`, used by the virtual stadium; without it nothing changes): TV cutting rules
over a T2 camera set with robotic cameras, driven by the pitch map's game situation (see `_tv`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.ball_tracker import BallState
from src.director import score_camera
from studio.cards import CamCard

IN_PLAY, REACTION = "IN_PLAY", "REACTION"
PLAY, ATTACK, STOPPAGE, RESTART = "PLAY", "ATTACK", "STOPPAGE", "RESTART"     # tv mode (MVP 4)
# Camera roles on SoccerTrack (the defaults). Other footage sets them in its own director settings:
# follow_cams (scored like MVP 1), wide_cam / close_cam (null = this footage has no such camera).
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
        self.challenger_since = 0
        self.reaction_start: float | None = None
        self.reaction_reason = ""
        self.handled: set[str] = set()            # story events already reacted to
        self.follow_cams = tuple(cfg.get("follow_cams", FOLLOW_CAMS))
        self.wide_cam = cfg.get("wide_cam", WIDE)
        self.close_cam = cfg.get("close_cam", CLOSE)
        if cfg.get("mode") == "tv":                # MVP 4: the game situation the TV rules follow
            self.state = PLAY
            self.state_since = 0.0
            self.situation = ""                     # why we are in this state (for the reasons)
            self.attack_side: str | None = None
            self.out_of_box_since: float | None = None
            self.stoppage_kind = ""
            self.last_map_t: float | None = None
            self.lost_reported = False
            self.last_seen_t: float | None = None   # the map last SAW the ball (its own age resets once it is lost)
            self.attack_over_t = -1e9               # when the last attack shot ended (cooldown)

    def start_at(self, i: int) -> None:
        """A live run that starts at frame i (not 0): the first shot starts there."""
        self.shot_start = self.challenger_since = i

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
            # 180-degree rule: a camera on the OTHER stand flips left and right for the viewer, so it must be
            # clearly better before we cut across (cameras -> stand in config; footage without stands: no rule).
            stands = c.get("stands") or {}
            if stands and stands.get(card.cam) != stands.get(self.current):
                bonus -= c.get("cross_line_penalty", 0)
            if card.cam == self.current:
                bonus += c.get("live_bonus", 0)   # a challenger must be clearly better, not just a little
            # Danger: the pitch map says play is hot near a goal -> that goal's cameras are the shot to have
            # (config danger_cams: zone -> cameras; footage without it: no rule).
            if card.map_intensity >= c.get("danger_min_intensity", 2.0) and card.cam in (c.get("danger_cams") or {}).get(card.zone, []):
                bonus += c["danger_bonus"]
            if card.cam in self.follow_cams:
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
            elif card.cam == self.wide_cam:
                score = c["wide_base"] + (c["wide_fast_bonus"] if card.play_speed >= c["fast_play"] else 0)
                out.append(float(score))
            elif card.cam == self.close_cam:
                slow = card.play_speed <= c["slow_play"] and card.action_in_view
                # The ball is in a penalty box (pitch map) and the close-up shows it: the danger moment.
                box = card.zone in ("left_box", "right_box") and card.ball is not None \
                    and card.map_intensity >= c["box_min_intensity"]
                out.append(float(c["close_box_score"] if box else c["close_slow_score"] if slow else 0))
            else:
                out.append(0.0)
        return out

    # ------------------------------------------------------------------ decide
    def decide(self, i: int, t: float, cards: list[CamCard], story_events: list[dict],
               story_updates: list[dict], phase: str | None, pitch=None) -> Decision:
        """pitch: the pitch map's latest MapState (tv mode only)."""
        if self.cfg.get("mode") == "tv":
            return self._tv(i, t, cards, pitch)
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
        close = self.close_cam
        has_close = close is not None and self.num_cams >= close and usable[close - 1]   # no close-up: best camera
        target = close if has_close else max(range(self.num_cams), key=lambda k: scores[k]) + 1
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
        # Counted in FRAMES (time), not in calls: live mode skips frames when the Mac is busy, and counting calls
        # made the 0.6 s wait last 1.2 s at 13 fps shown (and 15 s at 1 fps).
        if best == self.challenger:
            self.challenger_frames = i - self.challenger_since + 1
        else:
            self.challenger, self.challenger_frames, self.challenger_since = best, 1, i
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


    # ------------------------------------------------------------------ MVP 4: TV rules (mode: tv)
    def _tv(self, i: int, t: float, cards: list[CamCard], pitch) -> Decision:
        """TV cutting rules for the T2 camera set (docs/MVP4_STADIUM_PLAN.md Step 5), as a small state machine:
          PLAY      MAIN (the robotic camera at halfway that follows the ball): the default, ~70% of airtime
          ATTACK    the ball is in a penalty box and play is hot: GOAL L/R (behind that goal) if its card shows
                    the action, else BOX L/R; back to MAIN once the ball has left the box for a moment
          STOPPAGE  the ball went out / play stopped: CLOSE (the robotic close-up) for 2-4 s, then MAIN
          RESTART   a kick-off line-up, or a goal kick after the ball went over the goal line: WIDE until the
                    ball moves again
        Always: hold every shot `min_shot_seconds`; never cut TO a robotic camera that is still turning fast
        (`settle_deg_s`: a real director waits for the operator to frame the shot)."""
        c = self.cfg
        role = c["roles"]
        self._tv_situation(t, pitch)
        target, why = self._tv_target(t, cards)
        scores = [100.0 if card.cam == target else 0.0 for card in cards]
        events: list[dict] = []
        if target == self.current:
            return Decision(self.current, self.state, scores, f"{self.state.lower()}: holding {why}", events)
        if i - self.shot_start < self.min_shot:
            return Decision(self.current, self.state, scores, f"{why}: waiting (min shot length)", events)
        card = next((k for k in cards if k.cam == target), None)
        if card is not None and target in (role["main"], role["close"]) and card.moving_deg_s > c["settle_deg_s"]:
            return Decision(self.current, self.state, scores,
                            f"{why}: waiting for cam{target} to settle ({card.moving_deg_s:.0f} deg/s)", events)
        reason = self._cut(target, i, why, events)
        return Decision(self.current, self.state, scores, reason, events)

    def _tv_enter(self, state: str, t: float, situation: str) -> None:
        self.state, self.state_since, self.situation = state, t, situation

    def _tv_situation(self, t: float, pitch) -> None:
        """Update the game situation: the map's events when it has news (5 times a second), the timers every frame."""
        c = self.cfg
        if pitch is None:
            return
        fresh = pitch.t != self.last_map_t
        self.last_map_t = pitch.t
        for e in pitch.events if fresh else []:
            if e["type"] == "KICKOFF_LINEUP":
                self._tv_enter(RESTART, t, "kick-off")
            elif e["type"] in ("BALL_OUT", "GOAL_LINE") and self.state != RESTART:
                self.stoppage_kind = "goal line" if e["type"] == "GOAL_LINE" else e.get("line", "touchline")
                self._tv_enter(STOPPAGE, t, f"ball out over the {self.stoppage_kind}")
        seen = pitch.ball is not None and pitch.ball_status != "lost"
        if pitch.ball_status == "seen":
            self.last_seen_t = pitch.t
            self.lost_reported = False
        unseen = t - self.last_seen_t if self.last_seen_t is not None else 0.0
        if unseen >= c["stoppage_lost_s"] and not self.lost_reported and self.state in (PLAY, ATTACK):
            self.lost_reported = True                  # no ball for a while: play has (most likely) stopped
            self.stoppage_kind = "lost"
            self._tv_enter(STOPPAGE, t, f"no ball for {c['stoppage_lost_s']}s")
        in_box = seen and pitch.zone in ("left_box", "right_box")
        if (self.state == PLAY and in_box and pitch.intensity >= c["attack_min_intensity"]
                and t - self.attack_over_t >= c["attack_cooldown_s"]):
            self.attack_side = "left" if pitch.zone == "left_box" else "right"
            self.out_of_box_since = None
            self._tv_enter(ATTACK, t, f"attack in the {self.attack_side} box")
        elif self.state == ATTACK:
            if in_box:
                self.out_of_box_since = None
            elif self.out_of_box_since is None:
                self.out_of_box_since = t
            if self.out_of_box_since is not None and t - self.out_of_box_since >= c["attack_exit_s"]:
                self.attack_over_t = t
                self._tv_enter(PLAY, t, "ball left the box")
            elif t - self.state_since >= c["attack_max_s"]:
                self.attack_over_t = t                 # goal-camera shots are short on TV: back to MAIN
                self._tv_enter(PLAY, t, "attack shot long enough")
        elif self.state == STOPPAGE:
            held = t - self.state_since
            moving = seen and pitch.speed_ms >= c["restart_move_ms"]
            if held >= c["close_max_s"] or (moving and held >= c["close_min_s"]):
                if self.stoppage_kind == "goal line":
                    self._tv_enter(RESTART, t, "goal kick")
                else:
                    self._tv_enter(PLAY, t, "play on")
        elif self.state == RESTART:
            held = t - self.state_since
            if held >= c["restart_max_s"] or (seen and pitch.speed_ms >= c["restart_move_ms"] and held >= c["close_min_s"]):
                self._tv_enter(PLAY, t, "ball in play again")

    def _tv_target(self, t: float, cards: list[CamCard]) -> tuple[int, str]:
        role = self.cfg["roles"]
        if self.state == STOPPAGE:
            return role["close"], f"CLOSE after a stoppage ({self.situation})"
        if self.state == RESTART:
            return role["wide"], f"WIDE for the {self.situation}"
        if self.state == ATTACK:
            goal = role["goall" if self.attack_side == "left" else "goalr"]
            box = role["boxl" if self.attack_side == "left" else "boxr"]
            card = next((k for k in cards if k.cam == goal), None)
            if card is not None and (card.ball is not None or card.action_in_view):
                return goal, f"GOAL {self.attack_side[0].upper()}: {self.situation}, action in its view"
            return box, f"BOX {self.attack_side[0].upper()}: {self.situation}"
        return role["main"], "MAIN follows play"
