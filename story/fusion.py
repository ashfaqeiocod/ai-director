"""fusion.py — merges what all layers say into ONE timeline, intensity and score.

- Events: the same event type from any layer within the dedupe window (story.fusion.dedupe_window_s,
  longer for goals) is ONE event. Confidences are combined ("noisy-or": two 60% sources -> 84%) and
  every source is kept (e.g. sources = ["specialist", "gemini"]).
- Replays: while the picture is a replay, new events are ignored (no double goals).
- Score: a goal counts once its fused confidence reaches goal_min_confidence (not during replays).
  The on-screen scoreboard (as read by Gemini) is the referee: a goal while the scoreboard still shows
  the old score waits ("pending") and is confirmed when the scoreboard goes up, or dropped after
  goal_confirm_s if it never does. Without a visible scoreboard, goals count straight away.
  The dashboard's +1 / -1 buttons override it.
- Intensity (0..100): a weighted mix of audio excitement, motion, recent specialist events and
  the latest Gemini reading (weights in config.yaml story.fusion.intensity_weights). Layers that
  aren't running in this mode are left out and the remaining weights are rescaled, so fast_only
  still uses the full 0..100 range.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from story.fast_layer import FastResult
from story.schema import HEADLINE_TYPES, Event

# Which layers each mode runs (the fast layer always runs).
MODE_LAYERS = {
    "fast_only": {"specialist": False, "gemini": False},
    "specialist_only": {"specialist": True, "gemini": False},
    "gemini_only": {"specialist": False, "gemini": True},
    "hybrid": {"specialist": True, "gemini": True},
}


@dataclass
class FusedEvent:
    id: str
    type: str
    event_time: float
    logged_at: float                    # when the FIRST layer reported it (honest delay)
    confidence: float
    sources: list[str]
    team: str | None = None
    note: str | None = None
    counted: bool = False               # goal already added to the score
    pending: bool = False               # goal waiting for the scoreboard to go up
    rejected: bool = False              # goal the scoreboard never confirmed (dropped)
    members: list[Event] = field(default_factory=list)

    @property
    def headline(self) -> bool:
        return self.type in HEADLINE_TYPES

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "event_time": round(self.event_time, 2),
                "logged_at": round(self.logged_at, 2), "delay": round(self.logged_at - self.event_time, 2),
                "confidence": round(self.confidence, 3), "sources": self.sources, "team": self.team,
                "note": self.note, "headline": self.headline and not self.rejected,
                "pending": self.pending, "rejected": self.rejected}


class Fusion:
    def __init__(self, cfg: dict, mode: str):
        self.cfg = cfg["fusion"]
        self.mode = mode
        self.layers = MODE_LAYERS[mode]
        self.events: list[FusedEvent] = []
        self._ids = itertools.count(1)
        self.score: dict[str, int] = {}          # team colour -> goals
        self.ignored_in_replay = 0
        self.llm_reading: tuple[float, float] | None = None       # (playhead when logged, intensity 0..1)
        self.specialist_level: tuple[float, float] | None = None  # (playhead when logged, strength 0..1)

    # ------------------------------------------------------------------ intensity
    def note_llm_intensity(self, logged_at: float, intensity: int) -> None:
        self.llm_reading = (logged_at, intensity / 100)

    def note_specialist(self, logged_at: float, strength: float) -> None:
        if self.specialist_level is None or strength >= self._faded(self.specialist_level, logged_at,
                                                                   self.cfg["specialist_hold_s"]):
            self.specialist_level = (logged_at, strength)

    @staticmethod
    def _faded(reading: tuple[float, float] | None, now: float, hold_s: float) -> float:
        """A reading loses strength linearly and is gone after hold_s seconds."""
        if reading is None:
            return 0.0
        at, value = reading
        return value * max(0.0, 1 - (now - at) / hold_s)

    def intensity(self, fast: FastResult) -> int:
        w = self.cfg["intensity_weights"]
        parts = {"audio": (w["audio"], fast.audio), "motion": (w["motion"], fast.motion)}
        if self.layers["specialist"]:
            parts["specialist"] = (w["specialist"],
                                   self._faded(self.specialist_level, fast.t, self.cfg["specialist_hold_s"]))
        if self.layers["gemini"] and self.llm_reading is not None:
            parts["llm"] = (w["llm"], self._faded(self.llm_reading, fast.t, self.cfg["llm_intensity_hold_s"]))
        total_weight = sum(weight for weight, _ in parts.values())
        value = sum(weight * v for weight, v in parts.values()) / total_weight
        return int(round(100 * min(1.0, value)))

    # ------------------------------------------------------------------ events
    def add_event(self, event: Event, is_replay: bool, force_new: bool = False,
                  scoreboard: str | None = None) -> tuple[FusedEvent | None, bool]:
        """Merge one event from a layer. Returns (fused event, is_new) or (None, False) if ignored.

        force_new: never merge. scoreboard: the score read in the same snippet (goals only): a higher
        total than the goals counted means a NEW goal; the same total makes a new goal "pending"."""
        if event.type == "none":
            return None, False
        if is_replay and event.source != "manual":
            self.ignored_in_replay += 1
            return None, False
        if event.type == "goal" and self.scoreboard_says_new_goal(scoreboard):
            force_new = True
        if force_new and event.type == "goal" and any(
                f.type == "goal" and abs(f.event_time - event.event_time) < self.cfg["goal_min_gap_s"] for f in self.events):
            # Two real goals need a kick-off in between: this close, it's the same goal seen twice
            # (overlapping looks), even if Gemini misread the scoreboard.
            force_new = False
        window = 0.0 if force_new else self.cfg["dedupe_overrides"].get(event.type, self.cfg["dedupe_window_s"])
        for fused in reversed(self.events):
            if fused.rejected:
                continue                     # a dropped goal must not swallow a real one
            if fused.type == event.type and abs(fused.event_time - event.event_time) <= window:
                self._merge(fused, event)
                return fused, False
        fused = FusedEvent(id=f"e{next(self._ids)}", type=event.type, event_time=event.event_time,
                           logged_at=event.logged_at, confidence=event.confidence, sources=[event.source],
                           team=event.team, note=event.note, members=[event])
        # A visible scoreboard that did NOT go up yet: wait for it before counting this goal.
        total = scoreboard_total(scoreboard)
        fused.pending = (event.type == "goal" and event.source != "manual" and total is not None
                         and total <= self.goals_counted())
        self.events.append(fused)
        return fused, True

    @staticmethod
    def _merge(fused: FusedEvent, event: Event) -> None:
        if event.confidence > max(m.confidence for m in fused.members):
            fused.event_time = event.event_time          # the surest source gives the time
            fused.note = event.note or fused.note
        if event.source not in fused.sources:
            # Independent sources agreeing: combine ("noisy-or").
            fused.confidence = 1 - (1 - fused.confidence) * (1 - event.confidence)
            fused.sources.append(event.source)
        else:
            fused.confidence = max(fused.confidence, event.confidence)
        fused.team = fused.team or event.team
        fused.logged_at = min(fused.logged_at, event.logged_at)
        fused.members.append(event)

    def recent_events(self, now: float, seconds: float) -> list[dict]:
        return [e.to_dict() for e in self.events if now - e.event_time <= seconds]

    # ------------------------------------------------------------------ score
    def count_goal(self, fused: FusedEvent) -> bool:
        """Add a goal to the score once it is sure enough. Returns True if the score changed."""
        if fused.type != "goal" or fused.counted or fused.pending or fused.rejected \
                or fused.confidence < self.cfg["goal_min_confidence"]:
            return False
        fused.counted = True
        team = fused.team or "?"
        self.score[team] = self.score.get(team, 0) + 1
        return True

    def goals_counted(self) -> int:
        return sum(self.score.values())

    def scoreboard_says_new_goal(self, scoreboard: str | None) -> bool:
        """The scoreboard is the referee: "2-0" (total 2) while we counted 1 goal = a NEW goal,
        even if it comes soon after the last one (highlights compress time). Same total = a replay."""
        total = scoreboard_total(scoreboard)
        return total is not None and total > self.goals_counted()

    def review_pending_goals(self, scoreboard: str | None, now: float) -> list[tuple[FusedEvent, str]]:
        """Called after every Gemini look. Returns [(goal, "confirmed" | "rejected")]."""
        total = scoreboard_total(scoreboard)
        changes = []
        for fused in self.events:
            if fused.type != "goal" or not fused.pending:
                continue
            if total is not None and total > self.goals_counted():
                fused.pending = False                 # the scoreboard went up: it was real
                if self.count_goal(fused):
                    changes.append((fused, "confirmed"))
            elif total is not None and now - fused.logged_at >= self.cfg["goal_confirm_s"]:
                fused.pending, fused.rejected = False, True
                changes.append((fused, "rejected"))
        return changes

    def adjust_score(self, team: str, delta: int) -> None:
        """Manual +1 / -1 from the dashboard (always wins)."""
        self.score[team] = max(0, self.score.get(team, 0) + delta)

    def add_team(self, team: str) -> None:
        """Make a known team show in the score (0 goals) and get +1/-1 buttons."""
        if team and team not in self.score and len(self.score) < 2:
            self.score[team] = 0

    def score_text(self) -> str | None:
        if not self.score:
            return None
        return " – ".join(f"{team} {goals}" for team, goals in self.score.items())

    def score_message(self) -> dict:
        return {"type": "score", "teams": self.score, "teams_order": list(self.score), "text": self.score_text()}


def scoreboard_total(scoreboard: str | None) -> int | None:
    """"2-1" / "2 - 1" / "2:1" -> 3. None if there is no readable score."""
    import re

    m = re.fullmatch(r"\s*(\d{1,2})\s*[-–:]\s*(\d{1,2})\s*", scoreboard or "")
    return int(m.group(1)) + int(m.group(2)) if m else None
