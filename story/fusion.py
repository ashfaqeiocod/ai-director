"""fusion.py — merges what all layers say into ONE timeline, intensity and score.

- Events: the same event type from any layer within the dedupe window (story.fusion.dedupe_window_s,
  longer for goals) is ONE event. Confidences are combined ("noisy-or": two 60% sources -> 84%) and
  every source is kept (e.g. sources = ["specialist", "gemini"]).
- Replays: while the picture is a replay, new events are ignored (no double goals).
- Score: a goal counts once its fused confidence reaches goal_min_confidence (not during replays).
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
    members: list[Event] = field(default_factory=list)

    @property
    def headline(self) -> bool:
        return self.type in HEADLINE_TYPES

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "event_time": round(self.event_time, 2),
                "logged_at": round(self.logged_at, 2), "delay": round(self.logged_at - self.event_time, 2),
                "confidence": round(self.confidence, 3), "sources": self.sources, "team": self.team,
                "note": self.note, "headline": self.headline}


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
    def add_event(self, event: Event, is_replay: bool) -> tuple[FusedEvent | None, bool]:
        """Merge one event from a layer. Returns (fused event, is_new) or (None, False) if ignored."""
        if event.type == "none":
            return None, False
        if is_replay and event.source != "manual":
            self.ignored_in_replay += 1
            return None, False
        window = self.cfg["dedupe_overrides"].get(event.type, self.cfg["dedupe_window_s"])
        for fused in reversed(self.events):
            if fused.type == event.type and abs(fused.event_time - event.event_time) <= window:
                self._merge(fused, event)
                return fused, False
        fused = FusedEvent(id=f"e{next(self._ids)}", type=event.type, event_time=event.event_time,
                           logged_at=event.logged_at, confidence=event.confidence, sources=[event.source],
                           team=event.team, note=event.note, members=[event])
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
        if fused.type != "goal" or fused.counted or fused.confidence < self.cfg["goal_min_confidence"]:
            return False
        fused.counted = True
        team = fused.team or "?"
        self.score[team] = self.score.get(team, 0) + 1
        return True

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
