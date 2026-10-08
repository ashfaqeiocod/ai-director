"""base.py — the StoryLLM interface: any "smart layer" provider must offer these two methods.

Gemini implements it today (gemini.py). The template narrator (template.py) implements it too and
is used when Gemini is off, missing a key or rate-limited. Swapping in another provider later
(OpenAI, a local model...) means writing one new class, nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from story.schema import Memory, StoryUpdate


@dataclass
class WindowContext:
    """Everything a provider may use to describe the last few seconds (all from the past)."""
    sport: str
    window_start: float
    window_end: float                       # = the playhead when the request was made
    video_path: Path | None = None          # small mp4 of [window_start, window_end] (Gemini only)
    memory: Memory | None = None            # latest "story so far"
    recent_events: list[dict] = field(default_factory=list)      # fused events of the last ~60 s
    specialist_hints: list[str] = field(default_factory=list)    # e.g. "SHOT at 12.4s (0.81)"
    fast: dict = field(default_factory=dict)                     # audio/motion/replay summary of the window
    teams: dict[str, str] = field(default_factory=dict)          # colour -> description
    score_text: str | None = None
    triggers: list[str] = field(default_factory=list)            # why this call happened now


@dataclass
class MemoryContext:
    sport: str
    playhead: float
    previous: Memory | None
    updates: list[StoryUpdate]              # story updates since the previous memory
    events: list[dict]                      # all headline events so far
    score_text: str | None
    teams: dict[str, str]


class StoryLLM(ABC):
    name = "base"

    @abstractmethod
    def describe_window(self, ctx: WindowContext) -> StoryUpdate:
        """Describe the window. The engine fills in id and logged_at afterwards."""

    @abstractmethod
    def summarize(self, ctx: MemoryContext) -> Memory:
        """Story so far, score, momentum, teams. The engine fills in logged_at afterwards."""
