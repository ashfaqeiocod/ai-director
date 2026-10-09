"""schema.py — the story data model (Pydantic = typed, validated data classes; like a TS interface + zod).

Main models (spec section 5):
  Event        one thing that happened (goal, shot, foul...) — with event_time AND logged_at
  StoryUpdate  what one look at a few seconds of video says (phase, mood, narration...)
  Memory       the long view: story so far, score, momentum, teams

Every item stores BOTH times (rule 3, "honest latency"):
  event_time / window_end = when it happened in the clip
  logged_at               = where the playhead was when the AI produced it
so the dashboard can show the real delay.

The LLM* models at the bottom are what we ask Gemini to return (it can't know logged_at,
so the engine adds the timing fields itself).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Phase = Literal["build_up", "attack", "chance", "goal", "celebration", "stoppage", "foul",
                "set_piece", "injury", "replay", "break", "unknown"]
Sentiment = Literal["calm", "building", "tense", "excited", "joy", "frustration",
                    "anger", "disappointment", "relief", "unknown"]
EventType = Literal["goal", "shot", "save", "chance", "foul", "card_yellow", "card_red", "corner",
                    "free_kick", "penalty", "throw_in", "offside", "substitution", "tackle", "cross",
                    "header", "out", "kickoff", "whistle", "injury", "celebration", "argument", "none"]
Source = Literal["fast", "specialist", "gemini", "fusion", "manual"]

# Event types shown as icons on the timeline. The rest (whistle, throw-in, header...) are kept
# in the logs but would clutter the headline view.
HEADLINE_TYPES = {"goal", "shot", "save", "chance", "foul", "card_yellow", "card_red", "corner",
                  "free_kick", "penalty", "offside", "substitution", "injury"}
MAX_NARRATION_WORDS = 30   # the prompt asks for < 25; anything longer is cut here


class Event(BaseModel):
    type: EventType
    event_time: float               # seconds in clip
    logged_at: float                # playhead when produced
    source: Source
    team: str | None = None         # jersey colour, e.g. "white"
    confidence: float = Field(ge=0, le=1)
    note: str | None = None


class StoryUpdate(BaseModel):
    id: str = ""                    # set by the engine, used for 👍/👎 ratings
    source: Literal["gemini", "template"] = "template"
    window_start: float
    window_end: float
    logged_at: float
    phase: Phase
    events: list[Event] = []
    team_in_control: str | None = None
    intensity: int = Field(ge=0, le=100)
    sentiment: Sentiment
    narration: str                  # max ~25 words, present tense, like a commentator
    is_replay: bool = False         # broadcast replays must NOT create new events (no double goals)
    score: str | None = None        # e.g. "1-0" if visible

    @field_validator("narration")
    @classmethod
    def _short(cls, text: str) -> str:
        words = text.split()
        return " ".join(words[:MAX_NARRATION_WORDS]) + ("…" if len(words) > MAX_NARRATION_WORDS else "")


class Memory(BaseModel):
    source: Literal["gemini", "template"] = "template"
    logged_at: float
    story_so_far: str               # 2–4 sentences
    score: str | None = None
    momentum: str = ""              # e.g. "white pressing for 5 minutes"
    teams: dict[str, str] = {}      # colour -> short description


# ----------------------------------------------------------------------------------------------
# What Gemini must return (structured output). Times are relative to the snippet it watched.
class LLMEvent(BaseModel):
    type: EventType
    at_s: float = Field(description="seconds from the START of this video snippet when it happened")
    team: str | None = Field(None, description="jersey colour of the team that did it, e.g. 'white'")
    confidence: float = Field(description="0..1, how sure you are")
    note: str | None = Field(None, description="a few words, e.g. 'low shot, saved by the keeper'")
    in_replay: bool = Field(False, description="true ONLY if you saw this event in a replay part of the snippet")


class LLMStory(BaseModel):
    phase: Phase
    events: list[LLMEvent]
    team_in_control: str | None = Field(None, description="jersey colour of the team with the ball, or null")
    intensity: int = Field(description="0..100, how intense the play is right now")
    sentiment: Sentiment
    narration: str = Field(description="one commentator line, present tense, under 25 words")
    is_replay: bool = Field(description="true only if the snippet is MAINLY a replay (slow motion, replay graphic, "
                                        "or an action shown again); normal-speed play with quick cuts is live")
    score: str | None = Field(None, description="only if a scoreboard is VISIBLE, e.g. '1-0'; else null")


class LLMTeam(BaseModel):
    colour: str
    description: str


class LLMMemory(BaseModel):
    story_so_far: str = Field(description="2-4 sentences: the story of the match so far")
    score: str | None = Field(None, description="score if known from a visible scoreboard or counted goals")
    momentum: str = Field(description="who is on top right now and for how long")
    teams: list[LLMTeam]
