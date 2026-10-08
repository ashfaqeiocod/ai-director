"""template.py — the fallback "narrator": simple sentences built from fast/specialist signals.

Used when the mode has no Gemini (fast_only, specialist_only), when the key is missing,
or while Gemini is rate-limited. It never invents anything: it only says what the signals show.
"""
from __future__ import annotations

from story.llm.base import MemoryContext, StoryLLM, WindowContext
from story.schema import Memory, StoryUpdate

# Strongest event in the window decides the phase + line (first match wins).
EVENT_LINES = [
    ("goal", "goal", "joy", "GOAL! The crowd erupts."),
    ("penalty", "set_piece", "tense", "Penalty!"),
    ("card_red", "foul", "anger", "Red card!"),
    ("card_yellow", "foul", "frustration", "Yellow card shown."),
    ("save", "chance", "excited", "Big save by the keeper!"),
    ("shot", "chance", "excited", "A shot at goal!"),
    ("chance", "chance", "excited", "A real chance here!"),
    ("foul", "foul", "frustration", "Foul — play stops."),
    ("corner", "set_piece", "building", "Corner kick coming up."),
    ("free_kick", "set_piece", "building", "Free kick."),
    ("cross", "attack", "building", "A cross into the box."),
    ("header", "attack", "building", "A header in the box."),
    ("tackle", "build_up", "tense", "Strong tackle wins the ball."),
    ("throw_in", "set_piece", "calm", "Throw-in."),
    ("out", "stoppage", "calm", "Ball goes out of play."),
]


def sentiment_for(intensity: int) -> str:
    return "calm" if intensity < 25 else "building" if intensity < 45 else "tense" if intensity < 65 else "excited"


class TemplateNarrator(StoryLLM):
    name = "template"

    def describe_window(self, ctx: WindowContext) -> StoryUpdate:
        intensity = int(ctx.fast.get("intensity", 0))
        types = {e["type"] for e in ctx.recent_events if e["event_time"] > ctx.window_start}
        phase, sentiment, line = "build_up" if intensity < 55 else "attack", sentiment_for(intensity), None
        for event_type, p, s, text in EVENT_LINES:
            if event_type in types:
                phase, sentiment, line = p, s, text
                break
        if line is None:
            if ctx.fast.get("replay"):
                phase, line = "replay", "Looks like a replay of the action."
            elif "cheer" in ctx.triggers or ctx.fast.get("cheer"):
                line = "Big roar from the crowd — something is happening!"
            elif ctx.fast.get("whistle"):
                phase, line = "stoppage", "The referee's whistle goes."
            elif intensity >= 65:
                line = "Play is intense right now."
            elif intensity >= 40:
                line = "The tempo is picking up."
            else:
                line = "A quieter spell; play is being built up."
        return StoryUpdate(source="template", window_start=ctx.window_start, window_end=ctx.window_end,
                           logged_at=ctx.window_end, phase=phase, events=[], intensity=intensity,
                           sentiment=sentiment, narration=line, is_replay=bool(ctx.fast.get("replay")),
                           score=None)

    def summarize(self, ctx: MemoryContext) -> Memory:
        counts: dict[str, int] = {}
        for e in ctx.events:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        minutes = ctx.playhead / 60
        if counts:
            listed = ", ".join(f"{n} {t.replace('_', ' ')}{'s' if n > 1 else ''}" for t, n in counts.items())
            story = f"After {minutes:.1f} minutes of the clip: {listed} so far."
        else:
            story = f"After {minutes:.1f} minutes of the clip, no headline events have been detected yet."
        if ctx.score_text:
            story += f" Score: {ctx.score_text}."
        recent = ctx.updates[-6:]
        avg = sum(u.intensity for u in recent) / len(recent) if recent else 0
        momentum = ("a lot of intensity in the last minute" if avg >= 55
                    else "a calm last minute" if avg < 30 else "steady play in the last minute")
        return Memory(source="template", logged_at=ctx.playhead, story_so_far=story, score=ctx.score_text,
                      momentum=momentum, teams=ctx.teams)
