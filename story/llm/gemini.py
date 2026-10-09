"""gemini.py — the SMART layer: Gemini watches the last few seconds and writes the story.

describe_window(): sends the last `window_s` seconds as a small mp4 (with sound if
send_audio_to_llm is on) plus context (teams, story so far, recent events, specialist hints)
and gets back JSON that matches LLMStory (structured output = the SDK enforces the schema).

summarize(): a text-only call every ~60 s for the "story so far".

Free-tier friendly:
  - never more than `max_requests_per_minute` requests in any 60 s (story + memory together);
  - on HTTP 429 (rate limited) it waits (backoff_start_s, doubling up to backoff_max_s) and reports
    "rate_limited"; meanwhile the engine uses template lines, so the clip keeps playing.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable

from story.common import gemini_key
from story.llm.base import MemoryContext, StoryLLM, WindowContext
from story.schema import Event, LLMMemory, LLMStory, Memory, StoryUpdate


class NoKeyError(Exception):
    """GEMINI_API_KEY is missing in .env."""


class RateLimiter:
    """At most N requests in any 60 s, plus a pause after HTTP 429. Thread-safe."""

    def __init__(self, per_minute: int, backoff_start: float, backoff_max: float):
        self.per_minute = per_minute
        self.backoff_start = backoff_start
        self.backoff_max = backoff_max
        self.sent: deque[float] = deque()
        self.blocked_until = 0.0
        self.backoff = backoff_start
        self.lock = threading.Lock()

    def _forget_old(self, now: float) -> None:
        while self.sent and now - self.sent[0] >= 60:
            self.sent.popleft()

    def allowed(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self.lock:
            self._forget_old(now)
            return now >= self.blocked_until and len(self.sent) < self.per_minute

    def record(self, now: float | None = None) -> None:
        with self.lock:
            self.sent.append(time.time() if now is None else now)

    def count(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self.lock:
            self._forget_old(now)
            return len(self.sent)

    def hit_429(self, now: float | None = None) -> float:
        """Start (or double) the pause. Returns how long we pause."""
        now = time.time() if now is None else now
        with self.lock:
            wait = self.backoff
            self.blocked_until = now + wait
            self.backoff = min(self.backoff * 2, self.backoff_max)
            return wait

    def success(self) -> None:
        with self.lock:
            self.backoff = self.backoff_start

    def rate_limited(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) < self.blocked_until


SYSTEM_STORY = """You are a live {sport} analyst-commentator watching a match AS IT HAPPENS.
You only get the last few seconds of video (and sound, if included). You cannot see the future.

Rules:
- Describe ONLY what is visible or audible in this snippet. If unsure, say less and use lower confidence.
- Replays: a replay has clear cues: slow motion, a replay graphic or wipe, or the SAME action shown again
  from another angle. Normal-speed play is live, even with quick camera cuts, close-ups or celebrations.
  If unsure, it is live. Set is_replay=true (and phase="replay") only if the snippet is mainly a replay.
  A snippet can be partly live and partly replay: mark each event with in_replay (true = seen in the
  replay part). Events seen live must be reported even if a replay follows them.
- Never invent a score. Set "score" only if a scoreboard is visible in the snippet; otherwise null.
- Teams are named by jersey colour (e.g. "white", "red"). Reuse the known colours when they fit.
- Events: only clear ones. "at_s" = seconds from the START of this snippet. Don't repeat events that
  are already listed as logged, unless it is clearly a NEW one.
- "goal" only if the ball clearly goes in (net bulges, celebration, scoreboard change). A big chance
  that misses is "chance" or "shot".
- narration: ONE line, present tense, under 25 words, like a TV commentator, factual.
- intensity: 0 (nothing happening) to 100 (goal / huge chance).
Answer only with JSON that matches the schema."""

SYSTEM_MEMORY = """You keep the running story of a {sport} match that is being watched live.
Write a short, factual "story so far" (2-4 sentences) from the notes you are given. Never invent
events or a score that the notes don't support. Name teams by jersey colour.
Answer only with JSON that matches the schema."""


def _fmt_events(events: list[dict]) -> str:
    if not events:
        return "none"
    return "; ".join(f"{e['type']} at {e['event_time']:.0f}s" + (f" ({e['team']})" if e.get("team") else "")
                     for e in events)


class GeminiStoryLLM(StoryLLM):
    name = "gemini"

    def __init__(self, cfg: dict, on_call: Callable[[dict], None] | None = None):
        key = gemini_key()
        if not key:
            raise NoKeyError("GEMINI_API_KEY is missing in .env")
        from google import genai
        from google.genai import types

        self.types = types
        self.cfg = cfg["llm"]
        self.sport = cfg["sport"]
        self.model = self.cfg["model"] or "gemini-2.5-flash"
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(self.cfg["timeout_s"] * 1000)))
        self.limiter = RateLimiter(self.cfg["max_requests_per_minute"], self.cfg["backoff_start_s"],
                                   self.cfg["backoff_max_s"])
        self.on_call = on_call or (lambda item: None)
        self.in_flight = 0
        self.last_latency: float | None = None
        self.last_error = ""
        self.exhausted: set[str] = set()       # models whose daily free quota is used up
        self.daily_quota_out = False
        self.use_thinking_setting = self.cfg.get("thinking_budget") is not None

    # ------------------------------------------------------------------ status for the engine/UI
    def available(self) -> bool:
        """True if a request may be sent now (rate limit + 429 pause + daily quota)."""
        return not self.daily_quota_out and self.limiter.allowed()

    def requests_last_minute(self) -> int:
        return self.limiter.count()

    def _daily_quota_used(self, item: dict) -> None:
        """Google's free tier has a small DAILY request quota per model (20 for gemini-3.5-flash in
        Oct 2026). Waiting won't help today: switch to the fallback model (its own quota), or stop
        calling Gemini for this run (templates take over)."""
        self.exhausted.add(item.get("model", self.model))
        fallback = self.cfg.get("fallback_model")
        if fallback and fallback not in self.exhausted:
            item.update(daily_quota=True, switched_to=fallback)
            self.model = fallback
        else:
            item.update(daily_quota=True, switched_to=None)
            self.daily_quota_out = True

    def state(self) -> str:
        if self.daily_quota_out:
            return "daily_quota"
        if self.limiter.rate_limited():
            return "rate_limited"
        if self.in_flight:
            return "busy"
        return "error" if self.last_error else "ok"

    # ------------------------------------------------------------------ the two calls
    def describe_window(self, ctx: WindowContext) -> StoryUpdate:
        types = self.types
        parts = []
        if ctx.video_path is not None:
            parts.append(types.Part(inline_data=types.Blob(data=ctx.video_path.read_bytes(), mime_type="video/mp4"),
                                    video_metadata=types.VideoMetadata(fps=self.cfg["video_fps"])))
        parts.append(types.Part(text=self._story_prompt(ctx)))
        story: LLMStory = self._generate(parts, SYSTEM_STORY.format(sport=self.sport), LLMStory, kind="story",
                                         extra={"window": [ctx.window_start, ctx.window_end],
                                                "triggers": ctx.triggers,
                                                "video_bytes": ctx.video_path.stat().st_size if ctx.video_path else 0})
        length = ctx.window_end - ctx.window_start
        events = [Event(type=e.type, event_time=round(ctx.window_start + min(max(e.at_s, 0.0), length), 2),
                        logged_at=ctx.window_end, source="gemini", team=(e.team or None),
                        confidence=min(max(e.confidence, 0.0), 1.0), note=e.note)
                  for e in story.events if e.type != "none" and not e.in_replay]   # replayed = already happened
        return StoryUpdate(source="gemini", window_start=ctx.window_start, window_end=ctx.window_end,
                           logged_at=ctx.window_end, phase=story.phase, events=events,
                           team_in_control=story.team_in_control, intensity=min(max(story.intensity, 0), 100),
                           sentiment=story.sentiment, narration=story.narration, is_replay=story.is_replay,
                           score=story.score)

    def summarize(self, ctx: MemoryContext) -> Memory:
        prompt = self._memory_prompt(ctx)
        memory: LLMMemory = self._generate([self.types.Part(text=prompt)], SYSTEM_MEMORY.format(sport=self.sport),
                                           LLMMemory, kind="memory", extra={"playhead": ctx.playhead})
        return Memory(source="gemini", logged_at=ctx.playhead, story_so_far=memory.story_so_far,
                      score=memory.score or ctx.score_text, momentum=memory.momentum,
                      teams={t.colour.strip().lower(): t.description for t in memory.teams if t.colour.strip()})

    # ------------------------------------------------------------------ prompts
    def _story_prompt(self, ctx: WindowContext) -> str:
        lines = [
            f"Snippet: clip time {ctx.window_start:.1f}s to {ctx.window_end:.1f}s "
            f"({ctx.window_end - ctx.window_start:.0f} s, ends NOW).",
            "Known teams (jersey colour: description): "
            + (", ".join(f"{c}: {d or '?'}" for c, d in ctx.teams.items()) or "not known yet"),
            f"Story so far: {ctx.memory.story_so_far if ctx.memory else 'just started'}",
            f"Score counted so far: {ctx.score_text or 'unknown'}",
            f"Events already logged in the last minute: {_fmt_events(ctx.recent_events)}",
        ]
        if ctx.specialist_hints:
            lines.append("Action-spotting model hints (may be wrong): " + "; ".join(ctx.specialist_hints))
        fast = ctx.fast
        # (The scene-cut "replay hint" is NOT passed on: it fires on all fast highlight editing and
        # made Gemini call live goals "replays".)
        lines.append(f"Audio/motion sensors for this snippet: crowd excitement max {fast.get('audio_max', 0):.2f} (0-1), "
                     f"motion {fast.get('motion_avg', 0):.2f}, whistle heard: {fast.get('whistle', False)}.")
        if ctx.triggers:
            lines.append(f"This look was triggered by: {', '.join(ctx.triggers)}.")
        if ctx.video_path is None:
            lines.append("(No video attached this time: describe cautiously from the sensors only.)")
        return "\n".join(lines)

    def _memory_prompt(self, ctx: MemoryContext) -> str:
        notes = "\n".join(f"- {u.window_end:.0f}s [{u.phase}{', replay' if u.is_replay else ''}] {u.narration}"
                          for u in ctx.updates[-20:]) or "- (no new notes)"
        return "\n".join([
            f"Clip time now: {ctx.playhead:.0f}s.",
            f"Previous story so far: {ctx.previous.story_so_far if ctx.previous else 'none yet'}",
            f"Previous momentum: {ctx.previous.momentum if ctx.previous else 'unknown'}",
            "Known teams: " + (", ".join(f"{c}: {d or '?'}" for c, d in ctx.teams.items()) or "unknown"),
            f"Headline events so far: {_fmt_events(ctx.events)}",
            f"Score counted by the system: {ctx.score_text or 'unknown'}",
            "Commentary notes since then:", notes,
        ])

    # ------------------------------------------------------------------ one request
    def _send(self, model: str, parts: list, config: dict):
        return self.client.models.generate_content(
            model=model, contents=[self.types.Content(role="user", parts=parts)],
            config=self.types.GenerateContentConfig(**config))

    def _retry(self, error, parts: list, config: dict, item: dict):
        """One retry for two known, fixable errors; anything else is raised again."""
        if error.code == 400 and "thinking_config" in config:
            # Some models (e.g. Flash Lite) reject the "thinking" setting with a generic 400:
            # retry without it, and leave it out from now on.
            self.use_thinking_setting = False
            config.pop("thinking_config")
            item.update(thinking_setting_dropped=True)
            self.limiter.record()
            return self._send(self.model, parts, config)
        fallback = self.cfg.get("fallback_model")
        if error.code == 503 and fallback and fallback != self.model:
            # 503 = Google's model is overloaded right now: retry on the fallback model.
            item.update(model=fallback, overloaded=self.model)
            self.limiter.record()
            return self._send(fallback, parts, config)
        raise error

    def _generate(self, parts: list, system: str, schema, kind: str, extra: dict):
        from google.genai import errors

        types = self.types
        config = dict(system_instruction=system, response_mime_type="application/json", response_schema=schema,
                      temperature=self.cfg.get("temperature", 0.4),
                      media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
        if self.use_thinking_setting:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.cfg["thinking_budget"])
        self.limiter.record()
        self.in_flight += 1
        start = time.time()
        item = {"kind": kind, "model": self.model, "requested_at": round(start, 3), **extra}
        try:
            try:
                response = self._send(self.model, parts, config)
            except errors.APIError as error:
                response = self._retry(error, parts, config, item)
            parsed = response.parsed if isinstance(response.parsed, schema) else \
                schema.model_validate_json(response.text or "{}")
            usage = response.usage_metadata
            self.last_latency = time.time() - start
            self.last_error = ""
            self.limiter.success()
            item.update(latency_s=round(self.last_latency, 2),
                        prompt_tokens=getattr(usage, "prompt_token_count", None),
                        output_tokens=getattr(usage, "candidates_token_count", None),
                        thinking_tokens=getattr(usage, "thoughts_token_count", None),
                        total_tokens=getattr(usage, "total_token_count", None),
                        response=json.loads(parsed.model_dump_json()))
            return parsed
        except errors.APIError as error:
            item.update(latency_s=round(time.time() - start, 2), error=f"{error.code}: {str(error)[:300]}")
            if error.code == 429 and "PerDay" in str(error):
                self._daily_quota_used(item)
            elif error.code == 429:
                item.update(rate_limited=True, backoff_s=self.limiter.hit_429())
            elif error.code == 400 and "thinking" in str(error).lower() and self.use_thinking_setting:
                self.use_thinking_setting = False      # this model doesn't take the setting: drop it next time
            self.last_error = item["error"]
            raise
        except Exception as error:
            item.update(latency_s=round(time.time() - start, 2), error=f"{type(error).__name__}: {str(error)[:300]}")
            self.last_error = item["error"]
            raise
        finally:
            self.in_flight -= 1
            self.on_call(item)
