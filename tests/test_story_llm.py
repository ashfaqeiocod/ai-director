"""Smart layer: rate limiting, 429 backoff, and turning Gemini's JSON into story items (fake client)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from story.common import story_config
from story.llm.base import WindowContext
from story.llm.gemini import GeminiStoryLLM, NoKeyError, RateLimiter
from story.llm.template import TemplateNarrator
from story.schema import LLMStory


def test_rate_limiter_caps_requests_per_minute():
    rl = RateLimiter(per_minute=2, backoff_start=10, backoff_max=40)
    rl.record(0); rl.record(1)
    assert not rl.allowed(30)          # 2 in the last minute
    assert rl.allowed(61)              # the first one is older than 60 s now


def test_429_backoff_doubles_and_resets():
    rl = RateLimiter(per_minute=100, backoff_start=10, backoff_max=40)
    assert rl.hit_429(0) == 10 and not rl.allowed(5) and rl.allowed(11)
    assert rl.hit_429(11) == 20
    assert rl.hit_429(40) == 40 and rl.hit_429(90) == 40     # capped
    rl.success()
    assert rl.hit_429(200) == 10


def no_extra_keys(monkeypatch) -> None:
    """Only GEMINI_API_KEY counts in these tests (a real .env may also have GEMINI_API_KEY1 ...)."""
    import os
    for name in [k for k in os.environ if k.startswith("GEMINI_API_KEY") and k != "GEMINI_API_KEY"]:
        monkeypatch.delenv(name)


def test_missing_key_raises(monkeypatch):
    no_extra_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(NoKeyError):
        GeminiStoryLLM(story_config())


class FakeModels:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        parsed = LLMStory.model_validate_json(self.answer)
        usage = SimpleNamespace(prompt_token_count=900, candidates_token_count=60, thoughts_token_count=0,
                                total_token_count=960)
        return SimpleNamespace(parsed=parsed, text=self.answer, usage_metadata=usage)


def fake_llm(monkeypatch, answer: str) -> GeminiStoryLLM:
    no_extra_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []
    llm = GeminiStoryLLM(story_config(), on_call=calls.append)
    llm.client = SimpleNamespace(models=FakeModels(answer))
    llm.calls_logged = calls
    return llm


def ctx(**kw) -> WindowContext:
    return WindowContext(sport="soccer", window_start=100.0, window_end=106.0, **kw)


def test_gemini_answer_becomes_story_update_with_clip_times(monkeypatch):
    llm = fake_llm(monkeypatch, '{"phase":"chance","events":[{"type":"shot","at_s":4.0,"team":"white",'
                                '"confidence":0.8,"note":null}],"team_in_control":"white","intensity":75,'
                                '"sentiment":"excited","narration":"White shoots!","is_replay":false,"score":"1-0"}')
    update = llm.describe_window(ctx())
    assert update.source == "gemini" and update.phase == "chance"
    assert update.events[0].event_time == 104.0           # snippet start 100 + at_s 4
    assert update.events[0].source == "gemini"
    assert llm.calls_logged[0]["total_tokens"] == 960 and "latency_s" in llm.calls_logged[0]


def test_events_seen_in_a_replay_are_dropped_live_ones_kept(monkeypatch):
    llm = fake_llm(monkeypatch, '{"phase":"replay","events":[{"type":"goal","at_s":2,"team":"white",'
                                '"confidence":0.9,"note":null,"in_replay":true},{"type":"goal","at_s":1,"team":"white",'
                                '"confidence":0.9,"note":null,"in_replay":false}],"team_in_control":null,"intensity":40,'
                                '"sentiment":"joy","narration":"Goal, and another look at it.","is_replay":true,"score":null}')
    update = llm.describe_window(ctx())
    assert update.is_replay and [e.event_time for e in update.events] == [101.0]


def test_video_is_sent_inline_with_fps(monkeypatch, tmp_path: Path):
    llm = fake_llm(monkeypatch, '{"phase":"build_up","events":[],"team_in_control":null,"intensity":20,'
                                '"sentiment":"calm","narration":"Calm.","is_replay":false,"score":null}')
    clip = tmp_path / "w.mp4"
    clip.write_bytes(b"fake-mp4")
    llm.describe_window(ctx(video_path=clip))
    parts = llm.client.models.calls[0][1][0].parts
    assert parts[0].inline_data.mime_type == "video/mp4" and parts[0].video_metadata.fps == 2


def test_template_narrator_names_the_strongest_event():
    update = TemplateNarrator().describe_window(ctx(recent_events=[
        {"type": "shot", "event_time": 103.0}, {"type": "goal", "event_time": 104.0}], fast={"intensity": 80}))
    assert update.phase == "goal" and "GOAL" in update.narration and update.source == "template"


class QuotaModels:
    """Answers 429 'per day' for every model in `out`, normally otherwise."""
    def __init__(self, out):
        self.out, self.models_used = set(out), []

    def generate_content(self, model, contents, config):
        from google.genai import errors
        self.models_used.append(model)
        if model in self.out:
            raise errors.APIError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                  "message": "quota GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
        return FakeModels('{"phase":"build_up","events":[],"team_in_control":null,"intensity":20,'
                          '"sentiment":"calm","narration":"Calm.","is_replay":false,"score":null}'
                          ).generate_content(model, contents, config)


def test_daily_quota_switches_to_fallback_then_stops(monkeypatch):
    llm = fake_llm(monkeypatch, "{}")
    llm.model, llm.cfg = "main", {**llm.cfg, "fallback_model": "backup"}
    llm.client = SimpleNamespace(models=QuotaModels(out={"main"}))
    with pytest.raises(Exception):
        llm.describe_window(ctx())
    assert llm.model == "backup" and llm.available()
    assert llm.describe_window(ctx()).narration == "Calm."
    llm.client.models.out.add("backup")
    with pytest.raises(Exception):
        llm.describe_window(ctx())
    assert not llm.available() and llm.state() == "daily_quota"


def test_two_keys_take_turns_and_quota_moves_to_the_other_key(monkeypatch):
    no_extra_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY1", "key-b")
    llm = GeminiStoryLLM(story_config())
    assert llm.keys == 2
    llm._take_turn()
    first = llm.key_i
    llm._take_turn()
    assert llm.key_i != first                         # requests alternate between the keys
    item = {"key": llm.key_i, "model": llm.model}
    llm._daily_quota_used(item)                       # this key's daily quota for the model is gone...
    assert not llm.daily_quota_out and llm.key_i != item["key"] and llm.model == item["model"]  # ...other key, same model


class BusyModels(QuotaModels):
    """Answers 503 (overloaded) for every model in `busy`, normally otherwise."""
    def __init__(self, busy):
        super().__init__(out=())
        self.busy = set(busy)

    def generate_content(self, model, contents, config):
        from google.genai import errors
        if model in self.busy:
            self.models_used.append(model)
            raise errors.APIError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}})
        return super().generate_content(model, contents, config)


def test_overloaded_model_retries_on_the_next_in_the_chain(monkeypatch):
    llm = fake_llm(monkeypatch, "{}")
    llm.model = "lite-a"
    llm.cfg = {**llm.cfg, "model": "lite-a", "fallback_model": "lite-b", "extra_models": ["lite-c", "flash"]}
    llm.client = SimpleNamespace(models=BusyModels(busy={"lite-a"}))
    assert llm.describe_window(ctx()).narration == "Calm."
    assert llm.client.models.models_used == ["lite-a", "lite-b"]      # one retry, on the fallback
    assert llm.model == "lite-a"                                       # a 503 is momentary: the main model stays


def test_overload_skips_models_whose_daily_quota_is_used_up(monkeypatch):
    llm = fake_llm(monkeypatch, "{}")
    llm.model = "lite-a"
    llm.cfg = {**llm.cfg, "model": "lite-a", "fallback_model": "lite-b", "extra_models": ["lite-c"]}
    llm.exhausted.add((llm.key_i, "lite-b"))
    assert llm.overload_model("lite-a") == "lite-c"
    assert llm.overload_model("lite-c") == "lite-a"                    # wraps around the chain
