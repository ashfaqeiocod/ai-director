"""StoryEngine(start_s=...): the story starts like a new session at start_s (MVP 3 "start from anywhere")."""
from __future__ import annotations

import copy

import story.fast_layer
import story.run_log
from story.common import story_config
from story.engine import StoryEngine


def make_engine(tiny_clip, tmp_path, monkeypatch, start_s: float) -> StoryEngine:
    monkeypatch.setattr(story.run_log, "outputs_dir", lambda: tmp_path)
    # No YAMNet here: MediaPipe's classifier hangs when pytest shuts down. The fallback audio meter is enough.
    monkeypatch.setattr(story.fast_layer.AudioAnalyzer, "_load_yamnet", lambda self: (_ for _ in ()).throw(OSError("off")))
    cfg = copy.deepcopy(story_config())
    cfg["llm"]["interval_s"] = 1.5
    cfg["llm"]["window_s"] = 9.0
    cfg["memory"]["interval_s"] = 2.0
    return StoryEngine(tiny_clip, cfg, "fast_only", lambda m: None, start_s=start_s)


def test_default_start_is_zero(tiny_clip, tmp_path, monkeypatch):
    e = make_engine(tiny_clip, tmp_path, monkeypatch, 0.0)
    assert e.playhead.now() == 0.0 and e.fast_done_to == 0.0
    assert e.next_story_at == 1.5 and e.next_memory_at == 2.0


def test_engine_started_later_begins_there(tiny_clip, tmp_path, monkeypatch):
    e = make_engine(tiny_clip, tmp_path, monkeypatch, 2.0)
    assert e.playhead.now() == 2.0 and e.fast_done_to == 2.0
    assert e.next_story_at == 3.5 and e.next_memory_at == 4.0
    # The first story window never reaches back before the start (a new session has no past).
    ctx = e._window_context(3.5, [])
    assert ctx.window_start == 2.0 and ctx.window_end == 3.5


def test_fast_layer_starts_at_the_start_without_lag_skip(tiny_clip, tmp_path, monkeypatch):
    e = make_engine(tiny_clip, tmp_path, monkeypatch, 2.0)
    read_from = []
    original = e.reader.audio
    monkeypatch.setattr(e.reader, "audio", lambda t0, t1: (read_from.append(t0), original(t0, t1))[1])
    e.playhead.advance(3.2)
    assert e._step_fast() is True
    assert e.fast_done_to == 3.0                      # analysed (2.0, 3.0], not the whole clip from 0
    assert e.counters["lag_skips"] == 0
    assert min(read_from) >= 2.0                      # nothing from before the session start
