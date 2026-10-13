"""studio.latency: every AI answer's delay (happened -> known), as the live run records it."""
from __future__ import annotations

import json

from studio.latency import LatencyLog, derive


def test_items_file_and_summary(tmp_path):
    published = []
    log = LatencyLog(tmp_path, published.append)
    # Gemini: the call log brings the model + request time of the window that ends at 16.04
    log.note_llm_call({"kind": "story", "model": "gemini-3.5-flash-lite", "key": 1, "window": [7.04, 16.04], "latency_s": 2.1})
    log.story({"source": "gemini", "window_start": 7.04, "window_end": 16.04, "logged_at": 20.17, "narration": "build-up"})
    log.story({"source": "template", "window_end": 24.0, "logged_at": 24.0})                 # no AI: no delay item
    log.event({"new": True, "event_type": "throw_in", "event_time": 60.9, "logged_at": 63.4, "sources": ["specialist"]})
    log.event({"new": False, "event_type": "throw_in", "event_time": 60.9, "logged_at": 70.0, "sources": ["specialist"]})
    log.goal_look({"t": 72.8, "logged_at": 75.0, "side": "right", "save": True, "kind": "after"}, "gemini-3.5-flash")
    # detector: two frames of second 10 from the GPU, one late (cache) -> one summary item + one late item
    log.detection(10.0, 10.1, "colab", 80.0)
    log.detection(10.2, 10.5, "colab", 90.0)
    log.detection(10.4, 11.9, "cached_late", None)
    summary = log.close()

    items = [json.loads(x) for x in (tmp_path / "latency.jsonl").read_text().splitlines()]
    assert [i["source"] for i in items] == ["gemini", "colab", "goal_ai", "detector", "detector"]
    g = items[0]
    assert g["model"] == "gemini-3.5-flash-lite" and g["request_s"] == 2.1 and g["delay"] == 4.13
    assert items[1]["what"] == "event: throw in" and items[1]["delay"] == 2.5
    assert items[2]["what"] == "right goal: save"
    late, second = items[3], items[4]
    assert late["kind"] == "late" and late["used"] == "cached_late"
    assert second["kind"] == "second" and second["frames"] == 3 and second["late"] == 1 and second["rtt_ms"] == 85.0
    assert summary["detector"]["n"] == 2 and summary["detector"]["max_s"] == 0.3     # only the GPU answers count
    assert summary["gemini"]["avg_s"] == 4.13 and summary["goal_ai"]["n"] == 1
    assert len(published) == len(items) and all(m["type"] == "latency" for m in published)


def test_older_runs_get_the_same_bars():
    items = derive([{"source": "gemini", "window_end": 16.0, "logged_at": 20.0, "narration": "x"}],
                   [{"type": "shot", "event_time": 30.0, "known_at": 33.0, "sources": ["specialist"]}],
                   [{"t": 40.0, "logged_at": 44.5, "side": "left", "goal": False}])
    assert [(i["source"], i["delay"]) for i in items] == [("gemini", 4.0), ("colab", 3.0), ("goal_ai", 4.5)]
