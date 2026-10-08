"""Proof mode: one-to-one matching within the window, and the dashboard's truth markers."""
from __future__ import annotations

from story.evaluate import match, truth_status

TRUTH = [{"time_s": 60.0, "event": "goal", "team": "white"},
         {"time_s": 90.0, "event": "goal", "team": "white"},
         {"time_s": 30.0, "event": "shot", "team": "red"}]


def found(type_, t):
    return {"type": type_, "event_time": t, "logged_at": t + 5}


def test_each_truth_event_matches_at_most_one_found_event():
    pairs = match(TRUTH, [found("goal", 62.0), found("goal", 63.0), found("shot", 45.0)], window=10)
    result = {t["time_s"]: f for t, f in pairs}
    assert result[60.0]["event_time"] == 62.0
    assert result[90.0] is None                 # 63 is too far from 90
    assert result[30.0] is None                 # 45 is 15 s away (> 10)


def test_truth_markers_pending_missed_and_false_alarm():
    items = truth_status(TRUTH, [found("goal", 61.0), found("goal", 120.0)], window=10, until=95.0)
    by_time = {(i["time_s"], i["status"]) for i in items}
    assert (60.0, "matched") in by_time
    assert (90.0, "pending") in by_time         # still within the window at playhead 95
    assert (30.0, "missed") in by_time
    assert (120.0, "false_alarm") in by_time
