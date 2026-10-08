"""Specialist layer: class mapping, dedupe, play signals, and the server's peak finder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from story.common import story_config
from story.specialist_client import SpecialistClient, normalize_url

sys.path.insert(0, str(Path(__file__).parent.parent / "colab"))


class FakeReader:
    class playhead:
        @staticmethod
        def now():
            return 50.0


def client():
    got = []
    c = SpecialistClient(story_config(), FakeReader(), "abc.ngrok-free.app",
                         on_event=lambda e, s, n: got.append((e, s, n)), on_status=lambda s: None,
                         log_call=lambda i: None)
    return c, got


def test_classes_map_to_event_types_and_play_signals_are_not_events():
    c, got = client()
    c._handle({"class": "SHOT", "event_time": 40.0, "confidence": 0.8}, logged_at=50.0)
    c._handle({"class": "PASS", "event_time": 41.0, "confidence": 0.9}, logged_at=50.0)
    c._handle({"class": "PLAYER SUCCESSFUL TACKLE", "event_time": 42.0, "confidence": 0.5}, logged_at=50.0)
    shot, pass_, tackle = got
    assert shot[0].type == "shot" and shot[0].source == "specialist" and shot[0].logged_at == 50.0
    assert pass_[0] is None and pass_[1] > 0          # intensity only
    assert tackle[0].type == "tackle"


def test_same_class_within_window_is_reported_once_and_low_confidence_dropped():
    c, got = client()
    c._handle({"class": "SHOT", "event_time": 40.0, "confidence": 0.8}, logged_at=50.0)
    c._handle({"class": "SHOT", "event_time": 41.0, "confidence": 0.9}, logged_at=50.0)   # within 2 s
    c._handle({"class": "SHOT", "event_time": 45.0, "confidence": 0.2}, logged_at=50.0)   # below 0.3
    c._handle({"class": "SHOT", "event_time": 45.0, "confidence": 0.7}, logged_at=50.0)
    assert [e.event_time for e, _, _ in got] == [40.0, 45.0]


def test_url_is_normalized():
    assert normalize_url("abc.trycloudflare.com/") == "https://abc.trycloudflare.com"


def test_server_peak_finder_respects_report_range_and_nms():
    from specialist_server import find_peaks
    times = [i * 0.48 for i in range(20)]
    prob = np.zeros((20, 13))
    prob[5, 8] = 0.9          # SHOT (column 8 = class index 7 + 1 background)
    prob[6, 8] = 0.6          # neighbour, not a peak
    prob[15, 8] = 0.95        # after the report range
    events = find_peaks(times, prob, ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
                                      "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"],
                        start=-1, end=5.0, min_confidence=0.3, nms_s=1.0)
    assert events == [{"class": "SHOT", "event_time": 2.4, "confidence": 0.9}]
