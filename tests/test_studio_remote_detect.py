"""The Colab detector in live mode (studio.remote_detect + LiveStudio._apply_remote), with fake servers."""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import pytest

import studio.remote_detect as rd
from colab.detector_server import parse
from studio.latency import LatencyLog
from studio.live import LiveStudio

FPS = 30.0
CFG = {"max_age_s": 1.5, "max_in_flight": 2, "jpeg_quality": 85, "send_width": 0, "lost_after_s": 5, "cache_fallback": True}


# ------------------------------------------------------------------ message format
def test_encode_matches_the_server_parser():
    images = {1: np.zeros((96, 128, 3), np.uint8), 3: np.full((96, 128, 3), 200, np.uint8)}
    message, scales = rd.encode(42, 123.5, images, 85)
    header, items = parse(message)
    assert scales == {1: 1.0, 3: 1.0} and header["frame"] == 42 and header["sent"] == 123.5
    assert [c for c, _ in items] == [1, 3] and all(j[:2] == b"\xff\xd8" for _, j in items)   # two JPEGs


def test_smaller_pictures_come_back_in_original_pixels():
    # each camera keeps its own scale (a camera already small enough is not resized)
    _, scales = rd.encode(0, 0.0, {1: np.zeros((960, 1280, 3), np.uint8), 2: np.zeros((300, 400, 3), np.uint8)}, 85, width=640)
    assert scales == {1: 2.0, 2: 1.0}
    det = rd.rescale({"players": [{"box": [10, 20, 30, 40]}], "ball": {"box": [1, 2, 3, 4], "conf": 0.5}}, scales[1])
    assert det["players"][0]["box"] == [20, 40, 60, 80] and det["ball"]["box"] == [2, 4, 6, 8]


# ------------------------------------------------------------------ the live loop's rules
class FakeRemote:
    def __init__(self, on_its_way: bool = True):
        self.items = []
        self.way = on_its_way

    def on_its_way(self, frame):
        return self.way

    def results(self):
        out, self.items = self.items, []
        return out


class FakePipe:
    pitch_state = None

    def __init__(self):
        self.calls = []

    def detections(self, f, t, detector=None, images=None):
        self.calls.append((f, detector))
        return []


class FakeSource:
    shown_dets = None
    cam_ids = [1]

    def __init__(self, cached: dict):
        self.cached = cached

    def detection_at(self, f):
        return self.cached.get(f, {1: {"players": [], "ball": None}})


def live(cached=None) -> LiveStudio:
    lv = LiveStudio.__new__(LiveStudio)          # only what _apply_remote needs (no footage)
    lv.cfg = {"live": {"remote_detect": CFG}}
    lv.fps, lv.watcher, lv.colab_answers = FPS, None, {}
    lv.det_counts = {"colab": 0, "cached_late": 0, "cached_skipped": 0, "none": 0, "ignored_late": 0}
    lv.src, lv.pipe = FakeSource(cached or {}), FakePipe()
    lv.latency = LatencyLog(None, lambda m: None)
    return lv


def answer(frame: int, age: float) -> dict:
    return {"frame": frame, "age_s": age, "cams": {1: {"players": [{"box": [0, 0, 1, 1]}], "ball": None}},
            "sent": 0, "received": 0, "rtt_ms": 1, "infer_ms": 1}


def test_answers_are_applied_in_frame_order_with_their_own_frame():
    lv, remote = live(), FakeRemote()
    pending = deque([(0, True), (6, True)])
    remote.items = [answer(6, 0.3)]                       # frame 6 answered first: it must wait for frame 0
    lv._apply_remote(remote, pending, {}, now=0.3, files=None)
    assert lv.pipe.calls == [] and len(pending) == 2
    remote.items = [answer(0, 0.4)]
    lv._apply_remote(remote, pending, {}, now=0.4, files=None)
    assert [f for f, _ in lv.pipe.calls] == [0, 6] and not pending and lv.det_counts["colab"] == 2


def test_late_frame_falls_back_to_the_cache_and_its_answer_is_ignored():
    cached = {0: {1: {"players": [{"box": [5, 5, 6, 6]}], "ball": None}}}
    lv, remote = live(cached), FakeRemote()
    pending = deque([(0, True)])
    lv._apply_remote(remote, pending, {}, now=1.0, files=None)          # 1.0 s old: still waiting
    assert lv.pipe.calls == []
    lv._apply_remote(remote, pending, {}, now=1.6, files=None)          # > max_age_s: the cached detection
    assert lv.pipe.calls == [(0, cached[0])] and lv.det_counts["cached_late"] == 1
    remote.items = [answer(0, 1.7)]                                     # the answer arrives after all: ignored
    lv._apply_remote(remote, pending, {}, now=1.7, files=None)
    assert len(lv.pipe.calls) == 1 and lv.det_counts["ignored_late"] == 1


def test_skipped_frame_uses_the_cache_at_once_and_none_when_there_is_none():
    lv, remote = live({12: {1: {"players": [{"box": [1, 1, 2, 2]}], "ball": None}}}), FakeRemote()
    pending = deque([(12, False), (18, False)])
    lv._apply_remote(remote, pending, {}, now=0.7, files=None)
    # 18 has no cached detection: an EMPTY step still reaches the map (the ball can age to "lost")
    assert lv.pipe.calls == [(12, lv.src.cached[12]), (18, {1: {"players": [], "ball": None}})]
    assert lv.det_counts["cached_skipped"] == 1 and lv.det_counts["none"] == 1


# ------------------------------------------------------------------ the client against a fake server
def can_listen() -> bool:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not can_listen(), reason="no local sockets here (sandbox)")
def test_client_sends_receives_and_drops_stale_frames(monkeypatch):
    from websockets.sync.server import serve

    received = []

    def handler(ws):
        for message in ws:
            header, items = parse(message)
            received.append(header["frame"])
            time.sleep(0.2)                       # a slow GPU: frames pile up on the client
            ws.send(json.dumps({"type": "detections", "frame": header["frame"], "sent": header["sent"], "infer_ms": 30,
                                "cams": {str(c): {"players": [{"box": [10, 10, 20, 40], "cls": "player", "conf": 0.9}],
                                                  "ball": None} for c, _ in items}}))

    server = serve(handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(rd, "check_health", lambda url, timeout: {"detector": {"ready": True, "model": "fake"}})
    det = rd.RemoteDetector(f"http://127.0.0.1:{port}", FPS, lambda: 1.0, {**CFG, "max_in_flight": 1})
    det.start()
    try:
        for _ in range(50):
            if det.connected():
                break
            time.sleep(0.05)
        img = np.zeros((96, 128, 3), np.uint8)
        det.submit(0, {1: img, 2: img})
        time.sleep(0.1)                           # frame 0 is on its way (the only slot)
        for f in range(6, 60, 6):                 # 9 more at once: the waiting one is replaced again and again
            det.submit(f, {1: img, 2: img})
        time.sleep(1.0)
        out = det.results()
        assert out and out[0]["frame"] == 0 and set(out[0]["cams"]) == {1, 2}
        assert out[-1]["frame"] == 54                # the newest frame is always sent in the end
        assert det.status()["dropped"] >= 7          # the frames in between were stale: dropped, not queued
        assert abs(out[0]["age_s"] - (1.0 - 0 / FPS)) < 1e-6 and out[0]["rtt_ms"] >= 150
    finally:
        det.stop()
        server.shutdown()


def test_pure_live_never_uses_the_cache():
    cached = {0: {1: {"players": [{"box": [5, 5, 6, 6]}], "ball": None}}}
    lv, remote = live(cached), FakeRemote()
    lv.cfg = {"live": {"remote_detect": {**CFG, "cache_fallback": False}}}
    pending = deque([(0, True)])
    lv._apply_remote(remote, pending, {}, now=1.6, files=None)
    assert lv.pipe.calls == [(0, {1: {"players": [], "ball": None}})] and lv.det_counts["none"] == 1 and not pending


def test_a_frame_dropped_before_sending_does_not_hold_up_the_queue():
    cached = {0: {1: {"players": [{"box": [5, 5, 6, 6]}], "ball": None}}}
    lv, remote = live(cached), FakeRemote(on_its_way=False)          # replaced while waiting / link down
    remote.items = [answer(6, 0.2)]
    lv._apply_remote(remote, deque([(0, True), (6, True)]), {}, now=0.25, files=None)
    assert [f for f, _ in lv.pipe.calls] == [0, 6]                      # no 1.5 s wait for frame 0
    assert lv.det_counts["cached_late"] == 1 and lv.det_counts["colab"] == 1
