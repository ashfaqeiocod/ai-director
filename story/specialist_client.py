"""specialist_client.py — talks to the Colab specialist server (ball action spotting).

  1. GET <url>/health      -> the model's frame rate (~2.08 fps), frame size (398x224), classes.
  2. WS  <url>/stream      -> we send frames that have ALREADY PLAYED (CausalReader: never past the
                              playhead), at the model's frame rate, as JPEG + timestamp.
  3. The server answers every `stride_s` with events {class, event_time, confidence}.

Class names are mapped to our EventType (SHOT -> shot, GOAL -> goal, ...). PASS / DRIVE / HIGH PASS /
BALL PLAYER BLOCK are only "play" signals: they raise the intensity a little but are not events.
The same class within `dedupe_window_s` is reported once. The real-time factor (RTF) is
inference time / stride: under 1 means the GPU keeps up with live video.
"""
from __future__ import annotations

import json
import struct
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable

import cv2

from story.causal import CausalReader, FutureAccessError
from story.schema import Event

CLASS_MAP = {"SHOT": "shot", "GOAL": "goal", "FREE KICK": "free_kick", "THROW IN": "throw_in", "OUT": "out",
             "PLAYER SUCCESSFUL TACKLE": "tackle", "CROSS": "cross", "HEADER": "header"}
PLAY_SIGNALS = {"PASS", "DRIVE", "HIGH PASS", "BALL PLAYER BLOCK"}
# How much each class pushes the intensity meter (0..1).
STRENGTH = {"goal": 1.0, "shot": 0.9, "free_kick": 0.6, "cross": 0.6, "header": 0.6, "tackle": 0.5,
            "throw_in": 0.2, "out": 0.2}
PLAY_STRENGTH = 0.3
HEADERS = {"ngrok-skip-browser-warning": "1", "User-Agent": "story-reader"}


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def check_health(url: str, timeout: float = 10) -> dict:
    """GET /health. Raises on any problem (bad URL, tunnel down, server not started)."""
    request = urllib.request.Request(normalize_url(url) + "/health", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class SpecialistClient:
    def __init__(self, cfg: dict, reader: CausalReader, url: str,
                 on_event: Callable[[Event | None, float, str], None],
                 on_status: Callable[[dict], None], log_call: Callable[[dict], None], video_source=None,
                 start_s: float = 0.0):
        self.cfg = cfg["specialist"]
        self.reader = reader
        self.url = normalize_url(url)
        self.on_event = on_event
        self.on_status = on_status
        self.log_call = log_call
        # Where the frames come from: the clip (default), or anything with the same frames(t0, t1, fps) / close()
        # (MVP 3 live: the program, i.e. the camera on air). It must never return a frame after the playhead.
        self.video_source = video_source
        self.start_s = start_s                     # the session starts here: nothing earlier is sent (default 0)
        self.stop_flag = threading.Event()
        self.last_reported: dict[str, float] = {}
        self.rtfs: deque[float] = deque(maxlen=5)
        self.thread = threading.Thread(target=self._run, daemon=True, name="specialist")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()

    def status(self, state: str, message: str) -> None:
        rtf = sum(self.rtfs) / len(self.rtfs) if self.rtfs else None
        self.on_status({"state": state, "message": message, "url": self.url,
                        "rtf": round(rtf, 3) if rtf is not None else None})

    # ------------------------------------------------------------------ connection loop
    def _run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                self._session()
            except Exception as error:
                if self.stop_flag.is_set():
                    break
                self.status("error", f"{type(error).__name__}: {str(error)[:150]} (retrying in 5 s)")
                self.log_call({"error": f"{type(error).__name__}: {error}"})
            self.stop_flag.wait(5)

    def _session(self) -> None:
        from websockets.sync.client import connect

        self.status("connecting", f"connecting to {self.url}")
        health = check_health(self.url, self.cfg["connect_timeout_s"])
        fps = self.cfg["send_fps"] or health["fps"]
        size = (int(health.get("frame_width", 398)), int(self.cfg["frame_height"]))
        ws_url = self.url.replace("https://", "wss://").replace("http://", "ws://") + "/stream"
        with connect(ws_url, additional_headers=HEADERS, open_timeout=self.cfg["connect_timeout_s"],
                     max_size=2 ** 22) as ws:
            ws.send(json.dumps({"min_confidence": self.cfg["min_confidence"], "stride_s": self.cfg["stride_s"],
                                "report_delay_s": self.cfg["report_delay_s"], "nms_s": self.cfg["nms_s"]}))
            ready = json.loads(ws.recv(timeout=30))
            self.status("connected", f"{health['model'].split('/')[-1]} on {health.get('gpu')} · "
                                     f"{fps:.2f} fps · window {health.get('window_s')} s")
            self.log_call({"connected": True, "health": health, "ready": ready.get("type")})
            receiver = threading.Thread(target=self._receive, args=(ws,), daemon=True)
            receiver.start()
            self._send_loop(ws, fps, size)

    def _send_loop(self, ws, fps: float, size: tuple[int, int]) -> None:
        """Send every frame that has played since the last send (model frame rate)."""
        video = self.video_source() if self.video_source else self.reader.video()
        sent_until = self.start_s
        quality = [cv2.IMWRITE_JPEG_QUALITY, self.cfg["jpeg_quality"]]
        try:
            while not self.stop_flag.is_set():
                now = self.reader.playhead.now()
                if now - sent_until < 1.0 / fps:
                    time.sleep(0.05)
                    continue
                # Behind by a lot (slow upload)? Skip ahead like a live system: the model only needs
                # a continuous recent window.
                if now - sent_until > 10:
                    sent_until = now - 2.0
                try:
                    frames = list(video.frames(sent_until, now, fps=fps))
                except FutureAccessError:
                    continue
                for t, frame in frames:
                    ok, jpeg = cv2.imencode(".jpg", cv2.resize(frame, size, interpolation=cv2.INTER_AREA), quality)
                    ws.send(struct.pack("<d", t) + jpeg.tobytes())
                sent_until = now
        finally:
            video.close()

    def _receive(self, ws) -> None:
        try:
            for raw in ws:
                message = json.loads(raw)
                if message.get("type") == "error":
                    self.status("error", "server: " + message.get("message", "")[:150])
                    continue
                if message.get("type") != "events":
                    continue
                self.rtfs.append(message.get("rtf", 0))
                logged_at = self.reader.playhead.now()
                self.log_call({**message, "logged_at": round(logged_at, 2)})
                for item in message["events"]:
                    self._handle(item, logged_at)
                self.status("connected", f"last inference {message.get('infer_ms', 0):.0f} ms "
                                         f"on {message.get('frames')} frames")
        except Exception:
            pass   # the connection closed; _run reconnects

    def _handle(self, item: dict, logged_at: float) -> None:
        name, t, confidence = item["class"], float(item["event_time"]), float(item["confidence"])
        if confidence < self.cfg["min_confidence"]:
            return
        if t - self.last_reported.get(name, -1e9) < self.cfg["dedupe_window_s"]:
            return
        self.last_reported[name] = t
        if name in PLAY_SIGNALS:
            self.on_event(None, PLAY_STRENGTH * confidence, name)
            return
        event_type = CLASS_MAP.get(name)
        if event_type is None:
            return
        event = Event(type=event_type, event_time=round(t, 2), logged_at=round(logged_at, 2), source="specialist",
                      confidence=min(1.0, confidence), note=f"action spotting: {name}")
        self.on_event(event, STRENGTH.get(event_type, 0.3) * confidence, name)
