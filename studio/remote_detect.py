"""remote_detect.py — the football detector on Colab's GPU, used by the live loop while the match plays.

The M1 needs ~0.4 s per camera frame, live needs 3 cameras x 5 frames per second. So the live loop hands each
detection frame (full-resolution pictures of every camera) to a RemoteDetector, which sends it to the Colab
server (colab/detector_server.py behind WS /detect) and collects the answers as they come back.

  detector = RemoteDetector(url, fps, clock, cfg)   # clock() = the live match clock (seconds)
  detector.start()
  detector.submit(frame, {cam: picture})           # never blocks: the sending thread encodes + sends
  detector.submit(frame)                           # ... or the sending thread reads the full pictures itself
                                                   #     (FullFrameReader): the live loop never waits for them
  for r in detector.results(): ...                  # answers that arrived since the last call
  detector.status()                                 # for the page: connected, GPU ms, round trip, dropped

Live rules (like a real network link):
  - at most `max_in_flight` frames on their way; a newer frame replaces an older one still WAITING to be sent
    (the old one is dropped and counted: a stale picture is worth nothing live);
  - each answer says how old it is when it arrives (age_s = live clock - its frame time); the live loop
    ignores answers older than `max_age_s` (studio/live.py) and uses the cached detection instead, if any.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from story.specialist_client import HEADERS, check_health, normalize_url


def encode(frame: int, sent: float, images: dict[int, np.ndarray], quality: int,
           width: int = 0) -> tuple[bytes, dict[int, float]]:
    """One message: 4-byte header length + JSON header + the JPEGs one after another. Returns (message, scales):
    scales[cam] = original width / sent width, to turn that camera's boxes back into original pixels."""
    jpegs, scales = {}, {}
    for cam, img in images.items():
        scales[cam] = 1.0
        if width and img.shape[1] > width:
            scales[cam] = img.shape[1] / width
            img = cv2.resize(img, (width, round(img.shape[0] / scales[cam])), interpolation=cv2.INTER_AREA)
        jpegs[cam] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    header = json.dumps({"frame": frame, "sent": sent,
                         "items": [{"cam": c, "size": len(j)} for c, j in jpegs.items()]}).encode()
    return struct.pack("<I", len(header)) + header + b"".join(jpegs.values()), scales


def rescale(det: dict, scale: float) -> dict:
    """Boxes from the sent (smaller) picture back to the original camera's pixels."""
    if scale == 1.0:
        return det
    box = lambda b: [round(v * scale, 1) for v in b]                       # noqa: E731
    return {"players": [{**p, "box": box(p["box"])} for p in det.get("players", [])],
            "ball": {**det["ball"], "box": box(det["ball"]["box"])} if det.get("ball") else None}


class FullFrameReader:
    """Full-resolution pictures of every camera at a detection frame, read in the sender's thread (not the live
    loop: three full-HD decodes per detection step made the loop drop to ~8 frames/s with the browser open).
    Moves forward by grabbing (decodes, cheap); jumps (seek) only when far behind."""

    def __init__(self, videos: dict[int, Path], fps: float):
        self.caps = {c: [cv2.VideoCapture(str(v)), 0] for c, v in videos.items()}
        self.fps = fps

    def read(self, frame: int) -> dict[int, np.ndarray]:
        out = {}
        for c, item in self.caps.items():
            cap, pos = item
            if frame < pos or frame - pos > round(self.fps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                pos = frame
            while pos < frame:
                cap.grab()
                pos += 1
            ok, img = cap.read()
            item[1] = pos + 1
            if ok:
                out[c] = img
        return out

    def close(self) -> None:
        for cap, _ in self.caps.values():
            cap.release()


class RemoteDetector:
    def __init__(self, url: str, fps: float, clock: Callable[[], float], cfg: dict,
                 reader: FullFrameReader | None = None):
        self.reader = reader                                  # reads the pictures when submit() gets none
        self.url = normalize_url(url)
        self.fps, self.clock, self.cfg = fps, clock, cfg
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = threading.Event()
        self.waiting: tuple[int, dict] | None = None          # the newest frame not sent yet
        self.in_flight: dict[int, float] = {}                 # frame -> wall time sent
        self.scales: dict[int, dict[int, float]] = {}         # frame -> {cam: scale} (kept after "lost": late answers)
        self.arrived: list[dict] = []
        self.rtts: deque[float] = deque(maxlen=50)
        self.gpu_ms: deque[float] = deque(maxlen=50)
        self.sent_bytes: deque[tuple[float, int]] = deque(maxlen=50)
        self.counts = {"sent": 0, "answered": 0, "dropped": 0, "lost": 0, "errors": 0}
        self.state, self.message, self.health = "connecting", "", {}
        self.thread = threading.Thread(target=self._run, daemon=True, name="remote-detect")

    # ------------------------------------------------------------------ live loop side
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        self.wake.set()

    def submit(self, frame: int, images: dict[int, np.ndarray] | None = None) -> None:
        """Queue a detection frame (never blocks). A frame still waiting to be sent is replaced (dropped)."""
        with self.lock:
            if self.waiting is not None:
                self.counts["dropped"] += 1
            self.waiting = (frame, images)
        self.wake.set()

    def results(self) -> list[dict]:
        with self.lock:
            out, self.arrived = self.arrived, []
        return out

    def connected(self) -> bool:
        return self.state == "connected"

    def on_its_way(self, frame: int) -> bool:
        """Is an answer for this frame still possible? False if it was dropped before sending or the link is down
        (the live loop then uses the cache at once instead of waiting max_age_s)."""
        with self.lock:
            waiting = self.waiting is not None and self.waiting[0] == frame
            return self.state == "connected" and (frame in self.in_flight or waiting)

    def status(self) -> dict:
        with self.lock:
            now = time.time()
            recent = [(t, n) for t, n in self.sent_bytes if now - t <= 5]
            mbit = 8 * sum(n for _, n in recent) / 5 / 1e6 if recent else 0.0
            return {"state": self.state, "message": self.message, "url": self.url,
                    "gpu_ms": round(float(np.mean(self.gpu_ms)), 1) if self.gpu_ms else None,
                    "rtt_ms": round(float(np.mean(self.rtts)), 0) if self.rtts else None,
                    "upload_mbit": round(mbit, 1), "in_flight": len(self.in_flight), **self.counts}

    # ------------------------------------------------------------------ connection thread
    def _run(self) -> None:
        try:
            self._connect_loop()
        finally:
            if self.reader:
                self.reader.close()

    def _connect_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                self._session()
            except Exception as error:                 # noqa: BLE001 — reconnect, the live loop goes on (cached)
                if self.stop_flag.is_set():
                    break
                with self.lock:
                    self.state, self.message = "error", f"{type(error).__name__}: {str(error)[:150]} (retrying in 3 s)"
                    self.counts["errors"] += 1
                    self.in_flight.clear()
            self.stop_flag.wait(3)

    def _session(self) -> None:
        from websockets.sync.client import connect

        with self.lock:
            self.state, self.message = "connecting", f"connecting to {self.url}"
        health = check_health(self.url, 10).get("detector") or {}
        if not health.get("ready"):
            raise RuntimeError(f"the Colab server has no detector running ({health.get('error') or 'not ready'})")
        ws_url = self.url.replace("https://", "wss://").replace("http://", "ws://") + "/detect"
        with connect(ws_url, additional_headers=HEADERS, open_timeout=10, max_size=2 ** 24) as ws:
            with self.lock:
                self.health = health
                self.state, self.message = "connected", f"{health.get('model', '?').split('/')[0]} · " \
                                                        f"{', '.join(health.get('providers', [])[:1])}"
            receiver = threading.Thread(target=self._receive, args=(ws,), daemon=True, name="remote-detect-rx")
            receiver.start()
            while not self.stop_flag.is_set() and receiver.is_alive():
                self.wake.wait(0.2)
                self.wake.clear()
                self._forget_lost()
                with self.lock:
                    if self.waiting is None or len(self.in_flight) >= self.cfg["max_in_flight"]:
                        continue
                    frame, images = self.waiting
                    self.waiting = None
                if images is None:
                    images = self.reader.read(frame)
                message, scales = encode(frame, round(time.time(), 3), images, self.cfg["jpeg_quality"],
                                         self.cfg.get("send_width") or 0)
                with self.lock:
                    self.in_flight[frame] = time.time()
                    self.scales[frame] = scales
                    for old in [k for k in self.scales if k < frame - 3000]:
                        del self.scales[old]
                    self.counts["sent"] += 1
                    self.sent_bytes.append((time.time(), len(message)))
                ws.send(message)
            if not receiver.is_alive() and not self.stop_flag.is_set():
                raise ConnectionError("the detector connection closed")

    def _forget_lost(self) -> None:
        """An answer that never came (tunnel hiccup) must not block the in-flight slots forever."""
        now = time.time()
        with self.lock:
            for frame, sent in list(self.in_flight.items()):
                if now - sent > self.cfg["lost_after_s"]:
                    del self.in_flight[frame]
                    self.counts["lost"] += 1

    def _receive(self, ws) -> None:
        try:
            for raw in ws:
                message = json.loads(raw)
                if message.get("type") == "error":
                    with self.lock:
                        self.message = "server: " + message.get("message", "")[:150]
                        self.counts["errors"] += 1
                    continue
                if message.get("type") != "detections":
                    continue
                received = time.time()
                frame = int(message["frame"])
                with self.lock:
                    sent = self.in_flight.pop(frame, message.get("sent") or received)
                    scales = self.scales.pop(frame, {})
                    rtt = (received - sent) * 1000
                    self.rtts.append(rtt)
                    cams = len(message["cams"]) or 1
                    self.gpu_ms.append(message.get("infer_ms", 0) / cams)
                    self.counts["answered"] += 1
                    self.arrived.append({"frame": frame, "time": frame / self.fps, "sent": sent, "received": received,
                                         "rtt_ms": round(rtt, 1), "infer_ms": message.get("infer_ms"),
                                         "age_s": round(self.clock() - frame / self.fps, 3),
                                         "cams": {int(c): rescale(d, scales.get(int(c), 1.0))
                                                  for c, d in message["cams"].items()}})
                self.wake.set()                        # a slot is free: send the next frame
        except Exception:                              # noqa: BLE001 — the connection closed; _run reconnects
            pass
