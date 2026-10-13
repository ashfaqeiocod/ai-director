"""specialist_server.py — the SPECIALIST layer server (runs on a Colab GPU).

This file is the source of the server cells in colab/specialist_server.ipynb
(colab/build_notebook.py copies it into the notebook, so the notebook stays self-contained).
It can also run locally for testing:  python colab/specialist_server.py --port 8765   (CPU = slow)

Model: OpenSportsLab/OSL-loc-snbas-2025-e2e (OpenSportsLib, E2E-Spot: RegNetY-008 + GSM + GRU),
trained on SoccerNet Ball Action Spotting 2025, 12 classes. From its config.yaml:
  - frames: 224 x 398 (H x W), RGB, /255, ImageNet mean/std
  - sampling: input 25 fps, stride int(25 / extract_fps=2) = 12  ->  one frame every 0.48 s (~2.08 fps)
  - clip_len: 100 frames  ->  each forward pass sees the last ~48 s
  - output: per frame, softmax over 13 = background + 12 classes

API:
  GET /health   model, classes, expected fps, frame size, window length, GPU name
  WS  /stream   client sends binary messages: 8-byte little-endian float64 timestamp (s) + JPEG bytes.
                Every `stride_s` the server runs the model on the latest window (PAST frames only:
                it only has what was sent) and replies:
                {"type": "events", "events": [{"class", "event_time", "confidence"}], "infer_ms", "rtf", ...}
  WS  /detect   relayed to the football DETECTOR (colab/detector_server.py, its own Python environment on port
                DETECTOR_PORT, default 8766), so one tunnel URL serves both. /health shows it under "detector".

Live detail: the model's GRU looks both ways in time, so the newest frames (no "future" yet) are
unreliable. A frame is only reported once it is `report_delay_s` old (default 2 s): a small,
honest delay instead of many false alarms.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

DETECTOR_PORT = int(os.environ.get("DETECTOR_PORT", "8766"))
MODEL_REPO = "OpenSportsLab/OSL-loc-snbas-2025-e2e"
CLASSES = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
           "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Spotter:
    """Loads the model once; turns a list of frames into per-frame class probabilities."""

    def __init__(self, weights_dir: str | None = None):
        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from opensportslib.apis import LocalizationModel

        self.torch = torch
        config_path = os.path.join(weights_dir, "config.yaml") if weights_dir else \
            hf_hub_download(MODEL_REPO, "config.yaml")
        weights_path = os.path.join(weights_dir, "model.pt") if weights_dir else \
            hf_hub_download(MODEL_REPO, "model.pt")
        data = yaml.safe_load(open(config_path))["DATA"]
        self.classes = list(data["classes"])
        self.height, self.width = int(data["target_height"]), int(data["target_width"])
        self.clip_len = int(data["clip_len"])
        self.stride = max(1, int(data["input_fps"] / data["extract_fps"]))
        self.fps = data["input_fps"] / self.stride
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu = torch.cuda.get_device_name(0) if self.device == "cuda" else "none (CPU)"
        self.model = LocalizationModel(config=config_path, weights=weights_path).model   # E2EModel

    def health(self) -> dict:
        return {"model": MODEL_REPO, "classes": self.classes, "fps": round(self.fps, 4),
                "frame_height": self.height, "frame_width": self.width, "clip_len": self.clip_len,
                "window_s": round(self.clip_len / self.fps, 1), "device": self.device, "gpu": self.gpu}

    def prepare(self, jpeg: bytes) -> np.ndarray:
        """JPEG -> normalized CHW float32 array, exactly like the training loader (OpenCV path)."""
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - MEAN) / STD).transpose(2, 0, 1)

    def predict(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames (L x 3 x H x W) -> probabilities (L x 13). Short windows are padded at the start."""
        L = len(frames)
        if L < self.clip_len:   # pad with copies of the first frame (the model expects clip_len frames)
            frames = [frames[0]] * (self.clip_len - L) + list(frames)
        x = self.torch.from_numpy(np.stack(frames))
        _, prob = self.model.predict(x, use_amp=self.device == "cuda")
        return prob[0][-L:]

    def warm_up(self) -> float:
        start = time.time()
        self.predict([np.zeros((3, self.height, self.width), np.float32)] * self.clip_len)
        return time.time() - start


def find_peaks(times: list[float], prob: np.ndarray, classes: list[str], start: float, end: float,
               min_confidence: float, nms_s: float) -> list[dict]:
    """Events = frames in (start, end] where a class is >= min_confidence and the highest within +-nms_s."""
    events = []
    t = np.asarray(times)
    for c, name in enumerate(classes, start=1):          # column 0 = background
        p = prob[:, c]
        for i in np.where((t > start) & (t <= end) & (p >= min_confidence))[0]:
            near = np.abs(t - t[i]) <= nms_s
            if p[i] >= p[near].max():
                events.append({"class": name, "event_time": round(float(t[i]), 2), "confidence": round(float(p[i]), 3)})
    return events


class Session:
    """One connected client: a ring buffer of the latest frames + inference every stride_s."""

    def __init__(self, spotter: Spotter, settings: dict):
        self.spotter = spotter
        self.min_confidence = float(settings.get("min_confidence", 0.3))
        self.stride_s = float(settings.get("stride_s", 2.0))
        self.report_delay_s = float(settings.get("report_delay_s", 2.0))
        self.nms_s = float(settings.get("nms_s", 1.0))
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=spotter.clip_len)
        self.last_infer = -1e9
        self.reported_until = -1e9
        self.busy = False

    def add(self, t: float, jpeg: bytes) -> bool:
        """Store a frame. Returns True when an inference is due."""
        if self.frames and t <= self.frames[-1][0]:
            return False                    # out of order / duplicate: ignore
        self.frames.append((t, self.spotter.prepare(jpeg)))
        return not self.busy and t - self.last_infer >= self.stride_s

    def infer(self) -> dict:
        times = [t for t, _ in self.frames]
        start = time.time()
        prob = self.spotter.predict([f for _, f in self.frames])
        infer_s = time.time() - start
        now = times[-1]
        self.last_infer = now
        until = now - self.report_delay_s
        events = find_peaks(times, prob, self.spotter.classes, self.reported_until, until,
                            self.min_confidence, self.nms_s)
        self.reported_until = max(self.reported_until, until)
        return {"type": "events", "events": events, "infer_ms": round(infer_s * 1000, 1),
                "rtf": round(infer_s / self.stride_s, 3), "window": [round(times[0], 2), round(now, 2)],
                "frames": len(times), "reported_until": round(until, 2)}


def detector_health() -> dict:
    """The detector process's /health ({"ready": False} while it is not running or still loading)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{DETECTOR_PORT}/health", timeout=1) as r:
            return json.loads(r.read())
    except Exception as error:                       # noqa: BLE001 — shown on the dashboard
        return {"ready": False, "error": type(error).__name__}


def make_app(spotter: Spotter | None):
    app = FastAPI(title="Story Reader specialist")

    @app.get("/health")
    def health() -> dict:
        return {**(spotter.health() if spotter else {"model": "none", "fps": 0}), "detector": detector_health()}

    @app.websocket("/detect")
    async def detect(ws: WebSocket) -> None:
        """Relay to the detector process: frames in, detections back (both directions at once)."""
        import websockets

        await ws.accept()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{DETECTOR_PORT}/detect", max_size=2 ** 24) as inner:
                async def back() -> None:
                    async for reply in inner:
                        await ws.send_text(reply if isinstance(reply, str) else reply.decode())

                task = asyncio.create_task(back())
                try:
                    while True:
                        await inner.send(await ws.receive_bytes())
                finally:
                    task.cancel()
        except WebSocketDisconnect:
            pass
        except Exception as error:                   # noqa: BLE001 — the detector is not running
            await ws.send_json({"type": "error", "message": f"detector: {type(error).__name__}: {error}"})
            await ws.close()

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        session = Session(spotter, {})
        loop = asyncio.get_running_loop()
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text"):          # settings: {"min_confidence": .., "stride_s": .., ...}
                    session = Session(spotter, json.loads(message["text"]))
                    await ws.send_json({"type": "ready", **spotter.health()})
                    continue
                data = message.get("bytes") or b""
                if len(data) <= 8:
                    continue
                t = struct.unpack("<d", data[:8])[0]
                if session.add(t, data[8:]):
                    # Run the model in a thread so frames keep arriving meanwhile. While it is busy,
                    # new frames just go into the buffer (no backlog: like a live system).
                    session.busy = True

                    async def run() -> None:
                        try:
                            result = await loop.run_in_executor(None, session.infer)
                            await ws.send_json(result)
                        except Exception as error:
                            await ws.send_json({"type": "error", "message": f"{type(error).__name__}: {error}"})
                        finally:
                            session.busy = False

                    asyncio.create_task(run())
        except WebSocketDisconnect:
            pass

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weights-dir", default=None, help="folder with config.yaml + model.pt (default: download)")
    parser.add_argument("--detector-only", action="store_true", help="no action model (test the /detect relay)")
    args = parser.parse_args()
    import uvicorn

    spotter = None if args.detector_only else Spotter(args.weights_dir)
    if spotter:
        print("warm-up:", round(spotter.warm_up(), 2), "s |", spotter.health())
    uvicorn.run(make_app(spotter), host="127.0.0.1", port=args.port, log_level="warning")
