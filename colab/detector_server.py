"""detector_server.py — the football DETECTOR on a Colab GPU (MVP 3 live test: detection while the match plays).

The Mac needs ~0.4 s per camera frame; live needs 3 cameras x 5 frames/s = 15 frames/s. A T4 GPU does it.

It runs in its OWN Python environment (the notebook makes one with uv): Roboflow's `inference-gpu` pins many
packages and must not break the action model's torch next to it. The main server (specialist_server.py) relays
WS /detect to this process, so one tunnel URL serves both.
It can also run on the Mac for testing:  venv/bin/python colab/detector_server.py --port 8766  (CoreML, slow)

Same output as the Mac's `src/detect.Detector.detect` (the live director must see what the cached runs saw):
  - the frame is padded to 16:9 on a black canvas (pad_to_aspect), the model runs at its 1280 input;
  - players = player / goalkeeper / referee >= 0.3; ball = the single best ball >= 0.3;
  - boxes in the ORIGINAL picture's pixels, rounded to 0.1 px.

API:
  GET /health   {"model", "ready", "ms_per_frame", "providers"}
  WS  /detect   client sends binary: 4-byte little-endian header length + JSON header + JPEG bytes.
                header {"frame": int, "sent": float, "items": [{"cam": int, "size": int}, ...]} = several cameras of
                one frame, JPEGs one after another (sizes in the header); {"frame", "cam", "sent"} = one camera.
                Reply (JSON): {"type": "detections", "frame", "sent", "infer_ms", "cams": {"1": {"players": [...],
                "ball": {...} | null}, ...}}. The cameras of one message run as ONE batch on the GPU.
Key: ROBOFLOW_API_KEY in the environment (never printed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_CLASSES = {"player", "goalkeeper", "referee"}
ASPECT = 1.7778                      # src/detect: model_aspect
PLAYER_CONFIDENCE = 0.3              # config.yaml detection.player_confidence
BALL_CONFIDENCE = 0.3                # config.yaml detection.ball_confidence


def pad_to_aspect(frame: np.ndarray, aspect: float) -> tuple[np.ndarray, int]:
    """Copy of src/detect.pad_to_aspect: centre the frame on a black canvas `aspect` times wider than high."""
    h, w = frame.shape[:2]
    width = int(round(h * aspect))
    if w >= width:
        return frame, 0
    x = (width - w) // 2
    canvas = np.zeros((h, width, 3), dtype=frame.dtype)
    canvas[:, x:x + w] = frame
    return canvas, x


def _round(box, x_offset: int = 0) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [round(x1 - x_offset, 1), round(y1, 1), round(x2 - x_offset, 1), round(y2, 1)]


def parse(data: bytes) -> tuple[dict, list[tuple[int, bytes]]]:
    """One client message -> (header, [(cam, jpeg), ...])."""
    n = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4:4 + n])
    body = data[4 + n:]
    if "items" not in header:
        return header, [(int(header["cam"]), body)]
    out, k = [], 0
    for item in header["items"]:
        out.append((int(item["cam"]), body[k:k + int(item["size"])]))
        k += int(item["size"])
    return header, out


class FootballDetector:
    """Loads the Roboflow model once (GPU if onnxruntime has CUDA) and detects a batch of pictures."""

    def __init__(self):
        import onnxruntime as ort

        have = ort.get_available_providers()
        if "CUDAExecutionProvider" in have and hasattr(ort, "preload_dlls") and os.environ.get("DETECTOR_OWN_CUDA") == "1":
            # Colab: load the CUDA / cuDNN that came WITH this onnxruntime (pip nvidia-* packages in its own environment).
            # torch's newer cuDNN (9.24 on the first Colab run) crashed the first convolution: CUDNN_BACKEND_API_FAILED.
            ort.preload_dlls()
        print("onnxruntime", ort.__version__, "providers", have, flush=True)
        if "ONNXRUNTIME_EXECUTION_PROVIDERS" not in os.environ:      # GPU first (Colab), Apple chip (Mac), CPU
            wanted = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in have] + ["CPUExecutionProvider"]
            os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[" + ",".join(wanted) + "]"
        import supervision as sv
        from inference import get_model

        self.sv = sv
        self.model = get_model(MODEL_ID, api_key=os.environ["ROBOFLOW_API_KEY"])
        self.ms: list[float] = []

    def providers(self) -> list[str]:
        try:
            import onnxruntime as ort

            return ort.get_available_providers()
        except Exception:                            # noqa: BLE001 — informative only
            return []

    def detect_many(self, frames: list[np.ndarray]) -> list[dict]:
        padded = [pad_to_aspect(f, ASPECT) for f in frames]
        start = time.time()
        results = self.model.infer([p for p, _ in padded], confidence=min(PLAYER_CONFIDENCE, BALL_CONFIDENCE))
        self.ms.append((time.time() - start) * 1000 / max(1, len(frames)))
        self.ms = self.ms[-50:]
        return [self._one(self.sv.Detections.from_inference(r), x) for r, (_, x) in zip(results, padded)]

    def _one(self, dets, offset: int) -> dict:
        names = dets.data.get("class_name", np.array([]))
        players = [{"box": _round(b, offset), "cls": str(n), "conf": round(float(c), 3)}
                   for b, c, n in zip(dets.xyxy, dets.confidence, names)
                   if n in PLAYER_CLASSES and c >= PLAYER_CONFIDENCE]
        balls = [(b, c) for b, c, n in zip(dets.xyxy, dets.confidence, names) if n == "ball" and c >= BALL_CONFIDENCE]
        ball = None
        if balls:
            box, conf = max(balls, key=lambda bc: bc[1])
            ball = {"box": _round(box, offset), "conf": round(float(conf), 3)}
        return {"players": players, "ball": ball}

    def warm_up(self, n: int = 10, size: tuple[int, int] = (1280, 960), cams: int = 3) -> float:
        """ms per camera frame on random pictures (batches of `cams`)."""
        frame = np.random.randint(0, 255, (size[1], size[0], 3), np.uint8)
        self.detect_many([frame] * cams)            # first call compiles / allocates
        self.ms.clear()
        for _ in range(n):
            self.detect_many([frame] * cams)
        return float(np.mean(self.ms))

    def health(self) -> dict:
        return {"model": MODEL_ID, "ready": True, "ms_per_frame": round(float(np.mean(self.ms)), 1) if self.ms else None,
                "providers": self.providers()}


def make_app(detector: FootballDetector) -> FastAPI:
    app = FastAPI(title="football detector")

    @app.get("/health")
    def health() -> dict:
        return detector.health()

    @app.websocket("/detect")
    async def detect(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        try:
            while True:
                data = await ws.receive_bytes()
                header, items = parse(data)
                frames = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for _, j in items]
                start = time.time()
                found = await loop.run_in_executor(None, detector.detect_many, frames)
                await ws.send_json({"type": "detections", "frame": header["frame"], "sent": header.get("sent"),
                                    "infer_ms": round((time.time() - start) * 1000, 1),
                                    "cams": {str(c): r for (c, _), r in zip(items, found)}})
        except WebSocketDisconnect:
            pass

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    import uvicorn

    detector = FootballDetector()
    print(f"detector ready: {detector.warm_up(5):.0f} ms per camera frame | providers {detector.providers()}", flush=True)
    uvicorn.run(make_app(detector), host="127.0.0.1", port=args.port, log_level="warning")
