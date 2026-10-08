"""server.py — Step 9: the local web dashboard.

Start:  venv/bin/python -m dashboard.server      then open http://localhost:8000

How it works (frontend-dev view):
  - The Engine runs in a background thread (like a Web Worker), one frame at a time.
  - Every frame, the Renderer draws the program + 3 annotated camera tiles. They are
    JPEG-encoded and dropped into tiny "latest frame" buffers (max 2, old frames
    are thrown away), so a slow browser never makes the engine lag behind.
  - /stream/<name> serves those JPEGs as MJPEG: one never-ending HTTP response that
    keeps replacing the image. A plain <img src="/stream/cam1"> plays it.
  - /ws is a WebSocket. It sends the per-frame state (max ~10 per second) and every
    event (CUT, BALL_LOST, BALL_FOUND, INFO, ERROR) immediately.
  - REST endpoints control the run: POST /start, /pause, /stop, GET/POST /config, GET /files/{name}.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from src.common import cam_path, detections_path, load_config, path
from src.director import DecisionLog
from src.engine import LIVE, REPLAY, Engine
from src.render import Renderer

STATIC = Path(__file__).parent / "static"
STREAMS = ["program", "cam1", "cam2", "cam3"]
DOWNLOADS = {"program.mp4": "render.program_path", "debug.mp4": "render.debug_path",
             "director_log.csv": "director.log_csv"}


# ---------------------------------------------------------------- frame buffers
class FrameChannel:
    """Holds the newest JPEGs for one stream. maxlen=2: older frames are dropped automatically."""

    def __init__(self):
        self.frames: deque[bytes] = deque(maxlen=2)
        self.seq = 0
        self.cond = threading.Condition()

    def put(self, jpeg: bytes) -> None:
        with self.cond:
            self.frames.append(jpeg)
            self.seq += 1
            self.cond.notify_all()

    def wait_newer(self, seq: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """Block until a frame newer than `seq` exists; return (new seq, newest jpeg)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.seq, (self.frames[-1] if self.frames else None)


def encode_jpeg(image: np.ndarray, max_width: int, quality: int) -> bytes:
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ---------------------------------------------------------------- run controller
class Controller:
    """Owns the background engine thread and everything the UI can ask about it."""

    def __init__(self):
        self.cfg = load_config()
        self.channels = {name: FrameChannel() for name in STREAMS}
        self.status = "idle"            # idle / loading / running / paused / finished / stopped / error
        self.mode = REPLAY
        self.speed = 1.0
        self.overrides = {"min_shot_seconds": self.cfg["director"]["min_shot_seconds"],
                          "hysteresis_seconds": self.cfg["director"]["hysteresis_seconds"]}
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.detector = None            # loaded once, reused for every live run
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.history: list[dict] = []   # every message of the current run, for late joiners
        self.files_ready = False
        self.pool = ThreadPoolExecutor(max_workers=4)   # parallel JPEG encoding

    # ----- info for the UI
    def replay_available(self) -> bool:
        return all(detections_path(self.cfg, c).exists() for c in range(1, 4))

    def config_view(self) -> dict:
        return {**self.overrides, "status": self.status, "mode": self.mode, "speed": self.speed,
                "replay_available": self.replay_available(), "files_ready": self.files_ready,
                "cams_ready": all(cam_path(self.cfg, c).exists() for c in range(1, 4))}

    # ----- messages to browsers
    def publish(self, message: dict, keep: bool = True) -> None:
        """Send a JSON message to every connected WebSocket (thread-safe)."""
        if keep:
            self.history.append(message)
        if self.loop is None:
            return
        for queue in list(self.clients):
            self.loop.call_soon_threadsafe(_offer, queue, message)

    def event(self, kind: str, message: str, **extra) -> None:
        """An event goes to the browser immediately AND into outputs/run_log.jsonl."""
        item = {"type": "event", "kind": kind, "message": message,
                "wall_time": datetime.now().isoformat(timespec="seconds"), **extra}
        self.publish(item)
        log_path = path(self.cfg["dashboard"]["run_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(item) + "\n")

    # ----- controls
    def start(self, mode: str, speed: float) -> dict:
        if self.status == "paused":
            self.pause_flag.clear()
            self.status = "running"
            self.event("INFO", "resumed")
            return self.config_view()
        if self.thread and self.thread.is_alive():
            raise HTTPException(409, "A run is already in progress")
        if mode == REPLAY and not self.replay_available():
            raise HTTPException(400, "Replay needs cached detections: do a Live run first")
        self.mode, self.speed = mode, speed
        self.stop_flag.clear()
        self.pause_flag.clear()
        self.history = []
        self.files_ready = False
        self.status = "loading"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.config_view()

    def pause(self) -> dict:
        if self.status == "running":
            self.pause_flag.set()
            self.status = "paused"
            self.event("INFO", "paused")
        return self.config_view()

    def stop(self) -> dict:
        if self.thread and self.thread.is_alive():
            self.stop_flag.set()
            self.pause_flag.clear()
            self.thread.join(timeout=10)
        self.status = "idle"
        self.publish({"type": "reset"}, keep=False)
        self.publish({"type": "status", **self.config_view()}, keep=False)
        return self.config_view()

    # ----- the engine thread
    def _run(self) -> None:
        try:
            self._run_engine()
        except Exception as error:
            traceback.print_exc()
            self.status = "error"
            self.event("ERROR", f"{type(error).__name__}: {error}")
            self.publish({"type": "status", **self.config_view()})

    def _run_engine(self) -> None:
        cfg = load_config()
        cfg["director"].update(self.overrides)          # UI values apply to this run
        dash = cfg["dashboard"]
        self.cfg = cfg

        realtime = cfg.get("live", {}).get("realtime", False)
        if realtime and self.mode == LIVE:
            self.event("INFO", "real-time mode: starting the detector process (takes a few seconds)...")
        if self.mode == LIVE and self.detector is None and not realtime:
            self.event("INFO", "loading detection model (first live run only)...")
            from src.detect import Detector
            self.detector = Detector(cfg)
        if self.stop_flag.is_set():  # Stop was pressed while the model was loading
            return
        engine = Engine(cfg, mode=self.mode, detector=self.detector)
        renderer = Renderer(cfg, engine.fps)
        log = DecisionLog(path(cfg["director"]["log_csv"]))

        self.status = "running"
        self.publish({"type": "start", "mode": self.mode, "speed": self.speed, "fps": engine.fps,
                      "total_frames": engine.total_frames, "start_cam": cfg["director"]["start_cam"],
                      **self.overrides})
        self.event("INFO", f"{self.mode} run started (min shot {self.overrides['min_shot_seconds']}s, "
                           f"hysteresis {self.overrides['hysteresis_seconds']}s)")
        if self.mode == LIVE:
            live = cfg.get("live", {})
            self.event("INFO", f"live settings: parallel_cams={live.get('parallel_cams', False)}, "
                               f"detect_every_n={live.get('detect_every_n', 1)}, realtime={live.get('realtime', False)}")

        stats = RunStats(engine.fps)
        clock_start = time.time()
        last_state = 0.0
        for result in engine.frames():
            # Pause: wait here, then shift the replay clock so playback doesn't jump ahead.
            if self.pause_flag.is_set():
                paused_at = time.time()
                while self.pause_flag.is_set() and not self.stop_flag.is_set():
                    time.sleep(0.05)
                clock_start += time.time() - paused_at
            if self.stop_flag.is_set():
                engine.stop()
                break

            program, tiles = renderer.render(result)
            log.write(result.index, result.time, result.chosen, result.scores, result.reason)
            # Encode the 4 JPEGs in parallel (OpenCV releases the GIL while encoding).
            jobs = [("program", program, dash["program_max_width"])]
            jobs += [(f"cam{c + 1}", tile, dash["cam_max_width"]) for c, tile in enumerate(tiles)]
            jpegs = self.pool.map(lambda job: encode_jpeg(job[1], job[2], dash["jpeg_quality"]), jobs)
            for (name, _, _), jpeg in zip(jobs, jpegs):
                self.channels[name].put(jpeg)

            stats.add(result)
            for e in result.events:
                self.event(e["type"], e["message"], frame=result.index, time=round(result.time, 2),
                           **{k: v for k, v in e.items() if k not in ("type", "message")})

            now = time.time()
            if now - last_state >= 1.0 / dash["ws_max_rate"] or result.index == engine.total_frames - 1:
                last_state = now
                self.publish(stats.state(result), keep=False)

            # Replay plays at real time x speed. Live runs as fast as the M1 can go.
            if self.mode == REPLAY:
                target = clock_start + (result.index + 1) / engine.fps / self.speed
                delay = target - time.time()
                if delay > 0:
                    time.sleep(delay)

        if self.stop_flag.is_set():
            log.abort()
            renderer.abort()
            self.event("INFO", "run stopped")
            return
        self.event("INFO", "finishing videos (ffmpeg)...")
        log.close()
        renderer.close()
        self.files_ready = True
        self.status = "finished"
        self.event("INFO", f"run finished: {stats.cuts} cuts, avg shot {stats.avg_shot():.2f}s, "
                           f"avg processing {stats.avg_fps():.1f} fps")
        # "type" goes last so it is not overwritten by the "type" inside stats.state().
        self.publish({**stats.state(None), **self.config_view(), "type": "finished"})


def _offer(queue: asyncio.Queue, message: dict) -> None:
    """Put a message in a client's queue; if that client is hopelessly behind, drop it."""
    if queue.qsize() < 500:
        queue.put_nowait(message)


class RunStats:
    """Running numbers for the stats strip."""

    def __init__(self, fps: float):
        self.fps = fps
        self.frames = 0
        self.ball_frames = [0, 0, 0]
        self.cuts = 0
        self.started = time.time()
        self.processing_fps = 0.0
        self.detection_fps = 0.0

    def add(self, result) -> None:
        self.frames += 1
        for c, cam in enumerate(result.cams):
            self.ball_frames[c] += cam.ball_box is not None
        self.cuts += sum(e["type"] == "CUT" for e in result.events)
        self.processing_fps = result.processing_fps
        self.detection_fps = result.detection_fps

    def avg_shot(self) -> float:
        return self.frames / self.fps / (self.cuts + 1)

    def avg_fps(self) -> float:
        return self.frames / max(1e-6, time.time() - self.started)

    def state(self, result) -> dict:
        state = {"type": "state", "processing_fps": self.processing_fps, "detection_fps": self.detection_fps,
                 "cuts": self.cuts,
                 "avg_shot": round(self.avg_shot(), 2),
                 "ball_rate": [round(b / max(1, self.frames), 3) for b in self.ball_frames]}
        if result is not None:
            state.update({"frame": result.index, "time": round(result.time, 2), "chosen": result.chosen,
                          "scores": result.scores, "ball": [c.ball.status for c in result.cams],
                          "reason": result.reason})
        return state


# ---------------------------------------------------------------- web app
app = FastAPI(title="AI Sports Director")
ctl = Controller()


@app.on_event("startup")
async def _remember_loop() -> None:
    ctl.loop = asyncio.get_running_loop()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/stream/{name}")
def stream(name: str) -> StreamingResponse:
    if name not in ctl.channels:
        raise HTTPException(404, "unknown stream")
    channel = ctl.channels[name]

    def generate():
        seq = -1
        while True:
            seq, jpeg = channel.wait_newer(seq)
            if jpeg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + \
                  b"\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    ctl.clients.add(queue)
    try:
        # A browser that connects mid-run first gets everything that already happened.
        await ws.send_json({"type": "hello", **ctl.config_view()})
        for message in list(ctl.history):
            await ws.send_json(message)
        while True:
            await ws.send_json(await queue.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        ctl.clients.discard(queue)


@app.post("/start")
def start(body: dict | None = None) -> dict:
    body = body or {}
    mode = body.get("mode", ctl.mode)
    if mode not in (LIVE, REPLAY):
        raise HTTPException(400, "mode must be 'live' or 'replay'")
    speed = float(body.get("speed", ctl.speed))
    if speed not in (0.5, 1.0, 2.0):
        raise HTTPException(400, "speed must be 0.5, 1 or 2")
    return ctl.start(mode, speed)


@app.post("/pause")
def pause() -> dict:
    return ctl.pause()


@app.post("/stop")
def stop() -> dict:
    return ctl.stop()


@app.get("/config")
def get_config() -> dict:
    return ctl.config_view()


@app.post("/config")
def set_config(body: dict) -> dict:
    """Update min shot length / hysteresis (used on the next Start) and replay speed (immediate)."""
    for key in ("min_shot_seconds", "hysteresis_seconds"):
        if key in body:
            value = float(body[key])
            if not 0 <= value <= 30:
                raise HTTPException(400, f"{key} must be between 0 and 30")
            ctl.overrides[key] = value
    if "speed" in body:
        ctl.speed = float(body["speed"])
    return ctl.config_view()


@app.get("/files/{name}")
def files(name: str) -> FileResponse:
    # Only these 3 files can be downloaded; anything else is refused.
    if name not in DOWNLOADS:
        raise HTTPException(404, "unknown file")
    section, key = DOWNLOADS[name].split(".")
    file = path(ctl.cfg[section][key])
    if not file.exists():
        raise HTTPException(404, "file not ready yet")
    return FileResponse(file, filename=name)


if __name__ == "__main__":
    dash = ctl.cfg["dashboard"]
    print(f"Dashboard: http://localhost:{dash['port']}")
    uvicorn.run(app, host=dash["host"], port=dash["port"], log_level="warning")
