"""story_api.py — the dashboard's Story tab (MVP 2): web routes, all under /story.

server.py includes this router, so one server runs both tabs:
    http://localhost:8000/        MVP 1 (camera switching, unchanged)
    http://localhost:8000/story   MVP 2 (Live Story Reader)

How it works (frontend-dev view):
  - The page plays data/clips/<name>/clip.mp4 in a normal <video> tag, with sound.
  - Over the WebSocket /story/ws the page sends {"type": "tick", "t": currentTime, "paused": ...}
    about 5 times per second. That is the master clock: the StoryEngine never reads past it.
  - The same WebSocket carries everything back to the page (story updates, events, status...).
  - Plain REST for the rest: upload / URL intake, clip list, start / stop, settings.
"""
from __future__ import annotations

import asyncio
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from src.download import InputError
from story import intake
from story.common import clips_dir, load_config
from story.engine import StoryEngine

STATIC = Path(__file__).parent / "static"
MODES = ("fast_only", "specialist_only", "gemini_only", "hybrid")

router = APIRouter(prefix="/story")


class StoryController:
    """Owns the current story run and the list of connected browsers."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.engine: StoryEngine | None = None
        self.clip: str | None = None
        self.intake_jobs: dict[str, dict] = {}     # job id -> {"status", "message", "name"}
        self.lock = threading.Lock()

    # ----- messages to browsers (thread-safe: engine threads call this too)
    def publish(self, message: dict) -> None:
        if self.loop is None:
            return
        for queue in list(self.clients):
            self.loop.call_soon_threadsafe(_offer, queue, message)

    def status(self) -> dict:
        cfg = load_config()["story"]
        return {"type": "status", "running": self.engine is not None, "clip": self.clip,
                "mode": self.engine.mode if self.engine else cfg["mode"],
                "send_audio": self.engine.send_audio if self.engine else cfg["send_audio_to_llm"],
                "clips": intake.list_clips(), "intake_jobs": list(self.intake_jobs.values())}

    # ----- clip intake (runs in a background thread: downloads and ffmpeg take a while)
    def start_intake(self, source: str, name: str | None, original_name: str | None,
                     cleanup: Path | None = None) -> dict:
        job = {"id": uuid.uuid4().hex[:8], "status": "working", "name": name or original_name or source,
               "message": "importing (download / convert)..."}
        self.intake_jobs[job["id"]] = job
        self.publish({"type": "intake", **job})

        def work() -> None:
            try:
                meta = intake.intake(source, name, original_name)
                job.update(status="done", name=meta["name"],
                           message=f"ready: {meta['duration']:.0f} s, {meta['width']}x{meta['height']}")
            except InputError as error:
                job.update(status="error", message=str(error))
            except Exception as error:
                traceback.print_exc()
                job.update(status="error", message=f"{type(error).__name__}: {error}")
            finally:
                if cleanup:
                    shutil.rmtree(cleanup, ignore_errors=True)
            self.publish({"type": "intake", **job})
            self.publish(self.status())

        threading.Thread(target=work, daemon=True).start()
        return job

    # ----- runs
    def start_run(self, clip: str, mode: str, send_audio: bool) -> dict:
        folder = intake.clip_dir(clip)
        if not (folder / "meta.json").exists():
            raise HTTPException(404, f"clip '{clip}' not found")
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        with self.lock:
            self.stop_run()
            cfg = load_config()["story"]
            self.engine = StoryEngine(folder, cfg, mode, self.publish, send_audio=send_audio)
            self.clip = clip
            self.engine.start()
        self.publish({"type": "run_started", "clip": clip, "mode": mode, "send_audio": send_audio})
        self.publish(self.status())
        return self.status()

    def stop_run(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
            self.publish(self.status())

    def on_message(self, message: dict) -> None:
        """A message from the page. Ticks go straight to the engine."""
        if message.get("type") == "tick" and self.engine is not None:
            self.engine.on_tick(float(message.get("t", 0.0)), bool(message.get("paused", True)),
                                bool(message.get("ended", False)))


def _offer(queue: asyncio.Queue, message: dict) -> None:
    """Queue a message for one browser; drop it if that browser is hopelessly behind."""
    if queue.qsize() < 500:
        queue.put_nowait(message)


ctl = StoryController()


@router.get("", response_class=HTMLResponse)
def page() -> str:
    return (STATIC / "story.html").read_text()


@router.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    ctl.loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    ctl.clients.add(queue)

    async def sender() -> None:
        while True:
            await ws.send_json(await queue.get())

    send_task = asyncio.create_task(sender())
    try:
        await ws.send_json(ctl.status())
        while True:
            ctl.on_message(await ws.receive_json())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        send_task.cancel()
        ctl.clients.discard(queue)


@router.get("/clips")
def clips() -> list[dict]:
    return intake.list_clips()


@router.get("/clips/{name}/video")
def clip_video(name: str) -> FileResponse:
    # Only the normalized clip.mp4 of an existing clip folder can be served.
    file = intake.clip_dir(name) / "clip.mp4"
    if "/" in name or ".." in name or not file.exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(file, media_type="video/mp4")


@router.post("/upload")
async def upload(file: UploadFile) -> dict:
    """Save the upload to disk in chunks (never all in memory), then import it in the background."""
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in intake.VIDEO_EXTS:
        raise HTTPException(400, f"unsupported file type {suffix}; use one of {', '.join(intake.VIDEO_EXTS)}")
    temp_dir = clips_dir() / "_uploads" / uuid.uuid4().hex[:8]
    temp_dir.mkdir(parents=True)
    temp = temp_dir / f"upload{suffix}"
    with open(temp, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return ctl.start_intake(str(temp), None, file.filename, cleanup=temp_dir)


@router.post("/intake_url")
def intake_url(body: dict) -> dict:
    url = str(body.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "paste a full link starting with https://")
    return ctl.start_intake(url, body.get("name") or None, None)


@router.post("/start")
def start(body: dict) -> dict:
    cfg = load_config()["story"]
    return ctl.start_run(str(body.get("clip", "")), body.get("mode", cfg["mode"]),
                         bool(body.get("send_audio", cfg["send_audio_to_llm"])))


@router.post("/stop")
def stop() -> dict:
    ctl.stop_run()
    return ctl.status()


@router.get("/status")
def status() -> dict:
    return ctl.status()
