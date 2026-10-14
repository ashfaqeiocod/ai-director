"""studio_api.py — the dashboard's Studio tab (MVP 3): replay a directed run, all routes under /studio.

server.py includes this router next to the MVP 1 page (/) and the Story tab (/story).

How it works (frontend-dev view):
  - The page plays the run's program.mp4 in a <video>: that is the clock.
  - The 5 camera videos play muted next to it and are kept in sync with the program's currentTime.
  - Everything else (cards, cuts, story, ground truth) is fetched once as JSON and drawn for the current
    time. A story item is shown only once its known time has passed, exactly as the director saw it.
"""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from src.common import ROOT, load_config

STATIC = Path(__file__).parent / "static"
router = APIRouter(prefix="/studio")


def _outputs() -> Path:
    return ROOT / load_config()["studio"]["outputs_dir"]


def _run_dir(segment: str, run_id: str) -> Path:
    if ".." in segment + run_id or "/" in segment + run_id:
        raise HTTPException(404, "run not found")
    folder = _outputs() / segment / run_id
    if not (folder / "run.json").exists():
        raise HTTPException(404, "run not found")
    return folder


@router.get("", response_class=HTMLResponse)
def page() -> str:
    return (STATIC / "studio.html").read_text()


@router.get("/studio.css")
def page_css() -> FileResponse:
    """The page's Tailwind classes, built once (no internet needed). Rebuild after changing classes in studio.html:
    npx tailwindcss@3.4.17 -i in.css -o dashboard/static/studio.css --minify (content: studio.html; see TESTING_MVP3.md)."""
    return FileResponse(STATIC / "studio.css", media_type="text/css")


@router.get("/runs")
def runs() -> list[dict]:
    out = []
    # Newest first (run ids start with the date and time), whatever the segment.
    for path in sorted(_outputs().glob("*/*/run.json"), key=lambda p: p.parent.name, reverse=True):
        info = json.loads(path.read_text())
        if not (path.parent / "program.mp4").exists():
            continue
        _recheck_valid(info["summary"])
        out.append({"segment": info["segment"], "run_id": info["run_id"], "source": info["source"],
                    "story_mode": info.get("story_mode"), "summary": info["summary"],
                    "start_s": info.get("start_s", 0.0),
                    "has_report": (path.parent / "report.json").exists(),
                    "has_map": (path.parent / "map.jsonl").exists()})
    return out


def _recheck_valid(summary: dict) -> None:
    """Runs saved before the 'min shown fps' rule: judge them by it too (a starved run showed 0.9 fps, 'valid')."""
    shown, wall = summary.get("frames_shown"), summary.get("wall_seconds")
    if shown and wall and shown / wall < load_config()["studio"]["live"]["min_shown_fps"] \
            and summary.get("frames_skipped", 0) > 0:
        summary["valid"] = False


@router.get("/run/{segment}/{run_id}")
def run_data(segment: str, run_id: str) -> dict:
    folder = _run_dir(segment, run_id)
    info = json.loads((folder / "run.json").read_text())
    _recheck_valid(info["summary"])
    from studio.sources import is_folder_source

    if is_folder_source(segment):
        return folder_run_data(segment, folder, info)
    from studio.cameras import load_cams
    from studio.evaluate import truth_events
    from studio.prepare import load_segment
    from studio.story_pass import story_run
    from studio.cards import StoryFeed

    cams = load_cams(segment)
    meta = load_segment(segment)
    # The director log has one row per frame: keep only the rows where something changed.
    timeline = []
    with open(folder / "director_log.csv") as f:
        for row in csv.DictReader(f):
            key = (row["chosen_cam"], row["state"])
            if not timeline or (timeline[-1]["cam"], timeline[-1]["state"]) != (int(key[0]), key[1]):
                timeline.append({"t": float(row["time"]), "cam": int(row["chosen_cam"]), "state": row["state"],
                                 "reason": row["reason"]})
    cards: dict[str, list] = {}
    for line in (folder / "cards.jsonl").read_text().splitlines():
        c = json.loads(line)
        cards.setdefault(str(int(c["time"])), []).append(c)
    events = [json.loads(x) for x in (folder / "director_events.jsonl").read_text().splitlines() if x.strip()]
    story = {"updates": [], "events": []}
    if info.get("story_run"):
        s = story_run(segment, info["story_run"])
        feed = StoryFeed(s["folder"], s.get("wall_start", 0.0), s["speed"])
        if info.get("goal_looks"):                 # the goal watcher's answers, exactly as the director had them
            from studio.goalwatch import load_looks
            feed.add_goal_looks(load_looks(segment, info.get("goalwatch_run")), load_config()["studio"]["goalwatch"])
        story["updates"] = [{k: u[k] for k in ("logged_at", "window_start", "window_end", "phase", "sentiment",
                                               "narration", "intensity", "source")} for u in feed.updates]
        latest: dict[str, dict] = {}
        for e in feed.events:
            latest.setdefault(e["id"], e)          # first version: when it was first known
        story["events"] = [{k: e[k] for k in ("id", "type", "event_time", "known_at", "confidence", "sources")}
                           for e in latest.values() if e["type"] not in ("none",)]
    truth = [{"t": round(e["t"], 2), "label": e["label"], "team": e["team"]} for e in truth_events(meta)
             if e["label"] in ("GOAL", "SHOT")]
    report = (folder / "report.md").read_text() if (folder / "report.md").exists() else None
    on_air = load_config()["studio"]["director"].get("on_air_cams", [c["cam"] for c in cams["cams"]])
    return {"info": {k: info.get(k) for k in ("segment", "run_id", "source", "story_run", "story_mode", "summary",
                                                "goal_looks")},
            "cams": cams, "fps": meta["fps"], "start_s": info.get("start_s", 0.0),
            "duration": info.get("start_s", 0.0) + info["summary"]["seconds"], "timeline": timeline,
            "cards": cards, "director_events": events, "story": story, "truth": truth, "report": report,
            "on_air_cams": on_air, "overlays": overlays(segment, info["source"], cams),
            "pitch": pitch_data(folder, segment, info["source"], cams),
            "intensity": story_intensity(info, segment),
            "goal_cams": cams.get("goal_cams", []), "goal_looks": goal_looks(segment, info),
            "latency": latency_items(folder, story, goal_looks(segment, info))}


def _jsonl(path: Path, keep: tuple) -> list[dict]:
    if not path.exists():
        return []
    return [{k: x.get(k) for k in keep} for x in map(json.loads, path.read_text().splitlines()) if x]


def scores_by_second(folder: Path, fps: float) -> dict[str, list[float]]:
    """The director's score per camera, once per second (for the score bars under the camera tiles)."""
    out = {}
    with open(folder / "director_log.csv") as f:
        for row in csv.DictReader(f):
            if int(row["frame"]) % round(fps) == 0:
                out[str(int(int(row["frame"]) / fps))] = [round(float(v), 1) for k, v in row.items() if k.startswith("score_cam")]
    return out


def _timeline_cards_events(folder: Path) -> tuple[list, dict, list]:
    timeline = []
    with open(folder / "director_log.csv") as f:
        for row in csv.DictReader(f):
            if not timeline or (timeline[-1]["cam"], timeline[-1]["state"]) != (int(row["chosen_cam"]), row["state"]):
                timeline.append({"t": float(row["time"]), "cam": int(row["chosen_cam"]), "state": row["state"],
                                 "reason": row["reason"]})
    cards: dict[str, list] = {}
    for line in (folder / "cards.jsonl").read_text().splitlines():
        c = json.loads(line)
        cards.setdefault(str(int(c["time"])), []).append(c)
    events = [json.loads(x) for x in (folder / "director_events.jsonl").read_text().splitlines() if x.strip()]
    return timeline, cards, events


def folder_run_data(key: str, folder: Path, info: dict) -> dict:
    """A run on footage with real cameras in its own folder.
    MUVS (fans' phones): no pitch map or goal cameras; answer key = the human's camera picks.
    ISSIA (6 calibrated stadium cameras): pitch map from all cameras + the goal watcher's looks; no answer key."""
    from story.run_log import run_folder
    from studio.cards import StoryFeed
    from studio.sources import get_source

    src = _cached_source(key)
    timeline, cards, events = _timeline_cards_events(folder)
    story, intensity = {"updates": [], "events": []}, []
    if info.get("story_run") and info.get("story_clip"):
        sf = run_folder(info["story_clip"], info["story_run"])
        feed = StoryFeed(sf, info.get("story_wall_start", 0.0), 1.0)
        if info.get("goal_looks") and info.get("goalwatch_run"):
            from studio.goalwatch import load_looks
            feed.add_goal_looks(load_looks(key, info["goalwatch_run"]), load_config()["studio"]["goalwatch"])
        story["updates"] = [{k: u.get(k) for k in ("logged_at", "window_start", "window_end", "phase", "sentiment",
                                                   "narration", "intensity", "source", "team_in_control")}
                            for u in feed.updates]
        latest: dict[str, dict] = {}
        for e in feed.events:
            latest.setdefault(e["id"], e)
        story["events"] = [{k: e.get(k) for k in ("id", "type", "event_time", "known_at", "confidence", "sources", "team")}
                           for e in latest.values() if e["type"] != "none"]
        if (sf / "intensity.csv").exists():
            intensity = [[float(r["t"]), float(r["intensity"])] for r in csv.DictReader(open(sf / "intensity.csv"))]
        # The overall "story so far" (MVP 2 memory) and the Colab model's outputs, as they were known
        story["memory"] = _jsonl(sf / "memory.jsonl", ("logged_at", "story_so_far", "score", "momentum", "teams"))
        story["signals"] = [x for x in _jsonl(sf / "signals.jsonl", ("t", "kind", "message"))
                            if x["kind"] in ("SPECIALIST", "SPECIALIST_PLAY", "TRIGGER", "LLM")]
    cam_ids = [c["cam"] for c in src.cams["cams"]]
    w, h = src.cams["size"]
    rows = [json.loads(x) for x in (folder / "map.jsonl").read_text().splitlines()] if (folder / "map.jsonl").exists() else []
    frames = {}
    live_dets = folder / "detections_live.jsonl"        # a run with the Colab detector: what it really used
    if live_dets.exists() and info.get("summary", {}).get("detector") == "colab":
        dets = {c: {} for c in cam_ids}
        for row in map(json.loads, live_dets.read_text().splitlines()):
            for c, d in row["cams"].items():
                dets[int(c)][str(row["frame"] - row["frame"] % src.step)] = d
    else:
        dets = {c: src.dets[c]["frames"] for c in cam_ids}
    # only this run's stretch of the match (a full-half cache would ship 47 min of boxes for a 5-min run)
    t0 = info.get("start_s", 0.0)
    t1 = t0 + info["summary"]["seconds"]
    for f in sorted({int(k) for c in cam_ids for k in dets[c]}):
        if not t0 - 1 <= f / src.fps <= t1 + 1:
            continue
        frames[str(f)] = {str(c): {"players": [[round(v / q, 3) for v, q in zip(p["box"], (w, h, w, h))]
                                               for p in dets[c].get(str(f), {}).get("players", [])]}
                          for c in cam_ids}
    pitch = None
    if getattr(src, "calibrated", False):           # ISSIA, Alfheim
        import numpy as np
        from studio.multicam import to_picture
        # The map's ball, drawn into every camera it falls in (like live mode does)
        for r in rows:
            if not r["ball"] or r["status"] == "lost":
                continue
            f = str(round(r["t"] * src.fps) // src.step * src.step)
            for c in cam_ids:
                pt = to_picture(np.asarray(src.H[c]), tuple(r["ball"]))
                if pt and 0 <= pt[0] < w and 0 <= pt[1] < h and f in frames:
                    frames[f][str(c)]["ball"] = [round(pt[0] / w, 3), round(pt[1] / h, 3)]
        pitch = {"rows": rows, "step": src.step, "coverage": src.coverage(), "close_coverage": {}, "close_cam": None}
    report = (folder / "report.md").read_text() if (folder / "report.md").exists() else None
    looks = goal_looks(key, info) if getattr(src, "calibrated", False) else []
    latency = latency_items(folder, story, looks)
    return {"info": {k: info.get(k) for k in ("segment", "run_id", "source", "story_run", "story_mode", "summary",
                                                "goal_looks")},
            # A run may start later in the match (start_s): program.mp4 starts there, everything else is match time.
            "kind": src.kind, "cams": src.cams, "fps": src.fps, "start_s": info.get("start_s", 0.0),
            "duration": info.get("start_s", 0.0) + info["summary"]["seconds"],
            "timeline": timeline, "cards": cards, "director_events": events, "story": story, "truth": [],
            "human": src.human_picks() if src.kind == "muvs" else [], "report": report, "on_air_cams": src.on_air,
            "overlays": {"step": src.step, "frames": frames}, "pitch": pitch, "intensity": intensity,
            "goal_cams": src.cams.get("goal_cams", []), "goal_looks": looks, "credit": src.meta.get("credit", ""),
            "scores": scores_by_second(folder, src.fps), "layout": src.info().get("layout"),
            "teams": info.get("teams"), "latency": latency}


def latency_items(folder: Path, story: dict, looks: list[dict]) -> list[dict]:
    """Every AI answer's delay (studio.latency): latency.jsonl of the run, or derived from older runs' logs."""
    path = folder / "latency.jsonl"
    if path.exists():
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    from studio.latency import derive

    return derive(story.get("updates", []), story.get("events", []), looks)


def goal_looks(segment: str, info: dict) -> list[dict]:
    """The goal watcher's looks used by this run (studio.goalwatch): when the AI checked which goal, and its answer."""
    if not info.get("goal_looks"):
        return []
    from studio.goalwatch import load_looks

    keep = ("t", "logged_at", "side", "goal", "goal_confidence", "goal_time", "shot", "shot_time", "celebration", "save",
            "save_time", "kind", "trigger", "note")
    return [{k: x.get(k) for k in keep} for x in load_looks(segment, info.get("goalwatch_run"))]


def pitch_data(folder: Path, segment: str, source: str, cams: dict) -> dict:
    """The pitch map of the run (map.jsonl, one row per detection step) + what each camera sees on the pitch."""
    import numpy as np
    from studio.cameras import CLOSE_CAM, cams_dir
    from studio.pitchmap import Projection
    from studio.prepare import work_dir

    path = folder / "map.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    proj = Projection(np.load(work_dir(segment) / "pitch_to_image.npy"))
    fixed = {str(c["cam"]): proj.coverage(c["rect"]) for c in cams["cams"] if c["rect"] is not None}
    aim = "oracle" if source == "oracle" else "detector"
    close_rects = np.load(cams_dir(segment) / f"cam5_{aim}_rects.npy")
    step = round(25 / load_config()["studio"]["detect_fps"])
    # The close-up moves: its outline once per second is enough for the picture.
    close = {str(i): proj.coverage(tuple(close_rects[i]), n=4) for i in range(0, len(close_rects), 25)}
    return {"rows": rows, "step": step, "coverage": fixed, "close_coverage": close, "close_cam": CLOSE_CAM}


def story_intensity(info: dict, segment: str) -> list[list[float]]:
    """The story layer's fused intensity per second ([t, 0..100]), from its intensity.csv."""
    if not info.get("story_run"):
        return []
    from studio.story_pass import story_run

    path = story_run(segment, info["story_run"])["folder"] / "intensity.csv"
    if not path.exists():
        return []
    return [[float(r["t"]), float(r["intensity"])] for r in csv.DictReader(open(path))]


def overlays(segment: str, source: str, cams: dict) -> dict:
    """What the detector saw, to draw on the camera tiles (like MVP 1): per detection step and camera,
    player boxes and the ball, in 0..1 of that camera's picture. Detector runs only (oracle runs have no
    detector boxes). The ball is the match ball after the parked-spare-ball filter, mapped into every camera.
    """
    if source != "detector":
        return {"step": 0, "frames": {}}
    import numpy as np
    from studio.cameras import CLOSE_CAM, cam_rect, cams_dir
    from studio.detect import DETECT_CAMS, PanoramaDetections
    from studio.video import to_camera

    pano = PanoramaDetections(segment)
    w, h = cams["size"]
    close_rects = np.load(cams_dir(segment) / "cam5_detector_rects.npy")
    frames = {}
    for i in sorted(pano.dets[DETECT_CAMS[0]]["frames"]):
        here = pano.at(i)
        per_cam = {}
        for cam in cams["cams"]:
            c = cam["cam"]
            rect = cam_rect(cams, c, i, close_rects)
            item: dict = {}
            if c in DETECT_CAMS:
                det = pano.dets[c]["frames"].get(i, {})
                item["players"] = [[round(v / s, 3) for v, s in zip(p["box"], (w, h, w, h))] for p in det.get("players", [])]
            elif c == CLOSE_CAM:
                pts = [to_camera(rect, w, h, p) for p in here["players"]]
                item["dots"] = [[round(x / w, 3), round(y / h, 3)] for x, y in pts if 0 <= x < w and 0 <= y < h]
            if here["ball"]:
                bx, by = to_camera(rect, w, h, here["ball"][:2])
                if 0 <= bx < w and 0 <= by < h:
                    item["ball"] = [round(bx / w, 3), round(by / h, 3)]
            per_cam[str(c)] = item
        frames[str(i)] = per_cam
    return {"step": pano.step, "frames": frames}


@router.get("/video/{segment}/{run_id}/{name}")
def run_video(segment: str, run_id: str, name: str) -> FileResponse:
    if name not in ("program.mp4", "debug.mp4"):
        raise HTTPException(404)
    return FileResponse(_run_dir(segment, run_id) / name, media_type="video/mp4")


_SOURCES: dict[str, tuple[float, object]] = {}       # match key -> (detection files' time stamp, loaded source)
_SOURCES_LOCK = threading.Lock()


def _cached_source(key: str):
    """ONE loaded copy per folder match (ISSIA, Alfheim, MUVS), shared by Replay and live runs. A full Alfheim
    half's detections take ~0.5 GB in memory: several copies could fill the 8 GB Mac. Loaded again only when the
    detection files changed (a detection run still going); the old copy is dropped then. A live run keeps its own
    reference, so it is never swapped under it. SoccerTrack segments are loaded fresh (small)."""
    from studio.sources import folder_of, get_source, is_folder_source

    if not is_folder_source(key):
        return get_source(key)
    lv = live_ctl.live
    if lv and lv.status["state"] in BUSY and lv.segment == key:
        return lv.src
    folder = folder_of(key)
    stamp = max((f.stat().st_mtime for f in folder.glob("detections/cam*.json")), default=0.0)
    with _SOURCES_LOCK:
        cached = _SOURCES.get(key)
        if cached is None or cached[0] != stamp:
            _SOURCES[key] = (stamp, get_source(key))
        return _SOURCES[key][1]


@router.get("/cam/{segment}/{run_id}/{cam}")
def cam_video(segment: str, run_id: str, cam: int) -> FileResponse:
    from studio.cameras import cam_video as path_of

    info = json.loads((_run_dir(segment, run_id) / "run.json").read_text())
    from studio.sources import get_source, is_folder_source

    if is_folder_source(segment):
        # A playing video asks again and again (range requests): only the file path, never the loaded detections.
        from studio.sources import camera_file

        path = camera_file(segment, cam)
    else:
        aim = "oracle" if info["source"] == "oracle" else "detector"
        path = path_of(segment, cam, aim)      # cams 1-5, and 6/7 = the goal cameras
    if not path.exists():
        raise HTTPException(404, "camera video not found")
    return FileResponse(path, media_type="video/mp4")


# ============================================================================= LIVE MODE (studio.live)
# The page starts a live run; studio.live plays the match at real speed and pushes its state here.
# Browsers get: a WebSocket with state (~10/s) + every decision / story line / goal look, and MJPEG pictures.
import asyncio
import time

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse


BUSY = ("starting", "preparing", "live")      # a live run is going (or getting ready)


class LiveController:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.live = None                      # the running studio.live.LiveStudio
        self.history: list[dict] = []         # every message except "state", for a browser that joins late
        self.last_state: dict | None = None
        self.lock = threading.Lock()

    def publish(self, message: dict) -> None:
        """Called from the live threads: keep it, and pass it to every browser."""
        if message.get("type") == "state":
            self.last_state = message
        else:
            self.history.append(message)
        if self.loop is None:
            return
        for queue in list(self.clients):
            self.loop.call_soon_threadsafe(_offer_live, queue, message)

    def status(self) -> dict:
        lv = self.live
        return {"type": "live_status", "running": bool(lv and lv.status["state"] in BUSY),
                "segment": lv.segment if lv else None, "run_id": lv.run_id if lv else None,
                "status": dict(lv.status) if lv else None}


def _offer_live(queue: asyncio.Queue, message: dict) -> None:
    """Queue a message for one browser. State messages are dropped first when a browser falls behind."""
    if message.get("type") == "state" and queue.qsize() > 20:
        return
    if queue.qsize() < 2000:
        queue.put_nowait(message)


live_ctl = LiveController()


def _ready_segments() -> list[str]:
    from studio.sources import ready_sources

    return ready_sources()


@router.get("/live/sources")
def live_sources() -> dict:
    """Matches that can be played live (cameras + detections ready), and the defaults."""
    from studio.sources import source_label

    cfg = load_config()["story"]
    segments = _ready_segments()
    # The real multi-camera matches first: Alfheim (a full half), then ISSIA
    segments.sort(key=lambda k: (not k.startswith("alfheim-"), not k.startswith("issia-"), k))
    from story.common import gemini_keys

    llm = {**cfg["llm"], **load_config()["studio"].get("llm", {})}
    models = [m for m in [llm["model"], llm.get("fallback_model"), *(llm.get("extra_models") or [])] if m]
    return {"segments": segments, "labels": {k: source_label(k) for k in segments},
            "specialist_url": cfg["specialist"]["server_url"] or "",
            "gemini": {"keys": len(gemini_keys()), "models": models}}


@router.post("/live/colab")
def live_colab(body: dict) -> dict:
    """Test a Colab specialist URL (GET /health) before going live. Saves nothing: the URL is sent with Go live."""
    from story.specialist_client import check_health, normalize_url

    url = normalize_url(body.get("url", ""))
    if not url:
        return {"state": "off", "message": "no URL: the story runs without the Colab specialist"}
    try:
        health = check_health(url)
    except Exception as error:                     # noqa: BLE001 — shown on the page
        return {"state": "error", "url": url, "message": f"can't reach {url}/health: {type(error).__name__}: {error}"}
    det = health.get("detector") or {}
    detector = (f"detector ready · {det.get('ms_per_frame') or '?'} ms per camera frame"
                + ("" if "CUDAExecutionProvider" in det.get("providers", []) else " (NO GPU: too slow for live)")
                if det.get("ready") else "no detector (old notebook, or no ROBOFLOW_API_KEY secret)")
    return {"state": "connected", "url": url, "detector": bool(det.get("ready")),
            "message": f"{health['model'].split('/')[-1]} on {health.get('gpu') or 'CPU'} · {health['fps']:.2f} fps · {detector}"}


@router.get("/live/info/{segment}")
def live_info(segment: str) -> dict:
    """Everything static the page needs before going live: cameras, camera views on the pitch, answer key."""
    if segment not in _ready_segments():
        raise HTTPException(404, "not ready")
    from studio.sources import get_source

    return _cached_source(segment).info()


@router.post("/live/start")
def live_start(body: dict) -> dict:
    from studio.live import LiveStudio

    with live_ctl.lock:
        if live_ctl.live and live_ctl.live.status["state"] in BUSY:
            raise HTTPException(409, "a live run is already going (or getting ready): stop it first")
        segment = body.get("segment")
        if segment not in _ready_segments():
            raise HTTPException(400, f"segment '{segment}' is not ready")
        src = _cached_source(segment)
        duration = src.frames / src.fps
        try:
            start_s = float(body.get("start_s") or 0.0)
        except (TypeError, ValueError):
            raise HTTPException(400, "start_s must be a number of seconds") from None
        if not 0 <= start_s < duration - 10:
            raise HTTPException(400, f"start must be between 0:00 and {int(duration - 10) // 60}:{int(duration - 10) % 60:02d}")
        detector = body.get("detector", "cached")
        if detector == "colab" and not body.get("specialist_url", "").strip():
            raise HTTPException(400, "the Colab detector needs the Colab URL")
        live_ctl.history, live_ctl.last_state = [], None
        from studio.sources import is_folder_source

        source = src if is_folder_source(segment) else segment   # share the loaded copy
        try:
            live_ctl.live = LiveStudio(source, live_ctl.publish, story=body.get("story", "off"),
                                       specialist_url=body.get("specialist_url", "").strip(),
                                       goalwatch=bool(body.get("goalwatch", False)), record=True, start_s=start_s,
                                       detector=detector)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        live_ctl.live.start()
    return live_ctl.status()


@router.post("/live/stop")
def live_stop() -> dict:
    lv = live_ctl.live
    if lv:
        threading.Thread(target=lv.stop, daemon=True).start()   # stopping waits for the AI threads: don't block
    return live_ctl.status()


@router.get("/live/status")
def live_status() -> dict:
    return live_ctl.status()


@router.websocket("/live/ws")
async def live_ws(ws: WebSocket) -> None:
    await ws.accept()
    live_ctl.loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    live_ctl.clients.add(queue)

    async def sender() -> None:
        while True:
            await ws.send_json(await queue.get())

    task = asyncio.create_task(sender())
    try:
        await ws.send_json(live_ctl.status())
        for message in list(live_ctl.history):       # catch up: everything that already happened in this run
            await ws.send_json(message)
        while True:
            await ws.receive_text()                  # the page doesn't need to send anything; this keeps it open
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
        live_ctl.clients.discard(queue)


@router.get("/live/stream/{name}")
def live_stream(name: str) -> StreamingResponse:
    """MJPEG: program, cam1 ... cam5, goal_left, goal_right (shown with a plain <img>)."""
    def frames():
        seq = -1
        idle = 0
        while idle < 600:                            # ~60 s with no new picture: end the stream
            lv = live_ctl.live
            item = lv.jpeg(name) if lv else None
            if item and item[0] != seq:
                seq, data = item
                idle = 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            else:
                idle += 1
                time.sleep(0.1 if not item else 0.02)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- MVP 4: the virtual stadium (Step 4: 3D scene)
# The page dashboard/static/stadium.html draws the behind-goal cameras GOAL L / GOAL R in 3D (Three.js) from
# the "digital twin" (studio.zxy: everyone's position 10x per second) and the camera models (studio.stadium).
_twin_cache: dict[str, list[str]] = {}


def _stadium_folder(segment: str) -> Path:
    from studio.sources import folder_of

    folder = folder_of(segment) if segment.startswith("alfheim-") else None
    if folder is None or not (folder / "panorama.json").exists():
        raise HTTPException(404, "no stadium for this segment (needs studio.panorama)")
    return folder


@router.get("/stadium", response_class=HTMLResponse)
def stadium_page() -> str:
    return (STATIC / "stadium.html").read_text()


@router.get("/stadium/setup/{segment}")
def stadium_setup(segment: str) -> dict:
    """Every camera's model (position, where it looks, field of view) + match facts, for the 3D scene and the map."""
    from studio.panorama import Panorama
    from studio.stadium import Stadium, settings as stadium_settings

    folder = _stadium_folder(segment)
    cfg = stadium_settings()
    st = Stadium(cfg, Panorama.load(folder))
    meta = json.loads((folder / "meta.json").read_text())
    cams = []
    for cam_id, shot in st.shots().items():
        c = shot.cam
        hfov = shot.state.get("hfov")
        if hfov is None and shot.pinhole is None:      # a fixed real camera: its field of view from its width
            import math
            look = c.look_at_m
            dist = math.dist((look[0], look[1], 0), c.position_m)
            hfov = math.degrees(2 * math.atan(shot.state["field_width_m"] / 2 / dist))
        item = {"id": c.id, "name": c.name, "kind": c.kind, "built_from": c.built_from,
                "position_m": [round(v, 2) for v in c.position_m], "look_at_m": shot.state.get("look_at"),
                "field_width_m": shot.state.get("field_width_m"), "hfov_deg": round(hfov, 2)}
        if c.built_from == "source":                   # the real lens: exact direction (with its roll) + focal
            item["real"] = {"rotation": st.pano.data["rig_rotation"][str(c.source_cam)],
                            "focal_px": st.pano.data["rig_focal_px"][str(c.source_cam)],
                            "source_size": [1280, 960], "crop": list(c.crop)}
        cams.append(item)
    twin = folder / "twin.json"
    return {"segment": segment, "cameras": cams, "output_size": cfg["output_size"], "fps": meta["fps"],
            "duration": meta["length_s"], "teams": meta.get("teams", ""), "credit": meta.get("credit", ""),
            "twin": json.loads(twin.read_text()) if twin.exists() else None}


@router.get("/stadium/twin/{segment}")
def stadium_twin(segment: str, start: float = 0.0, seconds: float = 60.0) -> list[dict]:
    """The twin rows (one per 0.1 s) from `start` for `seconds` (the page asks for the next minute ahead of time)."""
    folder = _stadium_folder(segment)
    if segment not in _twin_cache:
        path = folder / "twin.jsonl"
        if not path.exists():
            raise HTTPException(404, "twin.jsonl missing: run `venv/bin/python -m studio.zxy`")
        _twin_cache[segment] = path.read_text().splitlines()
    lines = _twin_cache[segment]
    step = json.loads(lines[1])["t"] - json.loads(lines[0])["t"]
    a = max(0, int(start / step))
    b = min(len(lines), a + int(min(seconds, 300) / step) + 1)
    return [json.loads(line) for line in lines[a:b]]


@router.post("/stadium/upload/{segment}")
async def stadium_upload(segment: str, request: Request, cam: str, start: float, seconds: float,
                         run: str | None = None, part: int | None = None, area: str = "stadium") -> dict:
    """A 3D camera's recording (WebM from the page's MediaRecorder, one video frame per 1/fps of match time)
    -> cam_<cam>.mp4 in outputs/stadium/<segment>/<run> (default run: 3d-<start>-<seconds>s).
    part: a long recording comes in parts (the browser keeps a recording in memory): cam_<cam>_part<NNN>.mp4,
    joined by `POST /stadium/join`."""
    import subprocess

    from src.common import ffmpeg_exe

    if not cam.isalnum() or (run and (".." in run or "/" in run)) or area not in ("stadium", "broadcast"):
        raise HTTPException(400, "bad name")
    folder = _stadium_folder(segment)
    fps = json.loads((folder / "meta.json").read_text())["fps"]
    out = ROOT / load_config()["studio"][area]["outputs_dir"] / segment / (run or f"3d-{start:g}-{seconds:g}s")
    out.mkdir(parents=True, exist_ok=True)
    name = f"cam_{cam}" if part is None else f"cam_{cam}_part{part:03d}"
    webm = out / f"{name}.webm"
    webm.write_bytes(await request.body())
    mp4 = out / f"{name}.mp4"
    # Frames were recorded one per step of match time, not by the wall clock: renumber them at a steady fps, and
    # make the count exact (a recorder can drop a frame at the very start or end: the last one is repeated).
    frames = round(seconds * fps)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(webm),
           "-vf", f"setpts=N/({fps}*TB),fps={fps},tpad=stop_mode=clone:stop=30", "-frames:v", str(frames),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(mp4.with_suffix(".part.mp4"))]           # renamed when complete: a waiting run never reads half a file
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {done.stderr[-300:]}")
    mp4.with_suffix(".part.mp4").replace(mp4)
    count = subprocess.run([ffmpeg_exe(), "-i", str(webm), "-f", "null", "-"], capture_output=True, text=True).stderr
    recorded = int(count.rsplit("frame=", 1)[1].split()[0]) if "frame=" in count else None
    print(f"[stadium] {cam}: {recorded} of {frames} frames recorded -> {mp4.relative_to(ROOT)}", flush=True)
    webm.unlink()
    return {"file": str(mp4.relative_to(ROOT)), "start": start, "seconds": seconds, "frames": frames,
            "recorded_frames": recorded}


@router.post("/stadium/join/{segment}")
def stadium_join(segment: str, cam: str, run: str, area: str = "stadium") -> dict:
    """cam_<cam>_part000.mp4, part001 ... -> cam_<cam>.mp4 (no re-encoding), then the parts are deleted."""
    import subprocess

    from src.common import ffmpeg_exe

    if not cam.isalnum() or ".." in run or "/" in run or area not in ("stadium", "broadcast"):
        raise HTTPException(400, "bad name")
    out = ROOT / load_config()["studio"][area]["outputs_dir"] / segment / run
    parts = sorted(out.glob(f"cam_{cam}_part*.mp4"))
    if not parts:
        raise HTTPException(404, "no parts")
    listing = out / f"cam_{cam}_parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                           "-c", "copy", "-movflags", "+faststart", str(out / f"cam_{cam}.part.mp4")],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {done.stderr[-300:]}")
    (out / f"cam_{cam}.part.mp4").replace(out / f"cam_{cam}.mp4")
    for p in [*parts, listing]:
        p.unlink()
    return {"file": str((out / f"cam_{cam}.mp4").relative_to(ROOT)), "parts": len(parts)}


@router.post("/stadium/live/start")
def stadium_live_start(body: dict) -> dict:
    """Play the virtual stadium live (studio.stadium_live). Pictures and state come over the same routes as an MVP 3
    live run: /studio/live/ws and /studio/live/stream/<program | main | close | wide | boxl | boxr>."""
    from studio.stadium_live import StadiumLive

    with live_ctl.lock:
        if live_ctl.live and live_ctl.live.status["state"] in BUSY:
            raise HTTPException(409, "a live run is already going: stop it first")
        segment = body.get("segment") or "alfheim-2013-11-03-H1"
        folder = _stadium_folder(segment)
        if not (folder / "panorama.mp4").exists():
            raise HTTPException(400, "panorama.mp4 is missing: run `venv/bin/python -m studio.panorama`")
        src = _cached_source(segment)
        try:
            start_s = float(body.get("start_s") or 0.0)
        except (TypeError, ValueError):
            raise HTTPException(400, "start_s must be a number of seconds") from None
        if not 0 <= start_s < src.frames / src.fps - 10:
            raise HTTPException(400, "start is outside the match")
        live_ctl.history, live_ctl.last_state = [], None
        live_ctl.live = StadiumLive(src, live_ctl.publish, start_s=start_s)
        live_ctl.live.start()
    return live_ctl.status()



# ---------------------------------------------------------------- Virtual Broadcast (MVP 4 part 2)
# dashboard/static/broadcast.html renders EVERY camera in 3D from the twin, following the camera paths and director
# decisions of studio.broadcast (outputs/broadcast/<segment>/<run>/broadcast.json).
def _broadcast_dir(segment: str, run: str) -> Path:
    if ".." in segment + run or "/" in segment + run:
        raise HTTPException(404, "run not found")
    folder = ROOT / load_config()["studio"]["broadcast"]["outputs_dir"] / segment / run
    if not (folder / "broadcast.json").exists():
        raise HTTPException(404, "run not found")
    return folder


@router.get("/broadcast", response_class=HTMLResponse)
def broadcast_page() -> str:
    return (STATIC / "broadcast.html").read_text()


@router.get("/broadcast/runs")
def broadcast_runs() -> list[dict]:
    """Newest first: segment, run id, start, length, and whether a rendered program.mp4 exists."""
    root = ROOT / load_config()["studio"]["broadcast"]["outputs_dir"]
    out = []
    for path in sorted(root.glob("*/*/summary.json"), key=lambda p: p.parent.name, reverse=True):
        info = json.loads(path.read_text())
        out.append({"segment": path.parent.parent.name, "run": path.parent.name, "start_s": info["start_s"],
                    "seconds": info["seconds"], "rendered": (path.parent / "program.mp4").exists()})
    return out


@router.get("/broadcast/run/{segment}/{run}")
def broadcast_run(segment: str, run: str) -> FileResponse:
    return FileResponse(_broadcast_dir(segment, run) / "broadcast.json", media_type="application/json")


@router.get("/broadcast/model/{name}")
def broadcast_model(name: str) -> FileResponse:
    """3D models kept in data/models/ (downloaded, not in git): the rigged player figure."""
    path = ROOT / "data" / "models" / name
    if "/" in name or ".." in name or not path.exists():
        raise HTTPException(404, "no such model (see TESTING_MVP4.md: download data/models/Xbot.glb)")
    return FileResponse(path, media_type="model/gltf-binary")


@router.post("/broadcast/stream/{segment}/{run}")
async def broadcast_stream(segment: str, run: str, request: Request, reset: int = 0, part: int | None = None) -> dict:
    """The page's H.264 stream (WebCodecs, hardware encoder, one frame per 1/30 s of match time), appended piece by
    piece to program.h264 in the run folder (reset=1: the first piece starts the file). part: a long render comes in
    segments (program_part000.h264 ...), each in a fresh browser tab, joined by studio.broadcast."""
    folder = _broadcast_dir(segment, run)
    data = await request.body()
    name = "program.h264" if part is None else f"program_part{part:03d}.h264"
    with open(folder / name, "wb" if reset else "ab") as f:
        f.write(data)
    return {"bytes": len(data)}


@router.post("/broadcast/finish/{segment}/{run}")
def broadcast_finish(segment: str, run: str, frames: int, fps: float = 30, part: int | None = None) -> dict:
    """program.h264 -> program.mp4 (wrapped at exactly `fps`, not re-encoded); checks the frame count.
    With `part`: program_partNNN.h264 -> program_partNNN.mp4."""
    import subprocess

    from src.common import ffmpeg_exe

    folder = _broadcast_dir(segment, run)
    stem = "program" if part is None else f"program_part{part:03d}"
    raw, mp4 = folder / f"{stem}.h264", folder / f"{stem}.mp4"
    wrap = folder / f"{stem}.wrap.mp4"     # renamed to .mp4 when complete
    # the raw stream has no usable timing: number the frames at exactly `fps` (setts), not re-encoded
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-c", "copy",
                           "-bsf:v", f"setts=ts=N/({fps}*TB)", "-video_track_timescale", str(round(fps * 1000)),
                           "-movflags", "+faststart", str(wrap)], capture_output=True, text=True)
    if done.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {done.stderr[-300:]}")
    count = subprocess.run([ffmpeg_exe(), "-i", str(wrap), "-c", "copy", "-f", "null", "-"], capture_output=True,
                           text=True).stderr
    got = int(count.rsplit("frame=", 1)[1].split()[0]) if "frame=" in count else None
    wrap.replace(mp4)
    raw.unlink()
    print(f"[broadcast] program: {got} of {frames} frames -> {mp4.relative_to(ROOT)}", flush=True)
    return {"file": str(mp4.relative_to(ROOT)), "frames": got, "expected": frames}
