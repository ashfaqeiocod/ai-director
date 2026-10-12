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
from pathlib import Path

from fastapi import APIRouter, HTTPException
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


@router.get("/runs")
def runs() -> list[dict]:
    out = []
    # Newest first (run ids start with the date and time), whatever the segment.
    for path in sorted(_outputs().glob("*/*/run.json"), key=lambda p: p.parent.name, reverse=True):
        info = json.loads(path.read_text())
        if not (path.parent / "program.mp4").exists():
            continue
        out.append({"segment": info["segment"], "run_id": info["run_id"], "source": info["source"],
                    "story_mode": info.get("story_mode"), "summary": info["summary"],
                    "has_report": (path.parent / "report.json").exists(),
                    "has_map": (path.parent / "map.jsonl").exists()})
    return out


@router.get("/run/{segment}/{run_id}")
def run_data(segment: str, run_id: str) -> dict:
    folder = _run_dir(segment, run_id)
    info = json.loads((folder / "run.json").read_text())
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
            feed.add_goal_looks(load_looks(segment), load_config()["studio"]["goalwatch"])
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
            "cams": cams, "fps": meta["fps"], "duration": info["summary"]["seconds"], "timeline": timeline,
            "cards": cards, "director_events": events, "story": story, "truth": truth, "report": report,
            "on_air_cams": on_air, "overlays": overlays(segment, info["source"], cams),
            "pitch": pitch_data(folder, segment, info["source"], cams),
            "intensity": story_intensity(info, segment),
            "goal_cams": cams.get("goal_cams", []), "goal_looks": goal_looks(segment, info)}


def goal_looks(segment: str, info: dict) -> list[dict]:
    """The goal watcher's looks used by this run (studio.goalwatch): when the AI checked which goal, and its answer."""
    if not info.get("goal_looks"):
        return []
    from studio.goalwatch import load_looks

    keep = ("t", "logged_at", "side", "goal", "goal_confidence", "goal_time", "shot", "celebration", "save", "note")
    return [{k: x[k] for k in keep} for x in load_looks(segment)]


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


@router.get("/cam/{segment}/{run_id}/{cam}")
def cam_video(segment: str, run_id: str, cam: int) -> FileResponse:
    from studio.cameras import cam_video as path_of

    info = json.loads((_run_dir(segment, run_id) / "run.json").read_text())
    aim = "oracle" if info["source"] == "oracle" else "detector"
    path = path_of(segment, cam, aim)          # cams 1-5, and 6/7 = the goal cameras
    if not path.exists():
        raise HTTPException(404, "camera video not found")
    return FileResponse(path, media_type="video/mp4")
