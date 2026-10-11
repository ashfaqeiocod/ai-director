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
    for path in sorted(_outputs().glob("*/*/run.json"), reverse=True):
        info = json.loads(path.read_text())
        if not (path.parent / "program.mp4").exists():
            continue
        out.append({"segment": info["segment"], "run_id": info["run_id"], "source": info["source"],
                    "story_mode": info.get("story_mode"), "summary": info["summary"],
                    "has_report": (path.parent / "report.json").exists()})
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
    return {"info": {k: info[k] for k in ("segment", "run_id", "source", "story_run", "story_mode", "summary")},
            "cams": cams, "fps": meta["fps"], "duration": info["summary"]["seconds"], "timeline": timeline,
            "cards": cards, "director_events": events, "story": story, "truth": truth, "report": report}


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
    path = path_of(segment, cam, aim)
    if not path.exists():
        raise HTTPException(404, "camera video not found")
    return FileResponse(path, media_type="video/mp4")
