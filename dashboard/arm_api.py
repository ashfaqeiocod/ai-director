"""arm_api.py — the dashboard's Arm tab (Arm MVP): the arm-wrestling footage, all routes under /arm.

server.py includes this router next to the other tabs.
  /arm                the program viewer (live vs editor, 5 synced cameras, phase/tension timeline, cut log)
  /arm/label          the labeling page: you mark what really happened (GO, surges, the pin, celebrations)
  /arm/vote           the blind A/B page: two programs side by side without names, viewers pick the better one
  /arm/media/{name}   the synced videos and the outputs (only files from data/arm/synced and data/arm/out)
  /arm/labels         GET / POST data/arm/labels.json
  /arm/votes          GET / POST data/arm/votes.jsonl
  /arm/run            GET everything the viewer draws (shots, phases, tension, log) as one JSON
  /arm/matches        GET the recorded matches (match 1 + the blocks under arm_matches in config.yaml)
/arm/media and /arm/run take ?match=match2 to show another match (default: the one the server was started with).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from arm.common import athlete_word, out_dir, work_dir
from src.common import load_config, path

STATIC = Path(__file__).parent / "static"
router = APIRouter(prefix="/arm")

LABEL_KINDS = {"GO", "SURGE", "NEAR_PIN", "PIN", "REACTION", "HIGHLIGHT", "END"}


@router.get("")
def page() -> FileResponse:
    return FileResponse(STATIC / "arm.html")


@router.get("/label")
def label_page() -> FileResponse:
    return FileResponse(STATIC / "arm_label.html")


@router.get("/vote")
def vote_page() -> FileResponse:
    return FileResponse(STATIC / "arm_vote.html")


def _match_dir(match: str) -> Path:
    """The work folder of a match: "" = the server's own match, else a block under arm_matches in config.yaml."""
    if not match:
        return work_dir()
    matches = load_config().get("arm_matches", {})
    if match not in matches:
        raise HTTPException(404, f"unknown match {match!r}")
    return path(matches[match]["work_dir"])


@router.get("/matches")
def matches() -> list[dict]:
    """The matches the page can switch between (only those with a finished run)."""
    out = [{"key": "", "name": "Match 1"}]
    for i, (key, block) in enumerate(load_config().get("arm_matches", {}).items(), start=2):
        if (path(block["work_dir"]) / "out" / "run.json").exists():
            out.append({"key": key, "name": f"Match {i}"})
    return out


@router.get("/media/{name}")
def media(name: str, match: str = "") -> FileResponse:
    """A video or file from data/arm/synced or data/arm/out (no other folders)."""
    if "/" in name or ".." in name:
        raise HTTPException(404, "not found")
    base = _match_dir(match)
    for folder in (base / "synced", base / "out"):
        file = folder / name
        if file.is_file():
            return FileResponse(file)
    raise HTTPException(404, f"{name} not found (run venv/bin/python -m arm.main first)")


def _labels_file() -> Path:
    return work_dir() / "labels.json"


@router.get("/labels")
def get_labels() -> dict:
    file = _labels_file()
    return json.loads(file.read_text()) if file.exists() else {"labels": []}


@router.post("/labels")
async def save_labels(request: Request) -> dict:
    """Save the labels. Each one: {t: seconds on the synced timeline, kind, who (black/tan/""), note}."""
    body = await request.json()
    labels = []
    for item in body.get("labels", []):
        if item.get("kind") not in LABEL_KINDS:
            raise HTTPException(400, f"unknown label kind {item.get('kind')!r}")
        labels.append({"t": round(float(item["t"]), 2), "kind": item["kind"],
                       "who": str(item.get("who", ""))[:10], "note": str(item.get("note", ""))[:200]})
    labels.sort(key=lambda x: x["t"])
    data = {"labels": labels, "winner": str(body.get("winner", ""))[:10], "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _labels_file().write_text(json.dumps(data, indent=2))
    return {"ok": True, "count": len(labels)}


@router.get("/run")
def run_data(match: str = "") -> dict:
    """Everything the viewer page draws, written by arm.main (data/arm/out/run.json)."""
    file = _match_dir(match) / "out" / "run.json"
    if not file.exists():
        raise HTTPException(404, "no run yet: venv/bin/python -m arm.main")
    data = json.loads(file.read_text())
    # What each athlete is called on screen (match 2: the second athlete wears blue, not tan).
    words = load_config().get("arm_matches", {}).get(match, {}).get("athlete_words", {}) if match else {}
    data["names"] = {who: words.get(who, athlete_word(who)) for who in ("black", "tan")}
    data["has_proof"] = (file.parent / "proof_behind_the_ai.mp4").exists()
    return data


@router.get("/votes/pair")
def vote_pair() -> dict:
    """Two programs in random left/right order. The names stay on the server until the vote comes back."""
    data = run_data()
    programs = [p for p in data.get("vote_programs", []) if (out_dir() / p["file"]).exists()]
    if len(programs) < 2:
        raise HTTPException(404, "need at least 2 rendered programs")
    a, b = random.sample(programs, 2)
    return {"left": {"file": a["file"], "key": a["key"]}, "right": {"file": b["file"], "key": b["key"]}}


@router.post("/votes")
async def add_vote(request: Request) -> dict:
    body = await request.json()
    vote = {"left": str(body.get("left"))[:40], "right": str(body.get("right"))[:40],
            "winner": str(body.get("winner"))[:40], "why": str(body.get("why", ""))[:300],
            "voter": str(body.get("voter", ""))[:40], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(work_dir() / "votes.jsonl", "a") as f:
        f.write(json.dumps(vote) + "\n")
    return {"ok": True}


@router.get("/votes")
def votes() -> dict:
    """All votes, and per program: how many times it was shown and how many times it won."""
    file = work_dir() / "votes.jsonl"
    items = [json.loads(line) for line in file.read_text().splitlines() if line.strip()] if file.exists() else []
    table: dict[str, dict] = {}
    for v in items:
        for key in (v["left"], v["right"]):
            table.setdefault(key, {"shown": 0, "won": 0})["shown"] += 1
        if v["winner"] in table:
            table[v["winner"]]["won"] += 1
    return {"votes": items, "table": table}
