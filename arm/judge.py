"""judge.py — Arm MVP Step 9: a sparse "second opinion" from Gemini, every ~2 s, on ONE still per moment.

The local models measure faces, bodies, the grip and the crowd 10x per second, but they don't UNDERSTAND the
moment. Every `interval_s` seconds Gemini gets one contact sheet (the 5 cameras side by side at that moment,
a single JPEG, never video or sound) and answers in fixed JSON: the phase, who is leading, each athlete's
emotion, the crowd's mood, how dramatic it is, and which camera a TV director would show now.
  - live: an answer is only usable from `t + latency` (the real time the request took), as in MVP 2/3
    (every answer keeps event_time and logged_at);
  - offline: the editor may use it from `t`.
~16 requests per run (31 s / 2 s), all cached (arm/gemini.py): a rerun costs nothing. Hard total cap: 200.

Usage: venv/bin/python -m arm.judge   (prints the answers; data/arm/cache/judge.json)
"""
from __future__ import annotations

import json
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from arm.common import cache_dir, cam_letters, settings, synced_video
from arm.gemini import BudgetExceeded, ask, calls_used, jpeg


class Judgement(BaseModel):
    phase: Literal["setup", "struggle", "surge", "near_pin", "pin", "reaction"]
    leader: Literal["black", "tan", "even"]
    black_emotion: str
    tan_emotion: str
    crowd_mood: Literal["calm", "engaged", "excited", "laughing", "cheering"]
    drama: int
    best_camera: Literal["A", "B", "C", "D", "E"]
    why: str
    caption: str


PROMPT = """You are the director of a live TV broadcast of an office arm-wrestling match (right hands).
The picture shows the SAME moment from 5 cameras, labelled A to E:
A and D film the table from the side (opposite sides), B films over the TAN athlete's back,
C and E film over the BLACK-shirt athlete's shoulder (E is wide and shows the audience).
Athletes: one in a BLACK polo shirt, one in a TAN (beige) polo shirt.
The winner pins the opponent's hand toward the table, across the table from the camera B/C/E point of view.
Answer:
- phase: setup (no grip yet) / struggle (even fight) / surge (someone is pushing hard right now) /
  near_pin (a hand is close to the table) / pin (a hand is on the table) / reaction (after the win),
- leader: who is winning right now (black / tan / even),
- black_emotion, tan_emotion: 1-3 words each (e.g. "straining", "laughing", "confident", "head down"),
- crowd_mood, drama (0 = nothing happens .. 10 = the decisive moment),
- best_camera: the ONE camera a TV director should put on air now to give the viewer the most emotion and
  the clearest view of the fight, and why (one short sentence),
- caption: a 3-6 word on-screen caption for this moment."""


def contact_sheet(frames: dict[str, np.ndarray]) -> np.ndarray:
    """The 5 cameras in a 3x2 grid (640x360 each) with big letters; the 6th tile stays dark."""
    tile_w, tile_h = 640, 360
    sheet = np.zeros((tile_h * 2, tile_w * 3, 3), np.uint8)
    for k, cam in enumerate(cam_letters()):
        x, y = (k % 3) * tile_w, (k // 3) * tile_h
        sheet[y:y + tile_h, x:x + tile_w] = cv2.resize(frames[cam], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (x, y), (x + 70, y + 60), (0, 0, 0), -1)
        cv2.putText(sheet, cam, (x + 14, y + 48), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3)
    return sheet


def run_judge(duration: float) -> list[dict]:
    """Ask Gemini every interval_s seconds. Returns [{event_time, logged_at, latency, ...answer}]."""
    cfg = settings()["judge"]
    caps = {c: cv2.VideoCapture(str(synced_video(c))) for c in cam_letters()}
    fps = settings()["sync"]["fps"]
    answers = []
    t = cfg["first_s"]
    while t <= duration:
        frames = {}
        for cam, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            frames[cam] = cap.read()[1]
        try:
            answer, info = ask("judge", [jpeg(contact_sheet(frames), cfg["sheet_width"])], PROMPT, Judgement,
                               cache_key_extra=f"t={t:.2f}")
        except BudgetExceeded as error:
            print(f"  judge stopped: {error}")
            break
        except Exception as error:                     # one failed look must not stop the run
            print(f"  judge at {t:.1f} s failed: {error}")
            t += cfg["interval_s"]
            continue
        latency = info.get("seconds", cfg["assumed_latency_s"])
        answers.append({"event_time": round(t, 2), "logged_at": round(t + latency, 2), "latency": latency,
                        "model": info.get("model"), "cached": info.get("cached", False), **answer})
        t += cfg["interval_s"]
    for cap in caps.values():
        cap.release()
    (cache_dir() / "judge.json").write_text(json.dumps(answers, indent=1))
    return answers


def load_judge() -> list[dict]:
    file = cache_dir() / "judge.json"
    return json.loads(file.read_text()) if file.exists() else []


def known_at(answers: list[dict], t: float, live: bool) -> dict | None:
    """The newest answer usable at time t (live: once it has arrived; offline: from its moment on)."""
    usable = [a for a in answers if (a["logged_at"] if live else a["event_time"]) <= t]
    if not usable:
        return None
    newest = usable[-1]
    return newest if t - newest["event_time"] <= settings()["judge"]["fresh_s"] else None


def main() -> int:
    from arm.match import load_timeline

    duration = load_timeline()["states"][-1]["t"]
    answers = run_judge(duration)
    for a in answers:
        print(f"{a['event_time']:5.1f} (known {a['logged_at']:5.1f}) {a['phase']:<9} leader {a['leader']:<5} "
              f"black: {a['black_emotion']:<14} tan: {a['tan_emotion']:<14} crowd {a['crowd_mood']:<9} "
              f"drama {a['drama']:>2}  best {a['best_camera']}  \"{a['caption']}\"  ({a['why']})")
    print(f"Gemini calls used so far: {calls_used()} / {settings()['gemini']['max_total_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
