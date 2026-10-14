"""evaluate.py — Arm MVP Step 11: does the AI make a better broadcast than simple alternatives? Numbers, not taste.

Programs compared (each = one shot decision per look, 10 per second):
  AI live      arm/director.py (sees only the past)
  AI editor    arm/editor.py (sees the whole match)
  one phone    each camera alone, the whole time (what you get from a single phone)
  random       a random camera every 3 s (a bored vision mixer)
  round robin  A, B, C, D, E, A ... every 3 s
  motion       MVP 1-style: the camera whose picture changes most, held >= 2 s, with 0.5 s hysteresis
Measures (per program):
  rules        shots, average / shortest shot, cuts under 1 s, 180-degree breaks (A <-> D), jump cuts
  quality      share of air time that is blurry, shaking or with the grip blocked by someone
  faces        share of air time with an athlete's face that is readable, emotional (strain or joy) and big
               enough to be felt (>= 6% of the picture height)
  crowd        share of air time with >= 3 readable smiling audience faces
  variety      how many camera angles and which kinds of content (athlete face, crowd, grip, close grip, wide)
               the viewer got to see
  moments      for each moment (from YOUR labels in data/arm/labels.json if they exist, else from the AI's own
               detectors - then the test is not independent and says so) was the right thing on screen
               within 1.5 s? LOOK_UP / REACTION -> that athlete's face; CHEER -> audience faces;
               SURGE / NEAR_PIN / PIN -> the grip, seen from behind an athlete (where the pin is visible) or
               close; GESTURE -> that athlete in the picture.
Output: data/arm/out/report.md (+ metrics.json).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import cv2
import numpy as np

from arm.common import cache_dir, cam_letters, out_dir, settings, synced_video, work_dir
from arm.shots import FRAME_H, FRAME_W, Shot


@dataclass
class Plain:
    """A decision of a baseline program (same shape as the director's)."""
    t: float
    n: int
    shot: Shot
    change: str = ""
    reason: str = ""


def full(cam: str) -> Shot:
    return Shot(cam, "WIDE", (0.0, 0.0, float(FRAME_W), float(FRAME_H)), parts={"clarity": 1.0})


def single(states: list[dict], cam: str) -> list[Plain]:
    return [Plain(s["t"], s["n"], full(cam)) for s in states]


def every(states: list[dict], period_s: float, pick) -> list[Plain]:
    """A cut every period_s; pick(i, previous cam) chooses the next camera."""
    out, cam, since = [], None, -99.0
    for s in states:
        change = ""
        if s["t"] - since >= period_s - 1e-6:
            new = pick(len(out), cam)
            change = "CUT" if cam is not None and new != cam else ""
            cam, since = new, s["t"]
        out.append(Plain(s["t"], s["n"], full(cam), change))
    return out


def motion_energy() -> dict[str, list[float]]:
    """Per camera and look: how much the (small, grey) picture changed since the last look. Cached."""
    file = cache_dir() / "motion.json"
    if file.exists():
        return json.loads(file.read_text())
    step = round(settings()["sync"]["fps"] / settings()["perceive"]["analysis_fps"])
    energy = {}
    for cam in cam_letters():
        cap = cv2.VideoCapture(str(synced_video(cam)))
        values, prev, n = [], None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % step == 0:
                small = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                values.append(float(np.mean(cv2.absdiff(small, prev))) if prev is not None else 0.0)
                prev = small
            n += 1
        energy[cam] = values
        cap.release()
    file.write_text(json.dumps(energy))
    return energy


def motion_program(states: list[dict]) -> list[Plain]:
    """MVP 1's idea on these cameras: most picture change wins, min shot 2 s, hysteresis 0.5 s."""
    energy = motion_energy()
    cams = cam_letters()
    out, cam, since, challenger, challenger_since = [], None, 0.0, None, 0.0
    for k, s in enumerate(states):
        best = max(cams, key=lambda c: energy[c][min(k, len(energy[c]) - 1)])
        change = ""
        if cam is None:
            cam = best
        elif best != cam:
            if best != challenger:
                challenger, challenger_since = best, s["t"]
            if s["t"] - since >= 2.0 and s["t"] - challenger_since >= 0.5:
                cam, since, change = best, s["t"], "CUT"
        out.append(Plain(s["t"], s["n"], full(cam), change))
    return out


def baselines(states: list[dict]) -> dict[str, list[Plain]]:
    cams = cam_letters()
    rng = random.Random(7)
    programs = {f"one phone {c}": single(states, c) for c in cams}
    programs["random 3s"] = every(states, 3.0, lambda i, prev: rng.choice([c for c in cams if c != prev]))
    programs["round robin 3s"] = every(states, 3.0, lambda i, prev: cams[(cams.index(prev) + 1) % len(cams)] if prev else cams[0])
    programs["motion (MVP 1 style)"] = motion_program(states)
    return programs


# ---------- measuring one program ----------

def inside(box: list[float], rect: tuple[float, float, float, float]) -> bool:
    """The box's centre is inside the rect."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return rect[0] <= cx <= rect[0] + rect[2] and rect[1] <= cy <= rect[1] + rect[3]


def face_on_screen(look: dict, athlete: str, rect, emotional: bool) -> bool:
    cfg = settings()["evaluate"]
    a = look[athlete]
    if not a or not a["face"]:
        return False
    f = a["face"]
    big = (f["box"][3] - f["box"][1]) / rect[3] >= cfg["face_min_share"]
    feel = max(f["strain"], f["joy"]) >= cfg["emotion_min"] if emotional else True
    return inside(f["box"], rect) and big and f["frontal"] >= 0.35 and feel


def readable_crowd(look: dict, rect) -> int:
    """Audience faces inside the rect that are big enough to read (>= crowd_face_share of the picture height)."""
    share = settings()["evaluate"]["crowd_face_share"]
    return sum(1 for f in look["audience_faces"] if inside(f["box"], rect) and (f["box"][3] - f["box"][1]) / rect[3] >= share)


def grip_readable(look: dict, rect) -> bool:
    """The grip is in the picture and big enough: the forearm is >= grip_share of the picture height."""
    forearm = look["grip"]["forearm"] if look["grip"] else None
    if forearm is None:
        return False
    return inside([*look["grip_point"], *look["grip_point"]], rect) and forearm / rect[3] >= settings()["evaluate"]["grip_share"]


def shows(look: dict, shot: Shot, kind: str, who: str, view: str) -> bool:
    """Is the content of moment `kind` (about athlete `who`) visible AND big enough to read in this shot?"""
    rect = shot.rect
    if kind in ("LOOK_UP", "REACTION", "HIGHLIGHT"):
        athletes = [who] if who in ("black", "tan") else ["black", "tan"]
        if any(face_on_screen(look, a, rect, emotional=False) for a in athletes):
            return True
        return kind == "HIGHLIGHT" and readable_crowd(look, rect) >= 3
    if kind == "CHEER":
        return readable_crowd(look, rect) >= 3
    if kind in ("SURGE", "NEAR_PIN", "PIN", "GO", "END"):
        # Who wins is only visible from behind an athlete (the hands go across the table) or in a close-up.
        return grip_readable(look, rect) and (view.startswith("behind") or rect[3] <= 0.6 * FRAME_H or kind in ("GO", "END"))
    if kind == "GESTURE":
        return bool(look[who]) and inside(look[who]["box"], rect) and (look[who]["box"][3] - look[who]["box"][1]) / rect[3] >= 0.4
    return False


def measure(decisions: list, timeline: dict, moments: list[tuple[float, str, str]]) -> dict:
    cfg = settings()["evaluate"]
    setup = timeline["setup"]
    cams = timeline["cams"]
    dt = 1.0 / settings()["perceive"]["analysis_fps"]
    starts = [i for i, d in enumerate(decisions) if i == 0 or d.shot.key != decisions[i - 1].shot.key]
    lengths = np.diff(starts + [len(decisions)]) * dt
    side = {c: settings()["cameras"][c]["side"] for c in cam_letters()}
    breaks = jumps = 0
    for i in starts[1:]:
        a, b = decisions[i - 1].shot, decisions[i].shot
        if {side[a.cam], side[b.cam]} == {"north", "south"}:
            breaks += 1
        if a.cam == b.cam and abs(1 - min(a.rect[3], b.rect[3]) / max(a.rect[3], b.rect[3])) < 0.4:
            jumps += 1
    bad = faces = crowd = 0
    kinds_seen = set()
    for k, d in enumerate(decisions):
        look = cams[d.shot.cam][min(k, len(cams[d.shot.cam]) - 1)]
        if look["sharp"] < settings()["shots"]["min_sharp"] or look["shake"] > settings()["shots"]["max_shake"] \
                or (look["occluded"] and inside([*look["grip_point"], *look["grip_point"]], d.shot.rect)):
            bad += 1
        crowd_joy = [f["joy"] for f in look["audience_faces"] if inside(f["box"], d.shot.rect)
                     and (f["box"][3] - f["box"][1]) / d.shot.rect[3] >= cfg["crowd_face_share"]]
        if any(face_on_screen(look, a, d.shot.rect, emotional=True) for a in ("black", "tan")):
            faces += 1
            kinds_seen.add("athlete face")
        if len(crowd_joy) >= 3 and float(np.mean(crowd_joy)) >= cfg["emotion_min"]:
            crowd += 1
            kinds_seen.add("crowd")
        if grip_readable(look, d.shot.rect):
            kinds_seen.add("grip close" if d.shot.rect[3] <= 0.6 * FRAME_H else "grip")
        if d.shot.rect[3] >= 0.99 * FRAME_H:
            kinds_seen.add("wide")
    caught = []
    for t, kind, who in moments:
        window = [k for k, d in enumerate(decisions) if t <= d.t <= t + cfg["moment_window_s"]]
        hit = any(shows(cams[decisions[k].shot.cam][k], decisions[k].shot, kind, who, setup[decisions[k].shot.cam]["view"])
                  for k in window)
        caught.append(hit)
    return {"shots": len(starts), "avg_shot_s": round(float(lengths.mean()), 2),
            "min_shot_s": round(float(lengths[:-1].min()) if len(lengths) > 1 else float(lengths[0]), 2),
            "cuts_under_1s": int((lengths[:-1] < 1.0 - 1e-6).sum()), "breaks_180": breaks, "jump_cuts": jumps,
            "bad_air_pct": round(100 * bad / len(decisions), 1), "faces_pct": round(100 * faces / len(decisions), 1),
            "crowd_pct": round(100 * crowd / len(decisions), 1), "angles": len({d.shot.cam for d in decisions}),
            "content_kinds": sorted(kinds_seen),
            "moments_caught": int(sum(caught)), "moments": len(caught),
            "moments_pct": round(100 * sum(caught) / len(caught), 1) if caught else None,
            "cams_used": sorted({d.shot.cam for d in decisions})}


def moments_to_check(timeline: dict) -> tuple[list[tuple[float, str, str]], str]:
    """(t, kind, who) moments + where they come from: the user's labels if any, else the AI's own events."""
    labels = work_dir() / "labels.json"
    if labels.exists():
        data = json.loads(labels.read_text())
        items = [(l["t"], l["kind"], l.get("who", "")) for l in data.get("labels", [])]
        if items:
            return items, "your labels (data/arm/labels.json): an independent test"
    items = []
    for s in timeline["states"]:
        for e in s["events"]:
            kind, _, who = e.partition(" ")
            if kind not in ("STRUGGLE", "SETUP"):
                items.append((s["t"], kind, who))
    return items, "the AI's own detected moments (NOT independent: label the match at /arm/label for a real test)"


def report(results: dict[str, dict], source: str, judge: list[dict]) -> str:
    lines = ["# Arm MVP — how good is the AI's broadcast?", "",
             f"Moments checked: {source}.", "",
             "| program | shots | avg shot s | shortest s | cuts < 1 s | 180° breaks | jump cuts | bad picture on air % "
             "| athlete face % | crowd joy % | angles | content seen | moments caught |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        caught = f"{r['moments_caught']}/{r['moments']} ({r['moments_pct']}%)" if r["moments"] else "-"
        lines.append(f"| {name} | {r['shots']} | {r['avg_shot_s']} | {r['min_shot_s']} | {r['cuts_under_1s']} | "
                     f"{r['breaks_180']} | {r['jump_cuts']} | {r['bad_air_pct']} | {r['faces_pct']} | {r['crowd_pct']} | "
                     f"{r['angles']} | {', '.join(r['content_kinds'])} | {caught} |")
    if judge:
        lines += ["", "## Gemini's second opinion (one still of all 5 cameras every 2 s)", "",
                  "| moment s | known (live) s | phase | leader | black | tan | crowd | drama | best camera | caption |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for a in judge:
            lines.append(f"| {a['event_time']} | {a['logged_at']} | {a['phase']} | {a['leader']} | {a['black_emotion']} | "
                         f"{a['tan_emotion']} | {a['crowd_mood']} | {a['drama']} | {a['best_camera']} | {a['caption']} |")
    return "\n".join(lines) + "\n"


def evaluate(programs: dict[str, list], timeline: dict, judge: list[dict]) -> dict:
    moments, source = moments_to_check(timeline)
    results = {name: measure(d, timeline, moments) for name, d in programs.items()}
    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "metrics.json").write_text(json.dumps({"source": source, "results": results}, indent=1))
    (out_dir() / "report.md").write_text(report(results, source, judge))
    return {"source": source, "results": results}
