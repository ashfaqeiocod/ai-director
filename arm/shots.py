"""shots.py — Arm MVP Steps 6-7: every shot the 5 phones can give, and how good each one is RIGHT NOW.

Shot catalogue. Each phone gives several "virtual" shots by cropping (a digital zoom, like a robotic camera
operator framing a shot):
  WIDE       the whole picture
  TWO        both athletes (their upper bodies, the table and the grip)
  GRIP       the gripped hands and forearms, tight
  FACE_BLACK / FACE_TAN   one athlete's head and shoulders (only when his face can be read in this camera)
  CROWD      the audience faces (only cameras that see at least 3 of them)
A crop may not zoom in further than `max_zoom` (1080p phones: 2x, the 4K phone: 3x), so a shot is never soft.

Shot value (0..100) per look, from the camera cards (arm/match.py) and the match state:
  fit       does this kind of shot suit the moment? (config arm.shots.fit: e.g. STRUGGLE -> TWO / GRIP / FACE)
  emotion   how much feeling it shows: strain or joy on a readable, frontal face; the crowd's smiles
  clarity   can the viewer see it: athletes found, grip measured, not blocked, sharp, not shaking
  events    a fresh moment boosts the shot that shows it: LOOK_UP tan -> FACE_TAN, CHEER -> CROWD, ...
  angle     for SURGE / NEAR_PIN / PIN the cameras behind an athlete show who is winning (the hands tilt across
            the table); side cameras show the faces
Every number is in config.yaml -> arm.shots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm.common import settings

WIDE, TWO, GRIP, FACE_BLACK, FACE_TAN, CROWD = "WIDE", "TWO", "GRIP", "FACE_BLACK", "FACE_TAN", "CROWD"
SHOT_TYPES = (WIDE, TWO, GRIP, FACE_BLACK, FACE_TAN, CROWD)
FRAME_W, FRAME_H = 1920, 1080          # the synced copies' size: every rect is in these pixels


@dataclass
class Shot:
    cam: str
    kind: str
    rect: tuple[float, float, float, float]      # x, y, w, h in 1920x1080 pixels (16:9)
    value: float = 0.0
    parts: dict | None = None                    # fit / emotion / clarity / event / angle, for the log

    @property
    def key(self) -> str:
        return f"{self.cam}:{self.kind}"


def fit_16x9(cx: float, cy: float, h: float, min_h: float) -> tuple[float, float, float, float]:
    """A 16:9 rect centred on (cx, cy), at least min_h tall, pushed back inside the picture."""
    h = float(np.clip(h, min_h, FRAME_H))
    w = h * 16 / 9
    x = float(np.clip(cx - w / 2, 0, FRAME_W - w))
    y = float(np.clip(cy - h / 2, 0, FRAME_H - h))
    return (x, y, w, h)


def box_rect(boxes: list[list[float]], pad: float, min_h: float) -> tuple[float, float, float, float]:
    """The smallest 16:9 rect around these boxes, with `pad` (share of the size) around them."""
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w, h = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    return fit_16x9((x1 + x2) / 2, (y1 + y2) / 2, max(h, w * 9 / 16), min_h)


def min_height(cam: str) -> float:
    """The tightest crop this camera allows (in 1080p pixels): 1080 / max zoom."""
    cfg = settings()["shots"]
    return FRAME_H / cfg["max_zoom"].get(cam, cfg["max_zoom"]["default"])


def frame_shots(cam: str, look: dict) -> list[Shot]:
    """The shots camera `cam` can offer at this look (their framing only, no value yet)."""
    cfg = settings()["shots"]
    lo = min_height(cam)
    shots = [Shot(cam, WIDE, (0.0, 0.0, float(FRAME_W), float(FRAME_H)))]
    athletes = [look[a] for a in ("black", "tan") if look[a]]
    if len(athletes) == 2:
        # Upper bodies: the top 70% of each athlete's box (the table edge and the grip are in there).
        uppers = [[a["box"][0], a["box"][1], a["box"][2], a["box"][1] + 0.7 * (a["box"][3] - a["box"][1])] for a in athletes]
        two = box_rect(uppers, cfg["two_pad"], lo)
        if two[3] <= cfg["two_max_share"] * FRAME_H:
            shots.append(Shot(cam, TWO, two))
        else:            # the athletes fill the picture (side cameras): the whole picture IS the two-shot
            shots[0] = Shot(cam, TWO, shots[0].rect)
    gx, gy = look["grip_point"]
    forearm = look["grip"]["forearm"] if look["grip"] else None
    if forearm is None and athletes:
        forearm = 0.25 * np.mean([a["box"][3] - a["box"][1] for a in athletes])
    if forearm:
        shots.append(Shot(cam, GRIP, fit_16x9(gx, gy, cfg["grip_size"] * forearm, lo)))
    for name, kind in (("black", FACE_BLACK), ("tan", FACE_TAN)):
        face = look[name]["face"] if look[name] else None
        if face and face["frontal"] >= cfg["face_min_frontal"]:
            fx1, fy1, fx2, fy2 = face["box"]
            shots.append(Shot(cam, kind, fit_16x9((fx1 + fx2) / 2, (fy1 + fy2) / 2 + 0.35 * (fy2 - fy1),
                                                  cfg["face_size"] * (fy2 - fy1), lo)))
    faces = look["audience_faces"]
    if len(faces) >= cfg["crowd_min_faces"]:
        shots.append(Shot(cam, CROWD, box_rect([f["box"] for f in faces], cfg["crowd_pad"], lo)))
    return shots


def event_boosts(recent_events: list[tuple[float, str]], t: float) -> dict[str, float]:
    """Shot kind (or 'kind@athlete') -> extra value, from events of the last `event_window_s` seconds."""
    cfg = settings()["shots"]
    boosts: dict[str, float] = {}
    for when, name in recent_events:
        age = t - when
        if age < 0 or age > cfg["event_window_s"]:
            continue
        fade = 1.0 - age / cfg["event_window_s"]
        word, _, who = name.partition(" ")
        for kind, amount in cfg["event_boost"].get(word, {}).items():
            kind = kind.replace("WHO", who.upper()) if who else kind
            boosts[kind] = max(boosts.get(kind, 0.0), amount * fade)
    return boosts


def score(shot: Shot, look: dict, state: dict, view: str, boosts: dict[str, float]) -> Shot:
    """Fill in shot.value (0..100) and its parts. See the module docstring."""
    cfg = settings()["shots"]
    phase = state["phase"]
    fit = cfg["fit"].get(phase, {}).get(shot.kind, 0.0)

    # Emotion: what feeling this framing shows.
    def face_feel(athlete: str) -> float:
        a = look[athlete]
        if not a or not a["face"]:
            return 0.0
        f = a["face"]
        # A face in profile (up to ~60 degrees) reads perfectly well on TV; only beyond that it fades.
        readable = min(1.0, f["frontal"] / cfg["profile_frontal"])
        return max(f["strain"], f["joy"], f["open"]) * readable
    if shot.kind == FACE_BLACK:
        emotion = face_feel("black")
    elif shot.kind == FACE_TAN:
        emotion = face_feel("tan")
    elif shot.kind == TWO:
        feels = sorted([face_feel("black"), face_feel("tan")])
        emotion = min(1.0, 0.7 * feels[1] + 0.5 * feels[0])          # both faces visible beats one
    elif shot.kind == CROWD:
        joys = [f["joy"] for f in look["audience_faces"]]
        emotion = float(np.mean(joys)) * min(1.0, len(joys) / 4) if joys else 0.0
    elif shot.kind == GRIP:
        emotion = cfg["grip_emotion"] * state["tension"]
    else:                                                  # WIDE: the room's mood
        emotion = 0.6 * state["crowd"]
    # Effort is visible in the body too: an athlete straining over the table, if THIS camera sees it.
    if shot.kind in (TWO, GRIP) and any(look[a] and look[a]["head_down"] for a in ("black", "tan")):
        emotion = max(emotion, cfg["head_down_emotion"])

    # Clarity: can the viewer read the shot?
    athletes = sum(1 for a in ("black", "tan") if look[a])
    clarity = 1.0
    if shot.kind in (TWO, GRIP):
        clarity *= 0.5 + 0.25 * athletes
        clarity *= 1.0 if look["grip"] else cfg["grip_unmeasured"]
    if look["occluded"] and shot.kind in (TWO, GRIP):
        clarity *= 0.2
    if look["sharp"] < cfg["min_sharp"]:
        clarity *= 0.3
    if look["shake"] > cfg["max_shake"]:
        clarity *= 0.5

    # Angle: who-is-winning moments need a camera behind an athlete (it sees the hands go across the table).
    angle = 0.0
    if phase in ("SURGE", "NEAR_PIN", "PIN") and shot.kind in (TWO, GRIP, WIDE):
        angle = cfg["angle_bonus"] if view.startswith("behind") else 0.0
    angle += cfg["view_bonus"].get(view, {}).get(shot.kind, 0.0)

    event = boosts.get(shot.kind, 0.0)
    angle += boosts.get(f"CAM:{shot.cam}", 0.0)            # Gemini's "best camera now" (arm/judge.py)
    w = cfg["weights"]
    value = 100 * (w["fit"] * fit + w["emotion"] * emotion + w["event"] * event + w["angle"] * angle) * clarity
    shot.value = round(float(value), 2)
    shot.parts = {"fit": round(fit, 2), "emotion": round(emotion, 2), "clarity": round(clarity, 2),
                  "event": round(event, 2), "angle": round(angle, 2)}
    return shot


def all_shots(looks: dict[str, dict], state: dict, setup: dict, boosts: dict[str, float]) -> list[Shot]:
    """Every scored shot of every camera at one look."""
    out = []
    for cam, look in looks.items():
        for shot in frame_shots(cam, look):
            out.append(score(shot, look, state, setup[cam]["view"], boosts))
    return out
