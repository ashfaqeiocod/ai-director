"""zxy.py — where every player and the ball are, 10 times a second, for the 3D scene (MVP 4 Step 3).

Why: the behind-goal cameras GOAL L / GOAL R don't exist at Alfheim, so Step 4 draws them in 3D. A 3D scene needs
positions for everyone on the pitch. We build that "digital twin" from three sources:
  home team (Tromsø)    the dataset's ZXY sensor data (every home player wore a radio tag, 20 times a second;
                        tag ids are random, the dataset forbids re-identifying players: we never try)
  away team + referees  OUR detections (studio.intensity, 5 times a second) put on the pitch with the calibration,
                        minus the detected players that are home players (they sit on a ZXY position), smoothed
                        by a small tracker (constant speed; a lost player is held for up to 1 s)
  ball                  our pitch map's ball (studio.pitchmap via studio.multicam, the same ball the director
                        sees); gaps up to 2 s are filled in a straight line; long fast balls get a parabola height

The twin is a DRAWING aid for the 3D cameras. The director never reads it: its cards for the 3D cameras come
from the pitch map, live. Filling a ball gap uses up to 2 s of "future": a real production has a 5-10 s broadcast
delay (docs/MVP4_STADIUM_PLAN.md section 3), so the drawing may lag the live decisions by that much.

Clocks: the video's 3-second pieces are named with the wall-clock time (first piece 18:01:14.248, none missing),
the ZXY rows too (first row 18:01:09). The exact offset (and which way the ZXY axes point) is measured by matching
the ZXY players to our detected players over a few minutes.

Usage:
  venv/bin/python -m studio.zxy               # download (~87 MB), align, write twin.jsonl + check pictures (~5 min)

Output: data/alfheim/zxy/<file>.csv (the download), data/alfheim/<match>_H<half>/twin.jsonl (one row per 0.1 s:
{t, home:[{id,x,y,speed}], away:[{id,x,y}], ref:[{id,x,y}], ball:{x,y,z}|null}), twin.json (alignment numbers),
twin_check/cam2_<t>.jpg. Settings: config.yaml -> studio.stadium.twin.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass, field

from scipy.optimize import least_squares
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, load_config
from studio.alfheim import folder_for
from studio.multicam import feet, merge_points, to_pitch

ZXY_URL = "https://datasets.simula.no/downloads/alfheim/{match}/zxy/{name}"
ZXY_FILES = {("2013-11-03", 1): "2013-11-03_tromso_stromsgodset_first.csv",
             ("2013-11-03", 2): "2013-11-03_tromso_stromsgodset_second.csv",
             ("2013-11-07", 1): "2013-11-07_tromso_anji_first.csv",
             ("2013-11-07", 2): "2013-11-07_tromso_anji_second.csv"}
ZXY_HZ = 20
HALF_X_M = 52.5
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


def settings() -> dict:
    return load_config()["studio"]["stadium"]["twin"]


def download(match: str, half: int) -> Path:
    name = ZXY_FILES[(match, half)]
    dst = ROOT / load_config()["studio"]["alfheim"]["dir"] / "zxy" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name} (~87 MB)", flush=True)
        part = dst.with_suffix(".part")
        # The site's certificate chain is incomplete (see studio.alfheim): -k. It is a plain CSV, nothing is run.
        subprocess.run(["curl", "-sk", "--retry", "5", "-C", "-", "-o", str(part),
                        ZXY_URL.format(match=match, name=name)], check=True)
        part.replace(dst)
    return dst


# ---------------------------------------------------------------------------------------------------------------
# 1. The sensor data on a regular 20 Hz grid
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Zxy:
    start: datetime                    # wall-clock time of sample 0
    tags: list[int]
    xy: np.ndarray                     # (samples, tags, 2) metres from a pitch corner, NaN = no data
    speed: np.ndarray                  # (samples, tags) m/s

    def at(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tags, xy, speed) of the tags that have data at this many seconds after `start` (nearest sample)."""
        k = int(round(seconds * ZXY_HZ))
        if not 0 <= k < len(self.xy):
            return np.zeros(0, int), np.zeros((0, 2)), np.zeros(0)
        ok = np.isfinite(self.xy[k, :, 0])
        return np.array(self.tags)[ok], self.xy[k, ok], self.speed[k, ok]


def parse_time(text: str) -> datetime:
    return datetime.strptime(text if "." in text else text + ".0", "%Y-%m-%d %H:%M:%S.%f")


def load_zxy(path: Path) -> Zxy:
    """Rows: "timestamp", tag, x, y, heading, direction, energy, speed, total distance (dataset paper)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append((parse_time(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[7])))
    start = min(r[0] for r in rows)
    tags = sorted({r[1] for r in rows})
    n = int(round((max(r[0] for r in rows) - start).total_seconds() * ZXY_HZ)) + 1
    xy = np.full((n, len(tags), 2), np.nan)
    speed = np.zeros((n, len(tags)))
    col = {t: i for i, t in enumerate(tags)}
    for when, tag, x, y, s in rows:
        k = int(round((when - start).total_seconds() * ZXY_HZ))
        xy[k, col[tag]] = (x, y)
        speed[k, col[tag]] = s
    return Zxy(start, tags, xy, speed)


def video_start(folder: Path) -> datetime:
    """Wall-clock time of the video's first frame: the first raw piece's name ("0056_2013-11-03 18:01:14.248...")."""
    match, half = folder.name.rsplit("_H", 1)
    raw = ROOT / load_config()["studio"]["alfheim"]["dir"] / "raw" / f"{match}_H{half}" / "0"
    first = sorted(raw.glob("*.h264"))[0].name
    return parse_time(first.split("_", 1)[1][:26])


def to_centre(xy: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    """ZXY corner-origin metres -> our centre-origin metres (x right, y towards the near touchline)."""
    out = np.asarray(xy, np.float64) - [PITCH_LENGTH / 2, PITCH_WIDTH / 2]
    return out * [-1 if flip_x else 1, -1 if flip_y else 1]


# ---------------------------------------------------------------------------------------------------------------
# 2. Our detections on the pitch (all 3 cameras merged), with the detector's class
# ---------------------------------------------------------------------------------------------------------------
def detected_people(src, frame: int, merge_m: float) -> list[tuple[float, float, str]]:
    """(x, y, class) of everyone the 3 cameras detected at this frame; the same person seen twice = one point."""
    per_cam = {}
    for c, det in src.detection_at(frame).items():
        spots = to_pitch(np.linalg.inv(src.H[c]), feet(det))
        per_cam[c] = [(x, y, p.get("cls", "player")) for (x, y), p in zip(spots, det.get("players", []))
                      if abs(x) <= 55 and abs(y) <= 37]
    return [(p[0], p[1], p[2] or "player") for p in merge_points(per_cam, merge_m)]


def align(src, z: Zxy, nominal_s: float, cfg: dict) -> dict:
    """Find the clock offset (ZXY seconds = video seconds + offset) and the axis directions: the choice that puts
    the ZXY home players closest to our detected players over a few minutes."""
    lo, hi = cfg["align_minutes"]
    frames = [f - f % src.step for f in range(int(lo * 60 * src.fps), int(hi * 60 * src.fps), int(cfg["align_every_s"] * src.fps))]
    people = {f: np.array([p[:2] for p in detected_people(src, f, src.settings["merge_players_m"])]) for f in frames}

    def score(offset: float, fx: bool, fy: bool) -> float:
        d = []
        for f, det in people.items():
            if len(det) < 5:
                continue
            _, xy, _ = z.at(f / src.fps + offset)
            if not len(xy):
                continue
            home = to_centre(xy, fx, fy)
            dist = np.linalg.norm(home[:, None] - det[None], axis=2).min(axis=1)
            d.append(np.minimum(dist, cfg["align_cap_m"]))
        return float(np.mean(np.concatenate(d))) if d else 99.0

    best = None
    for fx in (False, True):
        for fy in (False, True):
            for off in np.arange(nominal_s - cfg["align_search_s"], nominal_s + cfg["align_search_s"] + 1e-9, 0.5):
                s = score(off, fx, fy)
                if best is None or s < best[0]:
                    best = (s, off, fx, fy)
    _, off, fx, fy = best
    for off2 in np.arange(off - 0.5, off + 0.5 + 1e-9, 0.05):                 # fine search
        s = score(off2, fx, fy)
        if s < best[0]:
            best = (s, off2, fx, fy)
    s, off, fx, fy = best
    # A small steady difference remains between the sensors' pitch and our calibrated one (~0.5 m shift): fit a
    # 6-number (affine) correction on ZXY/detection pairs closer than 4 m.
    pairs = []
    for f, det in people.items():
        _, xy, _ = z.at(f / src.fps + off)
        if len(det) < 5 or not len(xy):
            continue
        for q in to_centre(xy, fx, fy):
            d = np.linalg.norm(det - q, axis=1)
            if d.min() < 4:
                pairs.append((q, det[d.argmin()]))
    base, target = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    def gap(a: np.ndarray) -> np.ndarray:
        return (apply_correction(base, a.tolist()) - target).ravel()

    a = least_squares(gap, [1, 0, 0, 0, 1, 0], loss="soft_l1", f_scale=1.0).x
    before = float(np.median(np.linalg.norm(base - target, axis=1)))
    after = float(np.median(np.linalg.norm(gap(a).reshape(-1, 2), axis=1)))
    return {"offset_s": round(float(off), 3), "nominal_offset_s": round(nominal_s, 3), "flip_x": fx, "flip_y": fy,
            "mean_distance_m": round(s, 2), "samples": len(frames), "correction": [round(float(v), 5) for v in a],
            "pairs": len(pairs), "median_before_correction_m": round(before, 2),
            "median_after_correction_m": round(after, 2)}


def apply_correction(xy: np.ndarray, a: list[float]) -> np.ndarray:
    """The affine correction: x' = a0 x + a1 y + a2, y' = a3 x + a4 y + a5."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return np.column_stack([a[0] * xy[:, 0] + a[1] * xy[:, 1] + a[2], a[3] * xy[:, 0] + a[4] * xy[:, 1] + a[5]])


def claim(home: np.ndarray, people: list[tuple[float, float, str]], radius: float) -> list[tuple[float, float, str]]:
    """Each home player (ZXY) claims the ONE detection nearest to him within `radius` (nearest pairs first).
    What is left are away players and referees, also the ones marking a home player closely."""
    pairs = sorted((float(np.hypot(h[0] - p[0], h[1] - p[1])), i, k) for i, h in enumerate(home)
                   for k, p in enumerate(people))
    used_home, used = set(), set()
    for d, i, k in pairs:
        if d > radius:
            break
        if i in used_home or k in used:
            continue
        used_home.add(i)
        used.add(k)
    return [p for k, p in enumerate(people) if k not in used]


# ---------------------------------------------------------------------------------------------------------------
# 3. Away players + referees: a small tracker on the pitch
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Track:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    last_t: float
    hits: int = 1


@dataclass
class PitchTracker:
    """Constant-speed tracker in metres. A track that isn't seen is predicted forward for up to `hold_s`."""
    gate_m: float
    hold_s: float
    smooth: float
    tracks: list[Track] = field(default_factory=list)
    next_id: int = 1

    def update(self, t: float, points: list[tuple[float, float]]) -> None:
        free = list(range(len(points)))
        pairs = []
        for tr in self.tracks:
            guess = tr.pos + tr.vel * (t - tr.last_t)
            for k in free:
                d = math.dist(guess, points[k])
                if d <= self.gate_m:
                    pairs.append((d, tr, k))
        used_tracks, used_points = set(), set()
        for d, tr, k in sorted(pairs, key=lambda p: p[0]):          # nearest pairs first
            if id(tr) in used_tracks or k in used_points:
                continue
            used_tracks.add(id(tr))
            used_points.add(k)
            dt = max(t - tr.last_t, 1e-3)
            guess = tr.pos + tr.vel * dt
            new = guess + self.smooth * (np.array(points[k]) - guess)
            tr.vel = 0.5 * tr.vel + 0.5 * (new - tr.pos) / dt
            tr.vel *= min(1.0, 9.0 / max(np.linalg.norm(tr.vel), 1e-9))   # nobody runs faster than ~9 m/s
            tr.pos, tr.last_t, tr.hits = new, t, tr.hits + 1
        for k in range(len(points)):
            if k not in used_points:
                self.tracks.append(Track(self.next_id, np.array(points[k], float), np.zeros(2), t))
                self.next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.hold_s]

    def at(self, t: float, min_hits: int) -> list[tuple[int, float, float]]:
        out = []
        for tr in self.tracks:
            if tr.hits >= min_hits and 0 <= t - tr.last_t <= self.hold_s:
                p = tr.pos + tr.vel * (t - tr.last_t)
                out.append((tr.id, float(p[0]), float(p[1])))
        return out


# ---------------------------------------------------------------------------------------------------------------
# 4. The ball: the pitch map's ball, gaps filled, a height for long fast balls
# ---------------------------------------------------------------------------------------------------------------
def ball_track(sightings: list[tuple[float, tuple[float, float]]], cfg: dict):
    """sightings = [(t, (x, y))] when the map SAW the ball. Returns ball(t) -> (x, y, z) or None."""
    times = np.array([s[0] for s in sightings])
    pts = np.array([s[1] for s in sightings], np.float64).reshape(-1, 2)

    def ball(t: float):
        if not len(times):
            return None
        k = int(np.searchsorted(times, t))
        if k < len(times) and abs(times[k] - t) < 1e-6:
            return float(pts[k, 0]), float(pts[k, 1]), 0.0
        if k == 0 or k == len(times):
            return None
        t0, t1 = times[k - 1], times[k]
        gap = t1 - t0
        if gap > cfg["ball_gap_s"]:
            return None
        s = (t - t0) / gap
        x, y = pts[k - 1] + s * (pts[k] - pts[k - 1])
        dist = float(np.linalg.norm(pts[k] - pts[k - 1]))
        z = 0.0
        if dist / gap >= cfg["lofted_speed_ms"] and dist >= cfg["lofted_min_m"]:
            # a lofted ball in the air for `gap` seconds rises g * gap^2 / 8 (simple physics, no spin or air)
            z = min(cfg["lofted_max_m"], 9.81 * gap * gap / 8) * 4 * s * (1 - s)
        return float(x), float(y), round(float(z), 2)

    return ball


# ---------------------------------------------------------------------------------------------------------------
# 4b. Teams from shirt colours (twin v2)
# ---------------------------------------------------------------------------------------------------------------
def learn_kits(src, z: Zxy, al: dict, shirts: dict, cfg: dict) -> dict[str, list[float]]:
    """The 3 shirt colours (Lab) of this match, learned from the data itself:
      home  = detections standing on a ZXY home player (the sensors say who they are),
      ref   = what the detector calls "referee",
      away  = the other detections that look like neither.
    Then a few rounds of "every shirt to its nearest colour, colours = the middle of their shirts" (k-means)."""
    fx, fy, off, corr = al["flip_x"], al["flip_y"], al["offset_s"], al["correction"]
    samples, first = [], {"home": [], "ref": [], "other": []}
    for frame in range(0, src.frames, src.step * cfg["kit_sample_every"]):
        _, xy, _ = z.at(frame / src.fps + off)
        home = apply_correction(to_centre(xy, fx, fy), corr) if len(xy) else np.zeros((0, 2))
        for c, det in src.detection_at(frame).items():
            labs = shirts.get(str(frame), {}).get(str(c), [])
            spots = to_pitch(np.linalg.inv(src.H[c]), feet(det))
            for p, spot, lab in zip(det.get("players", []), spots, labs):
                if lab is None or abs(spot[0]) > 55 or abs(spot[1]) > 37:
                    continue
                samples.append(lab)
                if p.get("cls") == "referee":
                    first["ref"].append(lab)
                elif len(home) and np.linalg.norm(home - spot, axis=1).min() < 1.0:
                    first["home"].append(lab)
                else:
                    first["other"].append(lab)
    centres = {k: np.median(np.array(first[k]), axis=0) for k in ("home", "ref")}
    other = np.array(first["other"])
    far = (np.linalg.norm(other - centres["home"], axis=1) > 25) & (np.linalg.norm(other - centres["ref"], axis=1) > 25)
    centres["away"] = np.median(other[far], axis=0)
    data = np.array(samples)
    names = ["home", "away", "ref"]
    for _ in range(5):
        c = np.array([centres[n] for n in names])
        nearest = np.linalg.norm(data[:, None] - c[None], axis=2).argmin(axis=1)
        for k, n in enumerate(names):
            if (nearest == k).sum() > 20:
                centres[n] = np.median(data[nearest == k], axis=0)
    return {n: [round(float(v), 1) for v in centres[n]] for n in names}


def kit_of(lab, kits: dict, max_distance: float) -> str | None:
    if lab is None:
        return None
    d = {n: float(np.linalg.norm(np.array(lab) - c)) for n, c in kits.items()}
    best = min(d, key=d.get)
    return best if d[best] <= max_distance else None


def labelled_people(src, frame: int, merge_m: float, shirts: dict, kits: dict, cfg: dict) -> list[tuple]:
    """(x, y, label) of everyone the 3 cameras detected, label = home / away / ref (shirt colour; the detector's
    "referee" class counts as ref) or None; the same person seen by two cameras = one point (majority label)."""
    per_cam = {}
    for c, det in src.detection_at(frame).items():
        labs = shirts.get(str(frame), {}).get(str(c), [])
        spots = to_pitch(np.linalg.inv(src.H[c]), feet(det))
        pts = []
        for k, ((x, y), p) in enumerate(zip(spots, det.get("players", []))):
            if abs(x) > 55 or abs(y) > 37:
                continue
            label = "ref" if p.get("cls") == "referee" else kit_of(labs[k] if k < len(labs) else None, kits,
                                                                       cfg["kit_max_distance"])
            if label == "ref" and p.get("cls") != "referee":
                label = "home"          # a bright white shirt the detector doesn't call a referee: a Tromsø shirt
            pts.append((x, y, label))
        per_cam[c] = pts
    return list(merge_points(per_cam, merge_m))


def polish_tracks(rows: list[dict], group: str, cfg: dict) -> None:
    """Away players / referees from the detector, cleaned for drawing (offline, within the broadcast delay):
    tracks shorter than `min_track_s` are dropped (one-moment ghosts), the rest are smoothed over +-`smooth_s`."""
    step = rows[1]["t"] - rows[0]["t"]
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for k, row in enumerate(rows):
        for p in row[group]:
            tracks.setdefault(p["id"], []).append((k, p["x"], p["y"]))
    half = max(1, round(cfg["smooth_s"] / step))
    new: dict[int, list[dict]] = {}
    for tid, pts in tracks.items():
        if len(pts) * step < cfg["min_track_s"]:
            continue
        a = np.array(pts, np.float64)
        for j in range(len(a)):
            near = a[max(0, j - half):j + half + 1]
            near = near[np.abs(near[:, 0] - a[j, 0]) <= half]          # only rows that are really close in time
            new.setdefault(int(a[j, 0]), []).append({"id": tid, "x": round(float(near[:, 1].mean()), 2),
                                                     "y": round(float(near[:, 2].mean()), 2)})
    life = {tid: len(pts) for tid, pts in tracks.items()}
    for k, row in enumerate(rows):
        kept: list[dict] = []
        # the same player tracked twice (two dots close together): keep the longer-lived track
        for p in sorted(new.get(k, []), key=lambda p: -life[p["id"]]):
            if all(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) > cfg["duplicate_m"] for q in kept):
                kept.append(p)
        row[group] = kept


def drop_hotspots(sightings: list, cfg: dict) -> tuple[list, list]:
    """Places where the detector sees "the ball" far more often than a real ball ever stays: painted spots (on
    Alfheim the right penalty spot was 13% of all sightings). Cells of 1 m with more than `hotspot_min` sightings
    over the half; every sighting within `hotspot_radius_m` of one is dropped."""
    from collections import Counter

    cells = Counter((round(p[0]), round(p[1])) for _, p in sightings)
    hot = [c for c, n in cells.items() if n >= cfg["hotspot_min"]]
    # one centre per cluster of hot cells
    centres: list[list[float]] = []
    for c in sorted(hot, key=lambda c: -cells[c]):
        if all(math.dist(c, q) > 2.5 for q in centres):
            centres.append([float(c[0]), float(c[1])])
    keep = [(t, p) for t, p in sightings if all(math.dist(p, c) > cfg["hotspot_radius_m"] for c in centres)]
    return keep, centres


def carry_ball(rows: list[dict], cfg: dict) -> int:
    """Ball gaps longer than the straight-line fill: the ball stays at the feet of the player nearest its last
    sighting (dribbling), and in the last `carry_blend_s` glides to where it is seen next. Returns rows filled."""
    step = rows[1]["t"] - rows[0]["t"]
    filled, k, n = 0, 0, len(rows)
    while k < n:
        if rows[k]["ball"] is not None:
            k += 1
            continue
        a = k
        while k < n and rows[k]["ball"] is None:
            k += 1
        if a == 0 or (k - a) * step > cfg["carry_max_s"]:
            continue
        nxt = rows[k]["ball"] if k < n else None
        carrier = None
        # who had it: the player nearest its last sighting, else the one nearest its next sighting (the receiver)
        for ref_row, spot in ((rows[a - 1], rows[a - 1]["ball"]), (rows[k] if k < n else None, nxt)):
            if ref_row is None or spot is None:
                continue
            players = [(p["x"], p["y"], g, p["id"]) for g in ("home", "away") for p in ref_row[g]]
            if players:
                best = min(players, key=lambda p: math.hypot(p[0] - spot["x"], p[1] - spot["y"]))
                if math.hypot(best[0] - spot["x"], best[1] - spot["y"]) <= cfg["carrier_m"]:
                    carrier = best
                    break
        if carrier is None:
            continue                                    # nobody near it at either end: leave the gap
        blend = round(cfg["carry_blend_s"] / step)
        for j in range(a, k):
            me = next((p for p in rows[j][carrier[2]] if p["id"] == carrier[3]), None)
            if me is None:
                break
            x, y = me["x"], me["y"]
            if nxt is not None and k - j <= blend:
                s = 1 - (k - j) / blend
                x, y = x + (nxt["x"] - x) * s, y + (nxt["y"] - y) * s
            rows[j]["ball"] = {"x": round(x, 2), "y": round(y, 2), "z": 0.0}
            filled += 1
    return filled


# ---------------------------------------------------------------------------------------------------------------
# 5. Build twin.jsonl
# ---------------------------------------------------------------------------------------------------------------
def build(match: str, half: int) -> Path:
    from studio.sources import get_source
    from studio.stadium import NoStory

    cfg = settings()
    src = get_source(f"alfheim-{match}-H{half}")
    folder = src.folder
    z = load_zxy(download(match, half))
    nominal = (video_start(folder) - z.start).total_seconds()
    print(f"  ZXY: {len(z.tags)} tags, {len(z.xy) / ZXY_HZ / 60:.1f} min; video starts {nominal:.2f} s after it",
          flush=True)
    al = align(src, z, nominal, cfg)
    print(f"  alignment: {al}", flush=True)
    fx, fy, off, corr = al["flip_x"], al["flip_y"], al["offset_s"], al["correction"]
    shirts_file = folder / "shirts.json"
    shirts = json.loads(shirts_file.read_text()) if shirts_file.exists() else None
    kits = learn_kits(src, z, al, shirts, cfg) if shirts else None
    if kits:
        print(f"  shirt colours (Lab): {kits}", flush=True)
    else:
        print("  no shirts.json (run --shirts first): away players = everyone the sensors don't explain", flush=True)

    # the pitch map over the whole half (the director's ball), and the away/referee trackers
    pipe = src.pipeline(NoStory())
    merge_m = src.settings["merge_players_m"]
    away = PitchTracker(cfg["track_gate_m"], cfg["track_hold_s"], cfg["track_smooth"])
    refs = PitchTracker(cfg["track_gate_m"], cfg["track_hold_s"], cfg["track_smooth"])
    sightings = []
    duration = src.frames / src.fps
    out = folder / "twin.jsonl"
    rows = 0
    step_s = cfg["step_s"]
    detection_frames = range(0, src.frames, src.step)
    next_det = iter(detection_frames)
    upcoming = next(next_det, None)
    with open(out, "w") as f:
        k = 0
        while k * step_s <= duration:
            t = round(k * step_s, 3)
            # feed every detection step up to this time into the map and the trackers (past only)
            while upcoming is not None and upcoming / src.fps <= t + 1e-9:
                td = upcoming / src.fps
                pipe.detections(upcoming, td, detector=src.detection_at(upcoming))
                st = pipe.pitch_state
                tags, xy, _ = z.at(td + off)
                home = apply_correction(to_centre(xy, fx, fy), corr) if len(xy) else np.zeros((0, 2))
                if st.ball is not None and st.ball_status == "seen":
                    # a "ball" far from every player is a painted spot or a spare ball (the right penalty spot came
                    # back as the ball again and again); a real ball in flight is only far for a moment, and the
                    # straight-line fill covers that moment
                    everyone = [*map(tuple, home), *(p[:2] for p in pipe.pitch_state.players)]
                    if everyone and min(math.dist(st.ball, q) for q in everyone) <= cfg["ball_max_from_player_m"]:
                        sightings.append((td, st.ball))
                rest, ref = [], []
                if kits:
                    for x, y, label in claim(home, labelled_people(src, upcoming, merge_m, shirts, kits, cfg),
                                             cfg["home_match_m"]):
                        if label == "ref":
                            ref.append((x, y))
                        elif label == "away" or (label is None and abs(x) > HALF_X_M - 16.5 and abs(y) < 20.16):
                            rest.append((x, y))     # an away player, or a keeper (other colours) in a box
                else:
                    for x, y, cls in claim(home, detected_people(src, upcoming, merge_m), cfg["home_match_m"]):
                        (ref if cls == "referee" else rest).append((x, y))
                away.update(td, rest)
                refs.update(td, ref)
                upcoming = next(next_det, None)
            tags, xy, speed = z.at(t + off)
            home = apply_correction(to_centre(xy, fx, fy), corr) if len(xy) else np.zeros((0, 2))
            row = {"t": t,
                   "home": [{"id": int(g), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
                             "speed": round(float(s), 2)} for g, p, s in zip(tags, home, speed)],
                   "away": [{"id": i, "x": round(x, 2), "y": round(y, 2)} for i, x, y in away.at(t, cfg["track_min_hits"])],
                   "ref": [{"id": i, "x": round(x, 2), "y": round(y, 2)} for i, x, y in refs.at(t, cfg["track_min_hits"])],
                   "ball": None}
            f.write(json.dumps(row) + "\n")
            rows += 1
            k += 1
    # second pass (needs what comes AFTER a moment: allowed within the broadcast delay, see above): the ball,
    # cleaned away / referee tracks, the ball carried by a player through longer gaps
    sightings, hotspots = drop_hotspots(sightings, cfg)
    print(f"  ball: {len(sightings)} sightings kept; fixed false 'balls' (painted spots) at {hotspots}", flush=True)
    # the raw sightings, for the game reconstruction (studio.play)
    (folder / "ball_sightings.json").write_text(json.dumps([[round(t, 3), round(float(p[0]), 2), round(float(p[1]), 2)]
                                                           for t, p in sightings]))
    ball = ball_track(sightings, cfg)
    all_rows = [json.loads(line) for line in out.read_text().splitlines()]
    for row in all_rows:
        b = ball(row["t"])
        row["ball"] = None if b is None else {"x": round(b[0], 2), "y": round(b[1], 2), "z": b[2]}
    for group in ("away", "ref"):
        polish_tracks(all_rows, group, cfg)
    carried = carry_ball(all_rows, cfg)
    with open(out, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    info = {**al, "rows": rows, "step_s": step_s, "zxy_tags": z.tags, "ball_sightings": len(sightings),
            "kits_lab": kits, "ball_carried_rows": carried,
            "away_per_row_median": float(np.median([len(r["away"]) for r in all_rows])),
            "ball_rows": sum(1 for line in out.read_text().splitlines() if '"ball":{' in line)}
    (folder / "twin.json").write_text(json.dumps(info, indent=1))
    print(f"  wrote {out.relative_to(ROOT)}: {rows} rows; ball in {info['ball_rows']} of them", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 5b. Shirt colours of every detected player (which team is a detection?)
# ---------------------------------------------------------------------------------------------------------------
def measure_shirts(match: str, half: int) -> Path:
    """The shirt colour (Lab) of every detected player at every detection step, from the 3 camera videos: decided
    once, saved to shirts.json ({frame: {cam: [[L, a, b] or null per player box]}}), so the twin can tell the
    away team (Strømsgodset, dark blue) from home players seen twice and from the referees (yellow).
    Reads every frame (decoding can't skip) but measures only the detection frames (~20 min for a half)."""
    import threading

    from studio.sources import get_source
    from studio.teams import TeamClassifier, shirt_colour

    src = get_source(f"alfheim-{match}-H{half}")
    out = src.folder / "shirts.json"
    result: dict[str, dict] = {}

    def one(cam: int, store: dict) -> None:
        cap = cv2.VideoCapture(str(src.video(cam)))
        frames = src.dets[cam]["frames"]
        i = 0
        while True:
            if i % src.step == 0 and str(i) in frames:
                ok, image = cap.read()
                if not ok:
                    break
                labs = []
                for p in frames[str(i)].get("players", []):
                    c = shirt_colour(image, p["box"])
                    labs.append(None if c is None else [round(float(v), 1) for v in TeamClassifier._lab(c)])
                store[str(i)] = labs
            elif not cap.grab():
                break
            i += 1
        cap.release()

    stores = {c: {} for c in src.cam_ids}
    threads = [threading.Thread(target=one, args=(c, stores[c])) for c in src.cam_ids]
    t0 = __import__("time").time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for c, store in stores.items():
        for frame, labs in store.items():
            result.setdefault(frame, {})[str(c)] = labs
    out.write_text(json.dumps(result, separators=(",", ":")))
    print(f"  shirts: {len(result)} detection frames in {(__import__('time').time() - t0) / 60:.1f} min -> "
          f"{out.relative_to(ROOT)}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 6. Check pictures: the twin drawn onto real cam2 frames
# ---------------------------------------------------------------------------------------------------------------
def load_twin(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "twin.jsonl").read_text().splitlines()]


def check(match: str, half: int) -> dict:
    """twin_check/cam2_<t>.jpg at 5 moments + the median distance from each home (ZXY) player to the nearest
    player our detector found at that frame (metres)."""
    from studio.sources import get_source

    cfg = settings()
    src = get_source(f"alfheim-{match}-H{half}")
    twin = load_twin(src.folder)
    step = twin[1]["t"] - twin[0]["t"]
    outdir = src.folder / "twin_check"
    outdir.mkdir(exist_ok=True)
    H = src.H[2]
    report, dists = {}, []
    colours = {"home": (255, 255, 255), "away": (255, 120, 0), "ref": (0, 230, 255)}
    for minute in cfg["check_minutes"]:
        frame = int(minute * 60 * src.fps)
        frame -= frame % src.step
        row = twin[int(round(frame / src.fps / step))]
        cap = cv2.VideoCapture(str(src.video(2)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        det = np.array([p[:2] for p in detected_people(src, frame, src.settings["merge_players_m"])]).reshape(-1, 2)
        near = []
        for group, colour in colours.items():
            for p in row[group]:
                if group == "home" and len(det):
                    near.append(float(np.linalg.norm(det - [p["x"], p["y"]], axis=1).min()))
                q = cv2.perspectiveTransform(np.array([[[p["x"], p["y"]]]], np.float64), H)[0, 0]
                if 0 <= q[0] < img.shape[1] and 0 <= q[1] < img.shape[0]:
                    cv2.circle(img, (int(q[0]), int(q[1])), 9, colour, 2)
        if row["ball"]:
            q = cv2.perspectiveTransform(np.array([[[row["ball"]["x"], row["ball"]["y"]]]], np.float64), H)[0, 0]
            cv2.circle(img, (int(q[0]), int(q[1])), 6, (0, 140, 255), -1)
        cv2.putText(img, f"{int(minute)}:{int(minute % 1 * 60):02d}  white=home (ZXY)  blue=away  yellow=ref  orange=ball", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(outdir / f"cam2_{minute:05.1f}.jpg"), img)
        in_view = [d for d in near if d < 10]                # home players our detector also found (cam1-3)
        report[f"{minute:.0f}min"] = {"home_players": len(near),
                                      "median_m": round(float(np.median(in_view)), 2) if in_view else None}
        dists += in_view
    report["all_median_m"] = round(float(np.median(dists)), 2) if dists else None
    report["all_p90_m"] = round(float(np.percentile(dists, 90)), 2) if dists else None
    print(json.dumps(report, indent=1))
    print(f"  check pictures: {outdir.relative_to(ROOT)}/")
    return report


def main() -> None:
    alf = load_config()["studio"]["alfheim"]
    ap = argparse.ArgumentParser(description="Players + ball 10x per second for the 3D scene (MVP 4 Step 3).")
    ap.add_argument("--match", default=alf["match"])
    ap.add_argument("--half", type=int, default=alf["half"])
    ap.add_argument("--check-only", action="store_true", help="only the check pictures (twin.jsonl exists)")
    ap.add_argument("--shirts", action="store_true", help="measure every detected player's shirt colour (~20 min)")
    args = ap.parse_args()
    if args.shirts:
        measure_shirts(args.match, args.half)
        return
    if not args.check_only:
        build(args.match, args.half)
    check(args.match, args.half)


if __name__ == "__main__":
    main()
