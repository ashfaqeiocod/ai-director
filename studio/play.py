"""Game reconstruction (Virtual Broadcast): turn the twin into a match that PLAYS like football.

The twin (studio.zxy) is what our sensors and detector measured: the home team is exact (ZXY sensors), but the
away team, the referees and the ball come from the detector, so they flicker. Players pop in and out (3,575 short
tracks in a half), the ball is missing 28% of the time, it jumps (442 jumps faster than 35 m/s) and it floats with
nobody near it (38% of the time). Drawn in 3D, that does not look like football.

This module rebuilds the game from those measurements, with what we know about football:

  1. Fixed figures. Each team has 11 players (10 outfield + 1 keeper) and there are 3 referees. Each detected
     track piece is given to one fixed figure (the pieces of one figure never overlap in time, and the jump
     between two pieces is possible at running speed). The figure walks in a straight line through the gaps.
     Nobody pops in or out.
  2. Who has the ball. At every moment one player owns the ball. The owner is chosen for the whole half at once
     (a Viterbi search, like a phone keyboard guessing a word from sloppy taps): the ball sightings say who is
     near it, and "the ball goes to another player" has a cost, so the owner only changes when the sightings
     really say so. Sightings that are plain wrong (false balls) cost little and are outvoted.
  3. Touches and passes. While a player owns the ball it sits at his feet, pushed ahead a little on every
     touch as he runs. When the owner changes, the ball travels from the passer's foot to the receiver's foot at
     a pass speed that fits the distance: rolling and slowing on the ground, or in the air for long balls.
     The ball is never missing and never moves without a touch.

Offline: a moment uses sightings up to a few seconds later (the same as the twin's cleaning, inside a broadcast
delay). Writes play.jsonl (the twin's row format: one row per 0.1 s, plus `own` and `kick`) and play.json (the
checks). Run:  venv/bin/python -m studio.play
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.common import ROOT, load_config

HALF_X, HALF_Y = 52.5, 34.0


def settings() -> dict:
    return load_config()["studio"]["broadcast"]["play"]


# ---------------------------------------------------------------------------------------------------------------
# 1. Input
# ---------------------------------------------------------------------------------------------------------------
def load(folder: Path) -> tuple[list[dict], np.ndarray]:
    rows = [json.loads(line) for line in open(folder / "twin.jsonl")]
    path = folder / "ball_sightings.json"
    if not path.exists():
        raise SystemExit("ball_sightings.json missing: run `venv/bin/python -m studio.zxy` first")
    return rows, np.array(json.loads(path.read_text()), np.float64).reshape(-1, 3)


def on_pitch_home(p: dict) -> bool:
    """A sensor player standing still outside the lines is on the bench (or a substitute warming up)."""
    return not ((abs(p["x"]) > 53 or abs(p["y"]) > 34.6) and p.get("speed", 0) < 0.5)


# ---------------------------------------------------------------------------------------------------------------
# 2. Fixed figures from track pieces
# ---------------------------------------------------------------------------------------------------------------
def pieces(dots: list[list[tuple[int, float, float]]]) -> list[np.ndarray]:
    """Rows of (row, x, y) per detector track id."""
    by_id: dict[int, list] = {}
    for k, row in enumerate(dots):
        for tid, x, y in row:
            by_id.setdefault(tid, []).append((k, x, y))
    return [np.array(v, np.float64) for v in by_id.values()]


def stitch(dots: list[list[tuple[int, float, float]]], slots: int, step: float, cfg: dict,
           centre: np.ndarray) -> np.ndarray:
    """Give every track piece to one of `slots` fixed figures. Longest pieces first; a piece fits a figure when
    it does not overlap that figure's pieces in time and the jumps to the piece before / after are possible at
    running speed. Among the figures it fits, the one whose place in the team (position relative to the middle of
    the play, `centre`) matches best gets it. A piece that fits nowhere is a ghost or a duplicate and is dropped.
    Returns (slots, rows, 2) positions with NaN where the figure was not seen."""
    n = len(dots)
    owned: list[list[np.ndarray]] = [[] for _ in range(slots)]

    def reach(a: np.ndarray, b: np.ndarray) -> float | None:
        """How far his place in the team moved from the end of piece a to the start of piece b, or None if b is
        too far to be him (faster than running)."""
        ka, kb = int(a[-1, 0]), int(b[0, 0])
        gap = (kb - ka) * step
        if gap <= 0 or math.dist(a[-1, 1:], b[0, 1:]) > cfg["run_max_ms"] * gap + cfg["join_slack_m"]:
            return None
        return math.dist(a[-1, 1:] - centre[ka], b[0, 1:] - centre[kb])

    for piece in sorted(pieces(dots), key=lambda p: -len(p)):
        best, best_cost = None, math.inf
        for s in range(slots):
            mine = owned[s]
            before = [p for p in mine if p[-1, 0] < piece[0, 0]]
            after = [p for p in mine if p[0, 0] > piece[-1, 0]]
            if len(before) + len(after) != len(mine):
                continue                                         # overlaps in time
            cost = 0.0
            if before:
                d = reach(max(before, key=lambda p: p[-1, 0]), piece)
                if d is None:
                    continue
                cost += d
            if after:
                d = reach(piece, min(after, key=lambda p: p[0, 0]))
                if d is None:
                    continue
                cost += d
            if cost < best_cost:
                best, best_cost = s, cost
        if best is not None:
            owned[best].append(piece)
    out = np.full((slots, n, 2), np.nan)
    for s in range(slots):
        for p in owned[s]:
            out[s, p[:, 0].astype(int)] = p[:, 1:]
    return out


def fill(track: np.ndarray, smooth_rows: int, centre: np.ndarray | None = None) -> np.ndarray:
    """Positions through the gaps. With `centre` (the middle of the play): he keeps moving with the team, his
    place in it changing steadily from where he was last seen to where he is seen next (a straight walk across
    the pitch while everyone else moves looks robotic). Without: a straight walk. Then a light moving average."""
    seen = ~np.isnan(track[:, 0])
    if not seen.any():
        return track
    base = centre if centre is not None else np.zeros_like(track)
    rel = track - base
    k = np.arange(len(track))
    out = base + np.stack([np.interp(k, k[seen], rel[seen, a]) for a in (0, 1)], axis=1)
    np.clip(out, [-HALF_X - 3, -HALF_Y - 3], [HALF_X + 3, HALF_Y + 3], out=out)
    if smooth_rows > 0:
        w = 2 * smooth_rows + 1
        pad = np.pad(out, ((smooth_rows, smooth_rows), (0, 0)), mode="edge")
        kernel = np.ones(w) / w
        out = np.stack([np.convolve(pad[:, a], kernel, mode="valid") for a in (0, 1)], axis=1)
    return out


def bridge(track: np.ndarray, max_rows: int) -> np.ndarray:
    """A sensor that drops out for a moment (gaps up to `max_rows`) while the player is on the pitch: straight
    line across the gap (left as NaN = off the pitch when longer, e.g. a substitution)."""
    seen = ~np.isnan(track[:, 0])
    idx = np.flatnonzero(seen)
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_rows:
            for k in range(a + 1, b):
                track[k] = track[a] + (track[b] - track[a]) * (k - a) / (b - a)
    return track


def keeper(rows: list[dict], goal_x: float, inward: float, home_xy: list[np.ndarray], cfg: dict,
           check_home: bool) -> tuple[np.ndarray, list[set]]:
    """The keeper defending the goal at x = goal_x: per row the deepest detected dot near that goal (deeper than
    every outfield sensor player for the home keeper), kept only when it continues the keeper's path. Returns the
    keeper's positions (NaN when not seen) and, per row, the ids of the dots used (taken out of the away team)."""
    n = len(rows)
    out = np.full((n, 2), np.nan)
    used: list[set] = [set() for _ in range(n)]
    last, last_k = None, -1
    step = rows[1]["t"] - rows[0]["t"]
    for k, row in enumerate(rows):
        depth_home = min(((x - goal_x) * inward for x, _ in home_xy[k]), default=99) if check_home else 99
        best = None
        for p in row["away"]:
            depth = (p["x"] - goal_x) * inward
            if -3 <= depth <= cfg["keeper_box_m"] and abs(p["y"]) <= cfg["keeper_wide_m"] and depth < depth_home:
                if best is None or depth < best[0]:
                    best = (depth, p)
        if best is None:
            continue
        p = best[1]
        if last is not None and math.dist(last, (p["x"], p["y"])) > cfg["keeper_jump_m"] + cfg["run_max_ms"] * (k - last_k) * step:
            continue
        out[k] = (p["x"], p["y"])
        used[k].add(p["id"])
        last, last_k = (p["x"], p["y"]), k
    return out, used


def keeper_fallback(track: np.ndarray, goal_x: float, inward: float) -> np.ndarray:
    """Never seen at all: stand him 4 m off his goal line (a goal with no keeper looks wrong on TV)."""
    if np.isnan(track[:, 0]).all():
        track[:] = (goal_x + 4 * inward, 0.0)
    return track


def limit_speed(track: np.ndarray, max_m: float) -> np.ndarray:
    """Never faster than `max_m` per row: capped going forward in time and going backward, then the two averaged
    (so a jump becomes a run that starts a little early and ends a little late, not a lag)."""
    def one_way(t: np.ndarray) -> np.ndarray:
        t = t.copy()
        for k in range(1, len(t)):
            d = t[k] - t[k - 1]
            length = float(np.hypot(*d))
            if length > max_m:
                t[k] = t[k - 1] + d * max_m / length
        return t
    return (one_way(track) + one_way(track[::-1])[::-1]) / 2


def play_centre(home: np.ndarray, smooth_rows: int) -> np.ndarray:
    """The middle of the play per row: the mean of the home outfield players (sensors: always there), smoothed."""
    seen = ~np.isnan(home[:, :, 0])
    c = np.nansum(home, axis=0) / np.maximum(seen.sum(axis=0), 1)[:, None]
    return fill(c.copy(), smooth_rows)


def assistant(line_y: float, half: int, defenders: np.ndarray, ball: np.ndarray, smooth_rows: int,
              max_step: float) -> np.ndarray:
    """An assistant referee runs along his touchline (y = line_y) in his half (x * half >= 0), level with the
    offside line: the second-last defender of the team defending that half, or the ball when it is deeper."""
    x = np.sort(defenders[:, :, 0] * half, axis=0)              # deepest last
    line = np.maximum(np.nan_to_num(x[-2], nan=0.0), ball[:, 0] * half)
    line = np.clip(line, 0, HALF_X) * half
    out = np.stack([line, np.full(len(line), line_y)], axis=1)
    return limit_speed(fill(out, smooth_rows * 3), max_step)


# ---------------------------------------------------------------------------------------------------------------
# 3. Who has the ball (Viterbi over the whole half)
# ---------------------------------------------------------------------------------------------------------------
def loose_track(sight: np.ndarray, n: int, step: float, max_ms: float) -> np.ndarray:
    """Where a loose ball is (rolling with nobody on it, going out of play): the sightings joined by straight
    lines, smoothed over +-0.3 s (a sighting is a few metres off), never faster than `max_ms`."""
    track = np.full((n, 2), np.nan)
    for t, x, y in sight:
        k = int(round(t / step))
        if 0 <= k < n:
            track[k] = (x, y)
    track = fill(track, round(0.3 / step))
    most = max_ms * step
    for k in range(1, n):                                       # the detector's jumps, capped at a hard shot's speed
        d = track[k] - track[k - 1]
        length = float(np.hypot(*d))
        if length > most:
            track[k] = track[k - 1] + d * most / length
    return track


def owners(pos: np.ndarray, sight: np.ndarray, step: float, cfg: dict) -> np.ndarray:
    """pos = (players + 1, rows, 2) with NaN when a player is not on the pitch; the last one is the loose ball
    (nobody has it). sight = (m, 3) of (t, x, y). Returns the owner (index into pos) for every row."""
    J, n, _ = pos.shape
    emit = np.zeros((n, J))
    sig2 = 2 * cfg["sighting_sigma_m"] ** 2
    for t, x, y in sight:
        k = int(round(t / step))
        if 0 <= k < n:
            d2 = (pos[:, k, 0] - x) ** 2 + (pos[:, k, 1] - y) ** 2
            emit[k] += np.log(np.exp(-np.nan_to_num(d2, nan=1e6) / sig2) + cfg["sighting_floor"])
    # the loose ball explains any sighting as well as a player `loose_r_m` away from it would
    loose = math.log(math.exp(-cfg["loose_r_m"] ** 2 / sig2) + cfg["sighting_floor"])
    emit[:, -1] = -cfg["loose_per_s"] * step                    # a ball is not loose for long: it costs every second
    for t, _, _ in sight:
        k = int(round(t / step))
        if 0 <= k < n:
            emit[k, -1] += loose
    absent = np.isnan(pos[:, :, 0]).T                           # (rows, players)
    emit[absent] = -1e9
    # changing owner costs `switch_cost`; the ball becoming loose or a loose ball being picked up `loose_cost`
    sw, lc = cfg["switch_cost"], cfg["loose_cost"]
    players = np.arange(J - 1)
    back = np.zeros((n, J), np.int16)
    score = emit[0].copy()
    for k in range(1, n):
        order = np.argsort(score[:-1])
        b1, b2 = int(order[-1]), int(order[-2])
        other = np.where(players == b1, b2, b1)                  # the best OTHER player for each player
        cand = np.stack([score[:-1],                              # stay
                         score[other] - sw,                       # a pass from another player
                         np.full(J - 1, score[-1] - lc)])         # picks up the loose ball
        pick = cand.argmax(axis=0)
        new = np.empty(J)
        new[:-1] = cand[pick, players]
        back[k, :-1] = np.where(pick == 0, players, np.where(pick == 1, other, J - 1))
        if score[-1] >= score[b1] - lc:                            # the loose ball stays loose ...
            new[-1], back[k, -1] = score[-1], J - 1
        else:                                                      # ... or the best player loses it
            new[-1], back[k, -1] = score[b1] - lc, b1
        score = new + emit[k]
    own = np.zeros(n, np.int64)
    own[-1] = int(np.argmax(score))
    for k in range(n - 1, 0, -1):
        own[k - 1] = back[k, own[k]]
    return own


def guess_passes(own: np.ndarray, pos: np.ndarray, team: np.ndarray, sight: np.ndarray, dead: np.ndarray,
                 step: float, cfg: dict) -> int:
    """The cameras often lose the ball for a while in open play; the search then leaves it with one player for
    10+ s, which never happens in football. In such a blind stretch (no sighting confirms the owner) the owner
    plays it to his nearest team-mate and gets it back (or the team-mate keeps it when the stretch ends with a
    real pass), every `guess_hold_s`. Rows a sighting confirms are never changed. Returns the passes added."""
    n = len(own)
    hold = round(cfg["guess_hold_s"] / step)
    near_m, (lo_m, hi_m) = cfg["release_near_m"] + 1, cfg["guess_mate_m"]
    confirmed = np.zeros(n, bool)
    for t, x, y in sight:
        k = int(round(t / step))
        if 0 <= k < n and math.dist(pos[own[k], k], (x, y)) <= near_m:
            confirmed[max(0, k - hold // 2):k + hold // 2 + 1] = True
    added, k = 0, 0
    while k < n:
        if confirmed[k] or dead[k]:
            k += 1
            continue
        a = k
        while k < n and not confirmed[k] and not dead[k] and own[k] == own[a]:
            k += 1
        if (k - a) < cfg["max_hold_s"] / step:
            continue
        p, back = int(own[a]), k < n and confirmed[k] and own[k] == own[a]
        cur, j = p, a + hold
        while j + hold <= k:
            if cur == p:
                d = np.linalg.norm(pos[:, j] - pos[p, j], axis=1)
                ok = (team == team[p]) & (d >= lo_m) & (d <= hi_m) & ~np.isnan(d)
                ok[p] = False
                if not ok.any():
                    break
                cur = int(np.argmin(np.where(ok, d, np.inf)))
            else:
                cur = p
            end = k if j + 2 * hold > k else j + hold
            own[j:end] = cur
            added += 1
            j = end
        if back and own[k - 1] != p:                             # the stretch ends confirmed with him: give it back
            own[max(a, k - hold):k] = p
    return added


# ---------------------------------------------------------------------------------------------------------------
# 4. The ball: at the owner's feet, touches while he runs, passes between owners
# ---------------------------------------------------------------------------------------------------------------
def headings(pos: np.ndarray, step: float, start: float) -> tuple[np.ndarray, np.ndarray]:
    """Facing direction (unit vector) and speed of every player per row: where he runs; kept while he stands."""
    J, n, _ = pos.shape
    vel = np.zeros_like(pos)
    vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / (2 * step)
    speed = np.nan_to_num(np.linalg.norm(vel, axis=2))
    head = np.zeros_like(pos)
    cur = np.tile([start, 0.0], (J, 1))
    for k in range(n):
        moving = speed[:, k] > 0.8
        cur[moving] = vel[moving, k] / speed[moving, k][:, None]
        head[:, k] = cur
    return head, speed


def segments(own: np.ndarray) -> list[tuple[int, int, int]]:
    """(owner, first row, last row) of each spell on the ball."""
    out, a = [], 0
    for k in range(1, len(own) + 1):
        if k == len(own) or own[k] != own[a]:
            out.append((int(own[a]), a, k - 1))
            a = k
    return out


def passes(segs: list, pos: np.ndarray, sight: np.ndarray, step: float, cfg: dict) -> list[dict]:
    """When each pass leaves the passer and reaches the receiver: the passer's last sighting near him is the kick,
    else the pass is centred on the change of owner; its length sets how long it takes."""
    sp, (tmin, tmax) = cfg["pass_speed"], cfg["pass_time_s"]
    sight_rows = np.round(sight[:, 0] / step).astype(int)
    out, prev_end = [], 0
    for (p, a0, a1), (q, b0, b1) in zip(segs, segs[1:]):
        d = float(np.linalg.norm(pos[p, a1] - pos[q, b0]))       # (the passer may leave the pitch at b0)
        v = min(sp["max_ms"], sp["base_ms"] + sp["per_m"] * d)
        dur = min(tmax, max(tmin, d / v)) / step
        lo = int(max(a0, b0 - dur - 10))
        near = [k for k, (_, x, y) in zip(sight_rows, sight)
                if lo <= k <= a1 and math.dist(pos[p, k], (x, y)) <= cfg["release_near_m"]] if d > 3 else []
        r = max(near) if near else round(b0 - dur / 2)
        r = max(r, a0, prev_end)
        e = min(round(r + dur), b1)
        if e <= r:
            e = min(r + 1, b1)
        if e <= r:
            r = e - 1
        flown = float(np.linalg.norm(pos[p, r] - pos[q, e]))
        out.append({"from": p, "to": q, "r": int(r), "e": int(e), "len": round(d, 1),
                    "speed": flown / max((e - r) * step, step),
                    "lofted": d >= cfg["lofted_m"]})
        prev_end = e
    return out


def clean_touches(own: np.ndarray, pos: np.ndarray, sight: np.ndarray, step: float, cfg: dict) -> list[dict]:
    """A "touch" that leaves no time to get the ball there and away again at a real pass speed did not happen (the
    ball went past a player, or a sighting was off): the passer keeps it. Repeated until every pass is possible."""
    limit = cfg["pass_speed"]["max_ms"] * cfg["pass_speed_tolerance"]
    for _ in range(20):
        segs = segments(own)
        pas = passes(segs, np.nan_to_num(pos), sight, step, cfg)
        bad = [i for i, p in enumerate(pas) if p["speed"] > limit]
        if not bad:
            break
        done = set()
        for i in bad:
            if i in done or i - 1 in done:
                continue                                         # neighbours change: look again next round
            _, b0, b1 = segs[i + 1]
            own[b0:b1 + 1] = segs[i][0]
            done.add(i)
    return pas


def holders(own: np.ndarray, pas: list) -> list[tuple[int, int, int]]:
    """Who really has the ball, from the passes: the passer until his kick, the receiver from the moment the ball
    reaches him (not from the Viterbi's change of owner, which can come before the kick or after the arrival)."""
    out, start, who = [], 0, int(own[0])
    for p in pas:
        out.append((who, start, p["r"]))
        start, who = p["e"], p["to"]
    out.append((who, start, len(own) - 1))
    return out


def ball_path(own: np.ndarray, pas: list, pos: np.ndarray, head: np.ndarray, speed: np.ndarray,
              step: float, cfg: dict) -> np.ndarray:
    """(rows, 3) ball positions x, y, z."""
    n = len(own)
    segs = holders(own, pas)
    ball = np.zeros((n, 3))
    push = cfg["touch_push"]

    loose = pos.shape[0] - 1

    def feet(j: int, k: int, phase: float) -> np.ndarray:
        if j == loose:
            return pos[j, k].copy()
        ahead = cfg["foot_m"] + min(push["max_m"], push["per_ms"] * speed[j, k]) * math.sin(math.pi * phase)
        return pos[j, k] + head[j, k] * ahead

    # on the ball: dribbling touches every `touch_every_m` of running (the ball leaves the foot and is caught up)
    for j, a, b in segs:
        run = 0.0
        for k in range(a, b + 1):
            if k > a:
                run += float(np.linalg.norm(pos[j, k] - pos[j, k - 1]))
            ball[k, :2] = feet(j, k, (run / cfg["touch_every_m"]) % 1.0)
    # passes: from the passer's foot at the kick to the receiver's foot when it arrives
    for p in pas:
        r, e = p["r"], p["e"]
        start, end = ball[r, :2].copy(), feet(p["to"], e, 0.0)
        dur = (e - r) * step
        top = min(cfg["lofted_max_m"], 9.81 * dur * dur / 8) if p["lofted"] else 0.0
        for k in range(r + 1, e + 1):
            u = (k - r) / (e - r)
            s = u if p["lofted"] else (u - 0.25 * u * u) / 0.75     # a ground pass slows down as it rolls
            ball[k, :2] = start + (end - start) * s
            ball[k, 2] = top * 4 * u * (1 - u)
        ball[e, :2] = end
    return ball


def dead_rows(sight: np.ndarray, n: int, step: float, gap_s: float) -> np.ndarray:
    """Rows inside a sighting gap longer than `gap_s` (play stopped: the director's stoppage)."""
    dead = np.zeros(n, bool)
    times = np.concatenate([[0.0], sight[:, 0], [n * step]])
    for t0, t1 in zip(times, times[1:]):
        if t1 - t0 > gap_s:
            dead[int(t0 / step) + 1:int(t1 / step)] = True
    return dead


# ---------------------------------------------------------------------------------------------------------------
# 5. Build + check
# ---------------------------------------------------------------------------------------------------------------
def build(folder: Path) -> dict:
    from studio.broadcast import home_attacks

    cfg = settings()
    rows, sight = load(folder)
    n, step = len(rows), rows[1]["t"] - rows[0]["t"]
    hdir = home_attacks(folder)                         # +1: home attacks x > 0, defends x = -52.5
    smooth = round(cfg["smooth_s"] / step)

    # home: the sensors (bench players left out); keepers: the deepest dot at each goal
    tags = sorted({p["id"] for r in rows for p in r["home"]})
    home = np.full((len(tags), n, 2), np.nan)
    for k, r in enumerate(rows):
        for p in r["home"]:
            if on_pitch_home(p):
                home[tags.index(p["id"]), k] = (p["x"], p["y"])
    for i in range(len(tags)):
        home[i] = bridge(home[i], round(cfg["sensor_gap_s"] / step))
    home_xy = [home[:, k][~np.isnan(home[:, k, 0])] for k in range(n)]
    gk_home, used_h = keeper(rows, -hdir * HALF_X, hdir, home_xy, cfg, check_home=True)
    gk_away, used_a = keeper(rows, hdir * HALF_X, -hdir, home_xy, cfg, check_home=False)
    gk_home = keeper_fallback(fill(gk_home, smooth), -hdir * HALF_X, hdir)
    gk_away = keeper_fallback(fill(gk_away, smooth), hdir * HALF_X, -hdir)

    # away outfield + referees: fixed figures
    away_dots = [[(p["id"], p["x"], p["y"]) for p in r["away"] if p["id"] not in used_h[k] | used_a[k]]
                 for k, r in enumerate(rows)]
    centre = play_centre(home, round(1.0 / step))
    away = stitch(away_dots, cfg["away_slots"], step, cfg, centre)
    # referees: the referee in the middle; the 2 assistants on their touchlines (from where the detector saw them)
    side_dots = [p for r in rows for p in r["ref"] if abs(p["y"]) > cfg["assistant_y_m"]]
    upper = np.mean([p["x"] for p in side_dots if p["y"] > 0] or [1.0])   # the assistant at y > 0 runs in this half
    refs = stitch([[(p["id"], p["x"], p["y"]) for p in r["ref"] if abs(p["y"]) <= cfg["assistant_y_m"]] for r in rows],
                  1, step, cfg, centre)
    seen_pct = {"away": round(100 * float((~np.isnan(away[:, :, 0])).mean()), 1),
                "ref": round(100 * float((~np.isnan(refs[0, :, 0])).mean()), 1),
                "home_keeper": round(100 * float(sum(bool(u) for u in used_h) / n), 1),
                "away_keeper": round(100 * float(sum(bool(u) for u in used_a) / n), 1)}
    most = cfg["run_max_ms"] * step
    away = np.stack([limit_speed(fill(a, smooth, centre), most) for a in away])
    main_ref = limit_speed(fill(refs[0], smooth, centre), most)
    gk_home, gk_away = limit_speed(gk_home, most), limit_speed(gk_away, most)

    # everyone who can have the ball: home sensors (NaN when off the pitch), the keepers, the away figures
    keys = [("home", t) for t in tags] + [("home", 0), ("away", 0)] + [("away", s + 1) for s in range(len(away))]
    pos = np.concatenate([home, gk_home[None], gk_away[None], away, loose_track(sight, n, step, cfg["pass_speed"]["max_ms"])[None]])
    own = owners(pos, sight, step, cfg)
    dead = dead_rows(sight, n, step, cfg["dead_gap_s"])
    team = np.array([0] * (len(tags) + 1) + [1] * (1 + len(away)) + [2])
    guessed = guess_passes(own, pos, team, sight, dead, step, cfg)
    # the assistants (need the ball for the offside line; the first ball guess is close enough)
    pre = np.nan_to_num(pos[own, np.arange(n)])
    home_def = np.concatenate([home, gk_home[None]])
    away_def = np.concatenate([gk_away[None], away])
    hs = int(np.sign(upper)) or 1                                # the half the y > 0 assistant covers
    def_up = away_def if hdir == hs else home_def                # who defends that half
    def_dn = home_def if hdir == hs else away_def
    refs = np.stack([main_ref, assistant(HALF_Y + 0.8, hs, def_up, pre, smooth, most),
                     assistant(-HALF_Y - 0.8, -hs, def_dn, pre, smooth, most)])
    head, speed = headings(np.nan_to_num(pos), step, float(hdir))
    pas = clean_touches(own, pos, sight, step, cfg)
    ball = ball_path(own, pas, np.nan_to_num(pos), head, speed, step, cfg)
    # out of play: a long blind gap, or the ball over a line (until it is seen back on the pitch)
    outside = (np.abs(ball[:, 0]) > HALF_X + cfg["out_m"]) | (np.abs(ball[:, 1]) > HALF_Y + cfg["out_m"])
    dead |= outside
    flying = np.zeros(n, bool)
    for p in pas:
        flying[p["r"] + 1:p["e"]] = True
    for j, a, b in holders(own, pas):
        own[a:b + 1] = j
    kicks = {p["r"]: p for p in pas}

    def key(j: int) -> str | None:
        return f"{keys[j][0]}{keys[j][1]}" if j < len(keys) else None

    out = folder / "play.jsonl"
    with open(out, "w") as f:
        for k, r in enumerate(rows):
            row = {"t": r["t"],
                   "home": [{"id": t, "x": round(float(home[i, k, 0]), 2), "y": round(float(home[i, k, 1]), 2)}
                            for i, t in enumerate(tags) if not np.isnan(home[i, k, 0])]
                   + [{"id": 0, "gk": 1, "x": round(float(gk_home[k, 0]), 2), "y": round(float(gk_home[k, 1]), 2)}],
                   "away": [{"id": 0, "gk": 1, "x": round(float(gk_away[k, 0]), 2), "y": round(float(gk_away[k, 1]), 2)}]
                   + [{"id": s + 1, "x": round(float(away[s, k, 0]), 2), "y": round(float(away[s, k, 1]), 2)}
                      for s in range(len(away))],
                   "ref": [{"id": s + 1, "x": round(float(refs[s, k, 0]), 2), "y": round(float(refs[s, k, 1]), 2)}
                           for s in range(len(refs))],
                   "ball": {"x": round(float(ball[k, 0]), 2), "y": round(float(ball[k, 1]), 2),
                            "z": round(float(ball[k, 2]), 2), **({"dead": 1} if dead[k] else {})},
                   "own": None if flying[k] else key(int(own[k]))}
            if k in kicks and key(kicks[k]["from"]):
                p = kicks[k]
                row["kick"] = {"who": key(p["from"]), "to": key(p["to"]), "len": p["len"]}
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    seen_pct["guessed_passes"] = guessed
    seen_pct["out_of_play_pct"] = round(100 * float(dead.mean()), 1)
    seen_pct["loose_ball_pct"] = round(100 * float((own == len(pos) - 1).mean()), 1)
    info = check(folder, rows, sight, ball, pas, own, pos, step, seen_pct)
    (folder / "play.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(f"  wrote {out.relative_to(ROOT)}", flush=True)
    return info


def check(folder: Path, rows: list, sight: np.ndarray, ball: np.ndarray, pas: list, own: np.ndarray,
          pos: np.ndarray, step: float, seen_pct: dict) -> dict:
    """The numbers that say whether it plays like football (and how far it is from what the cameras saw)."""
    n = len(rows)
    jump = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) / step
    k = np.clip(np.round(sight[:, 0] / step).astype(int), 0, n - 1)
    miss = np.linalg.norm(ball[k, :2] - sight[:, 1:], axis=1)
    pos = pos[:-1]                                               # players only (not the loose ball)
    run = np.linalg.norm(np.diff(np.nan_to_num(pos), axis=1), axis=2) / step
    both = ~np.isnan(pos[:, 1:, 0]) & ~np.isnan(pos[:, :-1, 0])
    lens = np.array([p["len"] for p in pas]) if pas else np.zeros(1)
    held = np.array([b - a + 1 for a, b in zip([0] + [p["e"] for p in pas], [p["r"] for p in pas] + [n - 1])]) * step
    minutes = n * step / 60
    return {"rows": n, "ball_drawn_pct": 100.0,
            "ball_speed_ms_p50_p99_max": [round(float(np.percentile(jump, 50)), 1), round(float(np.percentile(jump, 99)), 1),
                                          round(float(jump.max()), 1)],
            "ball_faster_than_35ms_steps": int((jump > 35).sum()),
            "sightings": len(sight),
            "sighting_to_ball_m_median_p90": [round(float(np.median(miss)), 2), round(float(np.percentile(miss, 90)), 2)],
            "sightings_within_3m_pct": round(100 * float((miss <= 3).mean()), 1),
            "passes": len(pas), "passes_per_min": round(len(pas) / minutes, 1),
            "pass_m_median_p90": [round(float(np.median(lens)), 1), round(float(np.percentile(lens, 90)), 1)],
            "lofted": sum(p["lofted"] for p in pas),
            "possession_s_median": round(float(np.median(held)), 2),
            "players_per_row": {"home": int(round(np.median([len([p for p in r["home"] if on_pitch_home(p)]) for r in rows]))) + 1,
                                "away": 11, "ref": 3},
            "fastest_player_ms_p999": round(float(np.percentile(run[both], 99.9)), 1),
            "seen_pct": seen_pct}


def main() -> None:
    from studio.sources import folder_of

    ap = argparse.ArgumentParser(description="Rebuild the game (owners, passes, touches, fixed figures) -> play.jsonl")
    ap.add_argument("--segment", default="alfheim-2013-11-03-H1")
    args = ap.parse_args()
    build(folder_of(args.segment))


if __name__ == "__main__":
    main()
