"""editor.py — Arm MVP Step 8: the OFFLINE editor. It sees the whole match at once and picks the best sequence.

The live director must react: it cuts to tan's face a moment AFTER he looks up. An editor working on the
recording knows what comes next, so it can be on the right shot exactly when it happens. It uses the same shot
values (arm/shots.py) with two differences:
  - events boost their shot from `anticipate_s` BEFORE they happen (the editor knows they are coming);
  - the whole sequence is optimised at once (dynamic programming): the best total value over the match, minus a
    small cost per cut, under the same rules as the live director (minimum shot length per phase, a shot that
    lasts too long loses value, the 180-degree rule, no jump cuts).
How the optimisation works (plain English): for every moment t and every shot k, remember the best program that
ends at t with shot k on air. It is built from a shorter best program ending at s (< t) with another shot j,
plus the value of showing k from s to t. The answer is the best program ending at the last look.
"""
from __future__ import annotations

import numpy as np

from arm.common import settings
from arm.director import CUT, DISSOLVE, MOVE, Decision, allowed, judge_boost, transition
from arm.shots import Shot, all_shots, event_boosts


def value_table(timeline: dict, answers: list[dict] | None = None) -> tuple[list[str], np.ndarray, list[dict[str, Shot]]]:
    """(shot keys, values[look, key], the Shot objects per look). Events are known a bit early (anticipation)."""
    cfg = settings()["editor"]
    setup = timeline["setup"]
    cams = list(timeline["cams"])
    events = [(s["t"] - cfg["anticipate_s"], e) for s in timeline["states"] for e in s["events"]]
    per_look: list[dict[str, Shot]] = []
    for k, state in enumerate(timeline["states"]):
        looks = {c: timeline["cams"][c][k] for c in cams}
        boosts = judge_boost(event_boosts(events, state["t"]), answers or [], state["t"], live=False)
        shots = all_shots(looks, state, setup, boosts)
        per_look.append({s.key: s for s in shots if s.parts["clarity"] >= settings()["director"]["min_clarity"]})
    # The 180-degree line: the editor sees the whole match, so it keeps the side whose two-shot shows more emotion.
    sides = settings()["cameras"]
    totals: dict[str, list[float]] = {}
    for look in per_look:
        for shot in look.values():
            if sides[shot.cam]["side"] in ("north", "south") and shot.kind == "TWO":
                totals.setdefault(sides[shot.cam]["side"], []).append(shot.parts["emotion"])
    if len(totals) == 2:
        line = max(totals, key=lambda side: float(np.mean(totals[side])))
        per_look = [{k: v for k, v in look.items() if sides[v.cam]["side"] in ("neutral", line)} for look in per_look]
    keys = sorted({key for look in per_look for key in look})
    values = np.zeros((len(per_look), len(keys)))
    grace = round(settings()["director"]["lost_grace_s"] * settings()["perceive"]["analysis_fps"])
    for j, key in enumerate(keys):
        gap = grace + 1
        for t, look in enumerate(per_look):
            if key in look:
                values[t, j], gap = look[key].value, 0
                continue
            gap += 1
            if gap <= grace and t > 0:
                # A face turned away for a moment: like the live director, keep the last framing (worth less).
                values[t, j] = cfg["gap_factor"] * values[t - 1, j]
                prev = per_look[t - 1].get(key)
                if prev is not None:
                    look[key] = prev
            else:
                values[t, j] = -cfg["missing_penalty"]       # this shot can't be shown now
    return keys, values, per_look


def plan(timeline: dict, answers: list[dict] | None = None) -> list[Decision]:
    """The best sequence of shots for the whole match. One Decision per look, like the live director."""
    cfg, dcfg = settings()["editor"], settings()["director"]
    keys, values, per_look = value_table(timeline, answers)
    states = timeline["states"]
    T, K = values.shape
    dt = 1.0 / settings()["perceive"]["analysis_fps"]
    prefix = np.vstack([np.zeros(K), np.cumsum(values, axis=0)])
    min_len = [max(1, round(dcfg["min_shot_s"].get(s["phase"], dcfg["min_shot_s"]["default"]) / dt)) for s in states]
    max_phase = [dcfg["max_shot_s"].get(s["phase"], dcfg["max_shot_s"]["default"]) for s in states]
    kinds = [key.split(":")[1] for key in keys]
    # How long each shot stays fresh, per start look and shot (cut-aways are shorter), in looks.
    max_good = np.array([[round(min(m, dcfg["cutaway_max_s"].get(kind, m)) / dt) for kind in kinds] for m in max_phase])
    longest = round(cfg["longest_shot_s"] / dt)
    # A shot that shows a fresh event may be short (like the live director's event cuts).
    event_len = max(1, round(dcfg["event_min_shot_s"] / dt))
    key_index = {key: j for j, key in enumerate(keys)}
    event_start = np.zeros((T, K), dtype=bool)
    for i, look in enumerate(per_look):
        for key, shot in look.items():
            if shot.parts["event"] >= dcfg["event_cut_boost"]:
                event_start[i, key_index[key]] = True

    # Which switches are allowed (180-degree rule, jump cuts), from a representative framing of each shot.
    sample = {}
    for look in per_look:
        for key, shot in look.items():
            sample.setdefault(key, shot)
    ok = np.zeros((K, K), dtype=bool)
    for a, ka in enumerate(keys):
        for b, kb in enumerate(keys):
            ok[a, b] = a != b and allowed(sample[ka], sample[kb], dcfg) is None

    best = np.full((T + 1, K), -np.inf)
    back = np.zeros((T + 1, K, 2), dtype=int)               # (start of the last shot, previous shot)
    for t in range(1, T + 1):
        for length in range(1, min(longest, t) + 1):
            s = t - length
            shortest = min_len[s] if t < T else min(min_len[s], round(1.0 / dt))   # the last shot may end early
            if length < min(shortest, event_len):
                continue
            # Shorter than the phase minimum: only shots that start on an event.
            short_ok = event_start[s] if length < shortest else np.ones(K, dtype=bool)
            good = np.minimum(length, max_good[s])                         # per shot
            fresh = prefix[s + good, np.arange(K)] - prefix[s]
            seg = fresh + cfg["stale_factor"] * (prefix[t] - prefix[s] - fresh)
            if s == 0:
                total = seg
                prev = np.full(K, -1)
            else:
                # Best previous program ending at s with a shot we may switch from.
                options = np.where(ok, best[s][:, None], -np.inf)          # [from, to]
                prev = options.argmax(axis=0)
                total = options.max(axis=0) + seg - cfg["cut_cost"]
            total = np.where(short_ok, total, -np.inf)
            better = total > best[t]
            best[t][better] = total[better]
            back[t][better, 0] = s
            back[t][better, 1] = prev[better]

    # Walk back from the best final shot.
    segments = []
    t, k = T, int(best[T].argmax())
    while t > 0:
        s, j = back[t][k]
        segments.append((s, t, k))
        t, k = s, j
    segments.reverse()

    decisions: list[Decision] = []
    prev_shot = None
    for s, t, k in segments:
        key = keys[k]
        for i in range(s, t):
            shot = per_look[i].get(key) or (decisions[-1].shot if decisions and decisions[-1].shot.key == key else sample[key])
            change, reason = "", ""
            if i == s:
                if prev_shot is not None:
                    by_event = any(e for e in states[i]["events"]) or values[i, k] > 1.3 * values[i, keys.index(prev_shot.key)]
                    change = transition(prev_shot, shot, states[i]["phase"], by_event, dcfg)
                reason = (f"best sequence: {key} for {(t - s) * dt:.1f} s "
                          f"(avg value {values[s:t, k].mean():.0f}, phase {states[s]['phase']})")
            decisions.append(Decision(states[i]["t"], states[i]["n"], shot, change, reason))
        prev_shot = decisions[-1].shot
    return decisions


__all__ = ["plan", "CUT", "DISSOLVE", "MOVE"]
