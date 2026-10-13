"""compare_detectors.py — step 2d of the live test: is the Colab detector the same as the cached one?

  venv/bin/python -m studio.compare_detectors --segment alfheim-2013-11-03-H1 --colab-run <run id> [--cached-run <run id>]

For the detection frames a Colab run really used (detections_live.jsonl, source "colab") it compares, per camera,
with the cached detections of the SAME frames (made on the Mac by src/detect): ball rate, players per frame, and how
often both see the ball within 10 px. With --cached-run (the same stretch played with the cached detector) it also
compares the director's cuts. Prints a small table to paste into PROGRESS.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import ROOT, load_config
from studio.sources import get_source


def run_dir(segment: str, run_id: str):
    return ROOT / load_config()["studio"]["outputs_dir"] / segment / run_id


def compare(segment: str, colab_run: str) -> dict:
    src = get_source(segment)
    rows = [json.loads(x) for x in (run_dir(segment, colab_run) / "detections_live.jsonl").read_text().splitlines() if x]
    rows = [r for r in rows if r["source"] == "colab"]
    out = {"frames": len(rows), "cams": {}}
    for c in src.cam_ids:
        live_ball = cached_ball = both = near = 0
        live_players, cached_players = [], []
        n = 0
        for r in rows:
            cached = src.dets[c]["frames"].get(str(r["frame"]))
            if cached is None:
                continue                          # no cached detection there (the cache doesn't reach so far)
            live = r["cams"].get(str(c), {"players": [], "ball": None})
            n += 1
            live_players.append(len(live["players"]))
            cached_players.append(len(cached["players"]))
            lb, cb = live.get("ball"), cached.get("ball")
            live_ball += lb is not None
            cached_ball += cb is not None
            if lb and cb:
                both += 1
                centre = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])   # noqa: E731
                near += np.linalg.norm(centre(lb["box"]) - centre(cb["box"])) <= 10
        if n:
            out["cams"][c] = {"frames": n, "ball_rate_colab": round(100 * live_ball / n, 1),
                              "ball_rate_cached": round(100 * cached_ball / n, 1),
                              "same_ball_10px": round(100 * near / max(1, both), 1),
                              "players_colab": round(float(np.mean(live_players)), 2),
                              "players_cached": round(float(np.mean(cached_players)), 2)}
    return out


def cuts(segment: str, run_id: str) -> list[tuple[float, int]]:
    path = run_dir(segment, run_id) / "director_events.jsonl"
    return [(e["time"], e["to"]) for e in map(json.loads, path.read_text().splitlines()) if e.get("type") == "CUT"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--colab-run", required=True)
    parser.add_argument("--cached-run", default=None)
    args = parser.parse_args()
    result = compare(args.segment, args.colab_run)
    summary = json.loads((run_dir(args.segment, args.colab_run) / "run.json").read_text())["summary"]
    print(f"Colab run {args.colab_run}: {result['frames']} detection frames from the GPU, "
          f"used: {summary.get('detections')}, latency: {summary.get('latency', {}).get('detector')}")
    print("| cam | frames | ball % Colab | ball % cached | same ball (10 px) | players Colab | players cached |")
    print("|---|---|---|---|---|---|---|")
    for c, r in result["cams"].items():
        print(f"| {c} | {r['frames']} | {r['ball_rate_colab']} | {r['ball_rate_cached']} | {r['same_ball_10px']}% | "
              f"{r['players_colab']} | {r['players_cached']} |")
    if args.cached_run:
        a, b = cuts(args.segment, args.colab_run), cuts(args.segment, args.cached_run)
        matched = sum(any(abs(t - u) <= 1.0 and c == d for u, d in b) for t, c in a)
        print(f"cuts: Colab {len(a)}, cached {len(b)}, same camera within 1 s: {matched}")
        print(f"  Colab:  {a}\n  cached: {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
