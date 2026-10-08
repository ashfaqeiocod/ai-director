"""soccernet.py — turn SoccerNet "Labels-v2.json" (action spotting) into our truth.csv.

SoccerNet games come with labels like {"gameTime": "1 - 12:34", "label": "Goal", "team": "home",
"position": "754000"}. If your clip is a piece of one half, give the half and where the clip starts:

    venv/bin/python -m story.soccernet --labels Labels-v2.json --clip my-clip --half 1 --start 600
        (the clip begins 10:00 into the first half; times in truth.csv become clip seconds)

Team names: --home white --away red  (jersey colours, the same words Gemini uses).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from story.common import clips_dir

LABEL_MAP = {
    "Goal": "goal", "Shots on target": "shot", "Shots off target": "shot", "Corner": "corner", "Foul": "foul",
    "Yellow card": "card_yellow", "Red card": "card_red", "Yellow->red card": "card_red",
    "Substitution": "substitution", "Offside": "offside", "Direct free-kick": "free_kick",
    "Indirect free-kick": "free_kick", "Penalty": "penalty", "Kick-off": "kickoff", "Throw-in": "throw_in",
    "Ball out of play": "out",
}


def convert(labels: dict, half: int, start: float, duration: float | None, home: str, away: str) -> list[dict]:
    rows = []
    for a in labels.get("annotations", []):
        event = LABEL_MAP.get(a.get("label", ""))
        if event is None or not a.get("gameTime", "").startswith(f"{half} "):
            continue
        t = int(a["position"]) / 1000 - start
        if t < 0 or (duration is not None and t > duration):
            continue
        team = {"home": home, "away": away}.get(a.get("team", ""), "")
        rows.append({"time_s": round(t, 2), "event": event, "team": team})
    return sorted(rows, key=lambda r: r["time_s"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Import SoccerNet Labels-v2.json as truth.csv")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--clip", required=True, help="clip folder name under data/clips/")
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the half where the clip starts")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--home", default="home")
    parser.add_argument("--away", default="away")
    args = parser.parse_args()
    rows = convert(json.load(open(args.labels)), args.half, args.start, args.duration, args.home, args.away)
    out = clips_dir() / args.clip / "truth.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "event", "team"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[soccernet] {len(rows)} events -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
