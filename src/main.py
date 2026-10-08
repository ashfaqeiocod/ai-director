"""main.py — Step 8: run the whole pipeline with one command.

    venv/bin/python src/main.py           # skips any step whose output already exists
    venv/bin/python src/main.py --force   # redo every step

Steps: prepare the 3 cameras from the `input:` section of config.yaml (download/cut/sync)
-> detect (cached) -> engine (ball tracking + director, frame by frame)
-> render program.mp4 / debug.mp4 / director_log.csv.
Ends with a short summary.
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src.xxx" imports work
warnings.filterwarnings("ignore")

from src import detect, prepare
from src.common import cam_path, load_config, path, video_info
from src.director import DecisionLog, shot_lengths
from src.download import InputError
from src.engine import REPLAY, Engine
from src.render import Renderer


def direct_and_render(cfg: dict, force: bool) -> dict:
    """Steps 5-7: run the engine over the cached detections and write all outputs."""
    outputs = [path(cfg["render"]["program_path"]), path(cfg["render"]["debug_path"]),
               path(cfg["director"]["log_csv"])]
    if all(o.exists() for o in outputs) and not force:
        print("[render] outputs already exist, skipping (use --force to redo)")
        return summary_from_csv(cfg)

    engine = Engine(cfg, mode=REPLAY)
    renderer = Renderer(cfg, engine.fps)
    log = DecisionLog(path(cfg["director"]["log_csv"]))
    chosen: list[int] = []
    start = time.time()
    for result in engine.frames():
        renderer.render(result)
        log.write(result.index, result.time, result.chosen, result.scores, result.reason)
        chosen.append(result.chosen)
        for event in result.events:
            if event["type"] == "CUT":
                print(f"[director] {result.time:6.2f}s  {event['message']}")
        if result.index % 100 == 0:
            print(f"[render] frame {result.index}/{engine.total_frames}", flush=True)
    log.close()
    renderer.close()
    elapsed = time.time() - start
    print(f"[render] {len(chosen)} frames in {elapsed:.1f}s ({len(chosen) / elapsed:.1f} fps)")
    return {"chosen": chosen, "fps": engine.fps}


def summary_from_csv(cfg: dict) -> dict:
    """Rebuild the chosen-cam list from director_log.csv (used when rendering was skipped)."""
    with open(path(cfg["director"]["log_csv"])) as f:
        rows = list(csv.DictReader(f))
    return {"chosen": [int(r["chosen_cam"]) for r in rows], "fps": video_info(cam_path(cfg, 1))["fps"]}


def main() -> None:
    force = "--force" in sys.argv
    cfg = load_config()
    t0 = time.time()

    try:
        prepare.prepare(force)
    except InputError as error:
        sys.exit(f"\n[prepare] PROBLEM: {error}")
    rates = detect.run(force)
    run = direct_and_render(cfg, force)

    shots = shot_lengths(run["chosen"], run["fps"])
    print("\n========== SUMMARY ==========")
    print("Ball rate:      " + " | ".join(f"{k} {v:.0%}" for k, v in rates.items()))
    print(f"Cuts:           {len(shots) - 1}")
    print(f"Shots:          {', '.join(f'{s:.1f}s' for s in shots)}")
    print(f"Avg shot:       {sum(shots) / len(shots):.2f}s")
    if len(shots) > 1:
        print(f"Shortest shot:  {min(shots[:-1]):.2f}s (last shot ends with the clip, not counted)")
    print(f"Total time:     {time.time() - t0:.1f}s")
    print("Outputs:        outputs/program.mp4, outputs/debug.mp4, outputs/director_log.csv")


if __name__ == "__main__":
    main()
