"""main.py — Arm MVP: the whole pipeline in one command.

  venv/bin/python -m arm.main              # every step; a step whose output already exists is skipped
  venv/bin/python -m arm.main --no-render  # decisions + report only (fast: ~10 s once the caches exist)
  venv/bin/python -m arm.main --force      # redo the slow, cached steps too (perception, judge)

Steps: sync (arm.sync) -> perception (arm.perceive) -> camera setup (arm.setup, Gemini x5, cached) ->
match timeline (arm.match) -> Gemini judge (arm.judge, ~16 stills, cached) -> live director -> offline editor ->
baselines -> render (one pass for all programs) -> report.
Outputs in data/arm/out/:
  program_live.mp4 / program_edit.mp4   the two AI broadcasts (with graphics)
  vote_*.mp4                            clean 720p versions of the AI programs and the baselines (blind A/B page)
  director_log_live.csv / director_log_edit.csv   every cut with its reason
  report.md + metrics.json              the comparison;  run.json   everything the dashboard page draws
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time

from arm.common import cache_dir, cam_letters, out_dir, settings, sync_file
from arm.director import run_live
from arm.editor import plan
from arm.evaluate import baselines, evaluate
from arm.judge import load_judge


def step(name: str, module: str, done: bool, force: bool) -> None:
    """Run `python -m module` unless its output exists (it prints its own summary)."""
    if done and not force:
        print(f"[{name}] cached")
        return
    print(f"[{name}] running python -m {module}")
    subprocess.run([sys.executable, "-m", module, *(["--force"] if force and module == "arm.perceive" else [])],
                   check=True)


def write_log(decisions: list, timeline: dict, path) -> None:
    """One row per shot change: time, camera, shot, transition, value, reason (+ the match at that moment)."""
    states = timeline["states"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "camera", "shot", "transition", "value", "phase", "tension", "reason"])
        for k, d in enumerate(decisions):
            if k == 0 or d.shot.key != decisions[k - 1].shot.key:
                s = states[min(k, len(states) - 1)]
                w.writerow([f"{d.t:.1f}", d.shot.cam, d.shot.kind, d.change or "START", f"{d.shot.value:.0f}",
                            s["phase"], f"{s['tension']:.2f}", d.reason])


def segments(decisions: list) -> list[dict]:
    """The program as a list of shots for the dashboard: start, end, camera, kind, transition, reason."""
    out = []
    for k, d in enumerate(decisions):
        if k == 0 or d.shot.key != decisions[k - 1].shot.key:
            out.append({"start": d.t, "end": d.t, "cam": d.shot.cam, "kind": d.shot.kind, "change": d.change or "START",
                        "value": d.shot.value, "reason": d.reason})
        out[-1]["end"] = round(d.t + 0.1, 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-render", action="store_true", help="skip the videos (decisions + report only)")
    parser.add_argument("--force", action="store_true", help="redo perception and the Gemini judge too")
    args = parser.parse_args()
    started = time.time()
    cams = cam_letters()
    step("sync", "arm.sync", sync_file().exists(), False)
    step("perception", "arm.perceive", all((cache_dir() / f"cam{c}.json").exists() for c in cams), args.force)
    step("camera setup", "arm.setup", (cache_dir() / "setup.json").exists(), False)
    print("[match timeline] building")
    from arm.match import build_timeline
    timeline = build_timeline()
    step("judge", "arm.judge", (cache_dir() / "judge.json").exists(), args.force)
    judge = load_judge()

    print("[directors] live + editor")
    live = run_live(timeline, judge)
    edit = plan(timeline, judge)
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    write_log(live, timeline, out / "director_log_live.csv")
    write_log(edit, timeline, out / "director_log_edit.csv")
    base = baselines(timeline["states"])
    programs = {"AI live": live, "AI editor": edit, **base}
    result = evaluate(programs, timeline, judge)

    vote = {"AI live": "vote_live.mp4", "AI editor": "vote_edit.mp4", "random 3s": "vote_random.mp4",
            "motion (MVP 1 style)": "vote_motion.mp4", "one phone A": "vote_phone_A.mp4", "one phone E": "vote_phone_E.mp4"}
    run = {"duration": timeline["states"][-1]["t"] + 0.1, "cams": cams,
           "labels": {c: settings()["cameras"][c]["label"] for c in cams},
           "views": {c: timeline["setup"][c]["view"] for c in cams},
           "programs": {"live": {"file": "program_live.mp4", "shots": segments(live)},
                        "edit": {"file": "program_edit.mp4", "shots": segments(edit)}},
           "states": [{k: s[k] for k in ("t", "phase", "tension", "lean", "upright", "strain", "joy", "crowd", "events", "reason")}
                      for s in timeline["states"]],
           "judge": judge, "metrics": result, "vote_programs": [{"key": k, "file": f} for k, f in vote.items()]}
    (out / "run.json").write_text(json.dumps(run))

    if not args.no_render:
        from arm.render import Program, render_many
        print("[render] all programs in one pass over the 5 cameras")
        small = (1280, 720)
        jobs = [Program(live, out / "program_live.mp4", True, "AI LIVE"),
                Program(edit, out / "program_edit.mp4", True, "AI EDIT")]
        for name, file in vote.items():
            jobs.append(Program(programs[name], out / file, graphics=False, size=small,
                                push_in=name.startswith("AI")))
        render_many(jobs, timeline)

    print(f"\n{result['source']}")
    print(f"{'program':<22} {'shots':>5} {'avg s':>6} {'min s':>6} {'<1s':>4} {'180':>4} {'jump':>4} {'bad%':>5} "
          f"{'faces%':>6} {'crowd%':>6} {'kinds':>5} {'moments':>8}")
    for name, r in result["results"].items():
        m = f"{r['moments_caught']}/{r['moments']}" if r["moments"] else "-"
        print(f"{name:<22} {r['shots']:>5} {r['avg_shot_s']:>6} {r['min_shot_s']:>6} {r['cuts_under_1s']:>4} "
              f"{r['breaks_180']:>4} {r['jump_cuts']:>4} {r['bad_air_pct']:>5} {r['faces_pct']:>6} {r['crowd_pct']:>6} "
              f"{len(r['content_kinds']):>5} {m:>8}")
    print(f"\nDone in {time.time() - started:.0f} s. Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
