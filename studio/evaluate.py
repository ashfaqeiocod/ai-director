"""evaluate.py — MVP 3 Step 8: score a directed run against the match's ground truth.

  venv/bin/python -m studio.evaluate --segment eval                 # newest run of the segment
  venv/bin/python -m studio.evaluate --segment eval --run <run_id>
  venv/bin/python -m studio.evaluate --segment eval --compare       # table of all runs -> compare.md

Ground truth is used HERE ONLY (never by the director): the annotated ball actions (bas) and the
ball track (rebuilt from those actions, so approximate between them).

Writes outputs/studio/<segment>/<run_id>/report.md with:
  ball on air      % of time the ground-truth ball was inside the live camera's view (ball on the pitch)
  shots            shortest / average shot, cuts per minute (shortest must be >= min shot, except bad-feed escapes)
  reaction         for each real goal: did the close-up come, and how many seconds after the goal
  story            goals and shots found by the story layer vs the truth (precision / recall / delay)
  bad feed         % of the degraded interval the degraded camera was on air (target 0%)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.common import ROOT, load_config
from studio.cameras import CLOSE_CAM, cam_rect, cams_dir, load_cams
from studio.prepare import annotation_frame, gt_ball, load_segment, project, segment_time, work_dir
from studio.soccertrack import load_events

PITCH_HALF_X, PITCH_HALF_Y = 52.5, 34.0


def runs_dir(segment: str) -> Path:
    return ROOT / load_config()["studio"]["outputs_dir"] / segment


def resolve(segment: str, run_id: str | None) -> Path:
    runs = sorted(p for p in runs_dir(segment).glob("*") if (p / "run.json").exists())
    if run_id:
        runs = [p for p in runs if p.name == run_id]
    if not runs:
        raise SystemExit(f"no finished run for '{segment}'" + (f" with id {run_id}" if run_id else ""))
    return runs[-1]


def read_log(run: Path) -> tuple[list[int], list[str], list[str]]:
    cams, states, reasons = [], [], []
    with open(run / "director_log.csv") as f:
        for row in csv.DictReader(f):
            cams.append(int(row["chosen_cam"]))
            states.append(row["state"])
            reasons.append(row["reason"])
    return cams, states, reasons


def truth_events(meta: dict) -> list[dict]:
    """Ground-truth actions inside the segment, with time in segment seconds."""
    duration = meta["frames"] / meta["fps"]
    events = [{**e, "t": segment_time(meta, e["frame"])} for e in load_events(meta["match"], meta["half"])]
    return [e for e in events if 0 <= e["t"] < duration]


def ball_on_air(meta: dict, run_info: dict, cams_chosen: list[int]) -> dict:
    cams = load_cams(meta["segment"])
    aim = "oracle" if run_info["source"] == "oracle" else "detector"
    close_rects = np.load(cams_dir(meta["segment"]) / f"cam5_{aim}_rects.npy")
    matrix = np.load(work_dir(meta["segment"]) / "pitch_to_image.npy")
    ball = gt_ball(meta)
    rows = {int(f): k for k, f in enumerate(ball["frame"])}
    hits = total = 0
    per_cam = {c["cam"]: [0, 0] for c in cams["cams"]}
    wide = {c["cam"] for c in cams["cams"] if c["shot_type"] == "wide"}
    for i, cam in enumerate(cams_chosen):
        k = rows.get(annotation_frame(meta, i))
        if k is None or not np.isfinite(ball["x"][k]):
            continue
        bx, by = float(ball["x"][k]), float(ball["y"][k])
        if abs(bx) > PITCH_HALF_X or abs(by) > PITCH_HALF_Y:
            continue                                  # ball off the pitch: nothing to show
        px, py = project(matrix, np.array([[bx, by]]))[0]
        x, y, w, h = cam_rect(cams, cam, i, close_rects)
        seen = x <= px < x + w and y <= py < y + h
        hits += seen
        total += 1
        per_cam[cam][0] += seen
        per_cam[cam][1] += 1
    # WIDE shows the whole pitch, so it always "has the ball": staying on WIDE would score 100%.
    # The honest number is ball on air on the OTHER cameras, next to the share of time spent on WIDE.
    not_wide = [v for c, v in per_cam.items() if c not in wide]
    hits_nw, total_nw = sum(v[0] for v in not_wide), sum(v[1] for v in not_wide)
    return {"ball_on_air": hits / max(1, total), "frames_scored": total,
            "ball_on_air_not_wide": hits_nw / total_nw if total_nw else None,
            "wide_share": 1 - total_nw / max(1, total),
            "per_cam": {f"cam{c}": (v[0] / v[1] if v[1] else None) for c, v in per_cam.items()}}


def shot_stats(cams_chosen: list[int], reasons: list[str], fps: float, min_shot: float) -> dict:
    shots, start = [], 0
    for i in range(1, len(cams_chosen) + 1):
        if i == len(cams_chosen) or cams_chosen[i] != cams_chosen[i - 1]:
            escape = i < len(cams_chosen) and "unusable" in reasons[i]
            shots.append({"len": (i - start) / fps, "escape": escape, "last": i == len(cams_chosen)})
            start = i
    complete = [s for s in shots if not s["last"]]
    normal = [s["len"] for s in complete if not s["escape"]]
    return {"cuts": len(shots) - 1, "average_shot_s": float(np.mean([s["len"] for s in shots])),
            "shortest_shot_s": min(normal) if normal else None,
            "too_short": sum(1 for s in normal if s < min_shot - 1e-6),
            "bad_feed_escapes": sum(1 for s in complete if s["escape"]),
            "cuts_per_min": (len(shots) - 1) / (len(cams_chosen) / fps / 60)}


def false_reactions(truth: list[dict], states: list[str], fps: float, window: float) -> int:
    """REACTIONs that started with no real goal in the `window` seconds before them."""
    goals = [e["t"] for e in truth if e["label"] == "GOAL"]
    starts = [i / fps for i in range(len(states)) if states[i] == "REACTION" and (i == 0 or states[i - 1] != "REACTION")]
    return sum(1 for s in starts if not any(0 <= s - g <= window for g in goals))


def reactions(truth: list[dict], cams_chosen: list[int], states: list[str], fps: float, window: float) -> list[dict]:
    out = []
    for g in (e for e in truth if e["label"] == "GOAL"):
        start = round(g["t"] * fps)
        end = min(len(cams_chosen), start + round(window * fps))
        hit = next((i for i in range(start, end) if cams_chosen[i] == CLOSE_CAM and states[i] == "REACTION"), None)
        out.append({"goal_t": round(g["t"], 1), "team": g["team"], "reacted": hit is not None,
                    "delay_s": round((hit - start) / fps, 1) if hit is not None else None})
    return out


def story_scores(truth: list[dict], run_info: dict, window: float) -> dict | None:
    if not run_info.get("story_run"):
        return None
    from studio.story_pass import story_run

    folder = story_run(run_info["segment"], run_info["story_run"])["folder"]
    latest: dict[str, dict] = {}
    for line in (folder / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        latest[e["id"]] = e
    found = [e for e in latest.values() if not e.get("rejected")]
    # The goal watcher's goals (studio.goalwatch), merged like the director sees them.
    if run_info.get("goal_looks"):
        from studio.cards import StoryFeed
        from studio.goalwatch import load_looks

        feed = StoryFeed(None)
        feed.add_goal_looks(load_looks(run_info["segment"], run_info.get("goalwatch_run")),
                            load_config()["studio"]["goalwatch"])
        merged: dict[str, dict] = {}
        for e in feed.events:
            merged.setdefault(e["id"], e)
        found += list(merged.values())
    result = {}
    for ours, theirs in (("goal", "GOAL"), ("shot", "SHOT")):
        f = [e for e in found if e["type"] == ours]
        t = [e for e in truth if e["label"] == theirs]
        if ours == "shot":   # a goal is also a shot in the truth's eyes
            f += [e for e in found if e["type"] == "goal"]
            t += [e for e in truth if e["label"] == "GOAL"]
        matched_truth = [x for x in t if any(abs(e["event_time"] - x["t"]) <= window for e in f)]
        matched_found = [e for e in f if any(abs(e["event_time"] - x["t"]) <= window for x in t)]
        delays = [e["logged_at"] - e["event_time"] for e in matched_found]
        result[ours] = {"truth": len(t), "found": len(f), "recall": len(matched_truth) / len(t) if t else None,
                        "precision": len(matched_found) / len(f) if f else None,
                        "avg_delay_s": float(np.mean(delays)) if delays else None}
    return result


def confirmed_goals(truth: list[dict], found: list[float], window: float) -> dict:
    """Goals the system COUNTED (goal camera + kick-off line-up confirmation) vs the real goals."""
    real = [e["t"] for e in truth if e["label"] == "GOAL"]
    hit = [r for r in real if any(abs(f - r) <= window for f in found)]
    false = [f for f in found if not any(abs(f - r) <= window for r in real)]
    return {"real": len(real), "counted": len(found), "correct": len(hit), "false": len(false)}


def bad_feed(meta: dict, cams_chosen: list[int]) -> dict | None:
    test = load_config()["studio"]["segments"][meta["segment"]].get("degrade")
    if not test:
        return None
    fps = meta["fps"]
    frames = range(round(test["start_s"] * fps), min(len(cams_chosen), round(test["end_s"] * fps)))
    on_air = sum(1 for i in frames if cams_chosen[i] == test["cam"])
    # The quality meter needs a moment to notice: count the first second separately.
    late = sum(1 for i in frames if cams_chosen[i] == test["cam"] and i >= frames.start + round(fps))
    return {"cam": test["cam"], "seconds": len(frames) / fps, "on_air_share": on_air / max(1, len(frames)),
            "on_air_after_1s": late / max(1, len(frames))}


def evaluate(segment: str, run_id: str | None) -> Path:
    run = resolve(segment, run_id)
    info = json.loads((run / "run.json").read_text())
    meta = load_segment(segment)
    cfg = load_config()["studio"]
    cams_chosen, states, reasons = read_log(run)
    fps = meta["fps"]
    truth = [e for e in truth_events(meta) if e["t"] < len(cams_chosen) / fps]
    r = {"run": run.name, "source": info["source"], "story_mode": info.get("story_mode"),
         "ball": ball_on_air(meta, info, cams_chosen),
         "shots": shot_stats(cams_chosen, reasons, fps, cfg["director"]["min_shot_seconds"]),
         "reactions": reactions(truth, cams_chosen, states, fps, cfg["eval"]["reaction_window_s"]),
         "false_reactions": false_reactions(truth, states, fps, cfg["eval"]["reaction_window_s"]),
         "story": story_scores(truth, info, cfg["eval"]["match_window_s"]),
         "bad_feed": bad_feed(meta, cams_chosen), "summary": info["summary"],
         "confirmed_goals": confirmed_goals(truth, info["summary"].get("confirmed_goals", []),
                                            cfg["eval"]["match_window_s"])}
    (run / "report.json").write_text(json.dumps(r, indent=2))
    path = write_report(run, r, cfg)
    print(path.read_text())
    return path


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def write_report(run: Path, r: dict, cfg: dict) -> Path:
    s, b, story = r["shots"], r["ball"], r["story"]
    lines = [f"# MVP 3 report — {run.parent.name} / {r['run']}", "",
             f"Source: **{r['source']}**{' (oracle = annotated players, not the detector)' if r['source'] == 'oracle' else ''}"
             f" · story: **{r['story_mode'] or 'none'}** · {r['summary']['seconds']} s directed", "",
             "| Check | Result | Target |", "|---|---|---|",
             f"| Ball on air, all cameras (ground-truth ball inside the live camera) | {_pct(b['ball_on_air'])} "
             f"({b['frames_scored']} frames) | (WIDE always counts) |",
             f"| **Ball on air, excluding WIDE time** | **{_pct(b['ball_on_air_not_wide'])}** | ≥ 80% oracle / ≥ 60% detector |",
             f"| Time on WIDE | {_pct(b['wide_share'])} | (lower = more real directing) |",
             f"| Shortest shot (normal cuts) | {s['shortest_shot_s']:.2f} s ({s['too_short']} too short) "
             f"| ≥ {cfg['director']['min_shot_seconds']} s |" if s["shortest_shot_s"] is not None else
             "| Shortest shot | – (no complete shot) | |",
             f"| Cuts / average shot | {s['cuts']} cuts ({s['cuts_per_min']:.1f}/min), avg {s['average_shot_s']:.1f} s | |",
             f"| Bad-feed escapes (allowed to break min shot) | {s['bad_feed_escapes']} | |"]
    if r["bad_feed"]:
        f = r["bad_feed"]
        lines.append(f"| Degraded cam{f['cam']} on air during its bad {f['seconds']:.0f} s | "
                     f"{_pct(f['on_air_share'])} (after the first second: {_pct(f['on_air_after_1s'])}) | 0% |")
    reacted = sum(x["reacted"] for x in r["reactions"])
    lines.append(f"| Close-up within {cfg['eval']['reaction_window_s']} s after a real goal | "
                 f"{reacted}/{len(r['reactions'])} | ≥ 2/3 |")
    lines.append(f"| False reactions (no real goal before them) | {r['false_reactions']} | low |")
    cg = r.get("confirmed_goals")
    if cg:
        lines.append(f"| **Goals counted** (goal camera + kick-off confirmation) | **{cg['correct']} of {cg['real']} real, "
                     f"{cg['false']} false** | ≥ 2 of 3, 0 false |")
    if story:
        for k, v in story.items():
            lines.append(f"| Story {k}s: recall / precision / delay | {_pct(v['recall'])} / {_pct(v['precision'])} / "
                         f"{'–' if v['avg_delay_s'] is None else f'{v['avg_delay_s']:.1f} s'} "
                         f"({v['found']} found, {v['truth']} true) | |")
    lines += ["", "## Goals", ""]
    for x in r["reactions"]:
        lines.append(f"- goal at {x['goal_t']} s ({x['team']}): "
                     + (f"close-up after {x['delay_s']} s" if x["reacted"] else "no close-up reaction"))
    lines += ["", "## Ball on air per camera (when that camera was live)", ""]
    lines += [f"- {k}: {_pct(v)}" for k, v in b["per_cam"].items()]
    lines += ["", "## Time per camera / state", "", f"- {r['summary']['time_per_cam_s']}",
              f"- {r['summary']['time_per_state_s']}", "",
              "Ground truth: SoccerTrack v2 (CC BY 4.0). Its ball track is rebuilt from annotated actions "
              "(straight lines between them), so 'ball on air' is approximate."]
    path = run / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def baseline(segment: str, name: str, cam: int) -> dict:
    """A director that never cuts (always `cam`): what any real director must beat."""
    meta = load_segment(segment)
    chosen = [cam] * meta["frames"]
    return {"run": f"baseline: always {name}", "source": "–", "story_mode": None,
            "ball": ball_on_air(meta, {"source": "oracle"}, chosen), "shots": {"cuts": 0, "shortest_shot_s": None},
            "reactions": reactions(truth_events(meta), chosen, ["IN_PLAY"] * len(chosen), meta["fps"], 1),
            "false_reactions": 0, "bad_feed": bad_feed(meta, chosen), "story": None}


def compare(segment: str) -> Path:
    rows = ["| Run | Source | Story | Ball on air (not WIDE) | Time on WIDE | Cuts | Shortest | Goals reacted | "
            "False reactions | Bad cam on air | Story goals R/P |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    reports = [baseline(segment, "WIDE", 4), baseline(segment, "CENTRE", 2)]
    reports += [json.loads(p.read_text()) for p in sorted(runs_dir(segment).glob("*/report.json"))]
    for r in reports:
        g = r["story"]["goal"] if r["story"] else None
        b = r["ball"]
        rows.append(f"| {r['run']} | {r['source']} | {r['story_mode'] or '–'} | {_pct(b.get('ball_on_air_not_wide'))} | "
                    f"{_pct(b.get('wide_share'))} | {r['shots']['cuts']} | "
                    f"{(r['shots']['shortest_shot_s'] or 0):.1f} s | "
                    f"{sum(x['reacted'] for x in r['reactions'])}/{len(r['reactions'])} | {r.get('false_reactions', '–')} | "
                    f"{_pct(r['bad_feed']['on_air_share']) if r['bad_feed'] else '–'} | "
                    f"{(_pct(g['recall']) + ' / ' + _pct(g['precision'])) if g else '–'} |")
    path = runs_dir(segment) / "compare.md"
    path.write_text(f"# MVP 3 — runs on segment '{segment}'\n\nBaselines never cut: every real run must beat them on "
                    "'ball on air (not WIDE)' while reacting to goals.\n\n" + "\n".join(rows) + "\n")
    print(path.read_text())
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    compare(args.segment) if args.compare else evaluate(args.segment, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
