"""evaluate.py — PROOF MODE: compare a run with the ground truth and write report.md.

    venv/bin/python -m story.evaluate --clip eng-cze --run 20261008-223801-hybrid
    venv/bin/python -m story.evaluate --clip eng-cze --run latest
    venv/bin/python -m story.evaluate --clip eng-cze --compare      # table across all evaluated runs

Ground truth: data/clips/<clip>/truth.csv with columns time_s,event,team (or import SoccerNet labels
with story/soccernet.py). Event names are our EventType names (goal, shot, save, corner, foul...).

The report (outputs/story/<clip>/<run_id>/report.md + report.json) has:
  - per event type: precision / recall, matching within eval.match_window_s seconds
    (a found event matches a truth event of the same type at most match_window_s AFTER or BEFORE it;
    each truth event can be matched once);
  - delay: logged_at - truth time for matched events (average, max), and Gemini's delay per update;
  - narration accuracy from your 👍 / 👎 ratings;
  - intensity: is there an intensity peak right after each truth event?
  - the definition-of-done targets with PASS / FAIL.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from story.common import clips_dir, outputs_dir, story_config

TARGETS = {"goal_recall": 0.9, "headline_recall": 0.6, "avg_delay_s": 10.0, "narration_up": 0.8}


# ------------------------------------------------------------------ loading
def load_truth(clip: str) -> list[dict]:
    file = clips_dir() / clip / "truth.csv"
    if not file.exists():
        raise SystemExit(f"No ground truth: {file} (columns time_s,event,team). See TESTING_MVP2.md.")
    with open(file) as f:
        rows = [{"time_s": float(r["time_s"]), "event": r["event"].strip().lower(),
                 "team": (r.get("team") or "").strip().lower() or None}
                for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    return sorted(rows, key=lambda r: r["time_s"])


def read_jsonl(file: Path) -> list[dict]:
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def final_events(run: Path) -> list[dict]:
    """events.jsonl has a line every time a fused event is created or changes: keep the last one per id."""
    latest: dict[str, dict] = {}
    for item in read_jsonl(run / "events.jsonl"):
        if "id" in item:
            latest[item["id"]] = item
    # Goals the scoreboard never confirmed were dropped by fusion: they don't count as found.
    return sorted((e for e in latest.values() if not e.get("rejected")), key=lambda e: e["event_time"])


def resolve_run(clip: str, run_id: str) -> Path:
    base = outputs_dir() / clip
    if run_id == "latest":
        runs = sorted((p for p in base.glob("*") if (p / "run_meta.json").exists()), key=lambda p: p.name)
        if not runs:
            raise SystemExit(f"No runs in {base}")
        return runs[-1]
    run = base / run_id
    if not (run / "run_meta.json").exists():
        raise SystemExit(f"Run not found: {run}")
    return run


# ------------------------------------------------------------------ matching
def match(truth: list[dict], found: list[dict], window: float) -> list[tuple[dict, dict | None]]:
    """Greedy one-to-one matching of truth events to found events of the same type (closest first)."""
    pairs, used = [], set()
    candidates = sorted(((abs(f["event_time"] - t["time_s"]), ti, fi)
                         for ti, t in enumerate(truth) for fi, f in enumerate(found)
                         if f["type"] == t["event"] and abs(f["event_time"] - t["time_s"]) <= window),
                        key=lambda c: c[0])
    matched_truth: dict[int, int] = {}
    for _, ti, fi in candidates:
        if ti in matched_truth or fi in used:
            continue
        matched_truth[ti] = fi
        used.add(fi)
    for ti, t in enumerate(truth):
        pairs.append((t, found[matched_truth[ti]] if ti in matched_truth else None))
    return pairs


def truth_status(truth: list[dict], found: list[dict], window: float, until: float | None = None) -> list[dict]:
    """For the dashboard's proof-mode markers: matched / missed / pending, plus false alarms."""
    pairs = match(truth, found, window)
    used = {id(f) for _, f in pairs if f is not None}
    items = []
    for t, f in pairs:
        status = "matched" if f else ("pending" if until is not None and t["time_s"] + window > until else "missed")
        items.append({**t, "status": status})
    truth_types = {t["event"] for t in truth}
    for f in found:
        if id(f) not in used and f["type"] in truth_types:
            items.append({"time_s": f["event_time"], "event": f["type"], "team": f.get("team"), "status": "false_alarm"})
    return sorted(items, key=lambda i: i["time_s"])


def intensity_peaks(run: Path, truth: list[dict], window: float) -> dict:
    rows = list(csv.DictReader(open(run / "intensity.csv"))) if (run / "intensity.csv").exists() else []
    if not rows:
        return {}
    t = [float(r["t"]) for r in rows]
    v = [float(r["intensity"]) for r in rows]
    mean = sum(v) / len(v)
    threshold = sorted(v)[int(0.8 * (len(v) - 1))]          # top 20% of the run
    hits, at_events = 0, []
    for e in truth:
        after = [x for tt, x in zip(t, v) if e["time_s"] - 1 <= tt <= e["time_s"] + window]
        if after:
            at_events.append(max(after))
            hits += max(after) >= threshold
    return {"mean_intensity": round(mean, 1), "top20_threshold": threshold,
            "events_with_peak": hits, "events_checked": len(at_events),
            "mean_peak_after_events": round(sum(at_events) / len(at_events), 1) if at_events else None}


# ------------------------------------------------------------------ report
def evaluate(clip: str, run: Path) -> dict:
    window = story_config()["eval"]["match_window_s"]
    meta = json.loads((run / "run_meta.json").read_text())
    played_to = (meta.get("summary") or {}).get("played_to") or meta.get("clip_duration") or 1e9
    truth = [t for t in load_truth(clip) if t["time_s"] <= played_to]
    events = final_events(run)
    pairs = match(truth, events, window)
    found_types = {e["type"] for e in events}
    types = sorted({t["event"] for t in truth} | (found_types & {t["event"] for t in truth}))

    per_type = {}
    for event_type in types:
        t_list = [t for t in truth if t["event"] == event_type]
        f_list = [f for f in events if f["type"] == event_type]
        matched = [(t, f) for t, f in pairs if t["event"] == event_type and f is not None]
        per_type[event_type] = {
            "truth": len(t_list), "found": len(f_list), "matched": len(matched),
            "recall": round(len(matched) / len(t_list), 3) if t_list else None,
            "precision": round(len(matched) / len(f_list), 3) if f_list else None,
        }

    delays = [f["logged_at"] - t["time_s"] for t, f in pairs if f is not None]
    stories = read_jsonl(run / "story.jsonl")
    llm_delays = [s["logged_at"] - s["window_end"] for s in stories if s.get("source") == "gemini"]
    ratings = read_jsonl(run / "ratings.jsonl")
    latest_rating = {r["story_id"]: r["rating"] for r in ratings}    # the last click counts
    ups = sum(1 for r in latest_rating.values() if r > 0)

    goals = per_type.get("goal", {})
    others = [v for k, v in per_type.items() if k != "goal"]
    other_truth = sum(v["truth"] for v in others)
    other_matched = sum(v["matched"] for v in others)
    result = {
        "clip": clip, "run_id": run.name, "mode": meta.get("mode"), "model": meta.get("model"),
        "send_audio_to_llm": meta.get("send_audio_to_llm"), "played_to": played_to, "match_window_s": window,
        "per_type": per_type,
        "goal_recall": goals.get("recall"),
        "headline_recall": round(other_matched / other_truth, 3) if other_truth else None,
        "avg_delay_s": round(sum(delays) / len(delays), 2) if delays else None,
        "max_delay_s": round(max(delays), 2) if delays else None,
        "avg_llm_delay_s": round(sum(llm_delays) / len(llm_delays), 2) if llm_delays else None,
        "max_llm_delay_s": round(max(llm_delays), 2) if llm_delays else None,
        "narration_rated": len(latest_rating), "narration_up": round(ups / len(latest_rating), 3) if latest_rating else None,
        "score_final": (meta.get("summary") or {}).get("score"),
        "scoreboard_final": (meta.get("summary") or {}).get("scoreboard"),
        "goals_counted": sum(1 for e in events if e["type"] == "goal"),
        "ignored_in_replay": (meta.get("summary") or {}).get("ignored_in_replay"),
        "intensity": intensity_peaks(run, truth, window),
        "truth_status": truth_status(truth, events, window),
        "story_updates": {"gemini": sum(s.get("source") == "gemini" for s in stories),
                          "template": sum(s.get("source") == "template" for s in stories)},
    }
    result["targets"] = {
        "goals caught ≥ 90%": _check(result["goal_recall"], TARGETS["goal_recall"], higher=True),
        "other headline events ≥ 60%": _check(result["headline_recall"], TARGETS["headline_recall"], higher=True),
        "average delay ≤ 10 s": _check(result["avg_delay_s"], TARGETS["avg_delay_s"], higher=False),
        "narration 👍 ≥ 80%": _check(result["narration_up"], TARGETS["narration_up"], higher=True),
    }
    return result


def _check(value, target, higher: bool) -> str:
    if value is None:
        return "n/a"
    return "PASS" if (value >= target if higher else value <= target) else "FAIL"


def _pct(v) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def write_report(run: Path, r: dict) -> Path:
    lines = [f"# Evaluation — {r['clip']} / {r['run_id']}", "",
             f"Mode **{r['mode']}** · model `{r['model'] or '–'}` · audio to AI: {r['send_audio_to_llm']} · "
             f"played to {r['played_to']:.0f} s · match window ±{r['match_window_s']} s", "",
             "## Targets (definition of done)", "", "| Target | Value | Result |", "|---|---|---|",
             f"| Goals caught ≥ 90% | {_pct(r['goal_recall'])} | {r['targets']['goals caught ≥ 90%']} |",
             f"| Other headline events ≥ 60% | {_pct(r['headline_recall'])} | {r['targets']['other headline events ≥ 60%']} |",
             f"| Average delay ≤ 10 s | {r['avg_delay_s'] if r['avg_delay_s'] is not None else '–'} s | "
             f"{r['targets']['average delay ≤ 10 s']} |",
             f"| Narration 👍 ≥ 80% | {_pct(r['narration_up'])} ({r['narration_rated']} rated) | "
             f"{r['targets']['narration 👍 ≥ 80%']} |", "",
             "## Per event type", "", "| Event | Truth | Found | Matched | Recall | Precision |", "|---|---|---|---|---|---|"]
    for event_type, v in r["per_type"].items():
        lines.append(f"| {event_type} | {v['truth']} | {v['found']} | {v['matched']} | {_pct(v['recall'])} | {_pct(v['precision'])} |")
    lines += ["", "## Delay", "",
              f"- Event delay (logged_at − true time, matched events): average **{r['avg_delay_s']} s**, max {r['max_delay_s']} s",
              f"- Gemini delay per update (logged_at − window end): average {r['avg_llm_delay_s']} s, max {r['max_llm_delay_s']} s",
              f"- Story updates: {r['story_updates']['gemini']} from Gemini, {r['story_updates']['template']} from templates", "",
              "## Score and replays", "",
              f"- Final score graphic (as read): {r.get('scoreboard_final') or '–'} · goals counted per colour: "
              f"{r['score_final'] or '–'} · goal events found: {r['goals_counted']} · "
              f"events ignored during replays: {r['ignored_in_replay']}", "",
              "## Intensity vs truth", ""]
    i = r["intensity"]
    if i:
        lines.append(f"- {i['events_with_peak']} of {i['events_checked']} truth events have intensity in the run's top 20% "
                     f"(≥ {i['top20_threshold']:.0f}) within {r['match_window_s']} s; average peak after an event "
                     f"{i['mean_peak_after_events']} vs run average {i['mean_intensity']}.")
    lines += ["", "## Each truth event", "", "| Time | Event | Team | Result |", "|---|---|---|---|"]
    icon = {"matched": "✅ matched", "missed": "❌ missed", "false_alarm": "⚠️ false alarm", "pending": "▫️"}
    for item in r["truth_status"]:
        lines.append(f"| {item['time_s']:.1f} s | {item['event']} | {item.get('team') or '–'} | {icon[item['status']]} |")
    out = run / "report.md"
    out.write_text("\n".join(lines) + "\n")
    (run / "report.json").write_text(json.dumps(r, indent=2))
    return out


def compare(clip: str) -> Path:
    """compare.md: one row per evaluated run of this clip."""
    rows = []
    for report in sorted((outputs_dir() / clip).glob("*/report.json")):
        r = json.loads(report.read_text())
        rows.append(f"| {r['run_id']} | {r['mode']} | {r['send_audio_to_llm']} | {_pct(r['goal_recall'])} | "
                    f"{_pct(r['headline_recall'])} | {r['avg_delay_s'] if r['avg_delay_s'] is not None else '–'} | "
                    f"{_pct(r['narration_up'])} | {r['goals_counted']} | {r['score_final'] or '–'} |")
    text = "\n".join([f"# Mode comparison — {clip}", "",
                      "| Run | Mode | Audio to AI | Goals caught | Other headline | Avg delay (s) | Narration 👍 | Goal events | Final score |",
                      "|---|---|---|---|---|---|---|---|---|", *rows]) + "\n"
    out = outputs_dir() / clip / "compare.md"
    out.write_text(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Proof mode: compare a story run with the ground truth.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", help="run id (folder name under outputs/story/<clip>/) or 'latest'")
    parser.add_argument("--compare", action="store_true", help="write compare.md across all evaluated runs")
    args = parser.parse_args()
    if args.run:
        run = resolve_run(args.clip, args.run)
        result = evaluate(args.clip, run)
        out = write_report(run, result)
        print(out.read_text())
        print(f"[evaluate] wrote {out}")
    if args.compare or not args.run:
        print(compare(args.clip).read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
