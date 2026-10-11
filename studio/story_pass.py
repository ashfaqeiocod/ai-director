"""story_pass.py — MVP 3 Step 4: MVP 2's story engine on the wide panorama, at real speed, without a browser.

  venv/bin/python -m studio.story_pass --segment dev --mode gemini_only
  venv/bin/python -m studio.story_pass --segment dev --mode fast_only --speed 4    # quick test, no AI

In MVP 2 the browser's <video> is the clock. Here a simple loop plays that role: it moves the playhead
forward with the wall clock (1x = real time), exactly as the browser would. The story engine is unchanged:
it still reads nothing beyond the playhead, and every event keeps event_time + logged_at.

Why a separate pass: Gemini's delay only means something in real time, while detection on the M1 is slower
than real time. The director later uses an event only from its logged_at, so this stays causal.

Output: the usual MVP 2 run folder outputs/story/studio-<segment>/<run_id>/ (events.jsonl, story.jsonl, ...),
and data/studio/<segment>/story_runs.json lists the runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from src.common import load_config
from story.common import story_config
from story.intake import clip_dir, intake
from studio.prepare import load_segment, work_dir


def deep_merge(base: dict, extra: dict) -> dict:
    """base with extra's values on top (nested dicts merged), as a new dict."""
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def story_clip(segment: str) -> Path:
    """The segment's panorama as an MVP 2 clip folder (data/clips/studio-<segment>/), made once."""
    name = f"studio-{segment}"
    folder = clip_dir(name)
    if not (folder / "meta.json").exists():
        print(f"[story] making MVP 2 clip {name} from the panorama ...", flush=True)
        intake(str(work_dir(segment) / "source.mp4"), name=name)
    return folder


def run(segment: str, mode: str, speed: float = 1.0, specialist_url: str = "") -> str:
    from story.engine import StoryEngine          # imports the AI libraries

    meta = load_segment(segment)
    folder = story_clip(segment)
    cfg = deep_merge(story_config(), load_config()["studio"]["story"]["overrides"])
    events: list[dict] = []

    def publish(message: dict) -> None:
        # Print the interesting things as they happen, like the dashboard's log.
        kind = message.get("type")
        if kind == "story":
            print(f"  {message['logged_at']:7.1f}s  [{message.get('source')}] {message['phase']}: {message['narration']}")
        elif kind == "event" and message.get("new"):
            events.append(message)
            print(f"  {message['logged_at']:7.1f}s  EVENT {message['event_type']} at {message['event_time']:.1f}s "
                  f"({message['confidence']:.2f}, {'+'.join(message['sources'])})")
        elif kind == "log" and message.get("kind") in ("ERROR", "LAG", "SCORE"):
            print(f"  {message['t']:7.1f}s  {message['kind']}: {message['message']}")

    engine = StoryEngine(folder, cfg, mode, publish, send_audio=meta["has_audio"], specialist_url=specialist_url)
    engine.start()
    duration = engine.reader.duration
    print(f"[story] {segment}: {duration:.0f} s at {speed}x, mode {mode}, run {engine.run_id}", flush=True)
    start = time.time()
    try:
        while True:
            t = min(duration, (time.time() - start) * speed)
            engine.on_tick(t, paused=False, ended=t >= duration)
            if t >= duration:
                break
            time.sleep(0.2)                      # the browser sends ~5 ticks per second
        # Let the last AI look finish (it is stamped with the final playhead, honestly late).
        deadline = time.time() + 40
        while (engine.story_busy or engine.memory_busy) and time.time() < deadline:
            time.sleep(0.2)
    finally:
        engine.stop("ended")
    summary = engine.summary()
    print(f"[story] done: {summary['events']} events, headline {summary['events_by_type']}, "
          f"gemini looks {summary['story_gemini']}, template {summary['story_template']}, "
          f"avg delay {summary['avg_llm_delay']} s, errors {summary['llm_errors']}")
    record_run(segment, engine.run_id, engine.clip, mode, speed, summary, wall_start=start)
    return engine.run_id


def record_run(segment: str, run_id: str, clip: str, mode: str, speed: float, summary: dict, wall_start: float) -> None:
    path = work_dir(segment) / "story_runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    # wall_start = wall-clock time at playhead 0: turns log lines' wall_time into playhead time.
    runs.append({"run_id": run_id, "clip": clip, "mode": mode, "speed": speed, "wall_start": wall_start,
                 "summary": summary})
    path.write_text(json.dumps(runs, indent=2))


def story_run(segment: str, run_id: str | None) -> dict:
    """A recorded story run (the newest real-time one if run_id is None), with its MVP 2 run folder."""
    from story.run_log import run_folder

    path = work_dir(segment) / "story_runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    if run_id is not None:
        runs = [r for r in runs if r["run_id"] == run_id]
    else:
        runs = [r for r in runs if r["speed"] == 1.0]
    if not runs:
        raise SystemExit(f"no {'story run ' + run_id if run_id else 'real-time story run'} for '{segment}': run  "
                         f"venv/bin/python -m studio.story_pass --segment {segment} --mode gemini_only")
    return {**runs[-1], "folder": run_folder(f"studio-{segment}", runs[-1]["run_id"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--mode", default="gemini_only", choices=["fast_only", "specialist_only", "gemini_only", "hybrid"])
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time (needed for honest AI delays)")
    parser.add_argument("--specialist-url", default="", help="Colab URL for hybrid / specialist_only")
    args = parser.parse_args()
    run(args.segment, args.mode, args.speed, args.specialist_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
