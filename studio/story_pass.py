"""story_pass.py — MVP 3 Step 4: MVP 2's story engine on the wide panorama, at real speed, without a browser.

  venv/bin/python -m studio.story_pass --segment dev --mode gemini_only                       # watches the close-up
  venv/bin/python -m studio.story_pass --segment dev --mode gemini_only --feed wide           # the whole panorama
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
import subprocess
import time
from pathlib import Path

from src.common import ffmpeg_exe, load_config
from story.common import story_config
from story.intake import clip_dir, intake
from studio.prepare import load_segment, work_dir


def deep_merge(base: dict, extra: dict) -> dict:
    """base with extra's values on top (nested dicts merged), as a new dict."""
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


FEEDS = ("close_detector", "close_oracle", "wide")


def without_missing_sound(cfg: dict, has_sound: bool) -> dict:
    """A video with NO sound track: leave sound out of the intensity mix.

    Intensity is a weighted mix (story.fusion.intensity_weights: audio 0.4, motion, specialist, llm).
    With no sound track the audio part is stuck at 0 but still counted, so intensity could never pass 60
    and quiet looked the same as silent. Weight 0 makes the fusion share it among the other layers, the same
    way it already does for a layer that isn't running. Videos WITH sound (even quiet ones) are unchanged.
    """
    if has_sound:
        return cfg
    return deep_merge(cfg, {"fusion": {"intensity_weights": {"audio": 0.0}}})


def story_clip(segment: str, feed: str) -> Path:
    """The video the story layer watches, as an MVP 2 clip folder (data/clips/studio-<segment>-<feed>/), made once.

    wide            the whole panorama: the action is tiny (Gemini missed the dev goal on it)
    close_detector  the virtual close-up aimed by the MVP 1 detector (honest)
    close_oracle    the close-up aimed by the annotated players (oracle, labelled)
    The camera videos have no sound, so the match sound from the panorama is added.
    """
    name = f"studio-{segment}-{feed}"
    folder = clip_dir(name)
    if (folder / "meta.json").exists():
        return folder
    from studio.cameras import CLOSE_CAM, cam_video

    source = work_dir(segment) / "source.mp4"
    print(f"[story] making MVP 2 clip {name} ...", flush=True)
    if feed == "wide":
        intake(str(source), name=name)
        return folder
    video = cam_video(segment, CLOSE_CAM, feed.split("_")[1])
    if not video.exists():
        raise SystemExit(f"{video.name} is missing: render the close-up first (studio.cameras --only 5 --aim ...)")
    with_sound = work_dir(segment) / f"story_{feed}.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(source),
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_sound)], check=True)
    try:
        intake(str(with_sound), name=name)
    finally:
        with_sound.unlink(missing_ok=True)
    return folder


def play(folder: Path, cfg: dict, mode: str, send_audio: bool, speed: float = 1.0, specialist_url: str = ""):
    """Play one MVP 2 clip folder through the StoryEngine at `speed`, headless. Returns (engine, summary, start)."""
    from story.engine import StoryEngine          # imports the AI libraries

    def publish(message: dict) -> None:
        kind = message.get("type")
        if kind == "story":
            print(f"  {message['logged_at']:7.1f}s  [{message.get('source')}] {message['phase']}: {message['narration']}")
        elif kind == "event" and message.get("new"):
            print(f"  {message['logged_at']:7.1f}s  EVENT {message['event_type']} at {message['event_time']:.1f}s "
                  f"({message['confidence']:.2f}, {'+'.join(message['sources'])})")
        elif kind == "log" and message.get("kind") in ("ERROR", "LAG", "SCORE"):
            print(f"  {message['t']:7.1f}s  {message['kind']}: {message['message']}")

    has_sound = json.loads((folder / "meta.json").read_text()).get("has_audio", True)
    cfg = without_missing_sound(cfg, has_sound)
    engine = StoryEngine(folder, cfg, mode, publish, send_audio=send_audio, specialist_url=specialist_url)
    engine.start()
    duration = engine.reader.duration
    print(f"[story] {folder.name}: {duration:.0f} s at {speed}x, mode {mode}, sound to AI: {send_audio}, "
          f"run {engine.run_id}", flush=True)
    start = time.time()
    try:
        while True:
            t = min(duration, (time.time() - start) * speed)
            engine.on_tick(t, paused=False, ended=t >= duration)
            if t >= duration:
                break
            time.sleep(0.2)
        deadline = time.time() + 40
        while (engine.story_busy or engine.memory_busy) and time.time() < deadline:
            time.sleep(0.2)
    finally:
        engine.stop("ended")
    summary = engine.summary()
    print(f"[story] done: {summary['events']} events, headline {summary['events_by_type']}, "
          f"gemini looks {summary['story_gemini']}, template {summary['story_template']}, "
          f"avg delay {summary['avg_llm_delay']} s, errors {summary['llm_errors']}")
    return engine, summary, start


def run(segment: str, mode: str, speed: float = 1.0, specialist_url: str = "", feed: str = "close_detector") -> str:
    from story.engine import StoryEngine          # imports the AI libraries

    meta = load_segment(segment)
    folder = story_clip(segment, feed)
    feed_cfg = load_config()["studio"]["story"]["feeds"]["wide" if feed == "wide" else "close"]
    cfg = deep_merge(story_config(), feed_cfg)
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

    cfg = without_missing_sound(cfg, meta["has_audio"])
    engine = StoryEngine(folder, cfg, mode, publish, send_audio=meta["has_audio"], specialist_url=specialist_url)
    engine.start()
    duration = engine.reader.duration
    print(f"[story] {segment}: {duration:.0f} s at {speed}x, mode {mode}, feed {feed}, run {engine.run_id}", flush=True)
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
    # A run that fell behind the video (machine overloaded) or lost many AI answers is not a fair
    # measurement: it is kept on disk but marked invalid, and story_run() skips it.
    # Also invalid: the wall clock ran much longer than the clip (the Mac slept: the playhead jumped ahead
    # and the AI never looked), or a Gemini mode that got no Gemini answer at all.
    slept = summary["wall_seconds"] > 1.2 * duration / speed + 60
    no_ai = mode in ("gemini_only", "hybrid") and summary["story_gemini"] == 0
    valid = (summary["lag_skips"] == 0 and not slept and not no_ai
             and summary["llm_errors"] <= load_config()["studio"]["story"]["max_llm_errors"])
    if not valid:
        print(f"[story] run marked INVALID (lag skips {summary['lag_skips']}, AI errors {summary['llm_errors']}, "
              f"slept {slept}, no AI {no_ai}): run it again with nothing heavy alongside and the Mac plugged in")
    record_run(segment, engine.run_id, engine.clip, mode, speed, summary, wall_start=start, feed=feed, valid=valid)
    return engine.run_id


def record_run(segment: str, run_id: str, clip: str, mode: str, speed: float, summary: dict, wall_start: float,
               feed: str, valid: bool = True) -> None:
    path = work_dir(segment) / "story_runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    # wall_start = wall-clock time at playhead 0: turns log lines' wall_time into playhead time.
    runs.append({"run_id": run_id, "clip": clip, "feed": feed, "mode": mode, "speed": speed, "valid": valid,
                 "wall_start": wall_start,
                 "summary": summary})
    path.write_text(json.dumps(runs, indent=2))


def story_run(segment: str, run_id: str | None, feed: str | None = None) -> dict:
    """A recorded story run (the newest real-time one on `feed` if run_id is None), with its MVP 2 run folder."""
    from story.run_log import run_folder

    path = work_dir(segment) / "story_runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    if run_id is not None:
        runs = [r for r in runs if r["run_id"] == run_id]
    else:
        runs = [r for r in runs if r["speed"] == 1.0 and r.get("valid", True)
                and (feed is None or r.get("feed", "wide") == feed)]
    if not runs:
        raise SystemExit(f"no {'story run ' + run_id if run_id else 'real-time story run'} for '{segment}': run  "
                         f"venv/bin/python -m studio.story_pass --segment {segment} --mode gemini_only")
    return {**runs[-1], "folder": run_folder(runs[-1]["clip"], runs[-1]["run_id"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", help="a studio segment (dev / eval)")
    parser.add_argument("--clip", help="instead of a segment: any MVP 2 clip in data/clips/ (e.g. a SoccerNet broadcast)")
    parser.add_argument("--no-audio", action="store_true", help="--clip only: the AI gets no sound (no commentary)")
    parser.add_argument("--mode", default="gemini_only", choices=["fast_only", "specialist_only", "gemini_only", "hybrid"])
    parser.add_argument("--speed", type=float, default=1.0, help="1 = real time (needed for honest AI delays)")
    parser.add_argument("--specialist-url", default="", help="Colab URL for hybrid / specialist_only")
    parser.add_argument("--feed", default="close_detector", choices=FEEDS, help="which video the story layer watches")
    args = parser.parse_args()
    if args.clip:
        # Broadcast clips: plain MVP 2 settings. Prove-it mode: --no-audio (no commentary, no crowd).
        play(clip_dir(args.clip), story_config(), args.mode, not args.no_audio, args.speed, args.specialist_url)
    elif args.segment:
        run(args.segment, args.mode, args.speed, args.specialist_url, args.feed)
    else:
        parser.error("give --segment or --clip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
