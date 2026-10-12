"""run.py — MVP 3 Step 6: direct a segment. Cards -> director v2 -> program.mp4 + debug.mp4 + logs.

  venv/bin/python -m studio.run --segment dev                    # MVP 1 detector + newest story run
  venv/bin/python -m studio.run --segment dev --oracle           # annotated players instead of the detector
  venv/bin/python -m studio.run --segment dev --story-run <id>   # a specific story run (or --no-story)

Frame by frame (like MVP 1's engine), everything causal:
  every detection step (5 per s): new detections (or oracle players) -> World (ball, players, action point)
  every frame: picture quality per camera; cards; story events known by now; the director's decision
Writes outputs/studio/<segment>/<run_id>/: program.mp4, debug.mp4, director_log.csv, cards.jsonl,
director_events.jsonl, run.json (settings + summary).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, ffmpeg_exe, load_config
from src.director import shot_lengths
from studio.cameras import OracleAim, cam_video, cams_dir, load_cams
from studio.cards import StoryFeed
from studio.detect import PanoramaDetections, detect_step
from studio.pipeline import Pipeline, inside
from studio.prepare import annotation_frame, load_segment, work_dir
from studio.story_pass import story_run
from studio.video import VideoWriter, to_camera

RED, WHITE, GREEN, ORANGE, YELLOW = (40, 40, 230), (255, 255, 255), (80, 220, 80), (0, 160, 255), (0, 220, 255)


class OraclePlayers:
    """Oracle mode: the annotated players of frame i (panorama pixels + group centre on the pitch)."""

    def __init__(self, meta: dict, radius_m: float):
        self.aim = OracleAim(meta, radius_m)

    def at(self, i: int):
        """(player centres, group centre, player feet) in panorama pixels for frame i."""
        p, frame_no = self.aim.p, annotation_frame(self.aim.meta, i)
        lo, hi = np.searchsorted(p["frame"], [frame_no, frame_no + 1])
        players = [((p["x1"][k] + p["x2"][k]) / 2, (p["y1"][k] + p["y2"][k]) / 2) for k in range(lo, hi)]
        feet = [((p["x1"][k] + p["x2"][k]) / 2, p["y2"][k]) for k in range(lo, hi)]
        return players, self.aim.point(i), feet


def run(segment: str, oracle: bool, story_run_id: str | None, no_story: bool, max_seconds: float | None) -> Path:
    cfg = load_config()["studio"]
    meta = load_segment(segment)
    cams = load_cams(segment)
    fps, total = meta["fps"], meta["frames"]
    if max_seconds:
        total = min(total, round(max_seconds * fps))
    aim = "oracle" if oracle else "detector"
    close_rects = np.load(cams_dir(segment) / f"cam5_{aim}_rects.npy")
    step = detect_step(fps)
    source = "oracle" if oracle else "detector"
    oracle_players = OraclePlayers(meta, cfg["cameras"]["close"]["group_radius_m"]) if oracle else None
    detections = None if oracle else PanoramaDetections(segment)

    story = None if no_story else story_run(segment, story_run_id, None if story_run_id else f"close_{aim}")
    feed = StoryFeed(story["folder"] if story else None, story.get("wall_start", 0.0) if story else 0.0,
                     story["speed"] if story else 1.0)
    goal_looks = []
    if cfg["goalwatch"]["use_in_run"] and not oracle:
        from studio.goalwatch import load_looks
        goal_looks = load_looks(segment)
        feed.add_goal_looks(goal_looks, cfg["goalwatch"])

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{source}" + (f"-{story['mode']}" if story else "-nostory")
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {segment} -> {out.relative_to(ROOT)}  (story: {story['run_id'] if story else 'none'})", flush=True)

    # The per-frame brain, shared with live mode (studio.pipeline).
    pipe = Pipeline(cfg, cams, meta, close_rects, np.load(work_dir(segment) / "pitch_to_image.npy"), feed, source, step)
    caps = [cv2.VideoCapture(str(cam_video(segment, c["cam"], aim))) for c in cams["cams"]]
    w, h = cams["size"]
    program = VideoWriter(out / "program.mp4", w, h, fps)
    debug = VideoWriter(out / "debug.mp4", 1920, 1080, fps)
    log_file = open(out / "director_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "time", "state", "chosen_cam", *[f"score_cam{c['cam']}" for c in cams["cams"]], "reason"])
    cards_file = open(out / "cards.jsonl", "w")
    map_file = open(out / "map.jsonl", "w")
    events_file = open(out / "director_events.jsonl", "w")
    chosen, states = [], []
    start = time.time()
    try:
        for i in range(total):
            t = i / fps
            images = []
            for cap in caps:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"camera video ended early at frame {i}")
                images.append(image)
            events = []
            if i % step == 0:            # new detections (5x per second)
                if oracle:
                    events = pipe.detections(i, t, oracle=oracle_players.at(i))
                else:
                    events = pipe.detections(i, t, detector=detections.at(i))
                map_file.write(json.dumps(pipe.pitch_state.to_dict()) + "\n")
            r = pipe.frame(i, t, images, events)
            for e in r.events:
                events_file.write(json.dumps({"frame": i, "time": round(t, 2), **e}) + "\n")
                if e["type"] == "GOAL_CONFIRMED":
                    print(f"  {t:7.1f}s  GOAL_CONFIRMED: {e['message']}")
            if i % step == 0 and i % round(fps) == 0:
                for card in r.cards:
                    cards_file.write(json.dumps(card.to_dict()) + "\n")
            decision, cards, rects = r.decision, r.cards, r.rects
            chosen.append(decision.cam)
            states.append(decision.state)
            log.writerow([i, f"{t:.3f}", decision.state, decision.cam, *decision.scores, decision.reason])
            for e in decision.events:
                events_file.write(json.dumps({"frame": i, "time": round(t, 2), **e}) + "\n")
                print(f"  {t:7.1f}s  {e['type']}: {e['message']}")

            live = images[decision.cam - 1]
            program.write(_program_frame(live, cams["cams"][decision.cam - 1]))
            debug.write(_debug_frame(images, cards, decision, t, r.story_line, cams, rects, pipe.world.state))
            if i % 500 == 0:
                print(f"[run] frame {i}/{total} ({(i + 1) / (time.time() - start):.0f} fps)", flush=True)
    except BaseException:
        program.abort()
        debug.abort()
        raise
    finally:
        log_file.close()
        cards_file.close()
        map_file.close()
        events_file.close()
        for cap in caps:
            cap.release()
    program.close()
    debug.close()
    _add_audio(segment, out / "program.mp4", meta)

    shots = shot_lengths(chosen, fps)
    summary = {"frames": len(chosen), "seconds": round(len(chosen) / fps, 1), "cuts": len(shots) - 1,
               "shortest_shot_s": round(min(shots), 2), "average_shot_s": round(sum(shots) / len(shots), 2),
               "time_per_cam_s": {f"cam{c['cam']}": round(chosen.count(c["cam"]) / fps, 1) for c in cams["cams"]},
               "time_per_state_s": {s: round(states.count(s) / fps, 1) for s in sorted(set(states))},
               "processing_fps": round(len(chosen) / (time.time() - start), 1),
               "confirmed_goals": sorted(v["goal_time"] for v in pipe.confirmed.values())}
    (out / "run.json").write_text(json.dumps({"segment": segment, "run_id": run_id, "source": source,
                                              "story_run": story["run_id"] if story else None,
                                              "story_mode": story["mode"] if story else None,
                                              "goal_looks": len(goal_looks),
                                              "config": cfg, "summary": summary}, indent=2))
    print(f"[run] done: {summary}")
    return out


def _add_audio(segment: str, video: Path, meta: dict) -> None:
    """Copy the segment's sound (if any) into program.mp4."""
    if not meta["has_audio"]:
        return
    from studio.prepare import work_dir

    tmp = video.with_name("program.audio.mp4")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(work_dir(segment) / "source.mp4"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)], check=True)
    tmp.replace(video)


# ----------------------------------------------------------------------------- pictures
def _label(image, text, origin, scale=0.8, color=WHITE, bg=(0, 0, 0)):
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(2 * scale))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = origin
    cv2.rectangle(image, (x - 5, y - th - 6), (x + tw + 5, y + base + 3), bg, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _program_frame(image: np.ndarray, cam: dict) -> np.ndarray:
    frame = image.copy()
    _label(frame, f"CAM {cam['cam']} - {cam['name']}", (24, 44), 0.9)
    return frame


def _debug_frame(images, cards, decision, t, story_line, cams, rects, world) -> np.ndarray:
    """1920x1080: program (top-left), status panel (top-right), 5 camera tiles with their cards (bottom)."""
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    canvas[0:540, 0:960] = cv2.resize(images[decision.cam - 1], (960, 540), interpolation=cv2.INTER_AREA)
    _label(canvas, f"PROGRAM  CAM {decision.cam}", (16, 36), 0.8, WHITE, RED)
    # Status panel.
    x0, y = 990, 50
    lines = [f"time {int(t // 60):02d}:{t % 60:05.2f}   state {decision.state}",
             f"reason: {decision.reason[:70]}",
             "scores: " + "  ".join(f"c{k + 1} {s:.0f}" for k, s in enumerate(decision.scores)),
             f"play speed {world.play_speed:.2f}   ball {world.ball_status}"]
    if cards and cards[0].event:
        e = cards[0].event
        lines.append(f"story event: {e['type']} at {e['event_time']:.0f}s (known {e['known_at']:.0f}s)")
    if cards and cards[0].phase:
        lines.append(f"story phase: {cards[0].phase}")
    for k in range(0, len(story_line), 70):
        lines.append(("story: " if k == 0 else "       ") + story_line[k:k + 70])
    for line in lines[:12]:
        cv2.putText(canvas, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        y += 38
    # Camera tiles.
    tw, th = 384, 216
    w, h = cams["size"]
    for k, image in enumerate(images):
        tile = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        if world.ball is not None and inside(world.ball, rects[k]):
            bx, by = to_camera(rects[k], w, h, world.ball)
            color = GREEN if world.ball_status == "detected" else ORANGE
            cv2.circle(tile, (int(bx * tw / w), int(by * th / h)), 7, color, 2)
        border = RED if k + 1 == decision.cam else (70, 70, 70)
        cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), border, 5 if k + 1 == decision.cam else 1)
        cam = cams["cams"][k]
        _label(tile, f"{cam['cam']} {cam['name']}", (10, 26), 0.55)
        canvas[600:600 + th, k * tw:(k + 1) * tw] = tile
        if cards:
            c = cards[k]
            q = c.quality
            info = [f"{c.shot_type}  players {c.players}",
                    f"ball {c.ball.status if c.ball else '-'}  action {'yes' if c.action_in_view else 'no'}",
                    f"{'OK' if q.usable else 'BAD'}  blur {q.blur:.2f} shake {q.shake:.2f}",
                    f"score {decision.scores[k]:.0f}"]
            for j, line in enumerate(info):
                color = WHITE if q.usable or j != 2 else RED
                cv2.putText(canvas, line, (k * tw + 10, 600 + th + 32 + j * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--oracle", action="store_true", help="annotated players instead of the MVP 1 detector")
    parser.add_argument("--story-run", default=None, help="story run id (default: newest real-time run)")
    parser.add_argument("--no-story", action="store_true", help="direct without the story layer")
    parser.add_argument("--max-seconds", type=float, default=None, help="only the first N seconds (quick test)")
    args = parser.parse_args()
    run(args.segment, args.oracle, args.story_run, args.no_story, args.max_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
