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
from studio.cameras import CLOSE_CAM, OracleAim, cam_rect, cam_video, cams_dir, load_cams
from studio.cards import BallInfo, CamCard, QualityMeter, StoryFeed, World
from studio.detect import PanoramaDetections, detect_step
from studio.director import DirectorV2
from studio.prepare import annotation_frame, load_segment
from studio.story_pass import story_run
from studio.video import VideoWriter, to_camera

RED, WHITE, GREEN, ORANGE, YELLOW = (40, 40, 230), (255, 255, 255), (80, 220, 80), (0, 160, 255), (0, 220, 255)


def inside(point, rect) -> bool:
    x, y, w, h = rect
    return point is not None and x <= point[0] < x + w and y <= point[1] < y + h


class OraclePlayers:
    """Oracle mode: the annotated players of frame i (panorama pixels + group centre on the pitch)."""

    def __init__(self, meta: dict, radius_m: float):
        self.aim = OracleAim(meta, radius_m)

    def at(self, i: int) -> tuple[list[tuple[float, float]], tuple[float, float] | None]:
        p, frame_no = self.aim.p, annotation_frame(self.aim.meta, i)
        lo, hi = np.searchsorted(p["frame"], [frame_no, frame_no + 1])
        players = [((p["x1"][k] + p["x2"][k]) / 2, (p["y1"][k] + p["y2"][k]) / 2) for k in range(lo, hi)]
        return players, self.aim.point(i)


def build_cards(t: float, cams: dict, rects: list, world, quality: list, phase, event, source: str) -> list[CamCard]:
    w, h = cams["size"]
    cards = []
    for k, cam in enumerate(cams["cams"]):
        rect = rects[k]
        ball = None
        if world.ball is not None and inside(world.ball, rect):
            x, y = to_camera(rect, w, h, world.ball)
            ball = BallInfo(round(x / w, 3), round(y / h, 3), world.ball_status)
        players = sum(1 for p in world.players if inside(p, rect))
        cards.append(CamCard(cam=cam["cam"], name=cam["name"], time=round(t, 2), logged_at=round(t, 2),
                             shot_type=cam["shot_type"], ball=ball, players=players,
                             play_speed=round(world.play_speed, 2), quality=quality[k], event=event, phase=phase,
                             source=source, action_in_view=inside(world.action, rect)))
    return cards


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

    story = None if no_story else story_run(segment, story_run_id)
    feed = StoryFeed(story["folder"] if story else None, story.get("wall_start", 0.0) if story else 0.0,
                     story["speed"] if story else 1.0)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{source}" + (f"-{story['mode']}" if story else "-nostory")
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {segment} -> {out.relative_to(ROOT)}  (story: {story['run_id'] if story else 'none'})", flush=True)

    world = World(cfg, (meta["width"], meta["height"]), fps, step)
    meters = [QualityMeter(cfg["quality"]) for _ in cams["cams"]]
    director = DirectorV2(cfg["director"], fps, len(cams["cams"]))
    caps = [cv2.VideoCapture(str(cam_video(segment, c["cam"], aim))) for c in cams["cams"]]
    w, h = cams["size"]
    program = VideoWriter(out / "program.mp4", w, h, fps)
    debug = VideoWriter(out / "debug.mp4", 1920, 1080, fps)
    log_file = open(out / "director_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "time", "state", "chosen_cam", *[f"score_cam{c['cam']}" for c in cams["cams"]], "reason"])
    cards_file = open(out / "cards.jsonl", "w")
    events_file = open(out / "director_events.jsonl", "w")
    chosen, states = [], []
    cards, phase, story_line, latest_event = [], None, "", None
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
            rects = [cam_rect(cams, c["cam"], i, close_rects) for c in cams["cams"]]

            # New detections (5x per second).
            if i % step == 0:
                if oracle:
                    players, group = oracle_players.at(i)
                    world.update(t, None, players, group)
                else:
                    d = detections.at(i)
                    world.update(t, d["ball"][:2] if d["ball"] else None, d["players"])
            # Picture quality: shake every frame, blur on detection steps.
            quality = []
            for k, image in enumerate(images):
                move = (0.0, 0.0)
                if cams["cams"][k]["cam"] == CLOSE_CAM and i > 0:
                    prev = cam_rect(cams, CLOSE_CAM, i - 1, close_rects)
                    scale = w / rects[k][2]
                    move = ((rects[k][0] - prev[0]) * -scale, (rects[k][1] - prev[1]) * -scale)
                quality.append(meters[k].update(t, image, move, measure=i % step == 0))

            # Story known by now.
            if i % step == 0:
                known_events = feed.known_events(t)
                updates = feed.known_updates(t)
                phase = feed.latest_phase(t, cfg["director"]["reaction_max_s"])
                story_line = updates[-1]["narration"] if updates else ""
                headline = [e for e in known_events if e["type"] not in ("none", "whistle")]
                latest_event = max(headline, key=lambda e: e["known_at"]) if headline else None
                cards = build_cards(t, cams, rects, world.state, quality, phase,
                                    _event_summary(latest_event), source)
                if i % round(fps) == 0:
                    for card in cards:
                        cards_file.write(json.dumps(card.to_dict()) + "\n")
            decision = director.decide(i, t, cards, known_events, updates, phase)
            chosen.append(decision.cam)
            states.append(decision.state)
            log.writerow([i, f"{t:.3f}", decision.state, decision.cam, *decision.scores, decision.reason])
            for e in decision.events:
                events_file.write(json.dumps({"frame": i, "time": round(t, 2), **e}) + "\n")
                print(f"  {t:7.1f}s  {e['type']}: {e['message']}")

            live = images[decision.cam - 1]
            program.write(_program_frame(live, cams["cams"][decision.cam - 1]))
            debug.write(_debug_frame(images, cards, decision, t, story_line, cams, rects, world.state))
            if i % 500 == 0:
                print(f"[run] frame {i}/{total} ({(i + 1) / (time.time() - start):.0f} fps)", flush=True)
    except BaseException:
        program.abort()
        debug.abort()
        raise
    finally:
        log_file.close()
        cards_file.close()
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
               "processing_fps": round(len(chosen) / (time.time() - start), 1)}
    (out / "run.json").write_text(json.dumps({"segment": segment, "run_id": run_id, "source": source,
                                              "story_run": story["run_id"] if story else None,
                                              "story_mode": story["mode"] if story else None,
                                              "config": cfg, "summary": summary}, indent=2))
    print(f"[run] done: {summary}")
    return out


def _event_summary(e: dict | None) -> dict | None:
    if not e:
        return None
    return {"type": e["type"], "event_time": e["event_time"], "known_at": round(e["known_at"], 2),
            "confidence": round(e["confidence"], 2), "sources": e["sources"]}


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
