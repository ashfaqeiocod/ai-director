"""live.py — MVP 3 live mode: everything runs while the match plays, at real speed.

  venv/bin/python -m studio.live --segment dev --seconds 60                      # director only, headless
  venv/bin/python -m studio.live --segment eval --story gemini_only --goalwatch  # + AI story + goal watcher

A wall clock decides which frame is "now" (like a real broadcast). Each loop:
  1. read the newest frame of every on-air camera (frames we are too slow for are skipped: that's live);
  2. 5x per second: new detections (cached by default: the M1 detects ~1 frame/s, too slow for live);
  3. the shared Pipeline (studio.pipeline): pitch map, cards, goal confirmation, the director's choice;
  4. the AI layers run beside it, at real speed:
       story      MVP 2's StoryEngine on the close-up, with this clock as its playhead (Gemini, + Colab if hybrid)
       goalwatch  Gemini checks a goal camera when the map says play is near a goal (studio.goalwatch)
     Their answers count only from the moment they ARRIVE (honest delay), through a LiveFeed.
  5. publish a state message (~10/s) for the dashboard, JPEG pictures for its <img> streams, and write the
     same files as an offline run, so the live run can be replayed afterwards.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, load_config
from src.director import shot_lengths
from studio.cameras import cam_video, cams_dir, load_cams
from studio.detect import PanoramaDetections, detect_step
from studio.pipeline import Pipeline
from studio.prepare import load_segment, work_dir
from studio.video import VideoWriter

Publish = Callable[[dict], None]


class LiveFeed:
    """The story as it ARRIVES during a live run (same reading interface as studio.cards.StoryFeed).

    known_at = the live clock when the answer arrived, so the director only uses what it really knew."""

    def __init__(self, goal_cfg: dict):
        self.lock = threading.Lock()
        self.story_events: dict[str, dict] = {}     # fused story events by id (newest version)
        self.updates: list[dict] = []
        self.looks: list[dict] = []                  # goal watcher answers
        self.goal_cfg = goal_cfg

    def add_story_event(self, e: dict, now: float) -> None:
        with self.lock:
            old = self.story_events.get(e["id"])
            self.story_events[e["id"]] = {"id": e["id"], "type": e.get("event_type", e.get("type")),
                                          "event_time": e["event_time"], "logged_at": e["logged_at"],
                                          "known_at": old["known_at"] if old else max(now, e["logged_at"]),
                                          "confidence": e["confidence"], "sources": e["sources"],
                                          "rejected": e.get("rejected", False)}

    def add_update(self, u: dict) -> None:
        with self.lock:
            self.updates.append(u)

    def add_look(self, look: dict) -> None:
        with self.lock:
            self.looks.append(look)

    def _goal_items(self) -> tuple[list[dict], list[dict]]:
        from studio.cards import StoryFeed

        f = StoryFeed(None)
        f.add_goal_looks(list(self.looks), self.goal_cfg)
        return f.events, f.updates

    def known_events(self, t: float) -> list[dict]:
        with self.lock:
            story = [e for e in self.story_events.values() if e["known_at"] <= t and not e["rejected"]]
            goal_events, _ = self._goal_items()
        latest: dict[str, dict] = {}
        for e in goal_events:
            if e["known_at"] <= t:
                latest[e["id"]] = e
        return story + list(latest.values())

    def known_updates(self, t: float) -> list[dict]:
        with self.lock:
            _, goal_updates = self._goal_items()
            out = [u for u in self.updates + goal_updates if u["logged_at"] <= t]
        return sorted(out, key=lambda u: u["logged_at"])

    def latest_phase(self, t: float, max_age_s: float) -> str | None:
        known = self.known_updates(t)
        if known and t - known[-1]["logged_at"] <= max_age_s:
            return known[-1]["phase"]
        return None


class LiveStudio:
    def __init__(self, segment: str, publish: Publish | None = None, story: str = "off", specialist_url: str = "",
                 goalwatch: bool = False, seconds: float | None = None, speed: float = 1.0, record: bool = True):
        self.cfg = load_config()["studio"]
        self.segment, self.publish = segment, publish or (lambda m: None)
        self.story_mode, self.specialist_url, self.use_goalwatch = story, specialist_url, goalwatch
        self.speed, self.record = speed, record
        self.meta = load_segment(segment)
        self.cams = load_cams(segment)
        self.fps = self.meta["fps"]
        self.total = self.meta["frames"] if not seconds else min(self.meta["frames"], round(seconds * self.fps))
        self.step = detect_step(self.fps)
        self.on_air = self.cfg["director"]["on_air_cams"]
        self.feed = LiveFeed(self.cfg["goalwatch"])
        close_rects = np.load(cams_dir(segment) / "cam5_detector_rects.npy")
        self.pipe = Pipeline(self.cfg, self.cams, self.meta, close_rects,
                             np.load(work_dir(segment) / "pitch_to_image.npy"), self.feed, "detector", self.step)
        self.detections = PanoramaDetections(segment)
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpegs: dict[str, tuple[int, bytes]] = {}      # stream name -> (sequence number, JPEG)
        self.jpeg_lock = threading.Lock()
        self.story_engine = None
        self.watcher = None
        self.status = {"state": "idle", "t": 0.0, "lag_s": 0.0, "shown_fps": 0.0, "skipped": 0}
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-live"

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True, name="studio-live")
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        if self.thread and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=30)

    def jpeg(self, name: str) -> tuple[int, bytes] | None:
        with self.jpeg_lock:
            return self.jpegs.get(name)

    def _put_jpeg(self, name: str, image: np.ndarray, width: int, quality: int) -> None:
        h, w = image.shape[:2]
        if w > width:
            image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.jpeg_lock:
                seq = self.jpegs.get(name, (0, b""))[0] + 1
                self.jpegs[name] = (seq, buf.tobytes())

    # ------------------------------------------------------------------ AI layers
    def _start_ai(self, clock: Callable[[], float]) -> None:
        if self.story_mode != "off":
            from story.common import story_config
            from story.engine import StoryEngine
            from studio.story_pass import deep_merge, story_clip, without_missing_sound

            folder = story_clip(self.segment, "close_detector")
            cfg = without_missing_sound(deep_merge(story_config(), self.cfg["story"]["feeds"]["close"]),
                                        self.meta["has_audio"])

            def on_story(message: dict) -> None:
                kind = message.get("type")
                if kind == "event":
                    self.feed.add_story_event(message, clock())
                elif kind == "story":
                    self.feed.add_update({k: message.get(k) for k in ("window_start", "window_end", "logged_at", "phase",
                                                                      "sentiment", "narration", "intensity", "source",
                                                                      "events")})
                self.publish({"type": "story_msg", **{k: v for k, v in message.items() if k != "type"},
                              "kind": kind})

            self.story_engine = StoryEngine(folder, cfg, self.story_mode, on_story, send_audio=self.meta["has_audio"],
                                            specialist_url=self.specialist_url)
            self.story_engine.start()
        if self.use_goalwatch:
            from studio.goalwatch import LiveGoalWatcher

            llm = self.story_engine.llm if self.story_engine and self.story_engine.llm else None
            self.watcher = LiveGoalWatcher(self.segment, self.cams, clock, self.feed.add_look, self.publish,
                                           shared_llm=llm, out_dir=work_dir(self.segment) / "goalwatch" / self.run_id)

    def _stop_ai(self) -> None:
        if self.watcher:
            self.watcher.stop()
        if self.story_engine:
            self.story_engine.stop("live run ended")

    # ------------------------------------------------------------------ the loop
    def _run(self) -> None:
        try:
            self._loop()
        except Exception as error:      # noqa: BLE001 — report to the page, don't die silently
            import traceback

            traceback.print_exc()
            self.status["state"] = "error"
            self.publish({"type": "live_error", "message": f"{type(error).__name__}: {error}"})

    def _loop(self) -> None:
        cams, fps, step = self.cams, self.fps, self.step
        cam_ids = [c["cam"] for c in cams["cams"]]
        caps = {c: cv2.VideoCapture(str(cam_video(self.segment, c, "detector"))) for c in cam_ids if c in self.on_air}
        blank = np.zeros((36, 64, 3), np.uint8)      # cameras that are never aired are not decoded (CPU budget)
        out = ROOT / self.cfg["outputs_dir"] / self.segment / self.run_id
        files = self._open_files(out) if self.record else None
        w, h = cams["size"]
        program = VideoWriter(out / "program.mp4", w, h, fps) if self.record else None
        chosen, states, frames_shown, skipped, lags = [], [], 0, 0, []
        start = time.time()
        clock = lambda: min((time.time() - start) * self.speed, self.total / fps)   # noqa: E731
        self._start_ai(clock)
        self.status["state"] = "live"
        self.publish({"type": "live_started", "segment": self.segment, "run_id": self.run_id,
                      "duration": self.total / fps, "story": self.story_mode, "goalwatch": self.use_goalwatch})
        i, last_det, last_pub, last_tiles = -1, -1, 0.0, 0.0
        images = {}
        decision = None
        try:
            while not self.stop_flag.is_set():
                now_t = clock()
                target = min(int(now_t * fps), self.total - 1)
                if target <= i:
                    if i >= self.total - 1:
                        break
                    time.sleep(0.003)
                    continue
                # Skip the frames we are too late for (grab = advance without converting the picture).
                while i < target - 1:
                    for cap in caps.values():
                        cap.grab()
                    i += 1
                    skipped += 1
                    if chosen:                     # a skipped frame still shows the live camera
                        chosen.append(chosen[-1])
                        states.append(states[-1])
                for c, cap in caps.items():
                    ok, img = cap.read()
                    images[c] = img if ok else images.get(c, blank)
                i += 1
                t = i / fps
                lags.append(now_t - t)
                frames_shown += 1
                if self.story_engine:
                    self.story_engine.on_tick(now_t, paused=False, ended=i >= self.total - 1)
                # Detections: the newest 5 Hz step at or before this frame (causal).
                events = []
                det_frame = i - i % step
                if det_frame != last_det:
                    last_det = det_frame
                    events = self.pipe.detections(det_frame, det_frame / fps, detector=self.detections.at(det_frame))
                    if self.watcher:
                        self.watcher.consider(t, self.pipe.pitch_state)
                    if files:
                        files["map"].write(json.dumps(self.pipe.pitch_state.to_dict()) + "\n")
                frame_images = [images.get(c, blank) for c in cam_ids]
                r = self.pipe.frame(i, t, frame_images, events)
                decision = r.decision
                chosen.append(decision.cam)
                states.append(decision.state)
                all_events = r.events + decision.events
                for e in all_events:
                    # "kind" = CUT / REACTION / BOX_ENTRY / GOAL_CONFIRMED ... ("type" is the message type)
                    self.publish({"type": "decision", "time": round(t, 2), "kind": e["type"],
                                  **{k: v for k, v in e.items() if k != "type"}})
                if files:
                    files["log"].writerow([i, f"{t:.3f}", decision.state, decision.cam, *decision.scores, decision.reason])
                    for e in all_events:
                        files["events"].write(json.dumps({"frame": i, "time": round(t, 2), **e}) + "\n")
                    if det_frame == i and i % round(fps) == 0:
                        for card in r.cards:
                            files["cards"].write(json.dumps(card.to_dict()) + "\n")
                live_img = images.get(decision.cam, blank)
                if program:
                    program.write(live_img if live_img.shape[1] == w else cv2.resize(live_img, (w, h)))
                wall = time.time()
                self._put_jpeg("program", live_img, 960, 70)
                if wall - last_tiles >= 1 / self.cfg["live"]["tile_fps"]:
                    last_tiles = wall
                    for c, img in images.items():
                        self._put_jpeg(f"cam{c}", img, 480, 65)
                if wall - last_pub >= 1 / self.cfg["live"]["state_hz"]:
                    last_pub = wall
                    self.status.update(t=round(t, 2), lag_s=round(float(np.mean(lags[-25:])), 2),
                                       shown_fps=round(frames_shown / max(1e-6, wall - start), 1), skipped=skipped)
                    self.publish(self._state(t, r))
        finally:
            for cap in caps.values():
                cap.release()
            self._stop_ai()
            if program:
                program.close() if chosen else program.abort()
            summary = self._finish(out, files, chosen, states, frames_shown, skipped, lags, start)
            self.status["state"] = "ended"
            self.publish({"type": "live_ended", "run_id": self.run_id, "summary": summary})

    def _open_files(self, out: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        log_file = open(out / "director_log.csv", "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["frame", "time", "state", "chosen_cam", *[f"score_cam{c['cam']}" for c in self.cams["cams"]],
                      "reason"])
        return {"log_file": log_file, "log": log, "cards": open(out / "cards.jsonl", "w"),
                "map": open(out / "map.jsonl", "w"), "events": open(out / "director_events.jsonl", "w")}

    def _finish(self, out: Path, files: dict | None, chosen: list, states: list, shown: int, skipped: int,
                lags: list, start: float) -> dict:
        fps = self.fps
        shots = shot_lengths(chosen, fps) if chosen else [0.0]
        summary = {"frames": len(chosen), "seconds": round(len(chosen) / fps, 1), "cuts": max(0, len(shots) - 1),
                   "shortest_shot_s": round(min(shots), 2), "average_shot_s": round(sum(shots) / len(shots), 2),
                   "time_per_state_s": {s: round(states.count(s) / fps, 1) for s in sorted(set(states))},
                   "frames_shown": shown, "frames_skipped": skipped,
                   "avg_lag_s": round(float(np.mean(lags)), 2) if lags else 0.0,
                   "max_lag_s": round(float(np.max(lags)), 2) if lags else 0.0,
                   "wall_seconds": round(time.time() - start, 1),
                   "confirmed_goals": sorted(v["goal_time"] for v in self.pipe.confirmed.values()),
                   # A live run that fell far behind is not a fair measurement (like an invalid story pass).
                   "valid": (not lags) or float(np.max(lags)) <= self.cfg["live"]["max_lag_s"]}
        if files:
            for k in ("log_file", "cards", "map", "events"):
                files[k].close()
            story = None
            if self.story_engine:
                story = {"run_id": self.story_engine.run_id, "mode": self.story_mode}
            (out / "run.json").write_text(json.dumps({
                "segment": self.segment, "run_id": self.run_id, "source": "detector", "live": True,
                "story_run": story["run_id"] if story else None, "story_mode": story["mode"] if story else None,
                "goal_looks": len(self.feed.looks), "goalwatch_run": self.run_id if self.watcher else None,
                "config": self.cfg, "summary": summary}, indent=2))
            if self.story_engine:
                from studio.story_pass import record_run

                record_run(self.segment, self.story_engine.run_id, self.story_engine.clip, self.story_mode, 1.0,
                           self.story_engine.summary(), wall_start=start, feed="close_detector",
                           valid=summary["valid"])
            if self.watcher:
                self.watcher.save_run()
        return summary

    def _state(self, t: float, r) -> dict:
        """One state message for the dashboard (the same shape replay builds in the page)."""
        ps = self.pipe.pitch_state
        d = r.decision
        cards = {str(c.cam): {"players": c.players, "ball": c.ball.status if c.ball else None,
                              "usable": c.quality.usable} for c in r.cards if c.cam in self.on_air}
        i = round(t * self.fps)
        return {"type": "state", "t": round(t, 2), "cam": d.cam, "state": d.state, "reason": d.reason,
                "overlays": self._overlays(i, r.rects),
                "cards": cards, "map": ps.to_dict() if ps else None, "story_line": r.story_line,
                "lag_s": self.status["lag_s"], "shown_fps": self.status["shown_fps"],
                "skipped": self.status["skipped"], "confirmed": [v["goal_time"] for v in self.pipe.confirmed.values()],
                "asking": self.watcher.asking if self.watcher else None, "frame": i}

    def _overlays(self, i: int, rects: list) -> dict:
        """What the detector saw, per on-air camera, in 0..1 of the camera picture (boxes, dots, ball)."""
        from studio.cameras import CLOSE_CAM
        from studio.detect import DETECT_CAMS
        from studio.video import to_camera

        w, h = self.cams["size"]
        frame = i - i % self.step
        here = self.detections.at(frame)
        ball = self.pipe.world.state.ball                     # the map's ball (seen or remembered)
        out = {}
        for k, cam in enumerate(self.cams["cams"]):
            c = cam["cam"]
            if c not in self.on_air:
                continue
            item: dict = {}
            if c in DETECT_CAMS:
                det = self.detections.dets[c]["frames"].get(frame, {})
                item["players"] = [[round(v / s, 3) for v, s in zip(p["box"], (w, h, w, h))] for p in det.get("players", [])]
            elif c == CLOSE_CAM:
                pts = [to_camera(rects[k], w, h, p) for p in here["players"]]
                item["dots"] = [[round(x / w, 3), round(y / h, 3)] for x, y in pts if 0 <= x < w and 0 <= y < h]
            if ball is not None:
                bx, by = to_camera(rects[k], w, h, ball)
                if 0 <= bx < w and 0 <= by < h:
                    item["ball"] = [round(bx / w, 3), round(by / h, 3)]
            out[str(c)] = item
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--story", default="off", choices=["off", "fast_only", "gemini_only", "hybrid"])
    parser.add_argument("--specialist-url", default="")
    parser.add_argument("--goalwatch", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    counts = {"decision": 0, "state": 0}

    def publish(m: dict) -> None:
        kind = m.get("type")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "decision":
            print(f"  {m['time']:7.1f}s  {m['kind']}: {m.get('message', '')}")
        elif kind in ("live_error", "live_ended"):
            print(f"[live] {kind}: {m}")

    live = LiveStudio(args.segment, publish, args.story, args.specialist_url, args.goalwatch, args.seconds,
                      record=not args.no_record)
    live.start()
    try:
        while live.thread.is_alive():
            time.sleep(5)
            print(f"[live] t={live.status['t']:.0f}s lag {live.status['lag_s']}s, shown {live.status['shown_fps']} fps, "
                  f"skipped {live.status['skipped']}", flush=True)
    except KeyboardInterrupt:
        live.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
