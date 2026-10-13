"""live.py — MVP 3 live mode: everything runs while the match plays, at real speed.

  venv/bin/python -m studio.live --segment dev --seconds 60                      # director only, headless
  venv/bin/python -m studio.live --segment eval --story gemini_only --goalwatch  # + AI story + goal watcher
  venv/bin/python -m studio.live --segment muvs-P19 --seconds 60                 # real phone cameras (MUVS)
  venv/bin/python -m studio.live --segment issia-T2 --goalwatch                  # 6 real stadium cameras (ISSIA)
  venv/bin/python -m studio.live --segment alfheim-2013-11-03-H1 --start 23:40   # the broadcast starts at 23:40

A wall clock decides which frame is "now" (like a real broadcast). Each loop:
  1. read the newest frame of every on-air camera (frames we are too slow for are skipped: that's live);
  2. 5x per second: new detections. detector="cached": computed before the run (the M1 detects ~1 frame/s, too
     slow for live). detector="colab": REALLY live, the football model runs on Colab's GPU while the match plays
     (studio.remote_detect); each answer is used when it arrives, with its own frame time, in frame order;
     an answer later than max_age_s falls back to the cached detection of that frame (if there is one);
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
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, load_config
from src.director import shot_lengths
from src.render import letterbox
from studio.prepare import work_dir
from studio.sources import get_source
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


class ProgramFeed:
    """The program (camera on air) as a frame source for the Colab specialist: the action model was trained on TV
    broadcasts, so it should watch the directed picture, not a squeezed mosaic (ISSIA mosaic: 1 event in 2 min).
    The live loop pushes each shown program frame (already played = causal); frames() returns those in (t0, t1]."""

    def __init__(self, size: tuple[int, int] = (398, 224), keep_s: float = 20.0):
        self.size, self.keep_s = size, keep_s
        self.items: deque = deque()
        self.lock = threading.Lock()

    def push(self, t: float, image: np.ndarray) -> None:
        small = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        with self.lock:
            self.items.append((t, small))
            while self.items and t - self.items[0][0] > self.keep_s:
                self.items.popleft()

    def __call__(self):
        return self                                 # the specialist asks for "a video": this feed is it

    def frames(self, t0: float, t1: float, fps: float | None = None):
        with self.lock:
            items = [x for x in self.items if t0 < x[0] <= t1]
        last = -1e9
        for t, frame in items:
            if fps and t - last < 1.0 / fps - 1e-3:
                continue
            last = t
            yield t, frame

    def close(self) -> None:
        pass


class LiveStudio:
    def __init__(self, source: str, publish: Publish | None = None, story: str = "off", specialist_url: str = "",
                 goalwatch: bool = False, seconds: float | None = None, speed: float = 1.0, record: bool = True,
                 realtime: bool = True, start_s: float = 0.0, detector: str = "cached", detector_url: str = ""):
        self.cfg = load_config()["studio"]
        self.src = get_source(source)              # SoccerTrack segment ("eval") or MUVS period ("muvs-P19")
        self.segment, self.publish = self.src.key, publish or (lambda m: None)
        self.story_mode, self.specialist_url = story, specialist_url
        self.use_goalwatch = goalwatch and self.src.goalwatch
        self.speed, self.record, self.realtime = speed, record, realtime
        self.cams, self.fps, self.step, self.on_air = self.src.cams, self.src.fps, self.src.step, self.src.on_air
        # Start from anywhere: the broadcast begins at start_s, like a NEW session (teams, possession, score and
        # the story are learned again from there). All times in the recorded files stay ABSOLUTE match times.
        self.start_frame = max(0, min(round(start_s * self.fps), self.src.frames - 1))
        self.start_s = self.start_frame / self.fps
        end = self.src.frames if not seconds else self.start_frame + round(seconds * self.fps)
        self.total = min(self.src.frames, end)      # the frame where the run ends (absolute)
        # Where the detections come from: "cached" (computed before) or "colab" (the GPU, while the match plays)
        if detector not in ("cached", "colab"):
            raise ValueError(f"detector must be cached or colab, not {detector}")
        if detector == "colab" and not (getattr(self.src, "calibrated", False) and realtime):
            raise ValueError("the Colab detector is for live runs on calibrated stadium cameras (ISSIA, Alfheim)")
        self.detector, self.detector_url = detector, detector_url or specialist_url
        self.remote = None                          # studio.remote_detect.RemoteDetector (detector "colab")
        self.det_counts = {"colab": 0, "cached_late": 0, "cached_skipped": 0, "none": 0, "ignored_late": 0}
        self.feed = LiveFeed(self.cfg["goalwatch"])
        self.pipe = self.src.pipeline(self.feed)
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpegs: dict[str, tuple[int, bytes]] = {}      # stream name -> (sequence number, JPEG)
        self.jpeg_lock = threading.Lock()
        self.story_engine = None
        # Folder footage (ISSIA, MUVS): the Colab model watches the program. SoccerTrack keeps MVP 2's clip feed.
        self.program_feed = ProgramFeed() if self.src.kind != "soccertrack" else None
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
            from story.engine import StoryEngine

            folder, cfg = self.src.story_clip()        # what the story layer watches (+ its settings)

            def on_story(message: dict) -> None:
                kind = message.get("type")
                if kind == "event":
                    self.feed.add_story_event(message, clock())
                    # a cross / header / shot near a goal: the goal watcher looks at that goal again
                    if self.watcher and message.get("event_type") in self.cfg["goalwatch"]["recheck_events"] \
                            and not message.get("rejected"):
                        self.watcher.request(message["event_type"], message["event_time"], self.pipe.pitch_state)
                elif kind == "story":
                    self.feed.add_update({k: message.get(k) for k in ("window_start", "window_end", "logged_at", "phase",
                                                                      "sentiment", "narration", "intensity", "source",
                                                                      "events")})
                self.publish({"type": "story_msg", **{k: v for k, v in message.items() if k != "type"},
                              "kind": kind, "log_kind": message.get("kind") if kind == "log" else None})

            self.story_engine = StoryEngine(folder, cfg, self.story_mode, on_story, send_audio=self.src.has_audio,
                                            specialist_url=self.specialist_url, specialist_video=self.program_feed,
                                            start_s=self.start_s)
            self.story_engine.start()
        if self.use_goalwatch:
            from studio.goalwatch import LiveGoalWatcher

            llm = self.story_engine.llm if self.story_engine and self.story_engine.llm else None
            self.watcher = LiveGoalWatcher(self.segment, self.cams, clock, self.feed.add_look, self.publish,
                                           shared_llm=llm, out_dir=work_dir(self.src.segment) / "goalwatch" / self.run_id,
                                           snapshot=self._goal_snapshot, video_of=self._goal_video,
                                           cfg=self.src.cfg["goalwatch"] if self.src.kind != "soccertrack" else None)

    def _goal_video(self, cam: int) -> Path:
        """The goal camera's video: SoccerTrack renders its own goal views; other footage films them for real."""
        if self.src.kind == "soccertrack":
            from studio.cameras import cam_video

            return cam_video(self.src.segment, cam)
        return self.src.video(cam)

    def _goal_snapshot(self, side: str, video: Path, t: float) -> None:
        """The goal camera picture at the moment the AI starts checking it (one frame, read in the AI thread)."""
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(t * self.fps) - 1))
        ok, img = cap.read()
        cap.release()
        if ok:
            self._put_jpeg(f"goal_{side}", img, 480, 70)

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
        # Light decoding: with small copies (ISSIA 960x540), every camera is read small (tiles, quality, shirts) and
        # only the camera ON AIR is read in full HD for the program. 6 full-HD streams at once made the M1 drop
        # to ~13 fps shown (the program then looks jerky).
        small_of = getattr(self.src, "small_video", lambda c: None)
        light = all(small_of(c) for c in cam_ids if c in self.on_air)
        caps = {c: cv2.VideoCapture(str(small_of(c) if light else self.src.video(c))) for c in cam_ids if c in self.on_air}
        if self.start_frame:
            for cap in caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)   # exact on these H.264 files (tested)
        self.pipe.director.start_at(self.start_frame)
        full: dict[int, list] = {}                    # cam -> [capture, index of the frame it reads next]
        blank = np.zeros((36, 64, 3), np.uint8)      # cameras that are never aired are not decoded (CPU budget)
        out = ROOT / self.cfg["outputs_dir"] / self.segment / self.run_id
        files = self._open_files(out) if self.record else None
        w, h = 1280, 720                              # the recorded program (phones are 1080p: smaller = cheaper)
        program = VideoWriter(out / "program.mp4", w, h, fps) if self.record else None
        chosen, states, frames_shown, skipped, lags = [], [], 0, 0, []
        start = time.time()
        clock = lambda: min(self.start_s + (time.time() - start) * self.speed, self.total / fps)   # noqa: E731
        if not self.realtime:
            # Offline (tuning / scoring): every frame, as fast as the machine goes; the "clock" is the frame.
            frame_clock = {"t": self.start_s}
            clock = lambda: frame_clock["t"]                                             # noqa: E731
        self._start_ai(clock)
        if self.realtime:
            start = time.time()   # the match clock starts once the AI is ready (its start-up took seconds)
        remote, pending, ring = None, deque(), {}    # Colab detector: frames waiting for an answer, their small pictures
        if self.detector == "colab":
            from studio.remote_detect import RemoteDetector

            from studio.remote_detect import FullFrameReader

            reader = FullFrameReader({c: self.src.video(c) for c in caps}, fps) if light else None
            remote = self.remote = RemoteDetector(self.detector_url, fps, clock, self.cfg["live"]["remote_detect"], reader)
            remote.start()
            self.src.shown_dets = {}                  # tiles draw what the detector answered (not the cache)
            self.colab_answers: dict[int, dict] = {}
        self.status["state"] = "live"
        self.publish({"type": "live_started", "segment": self.segment, "run_id": self.run_id,
                      "duration": self.total / fps, "start_s": self.start_s, "story": self.story_mode,
                      "goalwatch": self.use_goalwatch})
        i, last_det, last_pub, last_tiles = self.start_frame - 1, -1, 0.0, 0.0
        images = {}
        decision = None
        last_program = None                          # the last picture written to program.mp4
        try:
            while not self.stop_flag.is_set():
                if not self.realtime:
                    frame_clock["t"] = min((i + 1) / fps, (self.total - 1) / fps)
                now_t = clock()
                # Offline: simply the next frame (computing it from the time can round back to the same frame).
                target = min(i + 1, self.total - 1) if not self.realtime else min(int(now_t * fps), self.total - 1)
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
                    if not chosen:                 # skipped before the first decision: the start camera
                        chosen.append(self.cfg["director"]["start_cam"] if self.src.kind == "soccertrack"
                                      else self.pipe.director.current)
                        states.append("IN_PLAY")
                    elif chosen:                   # a skipped frame still shows the live camera
                        chosen.append(chosen[-1])
                        states.append(states[-1])
                        if program and last_program is not None:
                            program.write(last_program)   # keep program.mp4 in real time (Replay's clock)
                    # A skipped frame's detections are still read (cached, cheap): the pitch map keeps the ball.
                    # Before: a run that showed ~13 fps lost half the map updates (324 of 600) and ball on air fell.
                    if remote is not None and i % step == 0:
                        # skipped on screen, still detected: the detector's thread reads its full pictures itself
                        if light:
                            remote.submit(i)
                        pending.append((i, light))
                    elif remote is None and i % step == 0 and i != last_det:
                        last_det = i
                        skip_events = self.pipe.detections(i, i / fps, detector=self.src.detection_at(i))
                        if files and self.pipe.pitch_state is not None:
                            files["map"].write(json.dumps(self.pipe.pitch_state.to_dict()) + "\n")
                        for e in skip_events:
                            self.publish({"type": "decision", "time": round(i / fps, 2), "kind": e["type"],
                                          **{k: v for k, v in e.items() if k != "type"}})
                            if files:
                                files["events"].write(json.dumps({"frame": i, "time": round(i / fps, 2), **e}) + "\n")
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
                if remote is not None:
                    # Colab: send this detection frame (full resolution, every camera), apply what has arrived.
                    if i % step == 0:
                        # full resolution: read by the detector's own thread (light), else these pictures already are
                        remote.submit(i, None if light else dict(images))
                        pending.append((i, True))
                        ring[i] = dict(images)
                        for f in [f for f in ring if f < i - 2 * fps]:
                            del ring[f]
                    events = self._apply_remote(remote, pending, ring, clock(), files)
                elif det_frame != last_det:
                    last_det = det_frame
                    if hasattr(self.pipe, "teams"):      # calibrated cameras: shirt colours from these pictures
                        events = self.pipe.detections(det_frame, det_frame / fps, detector=self.src.detection_at(det_frame),
                                                      images=images)
                    else:
                        events = self.pipe.detections(det_frame, det_frame / fps, detector=self.src.detection_at(det_frame))
                    if self.watcher:
                        self.watcher.consider(t, self.pipe.pitch_state)
                    if files and self.pipe.pitch_state is not None:
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
                live_img = self._full_frame(full, decision.cam, i) if light else None
                if live_img is None:
                    live_img = images.get(decision.cam, blank)
                if program:
                    # same size as the recording (a 4:3 camera like Alfheim's gets black side bars, never stretched)
                    last_program = live_img if live_img.shape[:2] == (h, w) else letterbox(live_img, w, h)
                    program.write(last_program)
                wall = time.time()
                if self.program_feed is not None and self.story_engine is not None:
                    self.program_feed.push(t, live_img)
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
            for item in full.values():
                item[0].release()
            if remote is not None:
                remote.stop()
            self._stop_ai()
            if program:
                program.close() if chosen else program.abort()
            summary = self._finish(out, files, chosen, states, frames_shown, skipped, lags, start)
            self.status["state"] = "ended"
            self.publish({"type": "live_ended", "run_id": self.run_id, "summary": summary})

    def _full_frame(self, full: dict, cam: int, i: int) -> np.ndarray | None:
        """Frame i of the on-air camera in full HD. The capture of each camera is kept; after a cut it jumps
        (keyframe every second, so a seek costs at most ~1 s of decoding once)."""
        if cam not in full:
            full[cam] = [cv2.VideoCapture(str(self.src.video(cam))), 0, None]
        cap, pos, last = full[cam]
        if i == pos - 1:
            return last                               # already read (the Colab detector and the program share it)
        if i < pos or i - pos > round(self.fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            pos = i
        while pos < i:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        full[cam][1], full[cam][2] = pos + 1, img if ok else None
        return img if ok else None

    def _apply_remote(self, remote, pending: deque, ring: dict, now: float, files: dict | None) -> list[dict]:
        """Colab detector: apply the detection frames whose turn has come, OLDEST FIRST (the pitch map only moves
        forward in time). A frame is applied when its answer has arrived, or with the cached detection once it is
        max_age_s old without one (or right away if it was never sent). Answers for frames already done are ignored."""
        for r in remote.results():
            self.colab_answers[r["frame"]] = r
        max_age = self.cfg["live"]["remote_detect"]["max_age_s"]
        events = []
        while pending:
            f, sent = pending[0]
            answer = self.colab_answers.pop(f, None)
            if answer is not None and answer["age_s"] <= max_age:
                det, source = answer["cams"], "colab"
            elif sent and answer is None and now - f / self.fps <= max_age:
                break                                 # still on its way: wait (keeps the order)
            else:
                # late or never sent: the cached detection of that frame, unless the run is "pure live"
                cached = self.src.detection_at(f) if self.cfg["live"]["remote_detect"]["cache_fallback"] else {}
                has = any(d.get("players") or d.get("ball") for d in cached.values())
                det = cached if has else None
                source = ("cached_late" if sent else "cached_skipped") if has else "none"
                if answer is not None:
                    self.det_counts["ignored_late"] += 1
            pending.popleft()
            self.det_counts[source] += 1
            if det is not None:
                events += self.pipe.detections(f, f / self.fps, detector=det, images=ring.get(f))
                self.src.shown_dets = det
                if self.watcher:
                    self.watcher.consider(now, self.pipe.pitch_state)
                if files and self.pipe.pitch_state is not None:
                    files["map"].write(json.dumps(self.pipe.pitch_state.to_dict()) + "\n")
            if files:
                files["dets"].write(json.dumps({"frame": f, "time": round(f / self.fps, 3), "source": source,
                    "applied_at": round(now, 3), **({k: answer[k] for k in ("sent", "received", "rtt_ms", "infer_ms", "age_s")}
                                                   if answer else {}),
                    "cams": {str(c): d for c, d in (det or {}).items()}}) + "\n")
        # answers for frames that were already applied (too late): forget them
        for f in [f for f in self.colab_answers if not pending or f < pending[0][0]]:
            del self.colab_answers[f]
            self.det_counts["ignored_late"] += 1
        return events

    def _open_files(self, out: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        log_file = open(out / "director_log.csv", "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["frame", "time", "state", "chosen_cam", *[f"score_cam{c['cam']}" for c in self.cams["cams"]],
                      "reason"])
        return {"log_file": log_file, "log": log, "cards": open(out / "cards.jsonl", "w"),
                "map": open(out / "map.jsonl", "w"), "events": open(out / "director_events.jsonl", "w"),
                # the Colab detector's answers as they were used (frame, source, round trip...), for replay + comparison
                "dets": open(out / "detections_live.jsonl", "w")}

    def _finish(self, out: Path, files: dict | None, chosen: list, states: list, shown: int, skipped: int,
                lags: list, start: float) -> dict:
        fps = self.fps
        shots = shot_lengths(chosen, fps) if chosen else [0.0]
        # The last shot is cut short by the end of the clip, not by the director: leave it out of "shortest".
        whole = shots[:-1] or shots
        summary = {"start_s": round(self.start_s, 2), "end_s": round((self.start_frame + len(chosen)) / fps, 2),
                   "frames": len(chosen), "seconds": round(len(chosen) / fps, 1), "cuts": max(0, len(shots) - 1),
                   "shortest_shot_s": round(min(whole), 2), "average_shot_s": round(sum(shots) / len(shots), 2),
                   "time_per_state_s": {s: round(states.count(s) / fps, 1) for s in sorted(set(states))},
                   "frames_shown": shown, "frames_skipped": skipped,
                   "avg_lag_s": round(float(np.mean(lags)), 2) if lags else 0.0,
                   "max_lag_s": round(float(np.max(lags)), 2) if lags else 0.0,
                   "wall_seconds": round(time.time() - start, 1),
                   "confirmed_goals": sorted(v["goal_time"] for v in self.pipe.confirmed.values()),
                   # A live run that fell far behind is not a fair measurement (like an invalid story pass).
                   "shown_fps": round(shown / max(1e-6, time.time() - start), 1),
                   "detector": self.detector,
                   **({"detections": dict(self.det_counts), "colab_link": self.remote.status()} if self.remote else {}),
                   "valid": ((not lags) or float(np.max(lags)) <= self.cfg["live"]["max_lag_s"])
                            and (not self.realtime or shown / max(1e-6, time.time() - start) >= self.cfg["live"]["min_shown_fps"])}
        if files:
            for k in ("log_file", "cards", "map", "events", "dets"):
                files[k].close()
            story = None
            if self.story_engine:
                story = {"run_id": self.story_engine.run_id, "mode": self.story_mode}
            (out / "run.json").write_text(json.dumps({
                "segment": self.segment, "run_id": self.run_id, "source": "detector", "live": True,
                "source_kind": self.src.kind,
                # program.mp4 starts at start_s (Replay plays it with that offset).
                "start_s": round(self.start_s, 2),
                # wall clock at match time 0 (so wall_time - story_wall_start = match time, as for runs from 0:00)
                "story_clip": self.story_engine.clip if self.story_engine else None,
                "story_wall_start": start - self.start_s / self.speed,
                "story_run": story["run_id"] if story else None, "story_mode": story["mode"] if story else None,
                "specialist_feed": "program" if self.program_feed is not None else "story clip",
                "goal_looks": len(self.feed.looks), "goalwatch_run": self.run_id if self.watcher else None,
                "teams": self.pipe.team_info() if hasattr(self.pipe, "team_info") else None,
                "config": self.cfg, "summary": summary}, indent=2))
            if self.story_engine and self.src.kind == "soccertrack":
                from studio.story_pass import record_run

                record_run(self.segment, self.story_engine.run_id, self.story_engine.clip, self.story_mode, 1.0,
                           self.story_engine.summary(), wall_start=start - self.start_s / self.speed, feed="close_detector",
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
                "scores": [round(float(x), 1) for x in d.scores],
                "teams": self.pipe.team_info() if hasattr(self.pipe, "team_info") else None,
                "overlays": self.src.overlays(i, self.pipe, r.rects),
                "cards": cards, "map": ps.to_dict() if ps else None, "story_line": r.story_line,
                "lag_s": self.status["lag_s"], "shown_fps": self.status["shown_fps"],
                "skipped": self.status["skipped"], "confirmed": [v["goal_time"] for v in self.pipe.confirmed.values()],
                "asking": self.watcher.asking if self.watcher else None, "frame": i,
                "detector": {"mode": self.detector, **({**self.remote.status(), "used": dict(self.det_counts)}
                                                       if self.remote else {})}}


def parse_time(text: str) -> float:
    """'23:40' -> 1420.0, '1:02:03' -> 3723.0, '95.5' -> 95.5"""
    seconds = 0.0
    for part in str(text).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True, help="a SoccerTrack segment (dev, eval, test), muvs-P19 / muvs-P15 or issia-T2")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--start", default="0", help="where the broadcast starts: seconds or mm:ss (e.g. 23:40)")
    parser.add_argument("--story", default="off", choices=["off", "fast_only", "gemini_only", "hybrid"])
    parser.add_argument("--specialist-url", default="")
    parser.add_argument("--detector", default="cached", choices=["cached", "colab"],
                        help="colab = the football model on Colab's GPU while the match plays (needs --detector-url)")
    parser.add_argument("--detector-url", default="", help="the Colab URL (default: --specialist-url)")
    parser.add_argument("--goalwatch", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--offline", action="store_true", help="every frame, as fast as possible (no AI): for scoring")
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
                      record=not args.no_record, realtime=not args.offline, start_s=parse_time(args.start),
                      detector=args.detector, detector_url=args.detector_url)
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
