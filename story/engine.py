"""engine.py — StoryEngine: runs the story layers in step with the live video.

The browser's <video> is the master clock. It sends its currentTime ~5x/s ("ticks"); on_tick()
moves the Playhead forward. The engine's own background thread then runs whatever is due, but
only up to the playhead (CausalReader refuses anything later):

  - FAST layer every `fast.tick_s` (1 s) of video: audio excitement, whistle, motion, cuts.
  - STORY every `llm.interval_s` (8 s), plus extra looks on triggers (cheer, whistle, specialist
    shot/goal), never closer than `trigger_min_gap_s`. Gemini when the mode has it and it is
    available, otherwise the template narrator.
  - MEMORY ("story so far") every `memory.interval_s` (60 s).
  - SPECIALIST (Colab) runs in its own thread and hands its events to the engine.

Slow AI calls run in worker threads, so the fast layer never waits for Gemini. Each result is
stamped with logged_at = the playhead when it ARRIVED (honest latency).

If the engine falls more than `playhead.max_lag_s` behind the video, it does NOT catch up on
old seconds: it jumps to the newest second, like a live system, and logs the lag ("LAG").
Everything goes to the dashboard via publish(message) and to outputs/story/<clip>/<run_id>/.
"""
from __future__ import annotations

import math
import tempfile
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from story.causal import CausalReader, Playhead
from story.fast_layer import FastLayer, FastResult
from story.fusion import MODE_LAYERS, Fusion, FusedEvent
from story.llm.base import MemoryContext, StoryLLM, WindowContext
from story.llm.template import TemplateNarrator
from story.run_log import RunLog
from story.schema import Event, Memory, StoryUpdate

Publish = Callable[[dict], None]


class StoryEngine:
    def __init__(self, clip_folder: Path, cfg: dict, mode: str, publish: Publish, send_audio: bool = True,
                 specialist_url: str = "", proof: bool = False, specialist_video=None):
        self.cfg = cfg
        self.specialist_video = specialist_video   # None = the specialist watches the clip (MVP 2)
        self.mode = mode
        self.layers = MODE_LAYERS[mode]
        self.publish_raw = publish
        self.send_audio = send_audio
        self.clip = Path(clip_folder).name
        self.playhead = Playhead()
        self.reader = CausalReader(clip_folder, self.playhead)
        self.fusion = Fusion(cfg, mode)
        self.fast = FastLayer(cfg, self.reader)
        self.narrator = TemplateNarrator()
        self.llm: StoryLLM | None = None          # Gemini (set up in _setup_llm)
        self.llm_state = "off"                    # off / ok / busy / rate_limited / no_key / error
        self.specialist = None                    # SpecialistClient (set up in _setup_specialist)
        self.specialist_url = specialist_url
        self.log_file = RunLog(self.clip, mode)
        self.run_id = self.log_file.run_id

        self.lock = threading.RLock()             # fusion + lists are touched by several threads
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="story-llm")
        self.paused = True
        self.ended = False
        self.ticks = 0
        self.fast_done_to = 0.0                   # the fast layer has analysed everything up to here
        self.recent_fast: deque[FastResult] = deque(maxlen=120)
        self.updates: list[StoryUpdate] = []
        self.memory: Memory | None = None
        self.teams: dict[str, str] = {}
        self.is_replay = False                    # latest replay status (Gemini, or the fast hint) — shown in UI
        self.llm_replay = False                   # Gemini's latest is_replay (blocks new events)
        self.replay_from_llm_at = -999.0
        self.story_busy = self.memory_busy = False
        self.next_story_at = cfg["llm"]["interval_s"]
        self.next_memory_at = cfg["memory"]["interval_s"]
        self.last_story_request = -999.0
        self.pending_triggers: list[str] = []
        self.delays: list[float] = []             # Gemini delays (logged_at - window_end)
        self.counters = {"story_gemini": 0, "story_template": 0, "memory": 0, "llm_errors": 0, "lag_skips": 0}
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="story-engine")
        self.started_wall = time.time()
        # Proof mode: ground truth from data/clips/<clip>/truth.csv, shown as markers on the timeline.
        # It is only ever compared with what the system already logged; it never feeds the AI.
        self.truth = self._load_truth() if proof else []
        self.last_truth_publish = -999.0

    def _load_truth(self) -> list[dict]:
        from story.evaluate import load_truth

        try:
            return load_truth(self.clip)
        except SystemExit:
            return []

    def _publish_truth(self, force: bool = False) -> None:
        """Proof mode: matched / missed / pending / false alarm, judged only up to the playhead."""
        if not self.truth or (not force and time.time() - self.last_truth_publish < 2):
            return
        from story.evaluate import truth_status

        self.last_truth_publish = time.time()
        with self.lock:
            found = [e.to_dict() for e in self.fusion.events]
        now = self.playhead.now()
        items = truth_status([t for t in self.truth if t["time_s"] <= now], found,
                             self.cfg["eval"]["match_window_s"], until=now)
        self.publish({"type": "truth", "items": items})

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self._setup_llm()
        self._setup_specialist()
        self.log_file.write_meta(
            run_id=self.run_id, clip=self.clip, mode=self.mode, send_audio_to_llm=self.send_audio,
            model=getattr(self.llm, "model", None), llm_state=self.llm_state,
            specialist_url=self.specialist_url or None, started_at=datetime.now().isoformat(timespec="seconds"),
            clip_duration=self.reader.duration, audio_model=self.fast.audio.model_name, proof=bool(self.truth),
            config=self.cfg)
        self.publish({"type": "run_started", "run_id": self.run_id, "clip": self.clip, "mode": self.mode,
                      "send_audio": self.send_audio, "model": getattr(self.llm, "model", None),
                      "duration": self.reader.duration, "proof": bool(self.truth),
                      "truth_events": len(self.truth)})
        self._publish_llm_state()
        self.publish(self.fusion.score_message())
        self.thread.start()

    def on_tick(self, t: float, paused: bool, ended: bool = False) -> None:
        """Called for every playhead message from the browser."""
        self.ticks += 1
        self.paused = paused
        self.ended = self.ended or ended
        self.playhead.advance(min(t, self.reader.duration))

    def stop(self, reason: str = "stopped") -> None:
        if self.stop_flag.is_set():
            return
        self.stop_flag.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=10)
        if self.specialist:
            self.specialist.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        summary = self.summary()
        self._publish_truth(force=True)
        self.log_file.write_meta(ended_at=datetime.now().isoformat(timespec="seconds"), end_reason=reason,
                                 summary=summary)
        self.publish({"type": "run_stopped", "run_id": self.run_id, "reason": reason, "t": self.playhead.now(),
                      "summary": summary})

    def summary(self) -> dict:
        with self.lock:
            headline = [e for e in self.fusion.events if e.headline]
            return {"played_to": round(self.playhead.now(), 2), "wall_seconds": round(time.time() - self.started_wall, 1),
                    "story_updates": len(self.updates), "events": len(self.fusion.events),
                    "headline_events": len(headline),
                    "events_by_type": {t: sum(e.type == t for e in headline) for t in sorted({e.type for e in headline})},
                    "ignored_in_replay": self.fusion.ignored_in_replay, "score": self.fusion.score_text(),
                    "scoreboard": self.fusion.scoreboard,
                    "avg_llm_delay": round(sum(self.delays) / len(self.delays), 2) if self.delays else None,
                    "max_llm_delay": round(max(self.delays), 2) if self.delays else None,
                    **self.counters}

    # ------------------------------------------------------------------ messages + logs
    def publish(self, message: dict) -> None:
        self.publish_raw(message)

    def log(self, kind: str, message: str, t: float | None = None) -> None:
        """A line in the dashboard's signal log (and signals.jsonl)."""
        item = {"kind": kind, "message": message, "t": round(self.playhead.now() if t is None else t, 2)}
        self.publish({"type": "log", **item})
        self.log_file.append("signals", item)

    # ------------------------------------------------------------------ setup
    def _setup_llm(self) -> None:
        if not self.layers["gemini"]:
            self.llm_state = "off"
            return
        try:
            from story.llm.gemini import GeminiStoryLLM, NoKeyError
        except ImportError as error:
            self.llm_state = "error"
            self.log("ERROR", f"Gemini module not available: {error}")
            return
        try:
            self.llm = GeminiStoryLLM(self.cfg, on_call=self._on_llm_call)
            self.llm_state = "ok"
            self.log("LLM", f"Gemini ready: {self.llm.model}")
        except NoKeyError as error:
            self.llm_state = "no_key"
            self.log("ERROR", f"{error} — using template narration instead")
        except Exception as error:
            self.llm_state = "error"
            self.log("ERROR", f"Gemini setup failed ({type(error).__name__}: {error}) — using template narration")

    def _setup_specialist(self) -> None:
        if not self.layers["specialist"]:
            self.publish({"type": "specialist", "state": "off", "message": f"not used in mode {self.mode}"})
            return
        if not self.specialist_url:
            self.log("SPECIALIST", "no Colab URL: running without the specialist layer")
            self.publish({"type": "specialist", "state": "off", "message": "no Colab URL (paste it and Connect)"})
            return
        self.connect_specialist(self.specialist_url)

    def connect_specialist(self, url: str) -> None:
        """Start (or restart) the Colab connection. Also used when Connect is pressed mid-run."""
        if not self.layers["specialist"]:
            return
        from story.specialist_client import SpecialistClient

        if self.specialist:
            self.specialist.stop()
        self.specialist_url = url
        self.log_file.write_meta(specialist_url=url)

        self.specialist = SpecialistClient(self.cfg, self.reader, self.specialist_url,
                                           on_event=self._on_specialist_event, on_status=self._on_specialist_status,
                                           log_call=lambda item: self.log_file.append("specialist_calls", item),
                                           video_source=self.specialist_video)
        self.specialist.start()

    # ------------------------------------------------------------------ main loop
    def _loop(self) -> None:
        self.log("INFO", f"audio model: {self.fast.audio.model_name}"
                         + (f" ({self.fast.audio.error})" if self.fast.audio.error else ""))
        while not self.stop_flag.is_set():
            try:
                did_work = self._step_fast()
                self._maybe_story()
                self._maybe_memory()
            except Exception as error:
                traceback.print_exc()
                self.log("ERROR", f"{type(error).__name__}: {error}")
                did_work = False
            if not did_work:
                time.sleep(0.02)

    def _step_fast(self) -> bool:
        """Run the fast layer for the next second, if it has played. Returns True if it ran."""
        tick_s = self.cfg["fast"]["tick_s"]
        now = self.playhead.now()
        # The last partial second at the end of the clip is analysed too.
        at_end = now >= self.reader.duration - 1e-3 and now > self.fast_done_to + 0.05
        if now - self.fast_done_to < tick_s and not at_end:
            return False
        t0 = self.fast_done_to
        lag = now - (t0 + tick_s)
        if lag > self.cfg["playhead"]["max_lag_s"]:
            # Too far behind: skip the old seconds and analyse the newest one, like a live system.
            self.counters["lag_skips"] += 1
            self.log("LAG", f"backend {lag:.1f}s behind the video: skipping {now - tick_s - t0:.1f}s")
            t0 = now - tick_s
        t1 = min(t0 + tick_s, now)
        result = self.fast.step(t0, t1)
        self.fast_done_to = t1
        self._on_fast(result, lag=max(0.0, self.playhead.now() - t1))
        return True

    # ------------------------------------------------------------------ fast layer
    def _on_fast(self, r: FastResult, lag: float) -> None:
        with self.lock:
            self.recent_fast.append(r)
            # Without a recent Gemini opinion, the fast replay hint decides.
            if r.t - self.replay_from_llm_at > self.cfg["llm"]["interval_s"] * 1.5:
                self.is_replay = r.replay
            intensity = self.fusion.intensity(r)
        self.publish({"type": "fast", "t": round(r.t, 2), "audio": round(r.audio, 3), "motion": round(r.motion, 3),
                      "crowd": round(r.crowd, 3), "whistle": round(r.whistle, 2), "loudness_db": r.loudness_db,
                      "cuts": r.cuts, "replay": self.is_replay, "intensity": intensity, "lag": round(lag, 2),
                      "audio_model": r.audio_model})
        self._publish_truth()
        self.log_file.intensity_row({"t": round(r.t, 2), "audio": round(r.audio, 3), "crowd": round(r.crowd, 3),
                                     "loudness_db": r.loudness_db, "motion": round(r.motion, 3), "cuts": r.cuts,
                                     "replay": int(self.is_replay), "whistle": round(r.whistle, 2),
                                     "intensity": intensity, "lag": round(lag, 2)})
        if "cheer" in r.triggers:
            self.log("CHEER", f"crowd/excitement jump to {r.audio:.2f} ({r.loudness_db:.0f} dB)", r.t)
            self.pending_triggers.append("cheer")
        if r.whistle_hit:
            self.log("WHISTLE", f"whistle (score {r.whistle:.2f})", r.t)
            self.pending_triggers.append("whistle")
            self.add_event(Event(type="whistle", event_time=r.t - 0.5, logged_at=self.playhead.now(),
                                 source="fast", confidence=min(1.0, r.whistle), note="referee whistle (audio)"))
        if r.cuts:
            self.log("CUT", f"{r.cuts} scene cut(s)" + (" → replay hint" if r.replay else ""), r.t)

    # ------------------------------------------------------------------ events (all layers)
    def blocking_replay(self) -> bool:
        """Is a replay on screen right now, sure enough to ignore new events?

        Gemini's is_replay (if recent) decides. The scene-cut hint alone only blocks events if
        fusion.fast_replay_blocks_events is on: on edited highlights it also fires on normal editing.
        """
        llm_recent = self.playhead.now() - self.replay_from_llm_at <= self.cfg["llm"]["interval_s"] * 1.5
        if llm_recent:
            return self.llm_replay
        return self.is_replay and self.cfg["fusion"]["fast_replay_blocks_events"]

    def add_event(self, event: Event, is_replay: bool | None = None, scoreboard: str | None = None) -> FusedEvent | None:
        """Hand one event to fusion; publish + log what changed.

        scoreboard: the score Gemini read in the same snippet (e.g. "2-0"), used to tell a new goal
        from a replayed one."""
        with self.lock:
            replay = self.blocking_replay() if is_replay is None else is_replay
            fused, is_new = self.fusion.add_event(event, replay, scoreboard=scoreboard)
            if fused is None:
                if replay and event.type != "none":
                    self.log("REPLAY", f"ignored {event.type} from {event.source} (replay)", event.event_time)
                return None
            if self.fusion.count_goal(fused):
                self.log("SCORE", f"goal counted for {fused.team or '?'} → {self.fusion.score_text()}")
                self.publish(self.fusion.score_message())
            elif is_new and fused.pending:
                self.log("SCORE", f"goal at {fused.event_time:.1f}s: waiting for the scoreboard to confirm it")
            item = {**fused.to_dict(), "new": is_new}
        # "type" is the message type for the page; the event's own type travels as "event_type".
        self.publish({**item, "type": "event", "event_type": item["type"]})
        self.log_file.append("events", {**item, "from": event.model_dump()})
        return fused

    # ------------------------------------------------------------------ story (Gemini / template)
    def _maybe_story(self) -> None:
        now = self.playhead.now()
        if self.story_busy or now < 1.0:
            return
        triggered = bool(self.pending_triggers) and \
            now - self.last_story_request >= self.cfg["llm"]["trigger_min_gap_s"]
        if now < self.next_story_at and not triggered:
            return
        triggers = self.pending_triggers if triggered else []
        self.pending_triggers = []
        self.last_story_request = now
        self.next_story_at = now + self.cfg["llm"]["interval_s"]
        ctx = self._window_context(now, triggers)
        use_gemini = self.llm is not None and self.llm.available()
        self.story_busy = True
        if use_gemini:
            self.pool.submit(self._run_story, self.llm, ctx)
        else:
            self._run_story(self.narrator, ctx)

    def _window_context(self, now: float, triggers: list[str]) -> WindowContext:
        # Round DOWN to 0.01 s: rounding up could point past the playhead (the causal guard refuses
        # that, rightly). logged_at is rounded to 0.01 s too, so delays never show "-0.0".
        now = math.floor(now * 100) / 100
        start = round(max(0.0, now - self.cfg["llm"]["window_s"]), 2)
        with self.lock:
            window_fast = [r for r in self.recent_fast if r.t > start]
            fast = {"intensity": self.fusion.intensity(window_fast[-1]) if window_fast else 0,
                    "audio_max": max((r.audio for r in window_fast), default=0),
                    "motion_avg": sum(r.motion for r in window_fast) / len(window_fast) if window_fast else 0,
                    "replay": self.is_replay, "whistle": any(r.whistle_hit for r in window_fast),
                    "cheer": any("cheer" in r.triggers for r in window_fast)}
            recent = self.fusion.recent_events(now, self.cfg["llm"]["recent_events_s"])
            hints = [f"{e['type'].upper()} at {e['event_time']:.1f}s ({e['confidence']:.2f})"
                     for e in recent if "specialist" in e["sources"]]
            return WindowContext(sport=self.cfg["sport"], window_start=start, window_end=now, memory=self.memory,
                                 recent_events=recent, specialist_hints=hints, fast=fast, teams=dict(self.teams),
                                 score_text=self.fusion.score_text(), triggers=triggers)

    def _run_story(self, provider: StoryLLM, ctx: WindowContext) -> None:
        """Runs in a worker thread for Gemini (inline for templates)."""
        try:
            if provider is not self.narrator:
                ctx.video_path = self._cut_window(ctx)
            try:
                update = provider.describe_window(ctx)
            except Exception as error:
                if provider is self.narrator:
                    raise
                self.counters["llm_errors"] += 1
                self.log("ERROR", f"Gemini: {str(error)[:200]} — template line instead")
                update = self.narrator.describe_window(ctx)
            finally:
                if ctx.video_path:
                    ctx.video_path.unlink(missing_ok=True)
            self._on_story(update, ctx)
        except Exception as error:
            traceback.print_exc()
            self.log("ERROR", f"story step failed: {type(error).__name__}: {error}")
        finally:
            self.story_busy = False
            self._publish_llm_state()

    def _cut_window(self, ctx: WindowContext) -> Path:
        llm = self.cfg["llm"]
        out = Path(tempfile.mkstemp(suffix=".mp4", prefix="story-window-")[1])
        # CausalReader refuses if window_end is after the playhead (it never is: it IS the playhead).
        return self.reader.window_clip(ctx.window_start, ctx.window_end, out, height=llm["video_height"],
                                       with_audio=self.send_audio)

    def _on_story(self, update: StoryUpdate, ctx: WindowContext) -> None:
        with self.lock:
            update.logged_at = round(self.playhead.now(), 2)        # where the video is NOW (honest delay)
            update.id = f"s{len(self.updates) + 1}"
            self.updates.append(update)
            if update.source == "gemini":
                self.counters["story_gemini"] += 1
                self.delays.append(update.logged_at - update.window_end)
                self.fusion.note_llm_intensity(update.logged_at, update.intensity)
                self.is_replay = self.llm_replay = update.is_replay
                self.replay_from_llm_at = update.logged_at
                for team in filter(None, [update.team_in_control, *[e.team for e in update.events]]):
                    self.fusion.add_team(team)
                    self.teams.setdefault(team, "")

            else:
                self.counters["story_template"] += 1
        item = update.model_dump()
        self.publish({"type": "story", **item, "triggers": ctx.triggers})
        self.log_file.append("story", {**item, "triggers": ctx.triggers})
        if update.source == "gemini":
            self.log("LLM", f"{update.phase} / {update.sentiment} / {update.intensity}"
                            + (" / replay" if update.is_replay else "")
                            + f" (delay {update.logged_at - update.window_end:.1f}s)", update.window_end)
        for event in update.events:
            event.logged_at = update.logged_at
            # Gemini already left out events it saw in a replay (in_replay); the rest it saw live.
            # (A replayed goal it mislabels is still caught by the 60 s goal merge window in fusion.)
            self.add_event(event, is_replay=False, scoreboard=update.score)
        self._review_goals(update)
        if update.source == "gemini" and not update.is_replay:
            with self.lock:
                changed = self.fusion.note_scoreboard(update.score)
            if changed:
                self.log("SCORE", f"score graphic reads {self.fusion.scoreboard}", update.window_end)
        self.publish(self.fusion.score_message())

    def _review_goals(self, update: StoryUpdate) -> None:
        """Pending goals: confirmed when the scoreboard goes up, dropped if it never does."""
        if update.source != "gemini" or update.is_replay:
            return
        with self.lock:
            changes = self.fusion.review_pending_goals(update.score, update.logged_at)
        for fused, action in changes:
            if action == "confirmed":
                self.log("SCORE", f"goal at {fused.event_time:.1f}s confirmed by the scoreboard ({update.score}) "
                                  f"→ {self.fusion.score_text()}")
            else:
                self.log("SCORE", f"goal at {fused.event_time:.1f}s dropped: the scoreboard stayed at {update.score}")
            item = {**fused.to_dict(), "new": False}
            self.publish({**item, "type": "event", "event_type": item["type"]})
            self.log_file.append("events", {**item, "review": action})

    # ------------------------------------------------------------------ memory
    def _maybe_memory(self) -> None:
        now = self.playhead.now()
        if self.memory_busy or now < self.next_memory_at:
            return
        self.next_memory_at = now + self.cfg["memory"]["interval_s"]
        with self.lock:
            since = self.memory.logged_at if self.memory else 0.0
            ctx = MemoryContext(sport=self.cfg["sport"], playhead=now, previous=self.memory,
                                updates=[u for u in self.updates if u.window_end > since],
                                events=[e.to_dict() for e in self.fusion.events if e.headline],
                                score_text=self.fusion.score_text(), teams=dict(self.teams))
        self.memory_busy = True
        use_gemini = self.llm is not None and self.llm.available()
        if use_gemini:
            self.pool.submit(self._run_memory, self.llm, ctx)
        else:
            self._run_memory(self.narrator, ctx)

    def _run_memory(self, provider: StoryLLM, ctx: MemoryContext) -> None:
        try:
            try:
                memory = provider.summarize(ctx)
            except Exception as error:
                if provider is self.narrator:
                    raise
                self.counters["llm_errors"] += 1
                self.log("ERROR", f"Gemini memory: {str(error)[:200]} — template summary instead")
                memory = self.narrator.summarize(ctx)
            with self.lock:
                memory.logged_at = round(self.playhead.now(), 2)
                self.memory = memory
                self.counters["memory"] += 1
                for colour, description in memory.teams.items():
                    self.teams[colour] = description
                    self.fusion.add_team(colour)
            self.publish({"type": "memory", **memory.model_dump()})
            self.log_file.append("memory", memory.model_dump())
            self.log("MEMORY", f"story so far updated ({memory.source})", memory.logged_at)
            self.publish(self.fusion.score_message())
        except Exception as error:
            traceback.print_exc()
            self.log("ERROR", f"memory step failed: {type(error).__name__}: {error}")
        finally:
            self.memory_busy = False
            self._publish_llm_state()

    # ------------------------------------------------------------------ Gemini status
    def _on_llm_call(self, item: dict) -> None:
        """Called by the Gemini provider after every request (success or error)."""
        self.log_file.append("llm_calls", {**item, "playhead": round(self.playhead.now(), 2)})
        if item.get("daily_quota"):
            if item.get("switched_to"):
                self.log("ERROR", f"Gemini daily free quota used up for {item.get('model')}: "
                                  f"switching to {item['switched_to']} for this run")
            else:
                self.log("ERROR", "Gemini daily free quota used up for all models: template lines for the rest of the run")
        elif item.get("rate_limited"):
            self.log("ERROR", f"Gemini rate limited (HTTP 429): pausing AI calls for {item.get('backoff_s', 0):.0f}s")
        self._publish_llm_state()

    def _publish_llm_state(self) -> None:
        state, rpm, max_rpm, latency = self.llm_state, 0, self.cfg["llm"]["max_requests_per_minute"], None
        if self.llm is not None:
            state = self.llm.state()
            rpm = self.llm.requests_last_minute()
            max_rpm = getattr(getattr(self.llm, "limiter", None), "per_minute", max_rpm)   # all keys together
            latency = self.llm.last_latency
        self.publish({"type": "gemini", "state": state, "rpm": rpm, "max_rpm": max_rpm, "latency": latency,
                      "avg_delay": round(sum(self.delays) / len(self.delays), 2) if self.delays else None})

    # ------------------------------------------------------------------ specialist (Colab)
    def _on_specialist_event(self, event: Event | None, strength: float, name: str) -> None:
        """An answer from the specialist. event is None for low-priority "play" signals (PASS, DRIVE...)."""
        with self.lock:
            self.fusion.note_specialist(self.playhead.now(), strength)
        if event is None:
            # Low-priority play signal (PASS, DRIVE...): only logged, so dashboards can show what the model sees
            self.log("SPECIALIST_PLAY", f"{name} ({strength:.2f})")
            return
        if event.type in ("shot", "goal", "free_kick"):
            self.pending_triggers.append(f"specialist:{event.type}")
        self.log("SPECIALIST", f"{event.type} @{event.event_time:.1f}s ({event.confidence:.2f})", event.logged_at)
        self.add_event(event)

    def _on_specialist_status(self, status: dict) -> None:
        self.publish({"type": "specialist", **status})

    # ------------------------------------------------------------------ dashboard actions
    def adjust_score(self, team: str, delta: int) -> dict:
        with self.lock:
            self.fusion.adjust_score(team, delta)
            message = self.fusion.score_message()
        self.log("SCORE", f"manual {team} {'+1' if delta > 0 else '-1'} → {self.fusion.score_text()}")
        self.log_file.append("events", {"type": "manual_score", "team": team, "delta": delta,
                                        "logged_at": round(self.playhead.now(), 2), "score": message["teams"]})
        self.publish(message)
        return message
