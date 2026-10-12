"""goalwatch.py — the AI's "goal eyes": when play is near a goal, ask Gemini to watch THAT goal's camera.

  venv/bin/python -m studio.goalwatch --segment dev

Why: on the far panorama (and even on the close-up) Gemini never saw a goal on this match, while on TV
footage, pictures only, it found 3 of 4: the ball going in must be shown big enough. So each goal gets a
fixed, sharp view (cam6 GOAL LEFT, cam7 GOAL RIGHT, studio.cameras), and the pitch map decides WHEN to look:
when the ball (or, if it is not known, the main group of players) is within `near_goal_m` of a goal.

As-if-live, honestly:
  - the pitch map is built from the detector, frame by frame (past only), exactly as in studio.run;
  - a look at time t sends the goal camera's last `window_s` seconds (ending at t), with NO sound;
  - one look at a time, like a live system: the answer is "known" at t + the real API latency, and the next
    look can only start after that. Every result keeps event_time + logged_at.
  - no scoreboard, no commentary, no ground truth.

Output: data/studio/<segment>/goalwatch/<run_id>/looks.jsonl (+ calls.jsonl) and data/studio/<segment>/goalwatch.json
(list of runs). studio.run reads the newest run and gives its goals/celebrations to the director.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.common import ffmpeg_exe, load_config
from story.common import story_config
from studio.cameras import cam_video, group_center, load_cams
from studio.detect import PanoramaDetections
from studio.pitchmap import HALF_X, PitchMap, Projection
from studio.prepare import load_segment, work_dir

SYSTEM = """You watch ONE fixed camera aimed at a soccer goal, at an amateur match, live.
You get the last few seconds of video (no sound). You cannot see the future.
Say only what you can see:
- goal: true only if the ball clearly ends up in THIS goal's net (crossing the line between the posts),
  or players clearly celebrate a goal right at this goal. A shot that misses or is saved is NOT a goal.
- shot: a shot or header towards this goal. save: the goalkeeper stops a shot.
- celebration: players clearly celebrating (arms up, running together, hugging).
- goal_at_s / shot_at_s: seconds from the START of the clip.
Be strict: when unsure, goal=false and a low confidence. Answer only with JSON matching the schema."""


AFTER = """A shot was just taken at this goal (amateur soccer, one fixed camera on the goal, no sound).
You see the few seconds AFTER the shot. Decide from what happens now whether it was a GOAL:
- goal signs: the ball is in the net, the goalkeeper picks the ball out of the net or kicks it away in
  frustration, attackers turn away / celebrate / run back to their own half together, defenders stand still.
- not a goal: the keeper holds the ball and plays on, a goal kick or corner is taken, play simply continues.
Be honest about uncertainty. Answer only with JSON matching the schema."""


class AfterLook(BaseModel):
    goal_scored: bool = False
    confidence: float = Field(0.0, description="0..1")
    keeper_takes_ball_from_net: bool = False
    attackers_celebrate: bool = False
    restart_from_goal_kick_or_corner: bool = False
    note: str = ""


class GoalLook(BaseModel):
    goal: bool = False
    goal_confidence: float = Field(0.0, description="0..1")
    goal_at_s: float = 0.0
    shot: bool = False
    shot_at_s: float = 0.0
    save: bool = False
    celebration: bool = False
    ball_visible: bool = False
    note: str = ""


def near_goal_side(state, cfg: dict) -> str | None:
    """Which goal play is near (pitch map), or None. Ball if known, else the main group of players."""
    x = None
    if state.ball is not None and state.ball_status != "lost":
        x = state.ball[0]
    elif state.players:
        pts = np.asarray(state.players, dtype=float)
        x = group_center(pts, pts, cfg["group_radius_m"])[0]
    if x is None:
        return None
    if x <= -HALF_X + cfg["near_goal_m"]:
        return "left"
    if x >= HALF_X - cfg["near_goal_m"]:
        return "right"
    return None


def cut_window(video: Path, start: float, end: float, out: Path, height: int) -> None:
    """[start, end) of a camera video as a small mp4 (no sound). end is "now": never later."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(video),
                    "-t", f"{end - start:.2f}", "-vf", f"scale=-2:{height}", "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "28", str(out)], check=True)


def ask(llm, video: Path, start: float, end: float, system: str, schema, text: str, cfg: dict, log) -> tuple:
    """One look: cut [start, end) of a goal camera, send it with HIGH picture detail, return (answer, latency).

    MVP 2's calls use LOW media resolution (64 tokens a frame): fine for "what is happening", but a small
    ball and net vanish. Here detail matters (tested on dev: LOW said "no shot", HIGH said "shot + save").
    A 429 "daily quota" switches to the fallback model (its own free quota); other errors are raised."""
    from google.genai import errors

    types = llm.types
    clip = Path(tempfile.mkstemp(suffix=".mp4", prefix="goalcam-")[1])
    try:
        cut_window(video, start, end, clip, cfg["video_height"])
        parts = [types.Part(inline_data=types.Blob(data=clip.read_bytes(), mime_type="video/mp4"),
                            video_metadata=types.VideoMetadata(fps=cfg["video_fps"])), types.Part(text=text)]
        config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                             response_schema=schema, temperature=0.2,
                                             media_resolution=getattr(types.MediaResolution, cfg["media_resolution"]))
        for attempt in range(2):
            asked = time.time()
            if hasattr(llm, "_take_turn"):
                llm._take_turn()                  # share the requests between the Gemini keys
            item = {"kind": schema.__name__, "model": llm.model, "key": getattr(llm, "key_i", 0),
                    "window": [round(start, 2), round(end, 2)],
                    "requested_at": round(asked, 3)}
            try:
                r = llm.client.models.generate_content(model=llm.model, contents=[types.Content(role="user", parts=parts)],
                                                       config=config)
                item.update(latency_s=round(time.time() - asked, 2), tokens=r.usage_metadata.total_token_count)
                log(item)
                return schema.model_validate_json(r.text or "{}"), time.time() - asked
            except errors.APIError as error:
                item.update(latency_s=round(time.time() - asked, 2), error=f"{error.code}: {str(error)[:200]}")
                log(item)
                if ((error.code == 429 and "PerDay" in str(error)) or error.code == 404) and hasattr(llm, "rate_limited_key"):
                    llm._daily_quota_used(item)   # this key + model is used up today: next key, then next model
                    if not llm.daily_quota_out:
                        continue
                if error.code == 429 and hasattr(llm, "rate_limited_key") and llm.rate_limited_key(item):
                    continue                      # per-minute limit on this key: try the next key now
                if error.code in (500, 503, 504) and attempt == 0:
                    continue
                raise
            except Exception as error:            # noqa: BLE001 — network timeouts: one retry
                item.update(latency_s=round(time.time() - asked, 2), error=f"{type(error).__name__}")
                log(item)
                if attempt == 0:
                    continue
                raise
    finally:
        clip.unlink(missing_ok=True)


class GoalPlanner:
    """WHEN to look at which goal camera, and what an answer means. Shared by the offline run and live mode.

    plan(t, state) -> a job (or None): an aftermath look if one is due, else a look at the goal play is near.
    answer(job, result, logged_at) -> the look record (and it schedules aftermath looks after a shot)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_look = {"left": -999.0, "right": -999.0}
        self.aftermath: list[dict] = []            # looks to do after a shot: [{due, side, shot_time}]

    def plan(self, t: float, state) -> dict | None:
        cfg = self.cfg
        # 1. Aftermath of a shot first: what happens in the seconds after it tells a goal from a save.
        due = [a for a in self.aftermath if a["due"] <= t]
        if due:
            a = due[0]
            self.aftermath.remove(a)
            start = max(0.0, t - cfg["window_s"])
            return {"kind": "after", "side": a["side"], "start": start, "end": t, "shot_time": a["shot_time"],
                    "zone": state.zone, "system": AFTER, "schema": AfterLook,
                    "text": f"The shot was about {max(0.0, start - a['shot_time']):.0f} s before this clip starts. "
                            "What happens now?"}
        # 2. Normal look when play is near a goal.
        side = near_goal_side(state, cfg)
        triggered = any(e["type"] in ("GOAL_LINE", "BOX_ENTRY") for e in state.events)
        if side is None or t < cfg["window_s"]:
            return None
        if not triggered and t - self.last_look[side] < cfg["interval_s"]:
            return None
        if any(a["due"] - t < cfg["aftermath_priority_s"] for a in self.aftermath):
            return None                    # a follow-up of a shot is due soon: keep the line free for it
        return {"kind": "look", "side": side, "start": t - cfg["window_s"], "end": t, "zone": state.zone,
                "trigger": "map event" if triggered else "near goal", "system": SYSTEM, "schema": GoalLook,
                "text": f"Goal camera: the {side} goal. Clip = the last {cfg['window_s']:.0f} s."}

    def answer(self, job: dict, result, logged_at: float) -> dict:
        cfg, t, side = self.cfg, job["end"], job["side"]
        if job["kind"] == "after":
            return {"t": round(t, 2), "logged_at": logged_at, "side": side, "zone": job["zone"], "kind": "after",
                    "trigger": f"after the shot at {job['shot_time']:.1f}s", "goal": result.goal_scored,
                    "goal_confidence": result.confidence, "goal_time": job["shot_time"], "shot": False,
                    "shot_time": 0, "save": False, "celebration": result.attackers_celebrate,
                    "ball_visible": False, "note": result.note}
        self.last_look[side] = t
        start = job["start"]
        item = {"t": round(t, 2), "logged_at": logged_at, "side": side, "zone": job["zone"], "kind": "look",
                "trigger": job["trigger"], **result.model_dump()}
        item["goal_time"] = round(start + min(max(result.goal_at_s, 0), cfg["window_s"]), 2)
        item["shot_time"] = round(start + min(max(result.shot_at_s, 0), cfg["window_s"]), 2)
        item["goal"] = result.goal and result.goal_confidence >= cfg["min_goal_confidence"]
        if result.shot or result.save or result.goal:
            # Look again at the aftermath (twice), unless this shot is already being followed up.
            st = item["shot_time"]
            if not any(abs(a["shot_time"] - st) < 3 for a in self.aftermath):
                for delay in cfg["aftermath_s"]:
                    self.aftermath.append({"due": st + delay, "side": side, "shot_time": st})
        return item


def describe(item: dict) -> str | None:
    """One line for the console when a look found something."""
    if item["kind"] == "after" and (item["goal"] or item["celebration"]):
        return f"{item['side']:5s} AFTER: {'GOAL' if item['goal'] else 'celebration'} (conf {item['goal_confidence']:.2f}, " \
               f"known {item['logged_at']:.1f}s) {item['note'][:70]}"
    if item["kind"] == "look" and (item["goal"] or item["celebration"] or item["shot"]):
        what = "GOAL" if item["goal"] else "celebration" if item["celebration"] else "shot"
        return f"{item['side']:5s} {what} (conf {item['goal_confidence']:.2f}, known {item['logged_at']:.1f}s) {item['note'][:70]}"
    return None


def save_run(segment: str, run_id: str, model: str, looks: list[dict], errs: int, played_to: float) -> None:
    path = work_dir(segment) / "goalwatch.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append({"run_id": run_id, "model": model, "looks": len(looks), "goals": sum(1 for x in looks if x["goal"]),
                 "errors": errs, "played_to": round(played_to, 1)})
    path.write_text(json.dumps(runs, indent=2))


def run(segment: str, max_seconds: float | None = None) -> str:
    """Offline (as-if-live): each answer is stamped with its real API latency; one look at a time."""
    from story.llm.gemini import GeminiStoryLLM

    cfg = load_config()["studio"]["goalwatch"]
    meta = load_segment(segment)
    cams = load_cams(segment)
    goal_cams = {g["side"]: g["cam"] for g in cams.get("goal_cams", [])}
    if len(goal_cams) < 2:
        raise SystemExit(f"goal cameras missing: venv/bin/python -m studio.cameras --segment {segment} --only 6 7")
    fps, total = meta["fps"], meta["frames"]
    if max_seconds:
        total = min(total, round(max_seconds * fps))
    pano = PanoramaDetections(segment)
    proj = Projection(np.load(work_dir(segment) / "pitch_to_image.npy"))
    pitch = PitchMap(load_config()["studio"]["map"], proj)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = work_dir(segment) / "goalwatch" / run_id
    out.mkdir(parents=True, exist_ok=True)
    looks_file, calls_file = open(out / "looks.jsonl", "w"), open(out / "calls.jsonl", "w")
    llm = GeminiStoryLLM(story_config())
    print(f"[goalwatch] {segment}: model {llm.model}, run {run_id}", flush=True)
    planner = GoalPlanner(cfg)
    busy_until = -1.0                      # one look at a time: the next starts after the last answer arrives
    looks, errs = [], 0
    log = lambda item: calls_file.write(json.dumps(item, default=str) + "\n")   # noqa: E731
    for i in range(0, total, pano.step):
        t = i / fps
        d = pano.at(i)
        state = pitch.update(t, d["ball"][:2] if d["ball"] else None, d["feet"])
        if t < busy_until:
            continue
        job = planner.plan(t, state)
        if job is None:
            continue
        try:
            result, latency = ask(llm, cam_video(segment, goal_cams[job["side"]]), job["start"], job["end"],
                                  job["system"], job["schema"], job["text"], cfg, log)
        except Exception as error:         # noqa: BLE001 — a failed look is logged, not fatal
            errs += 1
            print(f"  {t:7.1f}s  ERROR {type(error).__name__}: {str(error)[:120]}", flush=True)
            busy_until = t + cfg["error_pause_s"]
            continue
        logged_at = round(t + latency, 2)               # where a live video would be when the answer arrives
        busy_until = logged_at
        item = planner.answer(job, result, logged_at)
        looks.append(item)
        looks_file.write(json.dumps(item) + "\n")
        looks_file.flush()
        line = describe(item)
        if line:
            print(f"  {t:7.1f}s  {line}", flush=True)
    looks_file.close()
    calls_file.close()
    save_run(segment, run_id, llm.model, looks, errs, total / fps)
    print(f"[goalwatch] done: {len(looks)} looks, {sum(1 for x in looks if x['goal'])} saying goal, {errs} errors -> {out}")
    return run_id


def studio_story_config() -> dict:
    """MVP 2's settings with MVP 3's Gemini model chain (config studio.llm) on top."""
    from studio.story_pass import deep_merge

    return deep_merge(story_config(), {"llm": load_config()["studio"].get("llm", {})})


class LiveGoalWatcher:
    """Live mode: the same GoalPlanner, but the looks really run in the background while the match plays.

    consider(t, state) is called 5x per second by the live loop; when no look is running and the planner wants
    one, it starts in a worker thread. The answer is stamped with the live clock at the moment it ARRIVES.
    It can share MVP 2's Gemini client (and so its requests-per-minute limiter) with the live story layer."""

    def __init__(self, segment: str, cams: dict, clock, on_look, publish, shared_llm=None, out_dir: Path | None = None,
                 snapshot=None, video_of=None):
        from story.llm.gemini import GeminiStoryLLM

        self.cfg = load_config()["studio"]["goalwatch"]
        self.segment, self.clock, self.on_look, self.publish = segment, clock, on_look, publish
        self.snapshot = snapshot                    # snapshot(side, video, t): show the picture the AI is checking
        # video_of(cam) -> that camera's video file (default: SoccerTrack's rendered cameras)
        self.video_of = video_of or (lambda cam: cam_video(segment, cam))
        self.goal_cams = {g["side"]: g["cam"] for g in cams.get("goal_cams", [])}
        self.llm = shared_llm or GeminiStoryLLM(studio_story_config())
        self.planner = GoalPlanner(self.cfg)
        self.lock = threading.Lock()
        self.busy_until = -1.0
        self.asking: dict | None = None             # the look in progress (for the dashboard)
        self.looks: list[dict] = []
        self.errs = 0
        self.stopped = False
        self.out_dir = out_dir
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = out_dir.name if out_dir else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.calls = open(out_dir / "calls.jsonl", "w") if out_dir else None

    def consider(self, t: float, state) -> None:
        if self.stopped or len(self.goal_cams) < 2:
            return
        with self.lock:
            if self.asking is not None or t < self.busy_until:
                return
            if not self.llm.available():           # shared requests-per-minute budget (story + goal watch)
                return
            job = self.planner.plan(t, state)
            if job is None:
                return
            self.asking = {"side": job["side"], "kind": job["kind"], "since": round(t, 2)}
        self.publish({"type": "goal_asking", **self.asking})
        threading.Thread(target=self._look, args=(job,), daemon=True, name="goalwatch-look").start()

    def _look(self, job: dict) -> None:
        log = (lambda item: self.calls.write(json.dumps(item, default=str) + "\n")) if self.calls else (lambda item: None)
        if self.snapshot:
            self.snapshot(job["side"], self.video_of(self.goal_cams[job["side"]]), job["end"])
        try:
            self.llm.limiter.record()
            result, _ = ask(self.llm, self.video_of(self.goal_cams[job["side"]]), job["start"], job["end"],
                            job["system"], job["schema"], job["text"], self.cfg, log)
        except Exception as error:                 # noqa: BLE001 — a failed look is logged, not fatal
            with self.lock:
                self.errs += 1
                self.busy_until = self.clock() + self.cfg["error_pause_s"]
                self.asking = None
            self.publish({"type": "goal_look_error", "message": f"{type(error).__name__}: {str(error)[:120]}"})
            return
        logged_at = round(self.clock(), 2)          # the moment the answer arrived (honest delay)
        with self.lock:
            item = self.planner.answer(job, result, logged_at)
            self.looks.append(item)
            self.asking = None
        self.on_look(item)
        self.publish({"type": "goal_look", **item})

    def stop(self) -> None:
        self.stopped = True
        if self.calls:
            self.calls.close()

    def save_run(self) -> None:
        if not self.out_dir:
            return
        (self.out_dir / "looks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.looks))
        save_run(self.segment, self.run_id, self.llm.model, self.looks, self.errs,
                 max((x["t"] for x in self.looks), default=0.0))


def newest_offline_run(segment: str) -> str | None:
    path = work_dir(segment) / "goalwatch.json"
    runs = [r for r in json.loads(path.read_text()) if not r["run_id"].endswith("-live")] if path.exists() else []
    return runs[-1]["run_id"] if runs else None


def load_looks(segment: str, run_id: str | None = None) -> list[dict]:
    """The looks of a goal-watch run (the newest OFFLINE one if run_id is None); [] if there is none."""
    path = work_dir(segment) / "goalwatch.json"
    if not path.exists():
        return []
    runs = json.loads(path.read_text())
    if run_id:
        runs = [r for r in runs if r["run_id"] == run_id]
    else:
        runs = [r for r in runs if not r["run_id"].endswith("-live")]   # a live session's answers belong to it
    if not runs:
        return []
    f = work_dir(segment) / "goalwatch" / runs[-1]["run_id"] / "looks.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    run(args.segment, args.max_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
