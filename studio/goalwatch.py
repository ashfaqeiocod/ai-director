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
            item = {"kind": schema.__name__, "model": llm.model, "window": [round(start, 2), round(end, 2)],
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
                fallback = llm.cfg.get("fallback_model")
                if error.code == 429 and "PerDay" in str(error) and fallback and llm.model != fallback:
                    llm.model = fallback          # daily quota of this model used up: same key, other model
                    continue
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


def run(segment: str, max_seconds: float | None = None) -> str:
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

    busy_until = -1.0                      # one look at a time: the next starts after the last answer arrives
    last_look = {"left": -999.0, "right": -999.0}
    aftermath: list[dict] = []             # looks to do after a shot: [{due, side, shot_time}]
    looks = goals = errs = 0
    log = lambda item: calls_file.write(json.dumps(item, default=str) + "\n")   # noqa: E731

    def record(item: dict) -> None:
        looks_file.write(json.dumps(item) + "\n")
        looks_file.flush()

    for i in range(0, total, pano.step):
        t = i / fps
        d = pano.at(i)
        state = pitch.update(t, d["ball"][:2] if d["ball"] else None, d["feet"])
        if t < busy_until:
            continue
        # 1. Aftermath of a shot first: what happens in the seconds after it tells a goal from a save.
        due = [a for a in aftermath if a["due"] <= t]
        if due:
            a = due[0]
            aftermath.remove(a)
            start = max(0.0, t - cfg["window_s"])
            try:
                after, latency = ask(llm, cam_video(segment, goal_cams[a["side"]]), start, t, AFTER, AfterLook,
                                     f"The shot was about {max(0.0, start - a['shot_time']):.0f} s before this clip "
                                     "starts. What happens now?", cfg, log)
            except Exception as error:     # noqa: BLE001
                errs += 1
                print(f"  {t:7.1f}s  ERROR {type(error).__name__}: {str(error)[:120]}", flush=True)
                busy_until = t + cfg["error_pause_s"]
                continue
            logged_at = round(t + latency, 2)
            busy_until = logged_at
            looks += 1
            goal = after.goal_scored and after.confidence >= cfg["min_goal_confidence"]
            goals += goal
            record({"t": round(t, 2), "logged_at": logged_at, "side": a["side"], "zone": state.zone, "kind": "after",
                    "trigger": f"after the shot at {a['shot_time']:.1f}s", "goal": after.goal_scored,
                    "goal_confidence": after.confidence, "goal_time": a["shot_time"], "shot": False, "shot_time": 0,
                    "save": False, "celebration": after.attackers_celebrate, "ball_visible": False, "note": after.note})
            if after.goal_scored or after.attackers_celebrate:
                print(f"  {t:7.1f}s  {a['side']:5s} AFTER: {'GOAL' if after.goal_scored else 'celebration'} "
                      f"(conf {after.confidence:.2f}, known {logged_at:.1f}s) {after.note[:70]}", flush=True)
            continue
        # 2. Normal look when play is near a goal.
        side = near_goal_side(state, cfg)
        triggered = any(e["type"] in ("GOAL_LINE", "BOX_ENTRY") for e in state.events)
        if side is None or t < cfg["window_s"]:
            continue
        if not triggered and t - last_look[side] < cfg["interval_s"]:
            continue
        if any(a["due"] - t < cfg["aftermath_priority_s"] for a in aftermath):
            continue                       # a follow-up of a shot is due soon: keep the line free for it
        start = t - cfg["window_s"]
        try:
            look, latency = ask(llm, cam_video(segment, goal_cams[side]), start, t, SYSTEM, GoalLook,
                                f"Goal camera: the {side} goal. Clip = the last {cfg['window_s']:.0f} s.", cfg, log)
        except Exception as error:         # noqa: BLE001 — a failed look is logged, not fatal
            errs += 1
            print(f"  {t:7.1f}s  ERROR {type(error).__name__}: {str(error)[:120]}", flush=True)
            busy_until = t + cfg["error_pause_s"]
            continue
        logged_at = round(t + latency, 2)               # where a live video would be when the answer arrives
        busy_until, last_look[side] = logged_at, t
        looks += 1
        item = {"t": round(t, 2), "logged_at": logged_at, "side": side, "zone": state.zone, "kind": "look",
                "trigger": "map event" if triggered else "near goal", **look.model_dump()}
        item["goal_time"] = round(start + min(max(look.goal_at_s, 0), cfg["window_s"]), 2)
        item["shot_time"] = round(start + min(max(look.shot_at_s, 0), cfg["window_s"]), 2)
        item["goal"] = look.goal and look.goal_confidence >= cfg["min_goal_confidence"]
        record(item)
        if look.shot or look.save or look.goal:
            # Look again at the aftermath (twice), unless this shot is already being followed up.
            st = item["shot_time"]
            if not any(abs(a["shot_time"] - st) < 3 for a in aftermath):
                for delay in cfg["aftermath_s"]:
                    aftermath.append({"due": st + delay, "side": side, "shot_time": st})
        if look.goal or look.celebration or look.shot:
            goals += item["goal"]
            what = "GOAL" if item["goal"] else "celebration" if look.celebration else "shot"
            print(f"  {t:7.1f}s  {side:5s} {what} (conf {look.goal_confidence:.2f}, known {logged_at:.1f}s) {look.note[:70]}",
                  flush=True)
    looks_file.close()
    calls_file.close()
    path = work_dir(segment) / "goalwatch.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append({"run_id": run_id, "model": llm.model, "looks": looks, "goals": goals, "errors": errs,
                 "played_to": round(total / fps, 1)})
    path.write_text(json.dumps(runs, indent=2))
    print(f"[goalwatch] done: {looks} looks, {goals} saying goal, {errs} errors -> {out}")
    return run_id


def load_looks(segment: str, run_id: str | None = None) -> list[dict]:
    """The looks of a goal-watch run (the newest if run_id is None); [] if there is none."""
    path = work_dir(segment) / "goalwatch.json"
    if not path.exists():
        return []
    runs = json.loads(path.read_text())
    if run_id:
        runs = [r for r in runs if r["run_id"] == run_id]
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
