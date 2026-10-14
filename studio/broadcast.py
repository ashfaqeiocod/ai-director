"""broadcast.py — Virtual Broadcast: the director and camera operators of a fully virtual TV production.

Why (docs/VIRTUAL_BROADCAST_PLAN.md): MVP 4 cut between real Alfheim pictures and a 3D scene; they don't match, and
the real pictures can't zoom. Here EVERY camera is virtual: a camera plan like a real TV crew's (config.yaml ->
studio.broadcast.cameras: MAIN and TIGHT at halfway high in the main stand, HIGH 18 L/R at the box edges, GOAL L/R
behind the goals, WIDE), all rendered in 3D from the digital twin (studio.zxy) by the page
dashboard/static/broadcast.html. This module decides, for every frame:
  - where each camera looks and how wide (its "operator": framing rules, smooth fluid-head motion);
  - which camera is on air (DirectorV2 in tv mode, the MVP 4 rules).
It writes the camera paths + decisions as one JSON file the page plays back (preview) or renders (1080p video).

Where the game information comes from: the twin (home team = ZXY sensors, away team + referees + ball = our
detections) fed into MVP 3's pitch map 5 times a second: ball zone, box entries, ball out, kick-off line-up,
intensity. Possession = the team of the player nearest the ball; attacking direction = from the kick-off line-up.
Everything is causal except what the twin already smoothed (within the broadcast delay, see studio.zxy).

Usage:
  venv/bin/python -m studio.broadcast --start 23:40 --seconds 300
Writes outputs/broadcast/<segment>/<run>/broadcast.json, director_log.csv, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common import ROOT, load_config
from studio.director import DirectorV2
from studio.panorama import parse_time
from studio.pitchmap import HALF_X, MapState, PitchMap, Projection
from studio.stadium import Stadium, view_towards

SEGMENT = "alfheim-2013-11-03-H1"


def settings() -> dict:
    return load_config()["studio"]["broadcast"]


# ---------------------------------------------------------------------------------------------------------------
# 1. The twin as a time series
# ---------------------------------------------------------------------------------------------------------------
class Twin:
    """The rebuilt game (play.jsonl from studio.play, else the raw twin.jsonl), one row per 0.1 s, for a stretch
    of the match; `at(t)` = the row at or before t."""

    def __init__(self, folder: Path, start_s: float, end_s: float):
        self.rows = []
        path = folder / "play.jsonl"
        if not path.exists():
            path = folder / "twin.jsonl"
        self.kind = path.stem
        with open(path) as f:
            for line in f:
                t = float(line[5:line.index(",")])          # '{"t":123.4,...' : cheap peek at the time
                if t < start_s - 30:
                    continue
                if t > end_s + 1:
                    break
                self.rows.append(json.loads(line))
        self.step = self.rows[1]["t"] - self.rows[0]["t"]
        self.t0 = self.rows[0]["t"]

    def at(self, t: float) -> dict:
        k = int((t - self.t0) / self.step + 1e-6)
        return self.rows[max(0, min(len(self.rows) - 1, k))]


def people(row: dict) -> list[tuple[float, float, str]]:
    """Everyone on the pitch in a twin row: (x, y, team) with team home / away (referees left out)."""
    out = [(p["x"], p["y"], "home") for p in row["home"] if abs(p["x"]) <= 53 and abs(p["y"]) <= 34.6]
    return out + [(p["x"], p["y"], "away") for p in row["away"]]


def home_attacks(folder: Path) -> int:
    """+1 if the home team attacks the right goal (x > 0) in this half, -1 if the left: at the kick-off (first
    seconds of the twin) each team stands in its own half."""
    xs = []
    with open(folder / "twin.jsonl") as f:
        for k, line in enumerate(f):
            if k > 50:
                break
            row = json.loads(line)
            xs += [p["x"] for p in row["home"] if abs(p["y"]) <= 34]
    return 1 if np.mean(xs) < 0 else -1


# ---------------------------------------------------------------------------------------------------------------
# 2. The camera operators
# ---------------------------------------------------------------------------------------------------------------
class Operators:
    """What each operator aims at (the heads then move there smoothly, like fluid heads):
      MAIN      the ball with room ahead of play (the attacking direction of the team on the ball), wide enough
                for the ball carrier's nearest team-mates and opponents; wider at restarts, tighter in the box
      TIGHT     the player on the ball (kept until another player clearly has it), body in the middle of the shot
      HIGH 18   its own half's play when the ball is there, else its penalty box
      GOAL L/R  between the ball and its goal when play comes near, else the box"""

    def __init__(self, cfg: dict, stadium: Stadium, home_dir: int):
        self.cfg = cfg["operators"]
        self.st = stadium
        self.home_dir = home_dir
        self.carrier: tuple[float, float] | None = None
        self.carrier_team: str | None = None
        self.challenger_since: float | None = None
        self.last_ball: tuple[float, tuple[float, float]] | None = None
        self.velocity = np.zeros(2)

    def update(self, t: float, row: dict, pitch: MapState | None, state: str) -> None:
        o = self.cfg
        heads = self.st.heads
        ball = (row["ball"]["x"], row["ball"]["y"]) if row["ball"] else None
        if ball is None and pitch is not None and pitch.ball is not None and pitch.ball_status != "lost":
            ball = pitch.ball
        everyone = people(row)
        if ball is not None:
            if self.last_ball is not None and t > self.last_ball[0]:
                v = (np.array(ball) - self.last_ball[1]) / (t - self.last_ball[0])
                self.velocity = 0.7 * self.velocity + 0.3 * v
            self.last_ball = (t, np.array(ball, float))
        self._carrier(t, ball, everyone)
        focus = ball if ball is not None else self.carrier or self._group(everyone)
        if focus is None:
            return
        # MAIN: room ahead of play
        direction = 0.0
        if self.carrier_team:
            direction = self.home_dir if self.carrier_team == "home" else -self.home_dir
        lead = np.array([direction * o["lead_m"], 0.0]) + np.clip(self.velocity * o["lead_s"], -o["lead_max_m"], o["lead_max_m"])
        aim = np.array(focus) + lead
        aim[0] = np.clip(aim[0], -HALF_X + 8, HALF_X - 8)          # don't frame the empty stand behind a goal
        aim[1] = np.clip(aim[1], -30, 30)
        near = sorted(everyone, key=lambda p: math.dist(p[:2], focus))[:o["main_players"]]
        spread = max((abs(p[0] - focus[0]) for p in near), default=0.0)
        width = max(o["main_open_m"], 2 * spread * o["main_margin"])
        width = {"ATTACK": min(width, o["main_attack_m"]), "RESTART": o["main_restart_m"]}.get(state, width)
        heads["main"].aim(*aim)
        heads["main"].zoom(width)
        # TIGHT: the player on the ball, his body in the middle (aim a bit beyond his feet)
        target = self.carrier or focus
        heads["tight"].aim(*self._body(heads["tight"], target))
        heads["tight"].zoom(o["tight_m"] if state != "STOPPAGE" else o["tight_stoppage_m"])
        # HIGH 18 L / R and GOAL L / R: their own half
        for side, sign in (("l", -1), ("r", 1)):
            box = (sign * (HALF_X - 11), 0.0)
            mine = focus[0] * sign > o["half_from_m"]
            high, goal = heads[f"high18{side}"], heads[f"goal{side}"]
            if mine:
                high.aim(*focus)
                high.zoom(o["high18_m"])
                goal_point = (sign * HALF_X, 0.0)
                share = o["goal_aim_share"]                 # aim at the play, a little towards the goal
                goal.aim(focus[0] + (goal_point[0] - focus[0]) * share, focus[1] + (goal_point[1] - focus[1]) * share)
                goal.zoom(min(o["goal_max_m"], max(o["goal_min_m"], math.dist(focus, goal_point) * 1.1)))
            else:
                high.aim(*box)
                high.zoom(o["idle_m"])
                goal.aim(*box)
                goal.zoom(o["idle_m"])

    def _carrier(self, t: float, ball, everyone) -> None:
        """The player on the ball, kept until another player has been clearly nearer for a moment."""
        o = self.cfg
        if ball is None or not everyone:
            return
        best = min(everyone, key=lambda p: math.dist(p[:2], ball))
        if math.dist(best[:2], ball) > o["carrier_m"]:
            return                                              # a pass in flight: keep the last carrier
        if self.carrier is None:
            self.carrier, self.carrier_team = best[:2], best[2]
            return
        current = min(everyone, key=lambda p: math.dist(p[:2], self.carrier))   # follow the same player
        if math.dist(best[:2], ball) + o["carrier_switch_m"] < math.dist(current[:2], ball):
            if self.challenger_since is None:
                self.challenger_since = t
            if t - self.challenger_since >= o["carrier_switch_s"]:
                current, self.challenger_since = best, None
        else:
            self.challenger_since = None
        self.carrier, self.carrier_team = current[:2], current[2]

    @staticmethod
    def _group(everyone) -> tuple[float, float] | None:
        if not everyone:
            return None
        pts = np.array([p[:2] for p in everyone])
        return tuple(np.median(pts, axis=0))

    @staticmethod
    def _body(head, point) -> tuple[float, float]:
        """The ground point behind a player where the ray through his chest (1 m up) lands: aiming there puts
        his body, not his feet, in the middle of a tight shot."""
        c = head.centre
        height = -c[2]
        scale = height / max(height - 1.0, 0.5)
        return c[0] + (point[0] - c[0]) * scale, c[1] + (point[1] - c[1]) * scale


# ---------------------------------------------------------------------------------------------------------------
# 3. A run: camera paths + decisions for a stretch of the match
# ---------------------------------------------------------------------------------------------------------------
def run(segment: str, start_s: float, seconds: float) -> Path:
    from studio.sources import folder_of

    cfg = settings()
    alf = load_config()["studio"]
    folder = folder_of(segment)
    fps = cfg["fps"]
    twin = Twin(folder, start_s, start_s + seconds)
    stadium = Stadium({**cfg, "output_size": [1920, 1080]}, None)
    home_dir = home_attacks(folder)
    ops = Operators(cfg, stadium, home_dir)
    pitch_map = PitchMap(alf["map"], Projection(np.eye(3)))
    director = DirectorV2({**alf["director"], **cfg["director"]}, fps, len(stadium.cameras))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[broadcast] {segment} {int(start_s // 60)}:{int(start_s % 60):02d} +{seconds:.0f}s -> {out.relative_to(ROOT)}",
          flush=True)
    # warm up: the map and the heads start a few seconds earlier (a crew is already filming)
    warm = cfg["warmup_s"]
    n = round(seconds * fps)
    first = round(start_s * fps)
    every = round(fps / 5)                                      # the map gets news 5 times a second
    paths = {c.id: {"pan": [], "tilt": [], "hfov": []} for c in stadium.cameras}
    on_air, states, cuts, airtime = [], [], [], {}
    dlog = open(out / "director_log.csv", "w")
    dlog.write("frame,time,cam,cam_name,state,reason\n")
    pitch = None
    cards = None
    t0 = time.time()
    for k in range(-round(warm * fps), n):
        i = first + k
        t = i / fps
        row = twin.at(t)
        if k % every == 0:
            b = row["ball"]
            seen = b is not None and not b.get("dead")          # play.jsonl: the ball out of play = no ball on the map
            pitch = pitch_map.update(t, (b["x"], b["y"]) if seen else None, [p[:2] for p in people(row)])
            ops.update(t, row, pitch, director.state)
            cards = None
        if k == -round(warm * fps):                             # start already framing the play
            for head in stadium.heads.values():
                head.view = view_towards(head.centre, *head.target_look, head.target_width)
        stadium.step(1 / fps)
        shots = stadium.shots()
        if cards is None:
            cards = stadium.cards(t, shots, pitch, None)
        for card, cam in zip(cards, stadium.cameras):
            card.moving_deg_s = round(shots[cam.id].state.get("speed_deg_s", 0.0), 2)
        if k == 0:
            director.start_at(i)
        decision = director.decide(i, t, cards, [], [], None, pitch=pitch)
        if k < 0:
            continue
        for cam in stadium.cameras:
            st = shots[cam.id].state
            paths[cam.id]["pan"].append(round(st["pan"], 3))
            paths[cam.id]["tilt"].append(round(st["tilt"], 3))
            paths[cam.id]["hfov"].append(round(st["hfov"], 3))
        cam = stadium.cameras[decision.cam - 1]
        on_air.append(decision.cam - 1)
        states.append(decision.state)
        airtime[cam.id] = airtime.get(cam.id, 0) + 1
        for e in decision.events:
            if e["type"] == "CUT":
                cuts.append({**e, "frame": k, "t": round(t, 2), "to_id": cam.id, "reason": decision.reason[5:]})
        reason = decision.reason.replace('"', "'")
        dlog.write(f'{i},{t:.3f},{decision.cam},{cam.name},{decision.state},"{reason}"\n')
    dlog.close()
    ids = [c.id for c in stadium.cameras]
    src, replays = add_replays(on_air, states, ids, cfg["replays"], fps, cfg["director"]["min_shot_seconds"])
    enforce_min_shot(on_air, src, round(cfg["director"]["min_shot_seconds"] * fps))
    for r in replays:
        cuts.append({"type": "REPLAY", "frame": r["at"], "t": round(start_s + r["at"] / fps, 2), "to_id": ids[r["cam"]],
                     "reason": f"replay of the attack at {fmt_time(start_s + r['from'] / fps)} from {stadium.cameras[r['cam']].name}"})
    cuts.sort(key=lambda c: c["frame"])
    data = {"segment": segment, "start_s": start_s, "seconds": seconds, "fps": fps, "frames": n,
            "src": src, "replays": replays,
            "home_attacks": home_dir, "teams": ["Tromsø IL", "Strømsgodset"], "twin": twin.kind,
            "cameras": [{"id": c.id, "name": c.name, "kind": c.kind, "position_m": list(c.position_m)}
                        for c in stadium.cameras],
            "paths": paths, "on_air": on_air, "states": states, "cuts": cuts,
            "credit": "Alfheim dataset (Simula), non-commercial research"}
    (out / "broadcast.json").write_text(json.dumps(data, separators=(",", ":")))
    summary = {"segment": segment, "start_s": start_s, "seconds": seconds, "run": run_id,
               **program_summary(on_air, src, ids, fps, cfg["director"]["min_shot_seconds"]),
               "replays": len(replays),
               "fastest_deg_s": {cid: round(max(abs(np.diff(paths[cid]["pan"])).max(initial=0),
                                                abs(np.diff(paths[cid]["tilt"])).max(initial=0)) * fps, 1)
                                 for cid in stadium.heads},
               "seconds_taken": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    crowd_sound(data, out / "crowd.wav", cfg["sound"])
    print(json.dumps(summary, indent=1))
    return out


# ---------------------------------------------------------------------------------------------------------------
# 4. Sound: a crowd that reacts (Alfheim has no sound)
# ---------------------------------------------------------------------------------------------------------------
def crowd_envelope(data: dict, cfg: dict) -> np.ndarray:
    """How excited the crowd is, per program frame (0..1): from the director's state (an attack in a box is loud),
    a roar when an attack ends in a stoppage (a shot or a near miss), quieter during replays."""
    fps = data["fps"]
    level = {"ATTACK": cfg["attack"], "STOPPAGE": cfg["stoppage"], "RESTART": cfg["restart"]}
    raw = np.array([level.get(s, cfg["play"]) for s in data["states"]], np.float64)
    for k in range(1, len(raw)):                         # roars: attack -> stoppage
        if data["states"][k] in ("STOPPAGE", "RESTART") and data["states"][k - 1] == "ATTACK":
            n = round(cfg["roar_s"] * fps)
            raw[k:k + n] = np.maximum(raw[k:k + n], cfg["roar"] * np.exp(-np.arange(min(n, len(raw) - k)) / (n / 3)))
    for r in data.get("replays", []):
        raw[r["at"]:r["to"]] = np.minimum(raw[r["at"]:r["to"]], cfg["replay"])
    out = np.empty_like(raw)                             # smooth: a crowd swells and calms over a second or two
    a = 1 - math.exp(-1 / (cfg["smooth_s"] * fps))
    v = raw[0]
    for k, x in enumerate(raw):
        v += a * (x - v)
        out[k] = v
    return out


def crowd_sound(data: dict, path: Path, cfg: dict) -> None:
    """A stadium crowd as a WAV file: a murmur (filtered noise, slowly drifting) + a brighter cheering layer that
    follows the excitement. Made in one-minute blocks, so a whole half fits in memory."""
    import wave

    from scipy.signal import butter, sosfilt, sosfilt_zi

    rate, fps = cfg["rate"], data["fps"]
    env = crowd_envelope(data, cfg)
    total = round((len(env) / fps + 1.0) * rate)        # 1 s longer than the picture: muxing cuts at the picture
    murmur = butter(4, [120, 1800], btype="band", fs=rate, output="sos")
    cheer = butter(4, [700, 4500], btype="band", fs=rate, output="sos")
    zm, zc = sosfilt_zi(murmur) * 0, sosfilt_zi(cheer) * 0
    rng = np.random.default_rng(7)
    drift = 0.0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        for a in range(0, total, rate * 60):
            n = min(rate * 60, total - a)
            t = (a + np.arange(n)) / rate
            e = np.interp(t * fps, np.arange(len(env)), env)
            m, zm = sosfilt(murmur, rng.standard_normal(n), zi=zm)
            c, zc = sosfilt(cheer, rng.standard_normal(n), zi=zc)
            # slow random drift of the murmur (a crowd is never perfectly steady)
            steps = np.cumsum(rng.standard_normal(n // rate + 2)) * 0.05
            drift_curve = np.interp(np.arange(n) / rate, np.arange(len(steps)), steps + drift)
            drift = float(drift_curve[-1])
            mono = m * cfg["murmur_gain"] * (0.8 + 0.2 * np.tanh(drift_curve)) * (0.5 + e) + c * cfg["cheer_gain"] * e ** 2
            # stereo: two slightly different mixes so it sounds wide
            left = mono + 0.15 * c * e
            right = mono - 0.15 * c * e
            # a soft limiter (tanh): loud roars get rounder instead of clipping
            pcm = (np.tanh(np.stack([left, right], axis=1) * cfg["volume"]) * 32000).astype("<i2")
            w.writeframes(pcm.tobytes())


def add_sound(video: Path, wav: Path, fps: float = 30) -> None:
    """program.mp4 + crowd.wav -> program.mp4 with sound (the picture is copied, not re-encoded)."""
    import subprocess

    from src.common import ffmpeg_exe

    tmp = video.with_name("program_sound.part.mp4")
    # the picture is also re-timed to exactly 30 fps (the browser's raw stream carries no usable timing)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-bsf:v", f"setts=ts=N/({fps}*TB)",
                    "-video_track_timescale", str(round(fps * 1000)), "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(video)


# ---------------------------------------------------------------------------------------------------------------
# 5. Rendering the 1080p program (headless Chrome renders the page; the dashboard server stores the video)
# ---------------------------------------------------------------------------------------------------------------
def render(segment: str, run_id: str, server: str, seconds: float) -> bool:
    """program.mp4 in segments of `segment_s`, each rendered by headless Chrome in a FRESH tab (one tab rendering a
    whole half ran out of memory on the 8 GB Mac after ~20 min). A segment whose tab dies or stalls is rendered
    again; at the end the segments are joined (not re-encoded)."""
    import shutil
    import subprocess
    import tempfile
    import urllib.request

    from src.common import ffmpeg_exe

    cfg = settings()
    rc = cfg["render"]
    out = ROOT / cfg["outputs_dir"] / segment / run_id
    try:
        urllib.request.urlopen(f"{server}/studio/broadcast/runs", timeout=10).read()
    except Exception as e:  # noqa: BLE001
        print(f"[broadcast] no dashboard at {server} ({e}). Start it: venv/bin/python -m dashboard.server", flush=True)
        return False
    chrome = load_config()["studio"]["stadium"]["program"]["chrome"]
    flags = load_config()["studio"]["stadium"]["program"]["chrome_flags"]
    fps = cfg["fps"]
    frames = round(seconds * fps)
    seg = round(rc["segment_s"] * fps)
    mbps = rc["mbps_short"] if seconds <= rc["short_s"] else rc["mbps_long"]
    parts = [(k, a, min(frames, a + seg)) for k, a in enumerate(range(0, frames, seg))]
    print(f"[broadcast] rendering program.mp4 (1080p) in {len(parts)} segment(s), headless Chrome...", flush=True)
    t0 = time.time()
    for k, a, b in parts:
        done = out / f"program_part{k:03d}.mp4"
        for attempt in range(rc["retries"] + 1):
            if done.exists():
                break
            raw = out / f"program_part{k:03d}.h264"
            raw.unlink(missing_ok=True)
            profile = tempfile.mkdtemp(prefix="broadcast-chrome-")
            url = (f"{server}/studio/broadcast?segment={segment}&run={run_id}&render=1&mbps={mbps}"
                   f"&from={a}&to={b}&part={k}")
            proc = subprocess.Popen([chrome, "--headless=new", *flags, "--window-size=1700,1200",
                                     f"--user-data-dir={profile}", "--remote-debugging-port=0", url],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            last_size, last_change = -1, time.time()
            try:
                while not done.exists():
                    time.sleep(3)
                    size = raw.stat().st_size if raw.exists() else 0
                    if size != last_size:
                        last_size, last_change = size, time.time()
                    if proc.poll() is not None or time.time() - last_change > rc["stall_s"]:
                        print(f"  segment {k}: the tab died or stalled (try {attempt + 1})", flush=True)
                        break
            finally:
                proc.kill()
                shutil.rmtree(profile, ignore_errors=True)
        if not done.exists():
            print(f"[broadcast] segment {k} failed {rc['retries'] + 1} times: stopping", flush=True)
            return False
        print(f"  segment {k + 1}/{len(parts)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # join the segments (same encoder settings: copied, not re-encoded), then delete them
    listing = out / "program_parts.txt"
    listing.write_text("".join(f"file 'program_part{k:03d}.mp4'\n" for k, _, _ in parts))
    joined = out / "program.join.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(joined)], check=True)
    for k, _, _ in parts:
        (out / f"program_part{k:03d}.mp4").unlink()
    listing.unlink()
    joined.replace(out / "program.mp4")
    print(f"[broadcast] program.mp4 rendered in {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


def enforce_min_shot(on_air: list[int], src: list[float], min_frames: int) -> None:
    """A last safety pass after the replays: a LIVE shot shorter than the minimum (a replay was put over the start of
    a director's shot) is merged into the live shot before it (or after it, if a replay comes before). Every camera
    has a path at every moment, so any camera can simply stay on air longer."""
    n = len(on_air)
    replay = [abs(src[i] - i) > 0.5 for i in range(n)]
    changed = True
    while changed:
        changed = False
        shots, k = [], 0
        while k < n:
            j = k
            while j < n and on_air[j] == on_air[k] and replay[j] == replay[k]:
                j += 1
            shots.append((k, j))
            k = j
        for idx, (a, b) in enumerate(shots[:-1]):
            if replay[a] or b - a >= min_frames:
                continue
            before = shots[idx - 1] if idx > 0 else None
            after = shots[idx + 1]
            if before is not None and not replay[before[0]]:
                cam = on_air[before[0]]
            elif not replay[after[0]]:
                cam = on_air[after[0]]
            else:
                continue                                # between two replays: leave it (very rare)
            for i in range(a, b):
                on_air[i] = cam
            changed = True
            break


def program_summary(on_air: list[int], src: list[float], ids: list[str], fps: float, min_shot: float) -> dict:
    """Airtime per camera and shot lengths of the FINAL program (replays included: a replay is its own shot)."""
    shots, k, n = [], 0, len(on_air)
    replay = [abs(src[i] - i) > 0.5 for i in range(n)]
    while k < n:
        j = k
        while j < n and on_air[j] == on_air[k] and replay[j] == replay[k]:
            j += 1
        shots.append((j - k) / fps)
        k = j
    air: dict[str, int] = {}
    for c in on_air:
        air[ids[c]] = air.get(ids[c], 0) + 1
    done = shots[:-1]                                   # the last shot is still running when the clip ends
    return {"airtime_pct": {c: round(100 * v / n, 1) for c, v in sorted(air.items(), key=lambda x: -x[1])},
            "cuts": len(shots) - 1, "avg_shot_s": round(n / fps / len(shots), 2),
            "shots_under_min": sum(1 for d in done if d < min_shot - 1e-6),
            "shortest_shot_s": round(min(done), 2) if done else None}


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def add_replays(on_air: list[int], states: list[str], ids: list[str], cfg: dict, fps: float,
                min_shot: float) -> tuple[list, list]:
    """TV replays: when an attack ends and play stops (STOPPAGE / RESTART), the attack is shown again in slow motion
    from a behind-goal camera (the HIGH 18 one if the goal camera already showed it live), starting
    `delay_s` after the stoppage. Live play goes on meanwhile and is not shown (as on TV).
    Changes `on_air` in place; returns (src = the match frame each program frame shows, the replays)."""
    n = len(on_air)
    src: list[float] = list(range(n))
    replays: list[dict] = []
    attack_start = attack_end = None
    k = 0
    while k < n:
        if states[k] == "ATTACK" and (k == 0 or states[k - 1] != "ATTACK"):
            attack_start = k
        if states[k] != "ATTACK" and k > 0 and states[k - 1] == "ATTACK":
            attack_end = k
        stopped = states[k] in ("STOPPAGE", "RESTART") and k > 0 and states[k - 1] not in ("STOPPAGE", "RESTART")
        if stopped and attack_end is not None and (k - attack_end) <= cfg["after_attack_s"] * fps \
                and (not replays or k - replays[-1]["to"] > cfg["gap_s"] * fps):
            a = max(attack_start - round(cfg["before_s"] * fps), 0)
            b = min(k + round(cfg["after_s"] * fps), n - 1)
            a = max(a, b - round(cfg["max_source_s"] * fps))
            # which camera: the goal camera of the attacked side, unless it was on air for most of the attack live
            side = next((ids[c][-1] for c in on_air[attack_start:attack_end] if ids[c][:4] in ("goal", "high")), "l")
            goal, high = ids.index(f"goal{side}"), ids.index(f"high18{side}")
            live_goal = sum(1 for c in on_air[attack_start:attack_end] if c == goal) / max(1, attack_end - attack_start)
            cam = high if live_goal > 0.5 else goal
            hold = round(min_shot * fps)
            at = k + round(cfg["delay_s"] * fps)
            shot_start = at                                  # the shot running when the replay would start ...
            while shot_start > 0 and on_air[shot_start - 1] == on_air[min(at, n - 1)]:
                shot_start -= 1
            at = max(at, shot_start + hold)                  # ... is never cut short
            length = round((b - a) / cfg["speed"])
            if at + length + hold < n:
                for j in range(length):
                    on_air[at + j] = cam
                    src[at + j] = round(a + j * cfg["speed"], 2)
                back = on_air[at + length]                  # back to live: hold that shot long enough too
                for j in range(at + length, at + length + hold):
                    on_air[j] = back
                replays.append({"at": at, "to": at + length, "from": a, "until": b, "cam": cam, "speed": cfg["speed"]})
                attack_end = None
                k = at + length
                continue
        k += 1
    return src, replays


def main() -> None:
    ap = argparse.ArgumentParser(description="Virtual Broadcast: camera paths + director decisions.")
    ap.add_argument("--segment", default=SEGMENT)
    ap.add_argument("--start", default="23:40")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--render", action="store_true", help="also render program.mp4 (1080p, with crowd sound)")
    ap.add_argument("--server", default=None, help="dashboard address (default: config studio.stadium.program.server)")
    args = ap.parse_args()
    out = run(args.segment, parse_time(args.start), args.seconds)
    if args.render:
        server = args.server or load_config()["studio"]["stadium"]["program"]["server"]
        if render(args.segment, out.name, server, args.seconds):
            add_sound(out / "program.mp4", out / "crowd.wav")
            print(f"[broadcast] done: {(out / 'program.mp4').relative_to(ROOT)} (with crowd sound)", flush=True)


if __name__ == "__main__":
    main()
