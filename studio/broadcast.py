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
from studio.stadium import Stadium, director_summary, view_towards

SEGMENT = "alfheim-2013-11-03-H1"


def settings() -> dict:
    return load_config()["studio"]["broadcast"]


# ---------------------------------------------------------------------------------------------------------------
# 1. The twin as a time series
# ---------------------------------------------------------------------------------------------------------------
class Twin:
    """twin.jsonl rows (one per 0.1 s) for a stretch of the match; `at(t)` = the row at or before t."""

    def __init__(self, folder: Path, start_s: float, end_s: float):
        self.rows = []
        with open(folder / "twin.jsonl") as f:
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
            pitch = pitch_map.update(t, (b["x"], b["y"]) if b else None, [p[:2] for p in people(row)])
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
    data = {"segment": segment, "start_s": start_s, "seconds": seconds, "fps": fps, "frames": n,
            "home_attacks": home_dir, "teams": ["Tromsø IL", "Strømsgodset"],
            "cameras": [{"id": c.id, "name": c.name, "kind": c.kind, "position_m": list(c.position_m)}
                        for c in stadium.cameras],
            "paths": paths, "on_air": on_air, "states": states, "cuts": cuts,
            "credit": "Alfheim dataset (Simula), non-commercial research"}
    (out / "broadcast.json").write_text(json.dumps(data, separators=(",", ":")))
    summary = {"segment": segment, "start_s": start_s, "seconds": seconds, "run": run_id,
               **director_summary(airtime, [c for c in cuts], n, fps, {"director": cfg["director"]}),
               "fastest_deg_s": {cid: round(max(abs(np.diff(paths[cid]["pan"])).max(initial=0),
                                                abs(np.diff(paths[cid]["tilt"])).max(initial=0)) * fps, 1)
                                 for cid in stadium.heads},
               "seconds_taken": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Virtual Broadcast: camera paths + director decisions.")
    ap.add_argument("--segment", default=SEGMENT)
    ap.add_argument("--start", default="23:40")
    ap.add_argument("--seconds", type=float, default=300)
    args = ap.parse_args()
    run(args.segment, parse_time(args.start), args.seconds)


if __name__ == "__main__":
    main()
