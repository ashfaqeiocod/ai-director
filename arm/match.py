"""match.py — Arm MVP Step 5: what is happening in the MATCH, from all 5 cameras, using only the past.

Per camera and per look (10 per second), from the perception cache (arm/perceive.py):
  athletes  who is who comes from the camera setup (arm/setup.py): each athlete has a "seat" (a box) in every
            camera. In each look, the person overlapping a seat best is that athlete, and the seat follows them
            slowly (they lean and move). Everybody else is audience.
  grip      the point between the two athletes' closest wrists, plus how this camera sees it:
              upright  how high the grip stands above the elbows: 1 = forearms vertical, 0 = a hand on the table
              lean     who is winning, -1 (tan) .. +1 (black). In arm wrestling (right hands) each athlete pushes
                       his hand toward HIS left, i.e. ACROSS the table. Side cameras (A, D) see that move toward
                       or away from the lens, so only cameras behind an athlete (B, C, E) measure the lean: the
                       forearms tilting to the picture's left = the athlete nearest the camera is winning.
  faces     each athlete's expression (strain: brows down, eyes squeezed, lips pressed; joy: smile)
  body      head_down (the nose at or below the shoulders: straining over the table) and gesture (the FREE hand
            raised above the shoulder: waving the crowd on, celebrating)
  audience  smiles, clapping, arms up (arm/crowd.py)
  quality   sharpness compared with the camera's own usual sharpness, and shake
Match (all cameras together, every 0.1 s): lean, upright, momentum, phase, tension, and events (see MatchTracker).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from arm.common import cache_dir, cam_letters, settings
from arm.crowd import audience_signals, crowd_value, sound_excitement

NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
SETUP, STRUGGLE, SURGE, NEAR_PIN, PIN, REACTION = "SETUP", "STRUGGLE", "SURGE", "NEAR_PIN", "PIN", "REACTION"
ATHLETES = ("black", "tan")


def load_cache(cam: str) -> dict:
    return json.loads((cache_dir() / f"cam{cam}.json").read_text())


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------- one camera, one look ----------

class Seats:
    """Where the two athletes are in one camera. Starts from the camera setup, then follows them (past only)."""

    def __init__(self, setup: dict):
        self.box = {a: list(setup[a]["seat"]) for a in ATHLETES}
        self.grip = [(setup["grip"][0] + setup["grip"][2]) / 2, (setup["grip"][1] + setup["grip"][3]) / 2]

    def assign(self, people: list[dict]) -> dict[str, int]:
        """{athlete: person index} by best overlap with the seats (each person used once)."""
        cfg = settings()["match"]
        pairs = sorted(((iou(self.box[a], p["box"]), a, k) for a in ATHLETES for k, p in enumerate(people)), reverse=True)
        found: dict[str, int] = {}
        for score, athlete, k in pairs:
            if score < cfg["seat_min_iou"]:
                break
            if athlete not in found and k not in found.values():
                found[athlete] = k
        follow = cfg["seat_follow"]
        for athlete, k in found.items():          # the seat follows the athlete a little every look
            self.box[athlete] = [s + follow * (b - s) for s, b in zip(self.box[athlete], people[k]["box"])]
        return found


def arm_points(kp: np.ndarray, wrist: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(wrist, elbow, confidence) of one arm (the elbow on the same side as the wrist)."""
    elbow = L_ELBOW if wrist == L_WRIST else R_ELBOW
    return kp[wrist, :2], kp[elbow, :2], float(min(kp[wrist, 2], kp[elbow, 2]))


def grip_wrists(kb: np.ndarray, kt: np.ndarray, scale: float) -> tuple[int, int, float] | None:
    """The two athletes' wrists that are closest together: (black wrist, tan wrist, gap / scale), or None."""
    best = None
    for wb in (L_WRIST, R_WRIST):
        for wt in (L_WRIST, R_WRIST):
            if kb[wb, 2] < 0.2 or kt[wt, 2] < 0.2:
                continue
            gap = float(np.hypot(*(kb[wb, :2] - kt[wt, :2]))) / scale
            if best is None or gap < best[2]:
                best = (wb, wt, gap)
    return best


def grip_geometry(kb: np.ndarray, kt: np.ndarray, wb: int, wt: int, near: str, side: float) -> dict:
    """upright / lean of the grip as this camera sees it (see the module docstring)."""
    cfg = settings()["match"]
    pb, eb, cb = arm_points(kb, wb)
    pt, et, ct = arm_points(kt, wt)
    g = (pb + pt) / 2
    forearm = float((np.linalg.norm(pb - eb) + np.linalg.norm(pt - et)) / 2) + 1e-6
    upright = float(np.clip(((eb[1] + et[1]) / 2 - g[1]) / forearm, -0.2, 1.2))
    # Forearm tilt in the picture: sin of the angle from vertical, + = the hand is right of the elbow.
    tilts = [(w[0] - e[0]) / (np.linalg.norm(w - e) + 1e-6) for w, e, c in ((pb, eb, cb), (pt, et, ct)) if c >= 0.3]
    tilt = float(np.mean(tilts)) if tilts else 0.0
    lean = 0.0
    if near in ATHLETES and side < 0.5:
        lean = -tilt if near == "black" else tilt          # tilting to the left = the near athlete wins
        if cfg["hand"] == "left":
            lean = -lean
    return {"x": round(float(g[0]), 1), "y": round(float(g[1]), 1), "forearm": round(forearm, 1),
            "lean": round(float(np.clip(lean, -1, 1)), 3), "upright": round(upright, 3), "conf": round(min(cb, ct), 2)}


def face_near(people: list[dict], person: dict) -> dict | None:
    """The readable face at this person's head: their own, or one found on a duplicate box of them."""
    if person.get("face"):
        return person["face"]
    kp = person["kp"]
    if kp[NOSE][2] < 0.4:
        return None
    for other in people:
        f = other.get("face")
        if f and f["box"][0] <= kp[NOSE][0] <= f["box"][2] and f["box"][1] <= kp[NOSE][1] <= f["box"][3]:
            return f
    return None


def expression(face: dict | None) -> dict | None:
    """strain / joy (0..1) of one face from its blendshapes, plus how frontal it is (1 = facing the camera)."""
    if not face:
        return None
    bs = face["bs"]
    g = lambda *names: float(np.mean([bs.get(n, 0.0) for n in names]))
    strain = 0.3 * g("browDownLeft", "browDownRight") + 0.25 * g("eyeSquintLeft", "eyeSquintRight") \
        + 0.15 * g("mouthPressLeft", "mouthPressRight") + 0.1 * g("mouthStretchLeft", "mouthStretchRight") \
        + 0.1 * g("noseSneerLeft", "noseSneerRight") + 0.1 * g("eyeBlinkLeft", "eyeBlinkRight")
    joy = 0.7 * g("mouthSmileLeft", "mouthSmileRight") + 0.3 * g("cheekSquintLeft", "cheekSquintRight")
    return {"strain": round(min(1.0, strain * 2.0), 3), "joy": round(min(1.0, joy * 1.2), 3),
            "open": round(bs.get("jawOpen", 0.0), 3), "yaw": face["yaw"],
            "frontal": round(max(0.0, float(np.cos(np.radians(face["yaw"])))), 3), "box": face["box"]}


def body_signals(kp: np.ndarray, grip_wrist: int | None) -> dict:
    """head_down: nose at or below the shoulder line. gesture: the free (non-gripping) hand above the shoulder."""
    out = {"head_down": None, "gesture": False}
    if min(kp[L_SHOULDER, 2], kp[R_SHOULDER, 2]) >= 0.4:
        shoulders_y = (kp[L_SHOULDER, 1] + kp[R_SHOULDER, 1]) / 2
        if kp[NOSE, 2] >= 0.4:
            out["head_down"] = bool(kp[NOSE, 1] >= shoulders_y - 0.05 * abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0]))
        free = R_WRIST if grip_wrist == L_WRIST else L_WRIST if grip_wrist == R_WRIST else None
        if free is not None and kp[free, 2] >= 0.5:
            out["gesture"] = bool(kp[free, 1] < min(kp[L_SHOULDER, 1], kp[R_SHOULDER, 1]))
    return out


def describe_look(look: dict, seats: Seats, cam_setup: dict, sharp_ref: float) -> dict:
    """Everything one camera tells about one moment (see the module docstring)."""
    cfg = settings()["match"]
    people = look["people"]
    side = 1.0 if cam_setup["view"] == "side" else 0.0
    out = {"t": look["t"], "n": look["n"], "grip": None, "black": None, "tan": None,
           "sharp": round(look["sharp"] / sharp_ref, 3) if sharp_ref else 1.0, "shake": look["shake"]}
    found = seats.assign(people)
    kps = {a: np.array(people[k]["kp"]) for a, k in found.items()}
    grip_wrist = {}
    if len(found) == 2:
        scale = min(people[k]["box"][3] - people[k]["box"][1] for k in found.values())
        pair = grip_wrists(kps["black"], kps["tan"], scale)
        if pair and pair[2] <= cfg["grip_max_gap"]:
            grip_wrist = {"black": pair[0], "tan": pair[1]}
            out["grip"] = grip_geometry(kps["black"], kps["tan"], pair[0], pair[1], cam_setup["nearest"], side)
            seats.grip = [out["grip"]["x"], out["grip"]["y"]]
    out["grip_point"] = [round(v, 1) for v in seats.grip]        # measured now, or the last known place
    for athlete, k in found.items():
        p = people[k]
        out[athlete] = {"box": p["box"], "face": expression(face_near(people, p)),
                        **body_signals(kps[athlete], grip_wrist.get(athlete))}
    athletes = set(found.values())
    out["audience"] = audience_signals(people, athletes)
    out["audience_faces"] = [{"box": p["face"]["box"], "joy": expression(p["face"])["joy"]}
                             for k, p in enumerate(people) if k not in athletes and p.get("face")]
    out["occluded"] = occlusion(people, athletes, out["grip_point"], found)
    return out


def occlusion(people: list[dict], athletes: set[int], grip: list[float], found: dict[str, int]) -> bool:
    """True if a non-athlete stands in front of the grip (their box covers it and they are nearer = taller)."""
    if not found:
        return False
    athlete_h = max(people[k]["box"][3] - people[k]["box"][1] for k in found.values())
    for k, p in enumerate(people):
        if k in athletes:
            continue
        x1, y1, x2, y2 = p["box"]
        if x1 <= grip[0] <= x2 and y1 <= grip[1] <= y2 and (y2 - y1) > 1.05 * athlete_h:
            return True
    return False


def describe_camera(cam: str, setup: dict) -> list[dict]:
    """All looks of one camera, in order (each look uses only itself and the looks before it)."""
    cache = load_cache(cam)
    sharp_values = [f["sharp"] for f in cache["frames"]]
    # Sharpness reference: the median of the first 5 seconds only (a live system knows only the past).
    ref = float(np.median(sharp_values[:50])) if sharp_values else 1.0
    seats = Seats(setup[cam])
    return [describe_look(frame, seats, setup[cam], ref) for frame in cache["frames"]]


# ---------- all cameras together ----------

@dataclass
class MatchState:
    """The match at one moment (every 0.1 s), from the past only."""
    t: float
    n: int
    phase: str = SETUP
    lean: float = 0.0                 # + = black is winning, - = tan is winning
    upright: float = 1.0
    momentum: float = 0.0             # change of lean / upright per second (smoothed)
    leader: str = ""                  # "black" / "tan" / "" (even)
    gripping: bool = False
    tension: float = 0.0
    strain: float = 0.0               # athletes' strain (the stronger of the two, best readable face)
    joy: float = 0.0                  # athletes' joy
    head_down: dict = field(default_factory=dict)    # athlete -> straining over the table?
    gesture: dict = field(default_factory=dict)      # athlete -> free hand raised?
    crowd: float = 0.0
    sound: float = 0.0
    winner: str = ""
    events: list = field(default_factory=list)       # what just happened: "SURGE black", "GESTURE black", "CHEER"
    reason: str = ""


@dataclass
class MatchTracker:
    """Turns the per-camera looks into one MatchState per 0.1 s, step by step (past only).

    Phases: SETUP (no grip) -> STRUGGLE -> SURGE (the grip moves fast) / NEAR_PIN (close to the table)
            -> PIN (a hand down: someone won) -> REACTION.
    Events (moments for the director, they do not change the phase):
            GESTURE <athlete>  his free hand goes up        LOOK_UP <athlete>  he lifts his head (face visible)
            CHEER              the crowd suddenly gets louder / happier
    """
    cfg: dict
    lean_s: float = 0.0
    upright_s: float = 1.0
    momentum_s: float = 0.0
    grip_since: float | None = None
    phase: str = SETUP
    phase_since: float = 0.0
    winner: str = ""
    who: str = ""
    last_seen_grip: float = -99.0
    votes: dict = field(default_factory=dict)        # (athlete, signal) -> recent True/False values
    crowd_s: float = 0.0
    last_event_t: dict = field(default_factory=dict)

    def step(self, t: float, n: int, looks: dict[str, dict], sound: float) -> MatchState:
        c = self.cfg
        st = MatchState(t=t, n=n, sound=sound)
        leans, lean_w, ups, up_w = [], [], [], []
        for look in looks.values():
            g = look["grip"]
            if g is None or look["occluded"]:
                continue
            ups.append(g["upright"]); up_w.append(g["conf"] + 1e-3)
            if g["lean"] != 0.0:
                leans.append(g["lean"]); lean_w.append(g["conf"] + 1e-3)
        dt = 1.0 / settings()["perceive"]["analysis_fps"]
        alpha = min(1.0, dt / c["smooth_s"])
        if ups:
            self.last_seen_grip = t
        st.gripping = t - self.last_seen_grip <= c["release_s"]
        if ups:
            prev_lean, prev_up = self.lean_s, self.upright_s
            if leans:
                self.lean_s += alpha * (float(np.average(leans, weights=lean_w)) - self.lean_s)
            self.upright_s += alpha * (float(np.average(ups, weights=up_w)) - self.upright_s)
            change = max(abs(self.lean_s - prev_lean), abs(self.upright_s - prev_up)) / dt
            signed = (self.lean_s - prev_lean) / dt
            self.momentum_s += alpha * ((change if abs(signed) < 1e-9 else np.sign(signed) * change) - self.momentum_s)
        else:
            self.momentum_s *= 1 - alpha
        if st.gripping and self.grip_since is None:
            self.grip_since = t
        if not st.gripping:
            self.grip_since = None
        st.lean, st.upright, st.momentum = round(self.lean_s, 3), round(self.upright_s, 3), round(self.momentum_s, 3)
        st.leader = "black" if st.lean > c["lead_min"] else "tan" if st.lean < -c["lead_min"] else ""

        faces = [look[a]["face"] for look in looks.values() for a in ATHLETES if look[a] and look[a]["face"]]
        st.strain = round(max((f["strain"] * f["frontal"] for f in faces), default=0.0), 3)
        st.joy = round(max((f["joy"] * f["frontal"] for f in faces), default=0.0), 3)
        for a in ATHLETES:
            st.head_down[a] = self._vote(a, "head_down", [look[a]["head_down"] for look in looks.values()
                                                          if look[a] and look[a]["head_down"] is not None])
            # A raised hand seen clearly by ONE camera is real (the others are often blocked): any camera counts.
            st.gesture[a] = self._vote(a, "gesture", [any(look[a]["gesture"] for look in looks.values() if look[a])])
        aud = [look["audience"] for look in looks.values()]
        smile = [x["smile"] for x in aud if x["smile"] is not None]
        clap = [x["clap"] for x in aud if x["clap"] is not None]
        arms = [x["arms_up"] for x in aud if x["arms_up"] is not None]
        st.crowd = round(crowd_value(sound, float(np.mean(smile)) if smile else None,
                                     float(np.max(clap)) if clap else None, float(np.max(arms)) if arms else None), 3)
        self._phase(st)
        self._events(st)
        struggle_s = t - self.grip_since if self.grip_since is not None else 0.0
        closeness = min(1.0, max(abs(st.lean) / c["near_pin_lean"], max(0.0, 1 - st.upright) / (1 - c["pin_upright"])))
        effort = max(st.strain, 0.6 if any(st.head_down.values()) else 0.0)
        tw = c["tension_weights"]
        st.tension = round(min(1.0, tw["strain"] * effort + tw["duration"] * min(1.0, struggle_s / c["long_struggle_s"])
                               + tw["closeness"] * closeness + tw["crowd"] * st.crowd
                               + (tw["surge"] if st.phase in (SURGE, NEAR_PIN, PIN) else 0.0)), 3)
        st.winner = self.winner
        return st

    def _vote(self, athlete: str, signal: str, values: list[bool]) -> bool:
        """A body signal counts once most cameras that see it agree, over the last few looks (fewer flickers)."""
        memory = self.votes.setdefault((athlete, signal), [])
        if values:
            memory.append(sum(values) / len(values))
        del memory[:-self.cfg["vote_looks"]]
        return bool(memory) and float(np.mean(memory)) >= 0.5

    def _events(self, st: MatchState) -> None:
        """Moments: gestures, looking up, a cheer. Each kind at most once per `event_gap_s`."""
        c, t = self.cfg, st.t
        prev = getattr(self, "_prev", None)
        if prev is not None:
            for a in ATHLETES:
                if st.gesture[a] and not prev.gesture.get(a):
                    self._event(st, f"GESTURE {a}", t)
                if prev.head_down.get(a) and not st.head_down[a]:
                    self._event(st, f"LOOK_UP {a}", t)
        jump = st.crowd - self.crowd_s
        self.crowd_s += min(1.0, 0.1 / c["crowd_baseline_s"]) * (st.crowd - self.crowd_s)
        if jump >= c["cheer_jump"]:
            self._event(st, "CHEER", t)
        self._prev = st

    def _event(self, st: MatchState, name: str, t: float) -> None:
        if t - self.last_event_t.get(name, -99) >= self.cfg["event_gap_s"]:
            st.events.append(name)
            self.last_event_t[name] = t

    def _phase(self, st: MatchState) -> None:
        """The phase state machine. Every change records a plain-English reason."""
        c, t = self.cfg, st.t
        old = self.phase
        held = t - self.phase_since
        pinned = st.gripping and (st.upright <= c["pin_upright"] or abs(st.lean) >= c["pin_lean"])
        near = st.gripping and (st.upright <= c["near_pin_upright"] or abs(st.lean) >= c["near_pin_lean"])
        surging = st.gripping and abs(st.momentum) >= c["surge_momentum"]
        if self.phase in (PIN, REACTION):
            if self.phase == PIN and held >= c["pin_hold_s"]:
                self._go(REACTION, t, st, f"{self.winner} won: reaction", self.winner)
            elif self.phase == REACTION and st.gripping and held >= c["reaction_min_s"] and st.upright > 0.6:
                self.winner = ""
                self._go(STRUGGLE, t, st, "a new grip: the match goes on")
        elif not st.gripping:
            if self.phase != SETUP:
                self._go(SETUP, t, st, "no grip seen")
        elif pinned and held >= c["pin_confirm_s"] and self.phase == NEAR_PIN:
            self.winner = st.leader or ("black" if st.lean > 0 else "tan")
            self._go(PIN, t, st, f"PIN {self.winner} (upright {st.upright:.2f}, lean {st.lean:+.2f})", self.winner)
        elif near and self.phase != NEAR_PIN:
            self._go(NEAR_PIN, t, st, f"{st.leader or 'someone'} close to a pin (lean {st.lean:+.2f}, upright {st.upright:.2f})",
                     st.leader)
        elif surging and self.phase == STRUGGLE:
            who = "black" if st.momentum > 0 else "tan"
            self._go(SURGE, t, st, f"{who} pushes (grip moving {abs(st.momentum):.2f}/s)", who)
        elif self.phase == SETUP:
            self._go(STRUGGLE, t, st, "grip locked: the struggle is on")
        elif self.phase in (SURGE, NEAR_PIN) and not near and not surging and held >= c["surge_min_s"]:
            self._go(STRUGGLE, t, st, "back to an even struggle")
        st.phase = self.phase
        if self.phase != old:
            st.events.append(f"{self.phase} {self.who}".strip())

    def _go(self, phase: str, t: float, st: MatchState, reason: str, who: str = "") -> None:
        self.phase, self.phase_since, self.who = phase, t, who
        st.reason = reason


def build_timeline() -> dict:
    """All cameras' looks + the match state every 0.1 s. Saved to data/arm/cache/match.json."""
    from arm.setup import load_setup

    setup = load_setup()
    cams = cam_letters()
    per_cam = {c: describe_camera(c, setup) for c in cams}
    count = min(len(v) for v in per_cam.values())
    duration = per_cam[cams[0]][count - 1]["t"] + 0.1
    sound_rows = sound_excitement(duration)
    tracker = MatchTracker(settings()["match"])
    states = []
    for k in range(count):
        t = per_cam[cams[0]][k]["t"]
        heard = [r["sound"] for r in sound_rows if r["t"] <= t + 1e-6]       # only what was already heard
        states.append(tracker.step(t, per_cam[cams[0]][k]["n"], {c: per_cam[c][k] for c in cams},
                                   heard[-1] if heard else 0.0).__dict__)
    data = {"cams": {c: per_cam[c][:count] for c in cams}, "states": states, "sound": sound_rows, "setup": setup}
    (cache_dir() / "match.json").write_text(json.dumps(data, separators=(",", ":")))
    return data


def load_timeline() -> dict:
    return json.loads((cache_dir() / "match.json").read_text())


def main() -> int:
    data = build_timeline()
    for s in data["states"][::5]:
        hd = "".join(a[0].upper() if s["head_down"][a] else "." for a in ATHLETES)
        print(f"{s['t']:5.1f}  {s['phase']:<9} lean {s['lean']:+.2f} up {s['upright']:.2f} mom {s['momentum']:+.2f} "
              f"strain {s['strain']:.2f} joy {s['joy']:.2f} down {hd} crowd {s['crowd']:.2f} tension {s['tension']:.2f}  "
              f"{s['reason']}")
    for cam in cam_letters():
        looks = data["cams"][cam]
        print(f"cam {cam}: grip measured {sum(1 for l in looks if l['grip']) / len(looks):.0%}, both athletes found "
              f"{sum(1 for l in looks if l['black'] and l['tan']) / len(looks):.0%}, occluded {sum(1 for l in looks if l['occluded'])}")
    print("Events:", [(s["t"], e) for s in data["states"] for e in s["events"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
