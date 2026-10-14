"""director.py — Arm MVP Step 7: the LIVE director. Every 0.1 s it picks the shot to put on air, using only the past.

Like a TV director it watches all shot values (arm/shots.py) and cuts when another shot is clearly better, but
it follows broadcast rules:
  - hold every shot long enough to be read: the minimum depends on the phase (calm struggle: longer shots,
    a surge or a pin: quick cuts are fine). A fresh event (a look up, a gesture, a cheer) may cut sooner.
  - hysteresis: the challenger must beat the shot on air by `margin` for `hysteresis_s` in a row.
  - variety: a shot loses value after it has been on air for `max_shot_s` (the viewer has seen it), and a shot
    that was on air a moment ago is worth less again (no ping-pong).
  - 180-degree rule: cameras A and D film from opposite sides of the table: cutting straight from one to the
    other swaps the athletes left/right and confuses the viewer. Go through another camera first.
  - never cut to a blurry, shaking or blocked shot (its clarity is in its value already, plus a floor).
  - transitions: a hard CUT on action and events, a short DISSOLVE in calm moments, and a MOVE (the robotic
    camera reframes) when the next shot is the same camera with a clearly different zoom; a same-camera change
    smaller than `min_zoom_change` would look like a jump cut and is not allowed.
Every decision carries a plain-English reason (director_log.csv).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from arm.common import settings
from arm.shots import Shot, all_shots, event_boosts

CUT, DISSOLVE, MOVE = "CUT", "DISSOLVE", "MOVE"


@dataclass
class Decision:
    t: float
    n: int
    shot: Shot
    change: str = ""          # "", CUT, DISSOLVE or MOVE
    reason: str = ""


def zoom_change(a: Shot, b: Shot) -> float:
    """How different two framings of the same camera are: |1 - height ratio| (0 = the same size)."""
    return abs(1 - min(a.rect[3], b.rect[3]) / max(a.rect[3], b.rect[3]))


def allowed(current: Shot | None, candidate: Shot, cfg: dict) -> str | None:
    """None if we may switch from `current` to `candidate`, else the rule that forbids it."""
    if current is None:
        return None
    if candidate.cam == current.cam:
        if candidate.kind == current.kind:
            return "same shot"
        if zoom_change(current, candidate) < cfg["min_zoom_change"]:
            return "jump cut (same camera, similar zoom)"
        return None
    sides = settings()["cameras"]
    a, b = sides[current.cam]["side"], sides[candidate.cam]["side"]
    if {a, b} == {"north", "south"}:
        return "180-degree rule (opposite sides of the table)"
    return None


def transition(current: Shot, nxt: Shot, phase: str, by_event: bool, cfg: dict) -> str:
    if current.cam == nxt.cam:
        return MOVE
    if by_event or phase in cfg["cut_phases"]:
        return CUT
    return DISSOLVE


@dataclass
class LiveDirector:
    """Picks one shot per look from the shot values, using only what is known at that moment."""
    cfg: dict = field(default_factory=lambda: settings()["director"])
    current: Shot | None = None
    since: float = 0.0
    challenger: str | None = None
    challenger_since: float = 0.0
    aired: dict = field(default_factory=dict)        # shot key -> last time it was on air
    line: str | None = None                          # the side of the table we film from, once chosen
    side_values: dict = field(default_factory=dict)  # side -> values of its cameras' shots so far

    def choose_line(self, t: float, shots: list[Shot]) -> None:
        """The 180-degree line: watch both side cameras for `line_choice_s`, then keep the better side for good
        (the opposite side's camera is not used again). A side camera is there for the athletes' faces, so the
        side whose two-shot shows more readable emotion wins. Uses only the past."""
        sides = settings()["cameras"]
        for s in shots:
            side = sides[s.cam]["side"]
            if side in ("north", "south") and s.kind == "TWO":
                self.side_values.setdefault(side, []).append(s.parts["emotion"])
        if self.line is None and t >= self.cfg["line_choice_s"] and len(self.side_values) == 2:
            self.line = max(self.side_values, key=lambda side: sum(self.side_values[side]) / len(self.side_values[side]))

    def min_shot(self, phase: str) -> float:
        return self.cfg["min_shot_s"].get(phase, self.cfg["min_shot_s"]["default"])

    def max_shot(self, phase: str, kind: str = "") -> float:
        """How long a shot stays fresh: per phase, and shorter for cut-aways (crowd, a face, the grip)."""
        by_phase = self.cfg["max_shot_s"].get(phase, self.cfg["max_shot_s"]["default"])
        return min(by_phase, self.cfg["cutaway_max_s"].get(kind, by_phase))

    def adjusted(self, shot: Shot, t: float, phase: str) -> float:
        """The shot's value for THIS director: staleness on the shot on air, a penalty on recent repeats."""
        c = self.cfg
        v = shot.value
        if self.current is not None and shot.key == self.current.key:
            over = t - self.since - self.max_shot(phase, shot.kind)
            if over > 0:
                v *= max(c["stale_floor"], 1 - c["stale_per_s"] * over)
        elif t - self.aired.get(shot.key, -99) < c["repeat_window_s"]:
            v *= c["repeat_factor"]
        return v

    def step(self, t: float, n: int, shots: list[Shot], phase: str, boosts: dict[str, float]) -> Decision:
        c = self.cfg
        self.choose_line(t, shots)
        sides = settings()["cameras"]
        usable = [s for s in shots if s.parts["clarity"] >= c["min_clarity"]
                  and (self.line is None or sides[s.cam]["side"] in ("neutral", self.line))]
        if self.current is None:      # first look: like TV, open on the best establishing (wide) shot
            wides = [s for s in usable if s.kind == "WIDE"] or usable or shots
            best = max(wides, key=lambda s: s.value)
            return self._take(t, n, best, "", f"open on the establishing shot {best.key} ({best.value:.0f})")
        now = {s.key: s for s in shots}
        on_air = now.get(self.current.key)
        if on_air is not None:
            self.current = on_air                             # same shot, new framing (the operator follows)
        cur_value = self.adjusted(on_air, t, phase) if on_air else 0.0
        held = t - self.since
        ranked = sorted(((self.adjusted(s, t, phase), s) for s in usable if s.key != self.current.key),
                        key=lambda vs: -vs[0])
        for value, best in ranked:
            if allowed(self.current, best, c) is None:
                break
        else:
            return Decision(t, n, self.current)
        # Never cut away in the middle of the decisive action: hold a side shot of the hands through the PIN,
        # the cut comes after it (to the reaction).
        if phase == "PIN" and on_air is not None and on_air.kind in ("TWO", "GRIP") \
                and settings()["cameras"][on_air.cam]["side"] != "neutral" and held < c["pin_hold_s"] + self.min_shot(phase):
            return Decision(t, n, self.current)
        lost = on_air is None or on_air.parts["clarity"] < c["min_clarity"]
        by_event = boosts.get(best.kind, 0.0) >= c["event_cut_boost"]
        if lost and held < c["lost_grace_s"]:
            return Decision(t, n, self.current)               # hold the last framing through a short gap
        if lost:
            why = "the shot on air can't be used any more" if on_air else "the shot on air is gone (face turned away)"
            return self._switch(t, n, best, phase, by_event, f"{why} -> {best.key} ({value:.0f})")
        if value < cur_value * (1 + c["margin"]):
            self.challenger = None
            return Decision(t, n, self.current)
        if best.key != self.challenger:
            self.challenger, self.challenger_since = best.key, t
        if by_event and held >= c["event_min_shot_s"]:
            return self._switch(t, n, best, phase, True,
                                f"event: {best.key} shows it ({value:.0f} vs {cur_value:.0f})")
        if held >= self.min_shot(phase) and t - self.challenger_since >= c["hysteresis_s"]:
            stale = " (shot on air is getting old)" if held > self.max_shot(phase, self.current.kind) else ""
            return self._switch(t, n, best, phase, by_event,
                                f"{best.key} better for {c['hysteresis_s']} s ({value:.0f} vs {cur_value:.0f}){stale}")
        return Decision(t, n, self.current)

    def _switch(self, t: float, n: int, shot: Shot, phase: str, by_event: bool, reason: str) -> Decision:
        change = transition(self.current, shot, phase, by_event, self.cfg)
        self.aired[self.current.key] = t
        return self._take(t, n, shot, change, reason)

    def _take(self, t: float, n: int, shot: Shot, change: str, reason: str) -> Decision:
        self.current, self.since, self.challenger = shot, t, None
        self.aired[shot.key] = t
        return Decision(t, n, shot, change, reason)


def judge_boost(boosts: dict[str, float], answers: list[dict], t: float, live: bool) -> dict[str, float]:
    """Add Gemini's "best camera now" (if a fresh answer is known at t) as a small bonus for that camera."""
    from arm.judge import known_at

    answer = known_at(answers, t, live)
    if answer:
        boosts = {**boosts, f"CAM:{answer['best_camera']}": settings()["judge"]["best_camera_bonus"]}
    return boosts


def run_live(timeline: dict, answers: list[dict] | None = None) -> list[Decision]:
    """The live director over the whole match: one decision per look (10 per second)."""
    setup = timeline["setup"]
    cams = list(timeline["cams"])
    director = LiveDirector()
    events: list[tuple[float, str]] = []
    decisions = []
    for k, state in enumerate(timeline["states"]):
        events += [(state["t"], e) for e in state["events"]]        # events are known when they happen
        boosts = judge_boost(event_boosts(events, state["t"]), answers or [], state["t"], live=True)
        looks = {c: timeline["cams"][c][k] for c in cams}
        decisions.append(director.step(state["t"], state["n"], all_shots(looks, state, setup, boosts),
                                       state["phase"], boosts))
    return decisions
