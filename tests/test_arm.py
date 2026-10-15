"""Arm MVP: audio sync, the director's rules, the offline editor's constraints, the Gemini budget.

All synthetic: no footage, no network.
"""
import json

import numpy as np
import pytest

from arm.common import settings
from arm.director import LiveDirector, allowed
from arm.editor import plan
from arm.shots import Shot
from arm.sync import gcc_phat, solve_offsets


# ---------- sync ----------

def test_gcc_phat_finds_a_known_shift():
    rate = 8000
    rng = np.random.default_rng(1)
    room = rng.normal(size=rate * 6).astype(np.float32)          # 6 s of "room sound"
    shift = 0.4375                                              # seconds: the sound comes this much later in b
    a = room[:rate * 5]
    b = np.concatenate([rng.normal(size=int(shift * rate)).astype(np.float32) * 0.3, room])[:rate * 5]
    lag, confidence = gcc_phat(a, b, rate, center_s=0.0, search_s=1.0)
    assert abs(lag - shift) < 1 / rate
    assert confidence > 3


def test_solve_offsets_is_consistent():
    truth = {"A": 0.0, "B": -0.088, "C": 0.127, "D": 1.01}
    pairs = {(i, j): (truth[j] - truth[i] + noise, 5.0)
             for (i, j), noise in zip([("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
                                      [0.001, -0.001, 0.0, 0.002, -0.001, 0.0])}
    offsets = solve_offsets(pairs, list(truth))
    for cam, value in truth.items():
        assert abs(offsets[cam] - value) < 0.003


# ---------- director rules ----------

def shot(cam, kind="TWO", h=1080.0, value=50.0, clarity=1.0):
    return Shot(cam, kind, (0.0, 0.0, h * 16 / 9, h), value, {"clarity": clarity, "event": 0.0, "emotion": 0.5})


def test_180_degree_rule_blocks_a_to_d():
    cfg = settings()["director"]
    assert allowed(shot("A"), shot("D"), cfg) is not None           # opposite sides of the table
    assert allowed(shot("A"), shot("E"), cfg) is None
    assert allowed(shot("E"), shot("D"), cfg) is None


def test_no_jump_cuts_within_one_camera():
    cfg = settings()["director"]
    assert allowed(shot("A", "TWO", 1080), shot("A", "GRIP", 900), cfg) is not None   # only 17% tighter
    assert allowed(shot("A", "TWO", 1080), shot("A", "GRIP", 400), cfg) is None       # a real zoom: MOVE


def test_live_director_holds_the_minimum_and_needs_hysteresis():
    d = LiveDirector()
    phase = "STRUGGLE"
    min_s = d.min_shot(phase)
    on_air = []
    for k in range(80):
        t = k * 0.1
        better_e = 90.0 if t >= 0.3 else 10.0                        # E becomes much better at 0.3 s
        shots = [shot("A", "WIDE", value=60.0), shot("E", "TWO", value=better_e)]
        on_air.append(d.step(t, k * 3, shots, phase, {}).shot.cam)
    first_cut = on_air.index("E") * 0.1
    assert first_cut >= min_s - 1e-9                                 # the opening shot was held long enough
    assert on_air[-1] == "E"


def test_live_director_never_airs_an_unusable_shot():
    d = LiveDirector()
    for k in range(40):
        shots = [shot("A", value=40.0), shot("E", value=99.0, clarity=0.1)]   # E is blurry / blocked
        assert d.step(k * 0.1, k * 3, shots, "STRUGGLE", {}).shot.cam == "A"


# ---------- offline editor ----------

def fake_timeline(seconds=12.0):
    """Looks with no athletes (only WIDE shots per camera), values changing over time."""
    cams = list(settings()["cameras"])
    n = int(seconds * 10)
    looks = {c: [{"t": k / 10, "n": 3 * k, "grip": None, "black": None, "tan": None, "sharp": 1.0, "shake": 0.0,
                  "grip_point": [960, 540], "audience": {}, "audience_faces": [], "occluded": False}
                 for k in range(n)] for c in cams}
    states = [{"t": k / 10, "n": 3 * k, "phase": "STRUGGLE", "tension": 0.5, "crowd": 0.2 + 0.6 * ((k // 30) % 2),
               "head_down": {}, "events": [], "lean": 0.0} for k in range(n)]
    setup = {c: {"view": "side" if settings()["cameras"][c]["side"] != "neutral" else "wide"} for c in cams}
    return {"cams": looks, "states": states, "setup": setup}


def test_editor_respects_minimum_length_and_the_line():
    decisions = plan(fake_timeline())
    starts = [i for i, d in enumerate(decisions) if i == 0 or d.shot.key != decisions[i - 1].shot.key]
    lengths = np.diff(starts + [len(decisions)]) / 10
    min_s = settings()["director"]["min_shot_s"]["STRUGGLE"]
    assert all(length >= min_s - 1e-9 for length in lengths[:-1])
    used = {d.shot.cam for d in decisions}
    assert not {"A", "D"} <= used                                      # one side of the table only


# ---------- Gemini budget ----------

def test_gemini_budget_is_a_hard_cap(tmp_path, monkeypatch):
    import arm.gemini as gem
    monkeypatch.setattr(gem, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(gem, "_calls_file", lambda: tmp_path / "gemini_calls.json")
    (tmp_path / "gemini_calls.json").write_text(json.dumps({"count": settings()["gemini"]["max_total_calls"], "log": []}))
    with pytest.raises(gem.BudgetExceeded):
        gem.ask("test", [b"not really a jpeg"], "hello", dict)
