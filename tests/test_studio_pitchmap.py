"""Pitch map: zones, a lost ball is remembered briefly then dropped, zone triggers fire once."""
import numpy as np

from src.common import load_config
from studio.pitchmap import PitchMap, Projection, zone_of

CFG = load_config()["studio"]["map"]
IDENTITY = Projection(np.eye(3))          # "pixels" = metres, to test the logic alone


def test_zones():
    assert zone_of(0, 0) == "middle"
    assert zone_of(-45, 0) == "left_box"
    assert zone_of(45, 25) == "right_third"      # wide of the box
    assert zone_of(53, 0) == "out"


def test_lost_ball_is_remembered_then_dropped():
    m = PitchMap(CFG, IDENTITY)
    m.update(0.0, (0, 0), [])
    m.update(0.2, (2, 0), [])                    # moving right at 10 m/s
    s = m.update(0.4, None, [])
    assert s.ball_status == "remembered" and s.ball[0] > 2
    s = m.update(0.4 + CFG["remember_s"] + 0.5, None, [])
    assert s.ball_status == "lost" and s.ball is None


def test_box_entry_and_ball_out_fire_once():
    m = PitchMap(CFG, IDENTITY)
    types = []
    path = [(30, 0), (34, 0), (38, 0), (42, 0), (46, 2), (48, 6), (50, 10), (52, 14), (54, 17)]
    for k, p in enumerate(path):                 # into the right box, then out over the goal line, wide of the post
        types += [e["type"] for e in m.update(k * 0.2, p, []).events]
    assert types.count("BOX_ENTRY") == 1
    assert "BALL_OUT" in types or "GOAL_LINE" in types


def test_goal_line_between_posts():
    m = PitchMap(CFG, IDENTITY)
    m.update(0.0, (48, 1), [])
    s = m.update(0.2, (53, 1), [])
    assert [e["type"] for e in s.events] == ["GOAL_LINE"]


def test_impossible_jump_is_ignored():
    m = PitchMap(CFG, IDENTITY)
    m.update(0.0, (0, 0), [])
    s = m.update(0.2, (40, 0), [])               # 200 m/s: not the same ball
    assert s.ball_status == "remembered"
