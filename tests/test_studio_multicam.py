"""ISSIA-style footage: 6 real calibrated cameras. Director roles from config, players merged across cameras,
pitch <-> picture mapping, and the stand-B "turn" of a calibration."""
import copy

import numpy as np
import pytest

from src.common import load_config
from studio.cards import BallInfo, CamCard, Quality
from studio.director import DirectorV2
from studio.issia import TURNS, local_homography, turned
from studio.multicam import merge_points, to_pitch, to_picture

FPS = 25


def issia_director_cfg() -> dict:
    cfg = copy.deepcopy(load_config()["studio"])
    cfg["director"].update(cfg["issia"]["director"])
    return cfg["director"]


def six_cards(ball_cam: int | None) -> list[CamCard]:
    return [CamCard(cam=c, name=f"c{c}", time=0, logged_at=0, shot_type="medium",
                    ball=BallInfo(0.5, 0.5, "detected") if c == ball_cam else None, players=4, play_speed=0.9,
                    quality=Quality(usable=True)) for c in range(1, 7)]


def test_every_issia_camera_follows_the_ball_no_wide_or_close():
    cfg = issia_director_cfg()
    for cam in (4, 5, 6):                      # on SoccerTrack these numbers mean WIDE / CLOSE: not here
        d = DirectorV2(cfg, FPS, 6)
        for i in range(FPS * 6):
            decision = d.decide(i, i / FPS, six_cards(cam), [], [], None)
        assert decision.cam == cam
        assert decision.scores[cam - 1] >= 100  # MVP 1 scoring: ball seen


def test_reaction_without_a_close_up_uses_the_best_camera():
    cfg = issia_director_cfg()
    d = DirectorV2(cfg, FPS, 6)
    for i in range(FPS * 6):
        d.decide(i, i / FPS, six_cards(5), [], [], None)
    goal = {"id": "g", "type": "card_red", "event_time": 6.0, "known_at": 6.0, "confidence": 0.9, "sources": ["llm"]}
    decision = d.decide(FPS * 6, 6.0, six_cards(5), [goal], [], None)
    assert decision.state == "REACTION" and decision.cam == 5


def test_same_player_seen_by_two_cameras_is_one_dot():
    merged = merge_points({1: [(40.0, 0.0), (45.0, 5.0)], 2: [(40.5, 0.3), (10.0, 10.0)]}, radius=1.5)
    assert len(merged) == 3
    assert any(abs(x - 40.25) < 0.01 for x, _ in merged)


def test_players_of_one_camera_are_never_merged():
    assert len(merge_points({1: [(0.0, 0.0), (0.5, 0.0)]}, radius=1.5)) == 2


def test_pitch_to_picture_round_trip_and_turns():
    # a made-up camera: 4 landmarks around the right box
    pts = [[36, -20.16, 1000, 200], [36, 20.16, 1020, 700], [52.5, -20.16, 1720, 200], [52.5, 20.16, 1900, 700]]
    H = local_homography(pts)
    back = to_pitch(np.linalg.inv(H), [(1000.0, 200.0)])[0]
    assert abs(back[0] - 36) < 0.01 and abs(back[1] + 20.16) < 0.01
    assert to_picture(H, (36, -20.16)) == pytest.approx((1000, 200), abs=0.5)
    # a camera on the other stand with a rotated picture sees the same pixel at the opposite pitch spot
    rot = turned(H, "rotated")
    assert TURNS["rotated"] == (-1, -1)
    x, y = to_pitch(np.linalg.inv(rot), [(1000.0, 200.0)])[0]
    assert abs(x + 36) < 0.01 and abs(y - 20.16) < 0.01

