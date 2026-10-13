"""Director v2: shot length, hysteresis, bad-feed escape, reactions only from what the story layer knew."""
import json

from src.common import load_config
from studio.cards import BallInfo, CamCard, Quality, StoryFeed
from studio.director import CLOSE, IN_PLAY, REACTION, WIDE, DirectorV2

FPS = 25
CFG = load_config()["studio"]["director"]


def cards(ball_cam=None, bad=(), speed=0.3, action_cam=None):
    out = []
    for cam, shot in zip(range(1, 6), ["medium", "medium", "medium", "wide", "close_up"]):
        ball = BallInfo(0.5, 0.5, "detected") if cam in (ball_cam, 4) and ball_cam else None
        out.append(CamCard(cam=cam, name=f"c{cam}", time=0, logged_at=0, shot_type=shot, ball=ball, players=5,
                           play_speed=speed, quality=Quality(usable=cam not in bad),
                           action_in_view=cam in (action_cam, 4, 5),
                           action=(0.5, 0.5) if cam in (action_cam, 4, 5) else None))
    return out


def run(director, frames, make, events=(), updates=(), phase=None):
    chosen = []
    for i in range(frames):
        chosen.append(director.decide(i, i / FPS, make(i), list(events), list(updates), phase).cam)
    return chosen


def shots(chosen):
    out, run_len = [], 1
    for a, b in zip(chosen, chosen[1:]):
        if a == b:
            run_len += 1
        else:
            out.append(run_len)
            run_len = 1
    return out


def test_never_cuts_faster_than_min_shot():
    d = DirectorV2(CFG, FPS)
    # The ball hops between cams 1 and 3 every 1.2 s: longer than the hysteresis, shorter than the min shot.
    chosen = run(d, 25 * 40, lambda i: cards(ball_cam=1 if (i // 30) % 2 else 3))
    assert len(shots(chosen)) >= 3                      # it does cut...
    assert min(shots(chosen)) >= CFG["min_shot_seconds"] * FPS   # ...but never too fast


def test_follows_the_ball_after_hysteresis():
    d = DirectorV2(CFG, FPS)
    chosen = run(d, 25 * 10, lambda i: cards(ball_cam=1))
    assert chosen[-1] == 1


def test_bad_feed_is_left_at_once_even_mid_shot():
    d = DirectorV2(CFG, FPS)
    run(d, 25 * 10, lambda i: cards(ball_cam=2))
    assert d.current == 2
    t0 = 25 * 10
    # cam2 goes bad 0.2 s after a fresh cut would have been impossible: must still leave immediately.
    decision = d.decide(t0, t0 / FPS, cards(ball_cam=2, bad=(2,)), [], [], None)
    assert decision.cam != 2 and "unusable" in decision.reason


def test_bad_feed_never_chosen():
    d = DirectorV2(CFG, FPS)
    chosen = run(d, 25 * 30, lambda i: cards(ball_cam=3, bad=(3,)))
    assert 3 not in chosen


def goal_event(t, conf=0.9):
    return {"id": "e1", "type": "goal", "event_time": t, "confidence": conf, "sources": ["gemini"], "known_at": t + 4}


def goal_update(t, logged):
    return {"window_end": t, "logged_at": logged, "phase": "celebration", "events": [{"type": "goal"}], "narration": ""}


def test_corroborated_goal_starts_a_reaction_on_the_close_up():
    d = DirectorV2(CFG, FPS)
    run(d, 25 * 10, lambda i: cards(ball_cam=2))
    events, updates = [goal_event(9)], [goal_update(9, 12), goal_update(15, 18)]
    chosen = [d.decide(i, i / FPS, cards(ball_cam=2), events, updates, "celebration").cam
              for i in range(25 * 10, 25 * 16)]
    assert d.state == REACTION or CLOSE in chosen
    assert CLOSE in chosen


def test_single_look_goal_is_not_enough():
    d = DirectorV2(CFG, FPS)
    events, updates = [goal_event(9)], [goal_update(9, 12)]
    for i in range(25 * 20):
        d.decide(i, i / FPS, cards(ball_cam=2), events, updates, None)
    assert d.state == IN_PLAY


def test_story_feed_uses_only_what_was_known(tmp_path):
    # Event first reported at playhead 10 with low confidence; raised by a second look written at playhead 20.
    wall0 = 1000.0
    lines = [{"id": "e1", "type": "goal", "event_time": 8, "logged_at": 10, "confidence": 0.4, "sources": ["gemini"],
              "wall_time": wall0 + 10},
             {"id": "e1", "type": "goal", "event_time": 8, "logged_at": 10, "confidence": 0.9, "sources": ["gemini"],
              "wall_time": wall0 + 20}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    (tmp_path / "story.jsonl").write_text("")
    feed = StoryFeed(tmp_path, wall_start=wall0, speed=1.0)
    assert feed.known_events(9) == []
    assert feed.known_events(15)[0]["confidence"] == 0.4
    assert feed.known_events(25)[0]["confidence"] == 0.9


def test_hysteresis_is_time_not_calls_when_live_skips_frames():
    d = DirectorV2(CFG, FPS)
    run(d, 25 * 10, lambda i: cards(ball_cam=2))
    assert d.current == 2
    # Live mode on a busy Mac: only every 5th frame is decided. The challenger must still win after ~0.6 s.
    start = 25 * 10
    for i in range(start, start + 25 * 3, 5):
        d.decide(i, i / FPS, cards(ball_cam=3), [], [], None)
    assert d.current == 3
