"""Goal planner: looks only near a goal, follows a shot up twice, keeps the line free for a due follow-up."""
from types import SimpleNamespace

from src.common import load_config
from studio.goalwatch import AfterLook, GoalLook, GoalPlanner

CFG = load_config()["studio"]["goalwatch"]


def state(ball_x, events=()):
    return SimpleNamespace(ball=(ball_x, 0.0), ball_status="seen", players=[], zone="x", events=list(events))


def test_no_look_far_from_goal():
    assert GoalPlanner(CFG).plan(30.0, state(0.0)) is None


def test_look_near_goal_then_interval():
    p = GoalPlanner(CFG)
    job = p.plan(30.0, state(-45.0))
    assert job and job["side"] == "left" and job["end"] == 30.0 and job["start"] == 30.0 - CFG["window_s"]
    p.answer(job, GoalLook(), 33.0)
    assert p.plan(31.0, state(-45.0)) is None                       # too soon after the last look
    assert p.plan(31.0, state(-45.0, [{"type": "BOX_ENTRY"}])) is not None   # a trigger looks at once


def test_shot_schedules_two_aftermath_looks():
    p = GoalPlanner(CFG)
    job = p.plan(30.0, state(48.0))
    item = p.answer(job, GoalLook(shot=True, shot_at_s=5.0), 33.0)
    shot = item["shot_time"]
    assert [a["due"] for a in p.aftermath] == [shot + d for d in CFG["aftermath_s"]]
    after = p.plan(shot + CFG["aftermath_s"][0], state(0.0))          # due even if play moved away
    assert after["kind"] == "after" and after["side"] == "right"
    rec = p.answer(after, AfterLook(goal_scored=True, confidence=0.9), shot + 12)
    assert rec["goal"] and rec["goal_time"] == shot


def test_story_event_asks_for_a_recheck_even_right_after_a_look():
    p = GoalPlanner(CFG)
    job = p.plan(63.0, state(48.0))
    p.answer(job, GoalLook(), 64.0)                                  # routine look just done at 63 s
    assert p.plan(64.0, state(48.0)) is None                         # interval: no new routine look
    assert p.request("right", 63.5, "cross", now=64.0)               # the Colab model reports a cross at 63.5 s
    assert not p.request("right", 64.0, "header", now=64.5)          # same moment: one re-check only
    assert p.plan(65.0, state(48.0)) is None                         # not due yet (outcome not visible)
    re = p.plan(63.5 + CFG["recheck_after_s"], state(0.0))           # due even if play moved away
    assert re["side"] == "right" and "re-check" in re["trigger"] and re["start"] <= 62.5


def test_keeper_holding_the_ball_after_a_shot_is_a_save():
    from studio.cards import StoryFeed

    p = GoalPlanner(CFG)
    item = p.answer(p.plan(66.0, state(48.0)), GoalLook(shot=True, shot_at_s=4.0), 67.0)
    after = p.plan(item["shot_time"] + CFG["aftermath_s"][0], state(48.0))
    rec = p.answer(after, AfterLook(keeper_holds_ball=True), 75.0)
    assert rec["save"] and not rec["goal"]
    feed = StoryFeed(None)
    feed.add_goal_looks([item, rec], CFG)
    saves = [e for e in feed.events if e["type"] == "save"]
    assert len(saves) == 1 and saves[0]["event_time"] == item["shot_time"] and saves[0]["known_at"] == 75.0


def test_looks_follow_the_map_intensity_when_enabled():
    cfg = {**CFG, "interval_by_intensity": {"hot": 0.5, "hot_s": 2, "calm": 0.2, "calm_s": 8}}
    hot, calm = state(48.0), state(48.0)
    hot.intensity, calm.intensity = 0.8, 0.1
    p = GoalPlanner(cfg)
    p.answer(p.plan(30.0, hot), GoalLook(), 31.0)
    assert p.plan(32.5, hot) is not None                              # hot: again after 2 s
    assert p.plan(32.5, calm) is None and p.plan(38.5, calm) is not None   # calm: waits 8 s
    assert GoalPlanner(CFG).interval(hot) == CFG["interval_s"]        # SoccerTrack / MUVS: unchanged


def test_a_save_needs_the_follow_up_look():
    from studio.cards import StoryFeed

    p = GoalPlanner(CFG)
    first = p.answer(p.plan(22.0, state(-48.0)), GoalLook(shot=True, save=True, shot_at_s=4.0), 23.0)
    after = p.answer(p.plan(first["shot_time"] + CFG["aftermath_s"][0], state(-48.0)), AfterLook(), 30.0)
    feed = StoryFeed(None)
    feed.add_goal_looks([first, after], CFG)          # the follow-up did NOT see the keeper with the ball
    assert not [e for e in feed.events if e["type"] == "save"]
