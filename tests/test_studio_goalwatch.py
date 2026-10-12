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
