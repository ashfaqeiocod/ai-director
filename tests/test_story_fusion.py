"""Fusion: dedupe across layers, replays never add events, goals counted once, intensity."""
from __future__ import annotations

import pytest

from story.common import story_config
from story.fast_layer import FastResult
from story.fusion import Fusion
from story.schema import Event


def ev(type_, t, source="gemini", conf=0.7, team=None, logged=None):
    return Event(type=type_, event_time=t, logged_at=logged if logged is not None else t + 3,
                 source=source, confidence=conf, team=team)


@pytest.fixture
def fusion():
    return Fusion(story_config(), "hybrid")


def test_same_event_from_two_layers_is_merged(fusion):
    a, new_a = fusion.add_event(ev("shot", 10.0, "specialist", 0.6), is_replay=False)
    b, new_b = fusion.add_event(ev("shot", 11.5, "gemini", 0.6), is_replay=False)
    assert new_a and not new_b and a is b
    assert a.sources == ["specialist", "gemini"]
    assert a.confidence == pytest.approx(0.84)          # noisy-or: 1 - 0.4 * 0.4
    assert len(fusion.events) == 1


def test_far_apart_or_different_events_stay_separate(fusion):
    fusion.add_event(ev("shot", 10.0), is_replay=False)
    fusion.add_event(ev("shot", 30.0), is_replay=False)
    fusion.add_event(ev("corner", 31.0), is_replay=False)
    assert [e.type for e in fusion.events] == ["shot", "shot", "corner"]


def test_logged_at_keeps_the_first_report(fusion):
    a, _ = fusion.add_event(ev("foul", 20.0, "gemini", logged=26.0), is_replay=False)
    fusion.add_event(ev("foul", 20.5, "specialist", logged=22.0), is_replay=False)
    assert a.logged_at == 22.0


def test_replays_never_create_events(fusion):
    fused, _ = fusion.add_event(ev("goal", 50.0, conf=0.9), is_replay=True)
    assert fused is None and fusion.events == [] and fusion.ignored_in_replay == 1
    manual, _ = fusion.add_event(ev("goal", 50.0, "manual", 1.0), is_replay=True)
    assert manual is not None                           # a human correction always counts


def test_goal_counted_once_even_if_reported_again_later(fusion):
    first, _ = fusion.add_event(ev("goal", 60.0, conf=0.9, team="white"), is_replay=False)
    assert fusion.count_goal(first)
    # The same goal shown again 30 s later (a replay Gemini didn't flag) merges into it.
    again, is_new = fusion.add_event(ev("goal", 90.0, conf=0.9, team="white"), is_replay=False)
    assert again is first and not is_new
    assert not fusion.count_goal(again)
    assert fusion.score == {"white": 1}


def test_low_confidence_goal_waits_for_a_second_source(fusion):
    goal, _ = fusion.add_event(ev("goal", 60.0, "gemini", 0.5, "red"), is_replay=False)
    assert not fusion.count_goal(goal)
    goal, _ = fusion.add_event(ev("goal", 59.0, "specialist", 0.5, "red"), is_replay=False)
    assert fusion.count_goal(goal) and fusion.score == {"red": 1}   # 0.75 >= 0.6


def test_manual_score_override(fusion):
    fusion.adjust_score("white", 1)
    fusion.adjust_score("white", -1)
    fusion.adjust_score("white", -1)
    assert fusion.score == {"white": 0}


def test_intensity_uses_only_running_layers():
    cfg = story_config()
    loud = FastResult(t=10, audio=1.0, motion=1.0)
    assert Fusion(cfg, "fast_only").intensity(loud) == 100       # rescaled: full range without other layers
    hybrid = Fusion(cfg, "hybrid")
    assert hybrid.intensity(loud) < 100                           # specialist quiet -> lower
    hybrid.note_specialist(10, 1.0)
    hybrid.note_llm_intensity(10, 100)
    assert hybrid.intensity(loud) == 100


def test_scoreboard_tells_a_new_goal_from_a_replayed_one(fusion):
    from story.fusion import scoreboard_total
    assert scoreboard_total("2-1") == 3 and scoreboard_total("2 : 0") == 2 and scoreboard_total(None) is None
    first, _ = fusion.add_event(ev("goal", 61.0, conf=0.95, team="white"), is_replay=False)
    fusion.count_goal(first)
    # 33 s later, scoreboard reads 2-0 (total 2 > 1 counted): a NEW goal, not merged.
    assert fusion.scoreboard_says_new_goal("2-0")
    second, is_new = fusion.add_event(ev("goal", 94.0, conf=0.95, team="white"), is_replay=False, force_new=True)
    assert is_new and fusion.count_goal(second) and fusion.score == {"white": 2}
    # A replay of it with the scoreboard still at 2-0 is not new: merged by the 60 s window.
    assert not fusion.scoreboard_says_new_goal("2-0")
    again, is_new = fusion.add_event(ev("goal", 105.0, conf=0.95, team="white"), is_replay=False)
    assert again is second and not is_new and fusion.score == {"white": 2}
