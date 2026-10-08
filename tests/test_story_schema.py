"""The story data model: validation and what Gemini is asked to return."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from story.schema import Event, LLMStory, StoryUpdate


def test_event_needs_a_known_type_and_valid_confidence():
    Event(type="goal", event_time=10, logged_at=14, source="gemini", confidence=0.9)
    with pytest.raises(ValidationError):
        Event(type="touchdown", event_time=10, logged_at=14, source="gemini", confidence=0.9)
    with pytest.raises(ValidationError):
        Event(type="goal", event_time=10, logged_at=14, source="gemini", confidence=1.5)


def test_story_update_keeps_both_times_and_cuts_long_narration():
    u = StoryUpdate(window_start=0, window_end=6, logged_at=9.5, phase="attack", intensity=70,
                    sentiment="tense", narration=" ".join(["word"] * 50))
    assert u.logged_at - u.window_end == pytest.approx(3.5)        # the honest delay
    assert len(u.narration.split()) == 30
    with pytest.raises(ValidationError):
        StoryUpdate(window_start=0, window_end=6, logged_at=6, phase="attack", intensity=150,
                    sentiment="tense", narration="x")


def test_llm_schema_parses_a_gemini_style_answer():
    answer = ('{"phase":"chance","events":[{"type":"shot","at_s":3.5,"team":"white","confidence":0.8,'
              '"note":"low shot"}],"team_in_control":"white","intensity":72,"sentiment":"excited",'
              '"narration":"White shoots low!","is_replay":false,"score":null}')
    story = LLMStory.model_validate_json(answer)
    assert story.events[0].type == "shot" and story.score is None
