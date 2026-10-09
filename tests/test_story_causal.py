"""Rule 1 of MVP 2: nothing may read video or audio after the playhead."""
from __future__ import annotations

import pytest

from story.causal import CausalReader, FutureAccessError, Playhead


def test_audio_up_to_playhead_is_allowed(tiny_clip):
    reader = CausalReader(tiny_clip, Playhead(2.0))
    samples = reader.audio(1.0, 2.0)
    assert len(samples) == 16000          # exactly 1 s at 16 kHz
    assert samples.dtype.name == "float32"


def test_audio_after_playhead_raises(tiny_clip):
    reader = CausalReader(tiny_clip, Playhead(2.0))
    with pytest.raises(FutureAccessError):
        reader.audio(1.5, 2.1)


def test_frames_stop_at_playhead(tiny_clip):
    reader = CausalReader(tiny_clip, Playhead(2.0))
    video = reader.video()
    times = [t for t, _ in video.frames(1.0, 2.0)]
    assert times[0] > 1.0 and times[-1] == pytest.approx(2.0)
    assert len(times) == 25
    with pytest.raises(FutureAccessError):
        list(video.frames(1.0, 2.04))


def test_frames_follow_a_moving_playhead(tiny_clip):
    playhead = Playhead(1.0)
    video = CausalReader(tiny_clip, playhead).video()
    assert len(list(video.frames(0.0, 1.0, fps=5))) == 5
    with pytest.raises(FutureAccessError):
        list(video.frames(1.0, 2.0, fps=5))
    playhead.advance(2.0)                  # the video played on: now 1..2 s is the past
    assert len(list(video.frames(1.0, 2.0, fps=5))) == 5


def test_window_clip_after_playhead_raises(tiny_clip, tmp_path):
    reader = CausalReader(tiny_clip, Playhead(3.0))
    out = reader.window_clip(1.0, 3.0, tmp_path / "w.mp4", height=120, with_audio=True)
    assert out.stat().st_size > 0
    with pytest.raises(FutureAccessError):
        reader.window_clip(1.0, 3.5, tmp_path / "x.mp4", height=120, with_audio=True)


def test_playhead_never_goes_back():
    playhead = Playhead(5.0)
    assert playhead.advance(3.0) == 5.0    # a seek backwards is ignored
    assert playhead.advance(6.0) == 6.0


def test_window_end_rounded_down_is_never_in_the_future(tiny_clip, tmp_path):
    """Regression: the engine once rounded 8.168 up to 8.17 and the guard (correctly) refused it."""
    import math
    reader = CausalReader(tiny_clip, Playhead(2.168))
    end = math.floor(reader.playhead.now() * 100) / 100
    reader.window_clip(end - 1, end, tmp_path / "w.mp4", height=120, with_audio=False)
    with pytest.raises(FutureAccessError):
        reader.window_clip(1.0, round(2.168, 2) + 0.001, tmp_path / "x.mp4", height=120, with_audio=False)
