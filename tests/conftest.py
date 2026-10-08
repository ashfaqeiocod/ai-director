"""Shared test fixtures: a tiny synthetic clip folder (5 s, 25 fps, a beep at 2-3 s)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from story.common import ffmpeg_exe


@pytest.fixture(scope="session")
def tiny_clip(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("clip")
    ff = ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=5",
                    "-f", "lavfi", "-i", "sine=frequency=3000:sample_rate=48000:duration=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "25", "-c:a", "aac",
                    str(folder / "clip.mp4")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(folder / "clip.mp4"), "-vn",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(folder / "audio.wav")], check=True)
    (folder / "meta.json").write_text(json.dumps(
        {"name": "tiny", "duration": 5.0, "fps": 25.0, "width": 320, "height": 180,
         "frames": 125, "has_audio": True}))
    return folder
