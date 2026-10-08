"""run_log.py — writes every run's logs to outputs/story/<clip>/<run_id>/ (gitignored).

  run_meta.json          mode, model, settings snapshot, start/end, summary numbers
  story.jsonl            every StoryUpdate (Gemini or template), with window + logged_at
  events.jsonl           every fused event each time it is created or changes
  intensity.csv          one row per second: audio, motion, fused intensity, replay hint
  llm_calls.jsonl        request time, latency, tokens, errors (Gemini)
  specialist_calls.jsonl Colab requests/answers, inference time, real-time factor
  memory.jsonl           every "story so far"
  signals.jsonl          fast-layer triggers (cheer, whistle, cuts), lag, errors
  ratings.jsonl          your 👍 / 👎 on commentary lines (used by the evaluation)
"""
from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from story.common import outputs_dir


class RunLog:
    def __init__(self, clip: str, mode: str):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mode}"
        self.folder = outputs_dir() / clip / self.run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.meta: dict = {}
        self._csv_started = False

    def write_meta(self, **fields) -> None:
        with self.lock:
            self.meta.update(fields)
            (self.folder / "run_meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def append(self, name: str, item: dict) -> None:
        """Append one JSON line to <name>.jsonl (thread-safe)."""
        with self.lock, open(self.folder / f"{name}.jsonl", "a") as f:
            f.write(json.dumps({"wall_time": round(time.time(), 3), **item}, default=str) + "\n")

    def intensity_row(self, row: dict) -> None:
        with self.lock, open(self.folder / "intensity.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not self._csv_started:
                writer.writeheader()
                self._csv_started = True
            writer.writerow(row)


def run_folder(clip: str, run_id: str) -> Path:
    return outputs_dir() / clip / run_id
