"""fusion.py — merges what all layers say into ONE timeline, intensity and score.

- Intensity (0..100): a weighted mix of audio excitement, motion, recent specialist events and
  the latest Gemini reading (weights in config.yaml story.fusion.intensity_weights). Layers that
  aren't running in this mode are left out and the remaining weights are rescaled, so fast_only
  still uses the full 0..100 range.
"""
from __future__ import annotations

from story.fast_layer import FastResult

# Which layers each mode runs (the fast layer always runs).
MODE_LAYERS = {
    "fast_only": {"specialist": False, "gemini": False},
    "specialist_only": {"specialist": True, "gemini": False},
    "gemini_only": {"specialist": False, "gemini": True},
    "hybrid": {"specialist": True, "gemini": True},
}


class Fusion:
    def __init__(self, cfg: dict, mode: str):
        self.cfg = cfg["fusion"]
        self.mode = mode
        self.layers = MODE_LAYERS[mode]
        self.llm_reading: tuple[float, float] | None = None       # (playhead when logged, intensity 0..1)
        self.specialist_level: tuple[float, float] | None = None  # (playhead when logged, strength 0..1)

    # ------------------------------------------------------------------ intensity
    def note_llm_intensity(self, logged_at: float, intensity: int) -> None:
        self.llm_reading = (logged_at, intensity / 100)

    def note_specialist(self, logged_at: float, strength: float) -> None:
        if self.specialist_level is None or strength >= self._faded(self.specialist_level, logged_at,
                                                                   self.cfg["specialist_hold_s"]):
            self.specialist_level = (logged_at, strength)

    @staticmethod
    def _faded(reading: tuple[float, float] | None, now: float, hold_s: float) -> float:
        """A reading loses strength linearly and is gone after hold_s seconds."""
        if reading is None:
            return 0.0
        at, value = reading
        return value * max(0.0, 1 - (now - at) / hold_s)

    def intensity(self, fast: FastResult) -> int:
        w = self.cfg["intensity_weights"]
        parts = {"audio": (w["audio"], fast.audio), "motion": (w["motion"], fast.motion)}
        if self.layers["specialist"]:
            parts["specialist"] = (w["specialist"],
                                   self._faded(self.specialist_level, fast.t, self.cfg["specialist_hold_s"]))
        if self.layers["gemini"] and self.llm_reading is not None:
            parts["llm"] = (w["llm"], self._faded(self.llm_reading, fast.t, self.cfg["llm_intensity_hold_s"]))
        total_weight = sum(weight for weight, _ in parts.values())
        value = sum(weight * v for weight, v in parts.values()) / total_weight
        return int(round(100 * min(1.0, value)))
