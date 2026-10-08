"""fast_layer.py — the FAST layer: cheap signals computed on the Mac every second.

For each 1-second step that has already played (never past the playhead):
  - audio excitement (0..1): crowd-type sounds from YAMNet (cheering, crowd, applause, roar...)
    plus how loud it is compared with the last ~30 s.
  - whistle: YAMNet's "Whistle" score PLUS a tonal check (a referee whistle is a strong, narrow
    tone around 2–4.5 kHz). Both are needed, because YAMNet often confuses whistles and cheering.
  - motion (0..1): how much the picture changes between small grayscale frames.
  - scene cuts: a big jump in the colour histogram = a hard cut. Many cuts in a few seconds is
    typical of a broadcast replay sequence -> "replay hint".
  - triggers: a sudden jump in excitement ("CHEER") or a whistle. The engine uses triggers to
    ask Gemini for an extra look right away.

If MediaPipe/YAMNet can't run, audio falls back to loudness + spectral-band features.
"""
from __future__ import annotations

import urllib.request
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from story.causal import CausalReader
from story.common import path

YAMNET_WINDOW_S = 0.975  # YAMNet looks at 0.975 s of audio at a time


@dataclass
class FastResult:
    t: float                      # end of the 1-second step (= the playhead it was computed at)
    audio: float = 0.0            # excitement 0..1
    crowd: float = 0.0            # YAMNet crowd-type score 0..1
    loudness_db: float = -90.0
    whistle: float = 0.0          # whistle score 0..1
    whistle_hit: bool = False
    motion: float = 0.0           # 0..1
    cuts: int = 0                 # hard scene cuts in this step
    replay: bool = False          # replay hint (many cuts close together)
    triggers: list[str] = field(default_factory=list)   # "cheer", "whistle"
    top_classes: list[tuple[str, float]] = field(default_factory=list)
    audio_model: str = "yamnet"   # "yamnet" or "fallback"


# --------------------------------------------------------------------------- audio
class AudioAnalyzer:
    """YAMNet (via MediaPipe) + loudness + whistle tone check."""

    def __init__(self, cfg: dict, sample_rate: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.crowd_classes = [c for c in cfg["audio_classes"] if c != "Whistle"]
        self.loudness = deque(maxlen=int(cfg["loudness_window_s"] / cfg["tick_s"]))
        self.classifier = None
        self.model_name = "fallback"
        self.error = ""
        try:
            self.classifier = self._load_yamnet()
            self.model_name = "yamnet"
        except Exception as error:   # no model file, no internet, MediaPipe problem...
            self.error = f"{type(error).__name__}: {error}"

    def _load_yamnet(self):
        from mediapipe.tasks.python import audio
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model = path(self.cfg["yamnet_model"])
        if not model.exists():   # first run: download the 4 MB model into models_cache/
            model.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.cfg["yamnet_url"], model)
        options = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(model)), max_results=-1,
            running_mode=audio.RunningMode.AUDIO_CLIPS)
        return audio.AudioClassifier.create_from_options(options)

    def classify(self, samples: np.ndarray) -> dict[str, float]:
        """YAMNet class scores for this snippet (max over its 0.975 s sub-windows)."""
        if self.classifier is None or len(samples) < int(0.5 * self.sample_rate):
            return {}
        from mediapipe.tasks.python.components import containers

        results = self.classifier.classify(containers.AudioData.create_from_array(samples, self.sample_rate))
        scores: dict[str, float] = {}
        for result in results:
            for c in result.classifications[0].categories:
                scores[c.category_name] = max(scores.get(c.category_name, 0.0), c.score)
        return scores

    def tonal_whistle(self, samples: np.ndarray) -> float:
        """0..1: how much this audio looks like a referee whistle.

        Cut the audio into short frames (64 ms). A frame is "tonal" when, inside the whistle band,
        one narrow peak holds most of the energy and sticks out far above the rest. Speech also has
        energy in this band, but spread over many harmonics, so it fails the "one narrow peak" test.
        A whistle must stay tonal for several frames in a row (~0.15 s or more).
        """
        low, high = self.cfg["whistle_band_hz"]
        n, hop = 1024, 512
        if len(samples) < n:
            return 0.0
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        band = (freqs >= low) & (freqs <= high)
        window = np.hanning(n)
        run = best_run = 0
        for start in range(0, len(samples) - n + 1, hop):
            power = np.abs(np.fft.rfft(samples[start:start + n] * window))[band] ** 2
            peak = int(power.argmax())
            near = power[max(0, peak - 8):peak + 9].sum()          # +-125 Hz around the peak
            concentration = near / (power.sum() + 1e-12)           # share of band energy in that peak
            prominence_db = 10 * np.log10(power[peak] / (np.median(power) + 1e-12))
            tonal = concentration >= 0.5 and prominence_db >= 25
            run = run + 1 if tonal else 0
            best_run = max(best_run, run)
        # 4 frames in a row (~0.15 s) = 0.5; 8 frames (~0.28 s) or more = 1.0.
        return float(np.clip(best_run / 8, 0, 1))

    def analyze(self, samples: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        db = 20 * np.log10(rms + 1e-6)
        baseline = float(np.median(self.loudness)) if self.loudness else db
        self.loudness.append(db)      # only the past goes into the baseline
        # +9 dB above the recent median = fully "loud"; at the median = 0.4.
        loud = float(np.clip(0.4 + (db - baseline) / 15, 0, 1)) if db > -70 else 0.0

        scores = self.classify(samples)
        crowd = float(np.clip(sum(scores.get(c, 0.0) for c in self.crowd_classes) * self.cfg["crowd_gain"], 0, 1))
        tonal = self.tonal_whistle(samples)
        if self.model_name == "yamnet":
            # Under TV commentary YAMNet mostly hears "Speech", so loudness carries more weight.
            excitement = self.cfg["crowd_weight"] * crowd + (1 - self.cfg["crowd_weight"]) * loud
            # YAMNet alone confuses whistles with cheering, the tone check alone can be fooled by
            # music: a strong tone counts on its own, a medium one needs YAMNet to agree.
            yam_whistle = scores.get("Whistle", 0.0)
            whistle = max(tonal * 0.8, 0.5 * min(1.0, yam_whistle * 3) + 0.5 * tonal)
        else:   # fallback: loudness + band energy only
            excitement = loud
            whistle = tonal
        # Music (intros, outros, ads) is full of pure tones: no whistles while YAMNet hears music.
        if scores.get("Music", 0.0) > self.cfg["whistle_max_music"]:
            whistle = 0.0
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        return {"audio": float(excitement), "crowd": crowd, "loudness_db": round(db, 1),
                "whistle": float(whistle), "tonal": tonal,
                "top_classes": [(k, round(v, 2)) for k, v in top]}


# --------------------------------------------------------------------------- video
class VideoAnalyzer:
    """Motion + hard scene cuts from small frames."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prev_gray: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None

    def analyze(self, frames: list[np.ndarray]) -> dict:
        diffs, cuts = [], 0
        for frame in frames:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.cfg["motion_width"], max(1, int(h * self.cfg["motion_width"] / w))),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if self.prev_hist is not None:
                distance = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if distance > self.cfg["cut_threshold"]:
                    cuts += 1            # a cut is not "motion": skip this pair
                else:
                    diffs.append(float(np.mean(cv2.absdiff(gray, self.prev_gray))))
            self.prev_gray, self.prev_hist = gray, hist
        motion = float(np.clip(np.mean(diffs) / self.cfg["motion_scale"], 0, 1)) if diffs else 0.0
        return {"motion": motion, "cuts": cuts}


# --------------------------------------------------------------------------- the layer
class FastLayer:
    """Runs audio + video analysis for one 1-second step at a time."""

    def __init__(self, cfg: dict, reader: CausalReader):
        self.cfg = cfg["fast"]
        self.reader = reader
        self.video = reader.video()        # its own frame reader (not shared with other layers)
        self.audio = AudioAnalyzer(self.cfg, reader.sample_rate)
        self.vision = VideoAnalyzer(self.cfg)
        self.excitement_history: deque[float] = deque(maxlen=4)
        self.cut_times: deque[float] = deque()
        self.replay_until = -1.0
        self.last_whistle = -999.0
        self.last_cheer = -999.0

    def step(self, t0: float, t1: float) -> FastResult:
        """Analyse the step (t0, t1]. t1 must not be after the playhead (CausalReader checks)."""
        # YAMNet needs at least 0.975 s, so the audio window is never shorter than that.
        audio_start = max(0.0, t1 - max(YAMNET_WINDOW_S, t1 - t0))
        a = self.audio.analyze(self.reader.audio(audio_start, t1))
        frames = [f for _, f in self.video.frames(t0, t1, fps=self.cfg["sample_fps"])]
        v = self.vision.analyze(frames)

        result = FastResult(t=t1, audio=a["audio"], crowd=a["crowd"], loudness_db=a["loudness_db"],
                            whistle=a["whistle"], motion=v["motion"], cuts=v["cuts"],
                            top_classes=a["top_classes"], audio_model=self.audio.model_name)

        # Cheer trigger: excitement jumps well above the quietest of the last 4 seconds
        # (crowds often build up over 2-3 s, so "the average" would hide the jump).
        # It must also be loud enough in absolute terms, and one crowd reaction counts once
        # (cheer_min_gap_s). Not in the first 10 s: the loudness baseline is still forming.
        if t1 >= 10 and len(self.excitement_history) == self.excitement_history.maxlen \
                and result.audio - min(self.excitement_history) >= self.cfg["excitement_trigger_jump"] \
                and result.audio >= self.cfg["cheer_min_level"] \
                and t1 - self.last_cheer >= self.cfg["cheer_min_gap_s"]:
            result.triggers.append("cheer")
            self.last_cheer = t1
        self.excitement_history.append(result.audio)

        # Whistle: needs the combined score AND a real tone (rejects loud cheering).
        if result.whistle >= self.cfg["whistle_min_score"] and a["tonal"] >= 0.5 \
                and t1 - self.last_whistle >= self.cfg["whistle_min_gap_s"]:
            result.whistle_hit = True
            result.triggers.append("whistle")
            self.last_whistle = t1

        # Replay hint: several hard cuts within a few seconds.
        for _ in range(v["cuts"]):
            self.cut_times.append(t1)
        while self.cut_times and self.cut_times[0] < t1 - self.cfg["replay_window_s"]:
            self.cut_times.popleft()
        if v["cuts"] and len(self.cut_times) >= self.cfg["replay_min_cuts"]:
            self.replay_until = t1 + self.cfg["replay_hold_s"]
        result.replay = t1 <= self.replay_until
        return result
