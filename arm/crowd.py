"""crowd.py — Arm MVP Step 4: how excited is the room? From SOUND (5 phone microphones) and from the AUDIENCE's
faces and hands (pose + face cache of arm/perceive.py). Only the past is used for every value.

Sound, every 0.5 s, from the last 1 s of audio of every phone:
  YAMNet (the same audio model as MVP 2's fast layer) scores ~500 sound classes. We add up the "people are
  excited" ones (cheering, applause, laughter, shouting...) and take the average over the 5 microphones, so one
  phone held close to a loud person does not decide alone. Plus loudness: how much louder than the last ~10 s.
Audience (everyone who is not one of the two athletes), every 0.1 s:
  smile     average mouthSmile of the audience faces we can read
  clap      share of audience members with both hands together in front of the chest (clapping pose)
  arms_up   share with a wrist above the shoulder (cheering)
Output: a list of rows (t, sound, smile, clap, arms_up, crowd) with crowd = weighted mix, 0..1.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from arm.common import cam_letters, settings, synced_video
from arm.sync import load_audio
from src.common import load_config

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10


def sound_excitement(duration: float) -> list[dict]:
    """Per 0.5 s step: {t, cheer, loud, sound} where t = the END of the 1 s window that was heard."""
    from story.fast_layer import AudioAnalyzer

    cfg = settings()["crowd"]
    story_fast = load_config()["story"]["fast"]
    rate = 16000                                 # YAMNet listens at 16 kHz
    analyzer = AudioAnalyzer(story_fast, rate)
    if analyzer.model_name != "yamnet":
        print(f"  YAMNet not available ({analyzer.error}): loudness only")
    tracks = {c: load_audio(synced_video(c), rate) for c in cam_letters()}
    step, window = cfg["sound_step_s"], cfg["sound_window_s"]
    history = {c: deque(maxlen=int(cfg["loudness_memory_s"] / step)) for c in tracks}
    rows = []
    t = window
    while t <= duration + 1e-6:
        cheers, louds = [], []
        for cam, wave in tracks.items():
            piece = wave[int((t - window) * rate):int(t * rate)]
            scores = analyzer.classify(piece)
            cheers.append(min(1.0, sum(scores.get(k, 0.0) for k in cfg["sound_classes"]) * cfg["sound_gain"]))
            db = 20 * np.log10(np.sqrt(np.mean(piece ** 2)) + 1e-6) if len(piece) else -120.0
            past = history[cam]
            base = float(np.median(past)) if past else db
            past.append(db)                      # only the past forms the baseline
            louds.append(float(np.clip(0.3 + (db - base) / 12, 0, 1)))
        cheer, loud = float(np.mean(cheers)), float(np.mean(louds))
        rows.append({"t": round(t, 2), "cheer": round(cheer, 3), "loud": round(loud, 3),
                     "sound": round(cfg["cheer_weight"] * cheer + (1 - cfg["cheer_weight"]) * loud, 3)})
        t += step
    return rows


def audience_signals(people: list[dict], athletes: set[int]) -> dict:
    """Smile / clap / arms-up of the audience in ONE look of one camera (athletes left out by index)."""
    smiles, claps, ups, counted = [], 0, 0, 0
    for k, person in enumerate(people):
        if k in athletes:
            continue
        face = person.get("face")
        if face:
            smiles.append((face["bs"].get("mouthSmileLeft", 0) + face["bs"].get("mouthSmileRight", 0)) / 2)
        kp = np.array(person["kp"])
        if min(kp[L_SHOULDER, 2], kp[R_SHOULDER, 2]) < 0.4:
            continue
        counted += 1
        shoulder_w = abs(kp[L_SHOULDER, 0] - kp[R_SHOULDER, 0]) + 1e-6
        if min(kp[L_WRIST, 2], kp[R_WRIST, 2]) >= 0.4:
            hands_gap = np.hypot(*(kp[L_WRIST, :2] - kp[R_WRIST, :2])) / shoulder_w
            chest_y = (kp[L_SHOULDER, 1] + kp[R_SHOULDER, 1]) / 2
            in_front = kp[L_WRIST, 1] > chest_y - 0.3 * shoulder_w      # hands not above the head
            if hands_gap < 0.45 and in_front:
                claps += 1
        top = min(kp[L_SHOULDER, 1], kp[R_SHOULDER, 1])
        if (kp[L_WRIST, 2] >= 0.4 and kp[L_WRIST, 1] < top) or (kp[R_WRIST, 2] >= 0.4 and kp[R_WRIST, 1] < top):
            ups += 1
    return {"smile": float(np.mean(smiles)) if smiles else None, "faces": len(smiles),
            "clap": claps / counted if counted else None, "arms_up": ups / counted if counted else None,
            "people": counted}


def crowd_value(sound: float, smile: float | None, clap: float | None, arms_up: float | None) -> float:
    """The 0..1 crowd excitement: a weighted mix of what we heard and what the audience shows."""
    w = settings()["crowd"]["weights"]
    parts = {"sound": sound, "smile": smile, "clap": clap, "arms_up": arms_up}
    used = {k: v for k, v in parts.items() if v is not None}
    total = sum(w[k] for k in used)
    return float(sum(w[k] * min(1.0, v * (2.0 if k in ("clap", "arms_up") else 1.0)) for k, v in used.items()) / total) if total else 0.0
