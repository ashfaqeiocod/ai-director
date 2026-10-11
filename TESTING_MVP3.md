# How to test MVP 3 — Directed Story

MVP 3 combines MVP 1 (camera switching) and MVP 2 (story reading) on a **real match**:
one 4K panoramic recording from the SoccerTrack v2 dataset, cut into 5 virtual cameras
(LEFT, CENTRE, RIGHT, WIDE, CLOSE-up). A director picks the camera every frame using a
"camera card" per camera (ball, players, speed, picture quality, story), and a report scores
the result against the dataset's ground truth.

Spec: `docs/MVP3_SPEC.md` · Plan: `docs/AI_LIVE_DIRECTOR_PLAN.md` · Progress + findings: `PROGRESS.md` (MVP 3 section).

Run every command from the project folder:

```bash
cd ~/Documents/ai-director
```

> **Important (Claude Code sandbox):** run detection in your own Terminal, not inside Claude Code's sandbox.
> Inside the sandbox, Apple's CoreML can't write its temp files, the football model fails to load, and the
> generic fallback finds almost no balls. `studio.detect` stops with a clear message if that happens.

---

## 1. One-time setup: the footage (~3.5 GB per half)

1. Hugging Face account → accept the terms on https://huggingface.co/datasets/atomscott/soccertrack-v2
2. Put a read token in `.env`: `HF_TOKEN=hf_...`
3. Download one half (resumable; the big annotation file is converted to a small one and deleted):

```bash
venv/bin/python -m studio.soccertrack --match 128057 --half 2
```

You should see `ok` / `get` lines and `wrote data/soccertrack/gsr/128057_2nd_players.npz: 1,547,700 boxes`.

## 2. Everything for one segment, in one command

Segments are set in `config.yaml` → `studio.segments`: `dev` (6 min, 1 goal, for tuning) and
`eval` (14 min, 3 goals, for the reported numbers).

```bash
venv/bin/python -m studio.main --segment dev
```

It skips what is already done. Times on the M1 (8 GB):

| Step | What you see | Time (dev / eval) |
|---|---|---|
| prepare | `annotation offset: ... frames`, 3 check pictures, `ground-truth goals in this segment` | 1 / 3 min |
| cameras | `[cameras] frame N/9000 (15 fps)` | 10 / 25 min |
| detect | `[detect] 100/1800 frames, 0.8 frames/s` | 35 / 90 min |
| story | Gemini lines as the clip "plays" in real time | 6 / 14 min (real time) |
| run (x2) | `CUT` / `REACTION` lines, then a summary | 15 / 35 min each |
| evaluate | the report table (below) | seconds |

Real-time story passes must run **alone** (no detection or rendering at the same time). A pass that falls
behind is marked invalid automatically and not used.

## 3. Check the pictures line up

Open `data/studio/dev/check_04500.png`: green boxes (annotated players) should sit on the players' feet,
the yellow circle (ground-truth ball) near a player.

## 4. Watch a run

```bash
venv/bin/python -m dashboard.server
```

Open http://localhost:8000/studio, pick a run, **Load**, press play:
- big video = the program (what viewers would see), with `CAM n · NAME` and the director's state;
- 5 tiles below = the 5 cameras, red border = live; under each, its card (shot type, players, ball, picture OK/BAD);
- right: the story **as the director knew it** (each line appears only after the AI produced it, with its delay),
  and every director decision;
- timeline: colour = live camera (bright = REACTION), ⚽ marks; tick "show ground truth" for the real goals.

Also in `outputs/studio/<segment>/<run_id>/`: `program.mp4`, `debug.mp4` (everything in one picture),
`director_log.csv` (one row per frame, with the reason), `cards.jsonl`, `report.md`.

## 5. Read the report

```bash
venv/bin/python -m studio.evaluate --segment dev            # newest run
venv/bin/python -m studio.evaluate --segment dev --compare  # all runs + baselines -> compare.md
```

| Line | Meaning | Good |
|---|---|---|
| Ball on air, excluding WIDE time | when the director was NOT on the whole-pitch camera, how often the real ball was in the shot | ≥ 80% oracle, ≥ 60% detector; must beat "always CENTRE" |
| Time on WIDE | WIDE always contains the ball, so a lazy director would sit there | lower is better |
| Shortest shot | no cut faster than `studio.director.min_shot_seconds` (3 s), except leaving a bad picture | ≥ 3.0 s |
| Degraded cam on air | one camera is made blurry on purpose for a while; it must never be live then | 0% (after the first second) |
| Close-up after a real goal / false reactions | did the director react to each goal, and how often did it react to nothing | ≥ 2 of 3 / low |
| Story goals / shots | the AI's goals and shots vs the real ones (within 10 s), with delay | |

"oracle" runs use the dataset's annotated player positions instead of the detector: they test the director
alone. The detector runs are the honest ones.

## 6. Tests

```bash
venv/bin/python -m pytest -q tests/
```

All tests should pass (MVP 2 + MVP 3: causal guard, streaming reader, close-up causality, director rules, story timing).

## 7. MVP 1 still works

```bash
rm outputs/program.mp4 && venv/bin/python src/main.py
```

Same 5 cuts as before (see `TESTING.md` section 1).
