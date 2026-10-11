# MVP 3 — Directed Story (build spec)

> Combines MVP 1 (camera switching) and MVP 2 (Live Story Reader) on **real full-match footage**.
> Big picture and reasons: `docs/AI_LIVE_DIRECTOR_PLAN.md` (sections 3, 6, 14, 15). The plan may change;
> this spec is the current working version. Last updated: 2026-10-11.

## 1. Goal
Prove the core idea of the plan on a real match: **a per-camera description ("camera card") about once a
second + a story layer + broadcast rules = a program that follows the play and reacts to the story.**

Demo in one sentence: a 10–15 minute stretch of a real match plays in the dashboard; 5 camera tiles
(left, centre, right, wide, close-up) each show their live card; the program cuts between them following the
ball, holds wide during fast play, goes to the close-up after a goal, never puts a bad feed on air, and a story
log runs alongside. Afterwards a report scores it against the match's ground truth.

## 2. Non-negotiable rules
All MVP 1 and MVP 2 rules still apply (`CLAUDE.md`). In addition:
1. **Causal everywhere.** Nothing at time t may use data after t: detections, the virtual close-up's pan
   (past-only smoothing, no centred averages), story events (an event may be used only once its `logged_at`
   ≤ t), and camera cards.
2. **Ground truth is for scoring only.** The dataset's events (`bas/`), and its ball track (`ball/`), never
   feed the director, the cards or the story layer. The ball track is rebuilt from the annotated events
   (interpolated between them), so it "knows the future".
   Exception, labelled clearly: an **oracle mode** may use the per-frame player positions (`gsr/`) instead
   of the detector, to test the director on its own. Every output of an oracle run says `oracle` in its
   name and report.
3. **Tune on one part, report on another.** Dev segment for tuning, eval segment for the reported numbers
   (section 4). Never tune on the eval segment.
4. **Don't break MVP 1 or MVP 2.** New code goes in a new package `studio/`. It **uses** `src/` and `story/`
   (Detector, BallTracker, Director scoring, StoryEngine, Fusion) without changing their behaviour. Any
   change to those must keep their defaults identical. Run the MVP 1 quick check and `pytest` after each step.
5. **Secrets and data.** `HF_TOKEN` only from `.env` (never printed, never in code or commands). Never
   commit videos, annotations, outputs or `.env`. Credit SoccerTrack v2 (CC BY 4.0) in the README.

## 3. Footage: SoccerTrack v2
- Source: https://huggingface.co/datasets/atomscott/soccertrack-v2 (CC BY 4.0, Hugging Face login + terms).
  10 university matches, one fixed panoramic 4K video per half, 25 fps, frame k of a half = annotation frame k.
- Per half: `videos/<id>/<id>_panorama_<n>_half.mp4` (~6.7 GB), `gsr/<id>/<id>_<n>.json` (players per frame,
  pitch metres), `ball/<id>_<n>_ball.npz` (reconstructed ball, pitch metres), `bas/<id>/<id>_12_class_events.json`
  (12 classes: PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK,
  PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL; `frame` + `position` ms), `raw/<id>/` (homography, intrinsics,
  distortion maps) to map pitch metres → video pixels.
- Pitch coordinates: 105 × 68 m, origin at the centre spot, x to the right, y towards the near touchline.
- Facts to remember: no scoreboard graphic, no replays, no commentary; audio to be checked (may be none).
- The SoccerNet NDA (requested 2026-10-11) will later add real broadcast angles (MVFouls) and human
  director cuts (SoccerNet-v2 camera labels). Not needed for MVP 3.

## 4. Segments (config: `studio.segments`)
| Name | Match / half | Time in half | Ground-truth goals | Use |
|---|---|---|---|---|
| `dev` | 128057 (M7) 2nd half | 00:30 – 06:30 | 1 (1:31) | tuning, quick checks |
| `eval` | 128057 (M7) 2nd half | 23:00 – 37:00 | 3 (24:57, 31:14, 35:54) | reported numbers |
| later | 132831 (M9) or a validation match | full half | 2–3 | final check |

M7 and M9 are the dataset's **test** matches. Check whether the Colab specialist model
(`OSL-loc-snbas-2025-e2e`) was trained on SoccerTrack footage; if so, note that its scores here are optimistic.

## 5. Architecture
```
 SoccerTrack 4K panorama (one half)                     ground truth (scoring only)
        │ prepare (segment cut, calibration check)               │
        ▼                                                         │
 VIRTUAL CAMERAS  cam1 left · cam2 centre · cam3 right            │
                  cam4 WIDE (whole pitch) · cam5 CLOSE (causal    │
                  virtual pan/zoom on the action, from 4K)        │
        │                                                         │
        ├─▶ DETECTION per cam (MVP 1 Detector, cached, N fps)     │
        │      or ORACLE players (gsr) — labelled                 │
        │                                                         │
        ├─▶ STORY LAYER on cam4 WIDE (MVP 2 StoryEngine,          │
        │      driven at 1× by a simulated playhead, headless)    │
        │      → events.jsonl with event_time + logged_at         │
        ▼                                                         │
 CAMERA CARDS (1 per cam per second, plan §3.1)                   │
        ▼                                                         │
 DIRECTOR v2 (state machine: IN_PLAY / REACTION, feed rejection)  │
        ▼                                                         ▼
 program.mp4 · debug grid · director_log.csv · cards.jsonl ──▶ EVALUATION report.md
        └─▶ dashboard "Studio" tab (replay from caches, browser video = clock)
```
Two speeds (plan §4.1): cards' fast fields (ball, players, quality, play speed) update every second from
detection; the story fields (event, phase) arrive seconds late from the story layer and are used only
from their `logged_at`.

Why the story layer runs in its own 1× pass: its delays only mean something in real time, while detection on
the M1 is slower than real time. Running it first and saving `logged_at` keeps everything causal and lets the
director run as fast as the machine allows.

## 6. Camera card (`studio/cards.py`, Pydantic)
```python
class CamCard(BaseModel):
    cam: int; time: float; logged_at: float
    shot_type: Literal["wide","medium","close_up","crowd","bench","other"]   # from the virtual camera config
    ball: BallInfo | None          # x,y 0..1 in this cam, status detected/predicted/lost, confidence
    players: int                   # in this cam
    play_speed: float              # 0..1, how fast play moves in this view (players' motion, past 1 s)
    event: StoryEvent | None       # latest story event with logged_at <= time (shared by all cams)
    phase: str | None              # latest story phase (in_play / stoppage / goal / celebration ...)
    quality: Quality               # usable, blur 0..1, shake 0..1
    source: Literal["detector","oracle"]
```
Saved to `outputs/studio/<segment>/<run_id>/cards.jsonl`.

## 7. Director v2 (`studio/director.py`)
States and rules (every number in `config.yaml` → `studio.director`):
- **IN_PLAY** — follow the ball between cam1–3 with MVP 1's scoring (ball +100 / predicted +60 / centred
  +0..30 / players). When `play_speed` ≥ `fast_play` and the ball is moving across thirds, prefer **cam4 WIDE**
  (plan rule 1). When the ball is lost: wide after `lost_fallback_seconds`.
- **REACTION** — entered when a story event of type goal (or `reaction_events`) becomes known (its
  `logged_at`), or the phase becomes stoppage/celebration: cut to **cam5 CLOSE** for `reaction_hold_s`, then back
  to IN_PLAY when play restarts (or after `reaction_max_s`).
- **Reject bad feeds** — a cam whose `quality.usable` is false can never be chosen; if it is live it is cut
  away from at once (this is the one cut allowed to break the minimum shot length).
- **Hold every shot** — `min_shot_seconds` (default 3.0, plan says 3–4 s; MVP 1 keeps its 2.0) and hysteresis.
- Every decision has a plain-English reason in `director_log.csv` (frame, time, state, chosen cam, 5 scores, reason).

Goal rule on footage without a scoreboard (MVP 2 relied on the score graphic): a goal counts for REACTION only
if the story layer reports it with confidence ≥ `goal_min_confidence` **and** it is corroborated by a
specialist SHOT/GOAL or a celebration phase within `goal_corroborate_s`. Set in `studio`, not by changing
MVP 2's defaults.

## 8. Build steps
After each step: run its check, fix until it passes, update the "MVP 3" section of `PROGRESS.md`, commit
`MVP3 step N: <name>`, and tell the user in 2–3 simple sentences what was built and how to check it.

### Step 0 — Plan, data access
- This spec; `CLAUDE.md` phases (MVP 2 done, MVP 3 current); PROGRESS.md "MVP 3" section; `.env.example`
  `HF_TOKEN=`; README credit for SoccerTrack v2; commit with the plan doc.
- Check: `HF_TOKEN` works (whoami), one half + its annotations downloaded.

### Step 1 — Intake + calibration check (`studio/soccertrack.py`, `studio/prepare.py`)
- Download script: `python -m studio.soccertrack --match 128057 --half 2` (video, gsr, ball, bas, raw; resumable;
  token from `.env`). Print resolution, fps, frame count (must equal the gsr frame count), audio yes/no.
- Cut the segments from config to `data/studio/<segment>/source.mp4` (keep 4K for the close-up; also a ≤1080p-budget
  copy for detection). Disk is tight (~19 GB free): delete the full half after cutting if needed.
- Project gsr players (and, for checking only, the gt ball) onto frames; save 3 overlay PNGs.
- Check: the overlays land on the players (visual check of 3 frames), frame counts match.

### Step 2 — Virtual cameras (`studio/cameras.py`)
- cam1–3: left/centre/right thirds; cam4: whole panorama letterboxed; cam5: causal virtual close-up from the 4K
  source, centred on the action point (detected ball, else the players' centre of mass weighted to the densest
  group), smoothed with past-only filters and speed limits like a camera operator.
- All cams 1280×720, 25 fps, same frame count (`data/studio/<segment>/cams/camN.mp4`) + a `cams.json` that records
  each cam's crop/zoom per frame (needed for scoring).
- Add a test that the close-up's crop at frame t is identical whether or not later frames exist (causality).
- Check: 5 videos play in sync; the close-up follows play without jumps.

### Step 3 — Detection + cards (`studio/detect.py`, `studio/cards.py`)
- MVP 1 Detector on every cam at `studio.detect_fps` (default 5), cached to disk; or `--oracle` from gsr.
- Feed quality: blur (variance of the Laplacian vs the cam's own recent median) + shake (global frame shift).
  A test mode makes one cam blurry for a configured interval (`studio.test_degrade`).
- Cards once per second per cam.
- Check: cards.jsonl exists; ball rate per cam printed; the degraded interval is marked unusable.

### Step 4 — Story layer on the wide feed (`studio/story_pass.py`)
- Normalise cam4 into a MVP 2 clip folder and run `StoryEngine` headless with a simulated playhead at 1×
  (no browser). Modes: `fast_only`, `gemini_only`, `hybrid` (specialist if the Colab URL is set).
- Gemini budget: ~1 call / 8 s → ~110 calls for the eval segment (free Flash Lite: 500/day).
- Prompt note for this footage: amateur panoramic, no scoreboard, no replays, teams by shirt colour.
- Check: events.jsonl with `event_time` + `logged_at`; delay printed; no crash with silent audio.

### Step 5 — Director v2 (`studio/director.py`)
- As section 7. Unit tests: min shot, hysteresis, rejection beats min shot, REACTION only after `logged_at`.
- Check: on `dev`, director_log.csv shows IN_PLAY following the ball, wide on fast play, and a REACTION after the goal.

### Step 6 — Run + render (`python -m studio.run --segment dev [--oracle] [--story-run <id>]`)
- program.mp4 (with "CAM n · TYPE" label), debug grid (5 tiles + cards + state), director_log.csv, summary print
  (cuts, shot lengths, time per state, time per cam).

### Step 7 — Dashboard "Studio" tab
- Replay of a finished run: program + 5 tiles with red border on the live cam, card under each tile (shot type,
  ball, players, speed, quality), story log, state timeline coloured by cam, truth markers in proof mode.
  Browser `<video>` of the program is the clock. MVP 1 and Story tabs unchanged.

### Step 8 — Evaluation (`python -m studio.evaluate --segment eval --run <id>`)
`report.md` with (gt = ground truth, used here only):
- **Ball on air**: % of in-play time the gt ball is inside the live cam's view.
- **Shots**: shortest shot (must be ≥ min shot, except bad-feed escapes), average shot, cuts per minute.
- **Reaction**: for each gt goal, did a close-up start within `eval.reaction_window_s` (default 20 s) and how late.
- **Story**: precision/recall for goals and shots vs gt within `eval.match_window_s`, average delay.
- **Bad feeds**: % of degraded time the degraded cam was on air (target 0%).
- Mode comparison table (oracle vs detector; fast_only vs gemini vs hybrid story).

### Step 9 — Docs
`TESTING_MVP3.md` (beginner steps), README section, update the plan doc's "what is built" section.

## 9. Definition of done
- One command runs `dev` and `eval` end to end; the Studio tab replays a run.
- Report on `eval` exists. Targets (first version, may change): ball on air ≥ 80% (oracle) / ≥ 60% (detector);
  no shot under min length except bad-feed escapes; degraded cam 0% on air; close-up within 20 s after ≥ 2 of 3 goals.
- MVP 1 quick check and all tests pass.

## 10. Out of scope (next steps in the plan)
NDI/RTSP ingest, OBS/vMix output, colour matching, face recognition, crowd/bench cameras (no such footage),
training models, the learned "sequence model" for shot choice (needs SoccerNet human cuts).
