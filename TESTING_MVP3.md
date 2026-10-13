# How to test MVP 3 — Directed Story

MVP 3 combines MVP 1 (camera switching) and MVP 2 (story reading) on a **real match**:
one 4K panoramic recording from the SoccerTrack v2 dataset, cut into virtual cameras
(LEFT, CENTRE, RIGHT, CLOSE-up on air; WIDE is kept for the record only). A director picks the camera every
frame using a "camera card" per camera (ball, players, picture quality, story) and a PITCH MAP (ball + players
on the pitch, ball remembered for 3 s, ball-in-the-box / out / goal-line triggers, map intensity). A report
scores the result against the dataset's ground truth.

Spec: `docs/MVP3_SPEC.md` · Plan: `docs/AI_LIVE_DIRECTOR_PLAN.md` · Progress + findings: `PROGRESS.md` (MVP 3 section).

Run every command from the project folder:

```bash
cd ~/Documents/ai-director
```

> **Important (Claude Code sandbox):** run detection in your own Terminal, not inside Claude Code's sandbox.
> Inside the sandbox, Apple's CoreML can't write its temp files, the football model fails to load, and the
> generic fallback finds almost no balls. `studio.detect` stops with a clear message if that happens.

---

## 0A. The full half (Alfheim): the final proof

A whole professional half (47 min) filmed by 3 real fixed cameras (Alfheim dataset, Tromsø IL - Strømsgodset,
2013; non-commercial research use). All three stand at the halfway line: cam1 = left third, cam2 = middle,
cam3 = right third. A match the director was never tuned on, 20x longer than ISSIA.

```bash
venv/bin/python -m studio.alfheim              # download ~2.8 GB (resumable) + join + colour fix (~45 min)
venv/bin/python -m studio.alfheim --calibrate  # pitch calibration (check: data/alfheim/2013-11-03_H1/calibration_check/)
venv/bin/python -c "from pathlib import Path; from studio.intensity import detect; detect(Path('data/alfheim/2013-11-03_H1'))"
                                               # own Terminal, ~4.6 h; saves every minute and resumes if stopped
```

Check the calibration pictures: the red lines must sit on the white pitch lines (also the near touchline at the
bottom). Then Studio -> ● Live -> match "Alfheim · Tromsø IL - Strømsgodset · full half 1" -> Hybrid -> Go live.
Expect: 3 tiles in one row, the director mostly on the midfield camera and switching to a goal camera when play
goes there, the pitch map with both teams (red/white vs dark blue), box entries, goal-camera looks.
Honest limits: players are small (~35 px), the ball is found less often than on ISSIA, all shots are wide.

## 0. The 6-camera match (ISSIA): the main demo

Six REAL fixed cameras of one Serie A match (ISSIA-CNR Soccer Dataset, 2 min, no sound), two per area:
right goal (cam1, cam2), midfield (cam3, cam4), left goal (cam5, cam6). Cameras 1/3/5 are on stand A,
2/4/6 on the opposite stand B. Every camera is calibrated to the pitch, so all six feed ONE pitch map.

### 0a. Prepare (only once; already done for take 2)

Put the six `Film Role-0 ID-<cam> T-<take> ....avi` files in the project folder, then:

```bash
venv/bin/python -m studio.issia            # moves the AVIs to data/issia/raw/, makes cam1-6.mp4 + story mosaic (~3 min)
venv/bin/python -c "from pathlib import Path; from studio.intensity import detect; detect(Path('data/issia/T2'))"
                                           # football detector on all 6 cams, 5x per second (~30 min, own Terminal)
venv/bin/python -m studio.issia --calibrate
```

Check: open `data/issia/T2/calibration_check/cam1.jpg` … `cam6.jpg`. The red lines (the calibrated pitch) must
sit on the painted lines. The `--calibrate` output prints, for each stand-B camera, how far apart it and its
stand-A partner put the same players (metres); the chosen "turn" has the smallest distance.

### 0b. Go live

```bash
venv/bin/python -m dashboard.server
```

Open http://localhost:8000/studio → **● Live**. The setup panel opens:
1. **Match**: ISSIA Serie A · 6 stadium cameras.
2. **AI story layer**: Hybrid (Gemini + Colab), Gemini, Colab only, or Off (saves quota). The line under it shows
   how many Gemini keys were found in `.env` (`GEMINI_API_KEY`, `GEMINI_API_KEY1`, …) and the model order.
3. **Colab**: paste the URL, press **Test** (green dot = connected; it switches the story to Hybrid).
4. **Goal watcher** on → **● Go live**. You should see:

- the program (big picture) and the six cameras in a 3 x 2 grid, laid out as they stand around the pitch
  (top row stand A: left goal · midfield · right goal; bottom row stand B), red border = on air;
- a coloured bar under each camera = the director's score for it; it cuts when another camera wins for 0.6 s
  and never before 3 s;
- the pitch map: dots from all six cameras merged, the ball (filled = seen, hollow = remembered), the live
  camera's view filled, the other cameras as dashed outlines;
- when play gets near a goal: "AI checking goal…" on that goal's camera (Gemini looks at its last seconds);
- the status strip on the right: Cameras, Detector, Director, Gemini, Colab, Goal AI (green = working);
- the TV scoreboard on the program: the two teams named after their shirt colours (learned in the first second),
  score and match clock; the match card: possession, intensity (🔥 hot / ⚡ building / 😌 calm), the AI's phase;
- the AI caption at the bottom of the program (what Gemini says is happening, with its delay), and the
  "📖 story so far" summary above the Live feed;
- the Live feed with emoji, filterable: 🧠 AI (Gemini + goal looks), 🤖 Colab (➡️ pass, 🏃 run with the ball,
  🎯 shot …), 🗺️ map (📦 ball into the box …), 🎬 cuts;
- the timeline: lanes for AI events, Colab model outputs and pitch-map triggers, the camera on air, and the
  intensity graph (blue = AI story, amber = pitch map, yellow ticks = goal-camera looks);
- **What MVP 3 proves** (top bar): the five claims, each with live evidence numbers.

When the 2 minutes end, the run appears in **Replay** (newest first): **Load**, press ▶.

### 0c. Connect Colab (optional, needs a free GPU)

1. Open `colab/specialist_server.ipynb` in Google Colab, Runtime → GPU, run all, copy the URL it prints.
2. In the Live bar paste it into **Colab URL** and press **Test**: green dot = connected (model, GPU, fps shown).
3. Choose story **hybrid** (Gemini + Colab) and Go live. The "Colab" light next to Gemini shows its state.

No GPU ("Cannot connect to GPU backend: usage limits")? Leave the URL empty: everything else works.
The Colab model watches the PROGRAM (the camera on air), like the TV pictures it was trained on; Gemini watches the
3-camera strip so it sees the whole pitch.

### 0d. Notes
- Open the page as `http://localhost:8000/studio` (or `127.0.0.1`): half the camera videos load from the other
  name, because a browser allows only 6 connections per host.
- The page's styles are pre-built in `dashboard/static/studio.css` (works offline). After adding new Tailwind
  classes to `studio.html`, rebuild it: `npx tailwindcss@3.4.17 -i in.css -o dashboard/static/studio.css --minify`
  with `content: ["dashboard/static/studio.html"]` and a safelist for `grid-cols-1..6` (the CDN covers it meanwhile).

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
| story | Gemini lines as the clip "plays" in real time (on the close-up) | 6 / 14 min (real time) |
| run (x2) | `CUT` / `REACTION` lines, then a summary | 15 / 35 min each |
| evaluate | the report table (below) | seconds |

Real-time story passes must run **alone** (no detection or rendering at the same time). A pass that falls
behind is marked invalid automatically and not used.

## 3. Check the pictures line up

Open `data/studio/dev/check_04500.png`: green boxes (annotated players) should sit on the players' feet,
the yellow circle (ground-truth ball) near a player.

## 3b. Fastest way to SEE it (everything is already prepared and run)

```bash
venv/bin/python -m dashboard.server
```
Open http://localhost:8000/studio, choose the newest **eval** run, **Load**, press ▶.

## 4. Watch a run

```bash
venv/bin/python -m dashboard.server
```

Open http://localhost:8000/studio, pick a run, **Load**, press play:
- big video = the program (what viewers would see), with the camera name and the director's state;
- tiles below = the on-air cameras with the detector's player boxes and the ball (green circle), red border = live;
  under each, its card (shot type, players, ball seen/remembered, picture OK/BAD);
- NOW panel = what the director is doing, in words, plus the latest AI commentary line;
- GOALS SEEN BY THE AI + INTENSITY (map = from the pitch map, story = from the AI), with a 60 s graph;
- PITCH MAP = players, the ball and its trail (hollow = remembered), what each camera sees on the pitch
  (live camera highlighted), the penalty box glows when the ball is in it; next to it, the MAP triggers
  (ball into the box, out of play, over the goal line) and every director decision;
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
