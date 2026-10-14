# Arm MVP — an AI director on REAL footage (a 5-phone arm-wrestling match)

Status: built (2026-10-14). Code: `arm/` (+ `dashboard/arm_api.py`, `dashboard/static/arm*.html`).
It only USES `src/`, `story/` and `studio/` and never changes their defaults. Progress: "Arm MVP" in `PROGRESS.md`.
How to test: `TESTING_ARM.md`. (Named "Arm MVP", not "MVP 5": `docs/MVP5_VIRTUAL_BROADCAST_PLAN.md` is a different,
football plan.)

## 1. Goal
MVP 1–4 proved camera switching on football datasets. This MVP proves it on **our own footage**: 5 phones filmed one
home-made arm-wrestling match (~32 s, office, real colleagues). The AI must turn the 5 phones into ONE broadcast that
gives the viewer the emotion of the match: the struggle, faces under strain, laughter, the decisive push and the
crowd. It must switch angles AND zoom smoothly, never show a bad shot, and follow TV rules. Two outputs:
- **AI live**: decides every 0.1 s using only the past (it could cut a real live stream),
- **AI editor**: sees the whole match and picks the best sequence (a highlight editor),
and both are compared with simple alternatives (one phone, random cuts, round robin, MVP 1-style "most motion").

## 2. Non-negotiable rules
- Footage shows real people: everything stays in `data/arm/` (gitignored). Nothing is uploaded except **still JPEGs**
  to Gemini, with a **hard cap of 200 requests in total** (counted on disk, `data/arm/cache/gemini_calls.json`).
  Every Gemini answer is cached: a rerun sends nothing.
- Local models do the per-frame work. Gemini is the sparse "second opinion".
- Live mode is causal: frame n uses only frames <= n; a Gemini answer is usable only from `t + latency`
  (`event_time` + `logged_at`, as in MVP 2/3).
- All numbers in `config.yaml -> arm`.

## 3. The footage (measured)
| cam | phone | video | view (found by the camera setup) |
|---|---|---|---|
| A | Samsung | 4K HEVC 30 fps | side (both athletes in profile) |
| B | iPhone 13 | 1080p 30 | behind the tan athlete (low, wide) |
| C | iPhone 15 Pro | 1080p ~30 (VFR) | over the black athlete's shoulder (tan's face) |
| D | iPhone 15 | 1080p ~60 | side, the OPPOSITE side to A |
| E | iPhone 17 Pro | 1080p ~30 | behind black, wide, with the audience |

## 4. Pipeline (`venv/bin/python -m arm.main`)
1. **Sync** (`arm/sync.py`). The metadata start times are accurate only to the second (and the Samsung writes its
   STOP time). Sound gives the exact offset: GCC-PHAT cross-correlation of every camera pair (10 pairs), solved for one
   offset per camera. Every copy is cut to the common window and re-timed to 30 fps CFR (`data/arm/synced/`), the 4K
   phone keeps a full-size copy for sharp digital zoom. Check: `sync_check.mp4` (5 cameras + sound waves).
2. **Labels** (`/arm/label`): the user marks GO, surges, near pins, the pin + winner, reactions, great moments. Used
   ONLY to score the AI.
3. **Perception** (`arm/perceive.py`, cached): YOLO26s-pose (17 body points per person), MediaPipe Face Landmarker
   on a crop around each head (52 blendshapes: strain, joy, mouth open; head yaw), chest colour, sharpness, shake.
   10 looks per second per camera.
4. **Camera setup** (`arm/setup.py`): Gemini looks at each camera's FIRST frame once (5 calls) and boxes the black
   and the tan athlete and the grip, says which athlete is nearest and what the view is. These boxes ("seats") are
   matched to the pose model's people; afterwards every look recognises the athletes locally by overlap.
5. **Match** (`arm/match.py`): per camera — athletes, grip (upright = how high above the elbows; lean = who wins,
   only from cameras behind an athlete, because the pin goes ACROSS the table), faces (strain, joy), head down,
   free hand raised (gesture), audience smiles / clapping / arms up. All cameras together — lean, upright, momentum,
   phase (SETUP → STRUGGLE → SURGE / NEAR_PIN → PIN → REACTION), tension 0..1, crowd 0..1 (YAMNet on all 5
   microphones + audience faces/hands) and moments: LOOK_UP, GESTURE, CHEER.
6. **Shots** (`arm/shots.py`): every phone gives virtual shots by cropping: WIDE, TWO (both athletes), GRIP,
   FACE_BLACK / FACE_TAN, CROWD (max zoom 2x, the 4K phone 3x). Value 0..100 = (fit to the phase + emotion shown +
   fresh-event boost + angle) x clarity.
7. **Live director** (`arm/director.py`): min shot per phase, hysteresis + margin, event cuts after 1 s, staleness
   (cut-aways get old sooner), no ping-pong, the **180° line** chosen once (the side whose two-shot shows more
   emotion), no jump cuts, never an unusable shot, transitions CUT / DISSOLVE / MOVE.
8. **Editor** (`arm/editor.py`): the same values, events known 0.6 s early, best sequence by dynamic programming
   under the same rules.
9. **Judge** (`arm/judge.py`): one 5-camera contact sheet every 2 s (~16 calls, cached) → phase, leader,
   emotions, crowd mood, drama, best camera, caption. Small bonus for its best camera when fresh.
10. **Render** (`arm/render.py`): one decode pass writes every program: smooth virtual operator (eased,
    speed-limited), slow push-in on tension, dissolves, moves, lower third (phase, tension meter, moment caption),
    ONE continuous sound track (phone E).
11. **Proof** (`arm/evaluate.py` → `data/arm/out/report.md`): rules, picture quality on air, athlete faces with
    emotion on screen, crowd joy on screen, variety, moments caught (your labels, else the AI's own events — then
    marked "not independent"); blind A/B vote page `/arm/vote`.
12. **Dashboard** `/arm`: program (live / editor), 5 synced cameras with the on-air one in red, timelines, cut log
    with reasons, Gemini's notes (live: only once known), scorecard.

## 5. Honest limits (see PROGRESS.md for numbers)
- The lean (who is winning) is measured only by the cameras behind an athlete, from 2D forearm tilt: a weak signal.
- Gemini's judge is slow on the free tier (3–27 s per answer: most arrive too late for live) and not selective
  (it calls almost everything "near pin, drama 9" and prefers camera A); its value is as an independent cross-check
  (it saw tan smiling exactly where the local face model fired LOOK_UP).
- A single fixed side phone shows more athlete faces than any mixing program; the AI wins on moments, variety and
  rules. Only the user's labels and the blind vote are independent proof.
- The camera setup uses Gemini on the first frame; no other frame of the match is used for setup.
