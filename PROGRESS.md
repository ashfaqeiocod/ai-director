# Progress — AI Sports Director MVP 1
Last updated: 2026-10-08 19:30

## Status
| Step | Status | Notes |
|---|---|---|
| 0. System check | done | brew, python3.13, git, node present. ffmpeg + yt-dlp come from pip inside the venv (see Decisions) |
| 1. Setup | done | venv, pinned requirements, config.yaml, .gitignore, .env.example, README. Imports OK, MPS True, ffmpeg found |
| 2. Download | done | 22.08 s, 29.97 fps, 1920x1080, 660 frames, audio yes |
| 3. Crop | done | 3 × 640x1080, 660 frames each, same fps as source (verified by counting packets) |
| 4. Detect | done | Roboflow football-players-detection-3zvbc/11 via CoreML, ~300 ms/frame/cam |
| 5. Ball tracking | done | detected/predicted/lost per frame; ByteTrack IDs; 1 outlier rejected |
| 6. Director | done | 0 cuts on this clip (correct, see below); forced test cuts at exactly 2.0 s |
| 7. Render | done | program.mp4 1280x720 h264 + aac, 660 frames; debug.mp4 1278x790 h264 |
| 8. One command | done | `venv/bin/python src/main.py`, ~14 s when cached |
| 9. Dashboard | done | Replay 1× ≈ real time, 2× ≈ 65 fps; Live ≈ 1 fps; controls, WS, MJPEG, downloads tested |

## Key results
### Clip 1: Veo follow-cam (RVWMU6p0hDQ), first test
- Ball rate (frames with a ball detection): cam1 1% · cam2 53% · cam3 0% · **any cam 54%**.
- Ball tracker statuses: cam2 detected 349 / predicted 118 / lost 193 frames.
- Cuts: **0** (cam2 live for the full 22.02 s). With `start_cam: 1` → 1 cut at frame 60 (2.00 s).
- Processing: detection ~300 ms per cam-frame (≈1 fps live for 3 cams); render ~49 fps; dashboard replay up to ~65 fps.
- Memory: dashboard process ~870 MB during a live run.

### Clip 2 (current): last 30 s of a static panoramic camera (HBTFZwMdcCw)
- Source 3840x960 → scaled to 2880x720 (1080p pixel budget), 25 fps, 747 frames; cams 960x720.
- Ball rate (threshold 0.3): cam1 24% · cam2 34% · cam3 15% · any 70%. Detection ~315 ms/frame/cam.
- **5 cuts**, following the ball: cam2 → cam1 (2.0 s) → cam2 (5.3 s) → cam3 (10.7 s) → cam2 (17.2 s) → cam1 (24.0 s).
  Shots 2.0–6.8 s, average 5.0 s. Checked against debug frames at 1/3/5/10/15/20/27 s.
- With threshold 0.15 there were 6 cuts, one wrong (4.2 s → cam3): a static pitch marking in cam3 detected as
  "ball" (conf 0.17–0.30) beat the real ball in cam1 (conf 0.8–0.96). Threshold 0.3 fixed it.
- Clip 1 (Veo follow-cam, 0 cuts) files are kept in `data/archive/RVWMU6p0hDQ/`.

## Decisions made
- `ball_confidence` raised 0.15 → 0.3 (see Clip 2). The engine applies the current threshold when reading
  cached detections too, so changing it needs no re-detection.
- New clip settings: `video.trim_last_seconds` (keep last N s), `video.max_pixels` (1080p pixel budget, so
  wide panoramas keep detail), `detection.model_aspect` (pad any strip to 16:9 before detection).
- Python 3.13 (already installed) is used: `inference` 1.7.3 supports it. No extra Python installed.
- Homebrew has no prebuilt ffmpeg for macOS 14.0 (it would compile for hours, pulling in extra Pythons),
  so ffmpeg comes from the `imageio-ffmpeg` pip package and yt-dlp from pip, both inside the venv.
  Video info uses PyAV instead of ffprobe. Nothing was installed system-wide.
- yt-dlp needs a JS runtime for YouTube now: we pass `--js-runtimes node` (Node v22 is installed).
- Source is re-encoded to constant frame rate so "frame N" means the same moment in every cam.
- Detection runs on CoreML (M1 GPU/Neural Engine): 300 ms/frame vs 5.4 s/frame on CPU.
- Each 640x1080 cam strip is padded (centred) onto a black 1920x1080 canvas before detection. The model
  squashes tall images into its square input and then found almost nothing (0 players). Padding fixed it
  (203 players / 11 balls on 60 sample frames). Each camera still only sees its own pixels.
- "Ball rate under 40%" is measured as **any-cam** rate (54%), not the per-cam average (~18%): with a
  follow-cam the ball is in one strip at a time, so a per-cam average can never pass 33%. Full-frame
  detection agreed with the strip detector on 57/60 sampled frames, so the ball-only model + slicer
  upgrade was not needed (`use_ball_model` / `use_slicer` remain available in config.yaml).
- 0 cuts is correct: on full frames the ball was in the middle third in 60/60 sampled frames (Veo follow-cam).
- `sv.ByteTrack` is deprecated in supervision 0.30 but still works; versions pinned in requirements.txt.
- Dashboard: video files are written by a background thread and the 4 JPEGs are encoded in parallel, so
  replay keeps real time. Stop discards the half-made videos/CSV (the last complete ones are kept).
- Model caches (inference, ultralytics, matplotlib) are kept inside the project (`models_cache/`, `.yolo_config/`).

- Live mode settings (config.yaml `live:` + `detection.onnx_providers`): parallel cams (~25% faster),
  detect_every_n (frame skipping), realtime (detector in a separate low-priority process, so the video
  isn't blocked by Python's GIL). M1 numbers: n=1 ≈ 1 fps; n=3 ≈ 2.6 fps; realtime ≈ 27 fps video with
  ~0.6 detector rounds/s. Only n=1 + realtime=false refreshes the Replay cache.

- **Any-video input** (`input:` in config.yaml, `src/prepare.py`): `split` = one video (link or file) cut in 3;
  `files` = 3 camera clips, each aligned with `start_seconds`, letterboxed to one size, converted to one fps,
  cut to the same frame count. A fingerprint (`data/cams/input.json`) makes main.py redo preparation +
  detection only when the input (or a local file) changed. Input is validated BEFORE old results are deleted.
  Tested: mixed sizes/fps (1280x960@30 + 960x720@25), a 2 s offset (frame-exact: error 2.5 at the right
  frame vs ≥17 one frame off), local-file split, skip/touch detection, 9 error cases, dashboard guard.
- Cross-platform: ONNX providers filtered to what the machine has (CPU fallback); `os.nice` guarded (Windows);
  `src/check_setup.py` checks a new device. Docs: docs/NEW_DEVICE_SETUP.md, docs/USING_YOUR_OWN_VIDEOS.md.

## Problems / open issues
- Ball on a camera border can be seen by 2 cams at once (54 of 747 frames); the director's hysteresis keeps
  this from flickering, but the ball-at-the-edge gets no "center" bonus, so the cut can come a little late.
- ByteTrack IDs churn a bit (79 IDs in cam2 for ~15 people) because players are small and occluded.

## Next action
MVP 1 complete. See TESTING.md.

---

# MVP 2 — Live Story Reader
Spec: `docs/MVP2_STORY_SPEC.md`. Last updated: 2026-10-09

## Status
| Step | Status | Notes |
|---|---|---|
| 0. Plan + environment | done | packages, YAMNet, MVP 1 check OK; Gemini key OK, model gemini-3.5-flash (2–3 s per 6 s window) |
| 1. Intake + live player | done | file/URL/upload → clip.mp4 (≤720p, 25 fps) + audio.wav; Story tab plays with sound; ~5 ticks/s; guard tests 6/6 |
| 2. Fast layer | done | YAMNet (MediaPipe) + loudness + tonal whistle + motion + cuts/replay hint; ~37 ms per 1-s step; cheers 3 s after each of the 3 goals |
| 3. Story engine + logs | done | engine cadence (fast 1 s, story 8 s + triggers, memory 60 s), template narrator, fusion events/score, logs in outputs/story/<clip>/<run_id>/; tests 17/17 |
| 4. Smart layer (Gemini) | done | gemini-3.5-flash-lite (free tier), 27–30 AI lines per 3:22 clip, avg delay 3.3–3.5 s, 429/503/400 + daily-quota fallbacks |
| 5. Memory layer | done | Gemini story so far every 60 s; final summary matches the match (3 goals, white dominated) |
| 6. Specialist (Colab) | done | user's Colab T4 via ngrok: 254 ms per inference, RTF 0.13; shot at 37.0 s reported 3.7 s later |
| 7. Fusion, score, modes | done | 4 modes compared on the same clip (compare.md): hybrid best; no double goals on replays; score white 3 – red 0 |
| 8. Story tab UI | done | full clip plays with sound, all panels update live, no freeze, lag ≤ 0.2 s; Colab off / Gemini quota out fall back |
| 9. Proof mode | done | report.md per run + compare.md; hybrid: goals 3/3, other headline 80%, avg delay 3.8 s; narration 👍 needs user ratings |
| 10. Docs | done | TESTING_MVP2.md, DEMO.md, README MVP 2 section, compare.md |

## Key results (MVP 2)
- Test clips: `eng-cze` (England 3-0 Czechia highlights, 201 s broadcast, youtube kJEY-UFTDk8), `pano` (MVP 1 panoramic, 30 s, no commentary).
- Ground truth `data/clips/eng-cze/truth.csv`: DRAFT written by Claude from frames + on-screen score
  (13 events: 3 goals at 61.5 / 94.0 / 148.5 s, 6 shots, 2 saves, 1 free kick, 1 corner). Please verify.
- specialist_only on the Mac CPU (not live-capable): shots at 37 s and 92.6 s and 123.8 s found; RTF ~7, delay ~35 s.
  6 of 7 truth events up to 100 s had intensity in the run's top 20%.
- Intake: 201 s YouTube clip downloaded + normalized in 67 s; 30 s local file in 3 s.
- Browser → backend playhead: ~5 ticks/s, follows play / pause (checked with headless Chrome).
- eng-cze goals (from the on-screen score): 1-0 at ~60 s, 2-0 at ~93 s, 3-0 at ~146 s. Fast layer CHEER triggers at
  63, 96, 150 s (+ big chances at 26, 39, 126 s; intro jingle at 10 s). Live run: lag ≤ 0.5 s.
- Whistle check: a synthetic referee whistle at commentary level is caught in 198/200 seconds; on the clean
  clip only 4/200 seconds fire (end music / final whistle area).

## Decisions made (MVP 2)
- `google-genai` pinned to 2.8.0: newer versions need pydantic ≥ 2.12.5, but Roboflow `inference` (MVP 1)
  needs pydantic < 2.12. 2.8.0 has everything we need (structured output, inline video, VideoMetadata fps).
- MediaPipe 1.1.0 runs YAMNet on the M1 CPU in ~4 ms per second of audio. Model in `models_cache/yamnet/` (gitignored).
- Fast layer, audio: YAMNet hears mostly "Speech" on TV highlights (crowd classes ≤ 0.09), so excitement is
  35% crowd classes (x3 gain) + 65% loudness vs the median of the last 30 s (past only). YAMNet "Whistle"
  was 0 on the whole clip, so the whistle needs a strict tone check: one narrow peak (≥ 50% of the 2–4.5 kHz
  band energy, ≥ 25 dB above the band median) for ≥ 4 frames in a row. Speech harmonics fail it.
- CHEER = excitement ≥ 0.45 and ≥ 0.25 above the quietest of the last 4 s, max once per 6 s, not in the first 10 s.
- Replay hint = ≥ 3 hard cuts within 8 s. On edited highlights this also fires on normal fast editing, so it is
  only a hint; Gemini's `is_replay` is the main replay signal when Gemini runs.
- Engine: AI calls run in a worker thread; `logged_at` = playhead when the answer ARRIVED. Templates (no AI) are
  used in fast_only/specialist_only and whenever Gemini is unavailable, so the commentary never stops.
- Fusion dedupe: same type within 4 s = one event (goals 60 s, so a goal shown again later can't count twice);
  confidences combine noisy-or; `logged_at` keeps the first report.
- Specialist model facts (read from opensportslib 0.3.1 source + the HF config.yaml): frames 224x398 RGB, /255,
  ImageNet mean/std; stride int(25 / extract_fps 2) = 12 -> ~2.08 fps; clip_len 100 (~48 s); output per frame =
  softmax over background + 12 classes. Server calls `E2EModel.predict()` on a rolling buffer (only frames already
  sent = past). The GRU looks both ways in time, so a frame is reported once it is `report_delay_s` (2 s) old.
- Server code lives in `colab/specialist_server.py` (tested locally on CPU, same code path as Colab) and is copied
  into the self-contained notebook by `colab/build_notebook.py`. Tunnel: ngrok (secret NGROK_AUTHTOKEN) or a
  Cloudflare quick tunnel without an account.
- Only Gemini's `is_replay` blocks new events. The scene-cut replay hint fired on normal highlight editing and
  would have dropped the specialist's 37 s shot (`fusion.fast_replay_blocks_events: false`).
- Default model switched to `gemini-3.5-flash-lite` (fallback `gemini-3.1-flash-lite`) after hitting the free
  tier's 20 requests/day on gemini-3.5-flash. Flash Lite rejects `thinking_budget: 0` with a generic 400: the
  provider retries once without it. A daily-quota 429 switches to the fallback model for the rest of the run.
- Bug found by the causal guard: the engine rounded the window end UP (8.168 -> 8.17 > playhead) -> refused.
  Now rounded down; regression test added. (It had silently dropped the cheer-triggered looks at the goals.)
- Replays: Flash Lite called live goals "replays" (and the scene-cut hint in the prompt pushed it that way). Hint
  removed from the prompt, stricter replay definition, per-event `in_replay` flag (only replayed events dropped).
- Goal dedupe: a 60 s merge window merged goal 2 (33 s after goal 1 in the highlights) into goal 1. Now the
  scoreboard decides: a goal with a scoreboard total above the counted goals is new; the 60 s window still
  merges replays when no new score is shown.
- Previous model note: `gemini-3.5-flash` (fallback `gemini-2.5-flash` on HTTP 503). 45 models on the key; newest Flash
  `gemini-3.8-flash` answered a one-word test in 65 s / 15 s and gave 503 "high demand"; `gemini-3.7-flash` gave
  503 on 2 of 3 real windows and 28 s on the third. `gemini-3.5-flash` on real 6 s windows: 2.1–2.7 s, ~1.5k tokens,
  goal found at 60.8 s (truth 61.5), replay at 78–84 s correctly flagged with no events, scoreboard read (2-0).
  `check_gemini` now skips models slower than `llm.max_test_latency_s` (5 s) on a warm test call.

## Problems / open issues (MVP 2)
- 2026-10-09 user runs on yt-lfzra492gdw (LaLiga Real Madrid–Barcelona, 2:22): 1st hybrid run 3 goals, 2nd only 1.
  Causes found and fixed: (1) 6 s looks every 8 s left a 2 s blind gap -> window_s 9 (overlap);
  (2) a second open Story tab parked at 2:20 sent its own playhead -> only the tab that pressed Go Live drives
  the clock now; (3) Flash Lite misreads the scoreboard sometimes ("3-1" for 2-1) -> goals < 20 s apart always
  merge; (4) Flash Lite invents goals (free kick on the crossbar at 38 s on eng-cze) -> with a visible scoreboard a
  goal is "pending" until the score graphic goes up, dropped after 30 s. The "goal at 115 s" on yt-lfzra492gdw
  was a disallowed goal (referee signalling 2:00–2:03, scoreboard stayed 2-1): now correctly not counted.
- 2026-10-09 longer test: esp-cro (Spain 4-1 Croatia extended highlights, 9:49, Sony LIV ZN7ZYybxpxM), goals-only
  truth drafted by Claude from the score graphic (75.5, 179.5, 214.5, 443.5, 521.5 s), fast + Gemini (Flash Lite),
  Colab offline. Score graphic read correctly every time (0-0 -> ... -> 4-1). Goals at the right moment (±10 s):
  run 1 0/5, run 2 2/5 after fixes (merged goals keep the EARLIEST time; the story prompt no longer gives Gemini the
  counted score, which it copied into its "score on screen" answer). Flash Lite mostly recognises goals in the
  replay 15-25 s later. Tried and REMOVED: re-timing a confirmed goal to the loudest crowd roar between the
  old/new score readings (errors -30 to +100 s, because Flash Lite's score readings lag).
  Conclusion: on the free tier, trust the score graphic for the score; for goal timing use gemini-3.5-flash
  (paid key) and/or the Colab specialist (hybrid).
- NOW card: big score = stable score-graphic reading (same value read twice in the last 3 looks); small line =
  goals counted per colour.
- Team attribution by Flash Lite is unreliable (Barcelona's goal reported as "white"): use the score ± buttons, or
  gemini-3.5-flash with a paid key.
- Specialist-only mode cannot find goals on highlights (the model never fires GOAL there), by design of the modes.
- Free Gemini tier: Flash models 5 req/min + 20 req/day (one clip needs ~30) -> default is Flash Lite (15/min, 500/day).
  Flash Lite over-labels replays as live (replayed shots become false alarms; replayed goals are merged by the
  scoreboard/60 s rule). With a paid key, gemini-3.5-flash recognised replays correctly in tests.
- Narration 👍 target (≥ 80%) is unmeasured until the user rates lines in a run.
- truth.csv for eng-cze is a Claude draft; the user should check it.
- Honest test (no audio to Gemini): 2/3 goals (the goal-2 window had no clear visual of the ball crossing the line).
- Specialist on the M1 CPU: 12–23 s per 48 s window (RTF 6–11): too slow for live, hence Colab.
- The spotting model never fires GOAL on eng-cze (offline, full context): it finds the SHOT for goals 1 and 3,
  misses goal 2. Goals therefore rely on Gemini (or specialist SHOT + Gemini goal).
- Seeking is blocked during a run (live means live; the AI already "saw" that part). The Playhead only moves forward.
- `.env` has no GEMINI_API_KEY yet.

## Next action (MVP 2)
MVP 2 complete. User: rate commentary lines (👍/👎) in a run to measure narration accuracy, check truth.csv,
and see TESTING_MVP2.md / DEMO.md. MVP 1 quick check passed on 2026-10-09 (5 cuts, shortest shot 2.00 s).
---

# MVP 3 — Directed Story
Spec: `docs/MVP3_SPEC.md`. Plan: `docs/AI_LIVE_DIRECTOR_PLAN.md`. Last updated: 2026-10-11

## Status
| Step | Status | Notes |
|---|---|---|
| 0. Plan + data access | done | HF token OK; 128057 2nd half (3.45 GB, 4096x1080, 25 fps, 70,350 frames, audio); SoccerNet NDA requested |
| 1. Intake + calibration check | done | dev segment cut; annotation offset measured (-23 frames); overlays checked by eye |
| 2. Virtual cameras | done | 5 cams x 9000 frames, ~15 fps render (libx264) |
| 3. Detection + cards | done | football model on CoreML (sandbox off), ~0.6-1 frames/s; ball ~16% of frames; parked spare ball ignored |
| 4. Story layer | done | close-up feed (delay ~4 s); goals 0/4 on the panorama, 3/4 camera-only on broadcast |
| 5. Director v2 | done (tuning) | 7 unit tests; oracle smoke test: cuts at exactly 3.0 s |
| 6. Run + render | done (tuning) | program.mp4 + debug.mp4 + logs, ~9 fps |
| 7. Studio tab | built, untested in browser | /studio |
| 8. Evaluation | done (first version) | ball on air (all / not WIDE), WIDE share, shots, reactions + false reactions, story R/P, bad feed, baselines |
| 9. Docs | in progress | TESTING_MVP3.md draft; plan doc to update with the findings |

## Key results (MVP 3)
- Baseline on dev (never cut): always CENTRE = 52% ground-truth ball on air; always WIDE = 100% but 100% WIDE time.
- Detector on this 4K panorama (60 sample frames, 3 cams): football model finds the ball in 9/57 frames (~16%),
  345 ms per cam-frame on CoreML. Tiling (InferenceSlicer 640) 2/57 at 2.8 s; ball-only model + tiling 2/57 at 1.4 s.
  So the plain model stays. Worse than the plan's "4 in 10": the ball is a few pixels on this camera.
- Story on the WIDE panorama (dev, Flash Lite, 44 looks, avg delay 6.2 s): missed the goal at 62 s completely and
  repeated "blue builds from the back" for 6 min. The goal is real (attack on the left goal at 61-62 s, kick-off
  at ~85 s) but players are a few pixels tall on the panorama.

## Decisions made (MVP 3)
- Footage: SoccerTrack v2 (CC BY 4.0). No open dataset has full matches from several real broadcast angles
  with sound (see plan section 14), so the 4K panorama is cut into 5 virtual cameras.
- Match 128057 (M7, a dataset TEST match), 2nd half: 5 goals at frames 2282, 37446, 46854, 53875, 67791.
  Dev segment 00:30-06:30, eval segment 23:00-37:00.
- The dataset's ball track is rebuilt from the annotated events (straight lines between them), so it knows
  the future: scoring only. Per-frame player positions (gsr) may drive an "oracle" mode, always labelled.

- Annotation offset: on dev the annotated players lined up ~23 frames EARLIER than expected. Cause found: MY cut,
  not the dataset. The half's video stream starts at timestamp 1.0 s and the fast seek didn't allow for it, so dev
  starts 24 frames early (checked by decoding the half frame by frame). The dataset IS in sync. dev keeps its
  measured offset (-23, corrects everything; no need to redo it); new cuts select frames by their original
  timestamps (-copyts) and were verified frame-exact (frames 0/250/499 of a 20 s test cut). prepare still measures
  the offset as a safety check (`annotation_offset` in segment.json); lookups use annotation_frame()/segment_time().
- Segment cuts stop decoding right after the segment (-frames:v): the dev cut took 5.5 min because ffmpeg read
  the whole 47-min half.
- Video encoding: libx264. VideoToolbox (hardware) refused 5 parallel sessions ("cannot create compression session").
- Ground-truth player boxes cover the feet area (small boxes), fine for counting and aiming.
- Director: the oracle "players group" gets MVP 1's centred bonus (else it tied with WIDE at 70 and never cut).
- Story feed: default is the CLOSE-up (detector-aimed for honest runs, oracle-aimed for oracle runs), with the
  match sound added. A story run with lag skips or > 3 AI errors is marked invalid and never used.
- Scoring: "ball on air" counts WIDE as always right, so the report leads with ball on air EXCLUDING WIDE time,
  plus WIDE share, false reactions and never-cut baselines (compare.md).

## Problems / open issues (MVP 3)
- SANDBOX: inside Claude Code's sandbox, CoreML cannot write its compiled model to the macOS temp folder, so
  MVP 1's Detector silently falls back to generic YOLO (ball rate ~2%). This also affects MVP 1's live mode when
  run from the sandbox. Detection was therefore run with the sandbox disabled. studio.detect now refuses the fallback.
- A story pass run at the same time as detection fell 240 s behind and lost Gemini calls: discarded. Real-time
  story passes must run alone.
- Detection speed with the real model while other jobs ran: 0.44 frames/s (3 cams); to re-measure alone.
- Gemini free tier: ~500 Flash Lite calls/day; a 14-min eval story pass needs ~120. Plan eval passes per day.
- The HF token was pasted in chat: regenerate it on huggingface.co when setup is done and update `.env`.
- Disk: ~19 GB free before the 6.7 GB download.

- dev results (detector + hybrid story, Colab T4): ball on air excl. WIDE 77% (always-CENTRE baseline 52%),
  WIDE time 20% (69% before the players' group could stand in for an unseen ball), shortest shot 3.0 s,
  degraded cam 0% on air, goal reaction 0/1 (no layer saw the goal).
- Story on the close-up (dev, oracle aim, Flash Lite): avg delay 4.0 s (wide: 6.2 s), saw the kick-off after the
  goal (87.6 s, true ~85 s) but not the goal. Colab specialist (hybrid): no GOAL; many headers/crosses that the
  truth doesn't have (dev truth: 76 passes, 62 dribbles, 1 free kick, 1 goal) -> it doesn't transfer to this camera.
- Audio: no crowd at this amateur match (YAMNet crowd ~0); "cheer" triggers every 10-30 s, so audio can't flag goals.
- eval, ORACLE run (annotated players aim + score; Gemini story on the oracle close-up, 109 looks, 0 errors,
  avg delay 5.0 s): ball on air excl. WIDE 71% (target 80% missed), WIDE 11%, 71 cuts (5.1/min, avg 11.7 s),
  shortest 3.0 s, degraded cam2 0% on air, goal reactions 0/3, story shots recall 40% / precision 67% (delay 10 s),
  goals 0/3. Gemini called "shot" 6 s and 5 s before goals 2 and 3, and "kick-off" 26 s after goal 1: it sees
  the attacks and restarts, never the goal itself.
- eval, DETECTOR run (final, honest; Gemini on the detector close-up, 117 looks, 0 errors, delay 4.2 s):
  ball on air excl. WIDE 74% (target 60% met; always-CENTRE baseline 43%), WIDE 24%, shortest normal shot 3.0 s
  (one 0.24 s shot = the allowed escape from the blurred camera), degraded cam2 0% on air, 135 cuts (9.6/min,
  avg 6.2 s: busier than a human would cut), goals 0/3 found -> 0/3 reactions, 0 false reactions.
  compare.md: outputs/studio/eval/compare.md.
- 2026-10-12 (user feedback): no WIDE on air + PITCH MAP (studio.pitchmap: ball + players on the 105x68 m pitch
  via the fixed camera's calibration, ball remembered 3 s, BOX_ENTRY/BOX_EXIT/BALL_OUT/GOAL_LINE triggers, map
  intensity). Cause of "no ball on air" (dev): 65% = live on CENTRE while the ball was on a wing; half within 2 s
  of the last sighting. Results: dev 75 -> 76% ball on air, 27 -> 31 cuts; eval 74 -> 75%, 135 -> 62 cuts
  (avg shot 6.2 -> 13.3 s); ball position known 56% of the time (16% from detection alone). Studio tab rebuilt as
  one combined dashboard (program, tracked cameras, pitch map with camera views + ball trail, intensity meter,
  goals, story, triggers).
- 2026-10-12 GOAL CAMERAS + GOAL WATCHER + KICK-OFF CONFIRMATION (studio.goalwatch, pitchmap KICKOFF_LINEUP):
  two fixed sharp views on the goals; when the map says play is near a goal, Gemini (no sound) checks that goal
  camera; after a shot it looks again at the aftermath (the celebration tells a goal from a save). MVP 2's LOW media
  resolution missed the dev goal; HIGH detail saw "shot + save", the aftermath look saw "goal, attackers celebrate".
  A goal is COUNTED only when a kick-off line-up (players evenly in both halves, some at the centre) follows within
  60 s. dev: goal found + confirmed, 1/1, 0 false counted (1 false alarm -> extra close-up, not counted).
  eval: close-up within 20 s after 2/3 real goals (target met); counted goals: 494 s correct, 775 s found but
  stamped 10.3 s early (shot at 765 s was the first attempt, the goal came from the second) -> strict 10 s check
  calls it false; goal at 118 s missed; 3 false alarms -> extra close-ups, none counted.
  CORRECTION (2026-10-13): the 60 s confirmation window was chosen AFTER looking at eval's kick-off periods and
  false alarms (100 s would have confirmed the false alarm at 418 s), so eval is no longer a held-out test of the
  confirmation rule. Settings are now frozen and a new held-out segment `test` (128057 2nd half, 2400-2814 s,
  1 goal at 2711.6 s) decides.
- CAMERA-ONLY TEST on real broadcast (SoccerNet, Arsenal 3-4 Liverpool 2016, 2nd half 0-20 min, 4 goals;
  NO sound to the AI, score graphic blacked out, Flash Lite, real time, 114 looks):
  goals 3/4 within +-10 s (608.5, 1051.3, 1123.3 s vs truth 608.4, 1051, 1123; the 4th at 226 s vs 209 s, with
  celebrations reported at 212/218 s), 2 false goals = REPLAYS of goals called live (none in raw camera feeds),
  shots 4/6. So the AI CAN see goals from the cameras alone when the camera shows them well. On SoccerTrack
  (one far panoramic camera) it found 0/4. Conclusion: camera placement/framing (close-ups, a view near the
  goal) matters more than commentary. Note: Gemini still read a score once (another banner), not used for goals.
- Two Claude sessions worked on the project at once (2026-10-12): the other one also ran eval detection, added the
  parked-ball filter (commit 8a8764a) and an eval story pass; they clashed on the Gemini quota. One session only.
- 2026-10-12 ~07:00: the MacBook was on BATTERY (4%) and kept sleeping: renders slowed (eval cams 74 min), and an
  eval story pass "played" while asleep with 0 Gemini looks (marked invalid; story_pass now detects this).
- REAL multi-camera footage (2026-10-12): MUVS (Zenodo 10.5281/zenodo.20708683, CC BY 4.0), fans' phones filming
  Chicago Fire 0-1 Charlotte FC (MLS, 2024-05-15, goal 60') from 3 seats at once, synced to 0.07 s, plus a human's
  camera pick per 3 s. No sound track. `python -m studio.muvs --period 19` -> data/muvs/P19/cam1-3.mp4 (1080p,
  25 fps, 2875 frames each, 115 s), human_cuts.csv, meta.json; 4K originals deleted. P19 = 2nd-half STOPPAGE
  TIME (ribbon: CHI 0 CLT 1, 48:43): Chicago attack, CORNER KICK (~75-105 s, on the ribbon) and a goalmouth
  scramble at ~105 s (likely the keeper's late double save from the match report). Cam views: cam1 high corner,
  cam2 side, cam3 low behind a fan's head (natural "bad feed"). Human picks: mostly cam2 (29 of the last 30).
  Next MUVS stretch: P15 (5.6 min, 3 cams, 7.6 GB). Other searches (PolyU/Zhejiang 5-7 cam set: author emailed
  2026-10-12; Alfheim; MVFouls NDA) are in the "MVP 3 Footage Recommendation" doc.
- No-sound intensity fix (2026-10-12, studio/story_pass.py `without_missing_sound`): story intensity is 0.4 audio +
  0.2 motion + 0.2 specialist + 0.2 llm; on a video with NO sound track audio was stuck at 0 but still counted
  (max 60; same strong motion scored 30). Now audio weight -> 0 for such videos only (same motion: 90). Videos
  with a sound track (MVP 2 clips, SoccerTrack) are unchanged; MVP 2 defaults untouched.
- Vision-only intensity (studio/intensity.py, settings studio.vision_intensity): detector at 5 fps on each cam,
  phone movement removed (phase correlation, sign checked), pressure near goal, player speed, STOP trigger.
  P19 (12.8 min detection, football model): v1 mean 49 inside the corner/scramble window (75-110 s) vs 47
  outside; v2 (only players standing on pitch-green grass, per-second max pressure) 67 vs 57, peak at 111 s
  (just after). Real hit: STOP triggers cluster at 86-93 s = play stopped for the corner. Weak part: the model's
  "goalkeeper" label is unreliable on fan footage (linesman in yellow, stewards = "goalkeeper"), so pressure
  saturates. Player bunching per cam rises on cam2 toward the corner (7.5 -> 12.4) but a high cam (cam1) looks
  bunched all the time. Verdict: a helper signal, not enough alone; meaning must come from specialist + Gemini.
  Tuned on P19 only: check on P15 before trusting any setting. The whole clip is late attacking play, so its
  "calm" baseline is not calm.
- P15 check (unseen, settings NOT retuned): 333 s, ribbon clock 43:34 at 168 s, so ~40:46-46:18 = end of the
  1st half incl. stoppage, 0-0, mostly midfield play (no big chance in sampled frames). Human picks: cam2 60,
  cam3 29, cam1 22 (3 s slots). 42.3 min detection. Mean intensity 38 (P19: 57-67), so a calmer stretch scores
  lower: good. STOP triggers 183 s (keeper alone in the box, likely goal kick), 312 s (ref pointing, players
  standing = foul) are real; 232 s unclear. Biggest peak (88 at 168 s, cam1) is FALSE: midfield play, keeper
  near the camera. Pressure still flips 0<->100 second to second (keeper label). Verdict holds: helper only;
  STOP trigger is the most useful part.


### 2026-10-12 evening: live mode, MUVS, held-out test
- Studio cleaned (old runs deleted, newest per match kept). Shared per-frame brain `studio/pipeline.py`
  (run.py refactored onto it: identical director_log over 3000 dev frames).
- LIVE MODE `studio/live.py` + Studio "Live" switch: wall clock decides the current frame, late frames are skipped
  (like live TV), same pipeline, story (StoryEngine on the close-up, playhead = live clock), Colab specialist
  (hybrid), live goal watcher (`GoalPlanner` shared with offline; one Gemini limiter for story + goal watch),
  MJPEG pictures + WebSocket state; a live run is recorded and appears in Replay. Measured on the M1:
  SoccerTrack dev ~20-23 frames/s shown, lag 0.0 s; MUVS 3x1080p ~19 fps, lag 0.0 s. `--offline` = every frame.
- MUVS in live + replay (`studio/sources.py`: SoccerTrackSource / MuvsSource; `IndependentPipeline`: each phone
  its own ball tracker; no pitch map / goal cams). Hand-held bad-picture check: shake = jitter around the recent
  smooth pan, in 1280-px units, only between back-to-back frames (P19 offline: phone 2 bad 2%, phone 1 19%,
  phone 3 15%; before: 63-72% "unusable" for all). Don't cut TO a phone shaking >= 0.7.
- MUVS score = agreement with the human's pick per 3 s. P19 (tuning): side-view bonus 0/15/30/50 -> 72/85/85/82%
  (always phone 2: 82%) -> bonus 15 locked. P15 (UNSEEN): 49% vs always phone 2 54% (phone 1 20%, phone 3 26%),
  49 cuts, shortest 3.0 s. HONEST RESULT: on real phone footage our rules don't beat "always the best seat";
  the human uses judgement (framing, following the ball across views) the ball/player rules don't capture.
  Neither MUVS period contains the match's goal (60'): MUVS tests camera choice, not goal detection.
- Held-out SoccerTrack `test` (2400-2814 s, goal at 2711.6 s), settings frozen, director without AI:
  ball on air 57% (target 60%: MISSED; dev 76%, eval 75%), always-CENTRE baseline 35%, 21 cuts (3/min), shortest
  3.0 s, degraded cam1 0% on air. So the camera choice generalises less well than dev/eval suggested. Gemini daily quota
  used up today (464/~500 on gemini-3.5-flash-lite): its goal test + the hybrid live test run after the reset
  (12:30 IST) on the same model, so the results stay comparable.

### 2026-10-12 night: ISSIA (6 real stadium cameras) = the main live demo; production-style Studio
- Footage: ISSIA-CNR Soccer Dataset (D'Orazio et al., AVSS 2009, research use), Serie A, 6 FIXED synced 1080p
  cameras, 2 per area (right goal cam1/2, midfield cam3/4, left goal cam5/6), take 2 = 2:00, no sound.
  `python -m studio.issia` moved the 6 AVIs from the repo root to data/issia/raw/, made cam1-6.mp4 (H.264,
  1088-row cameras cropped to 1080) + story_mosaic.mp4 (cams 5|3|1 side by side) in ~3 min.
- Detection (football model, 5x/s, 6 cams): 24.9 min on the M1. Ball rate 10-43% per cam, BUT cam1's 43% was
  mostly the PENALTY SPOT (149 of 600 frames): the parked-ball rule now also runs in pitch metres (0.6 m).
- Calibration: pitch landmarks clicked per camera (config studio.issia.calibration), red lines checked by eye
  (data/issia/T2/calibration_check). Stand B (cams 2/4/6) is on the other side; which way each picture is turned
  was MEASURED: median distance between the same players seen by both cameras of a pair: cam2 mirror_y 0.7 m
  (vs 7.2 / 50 m), cam4 rotated 0.5 m, cam6 mirror_y 0.6 m. So the 6 cameras agree to ~0.5 m and are in sync.
- `studio/multicam.py` CalibratedPipeline: all cameras -> one pitch map (players merged across cameras at 1.5 m,
  ball from the camera nearest the last ball), MVP 1 scoring per camera from the map's ball, DirectorV2.
  Director roles now config (follow_cams / wide_cam / close_cam; SoccerTrack + MUVS unchanged).
- Director tuning (offline, 2:00): first run cut cam3<->cam4 (same area, opposite stands) every 3 s ->
  180-degree rule (cross_line_penalty 40 for the other stand) -> then cam2<->cam4 ping-pong at the view overlap
  -> live_bonus 20 + stand A main side (+10). Result: 12 cuts, avg shot 9.2 s, shortest 3.0 s, all stand A.
  Ball on air when the map knows the ball: 79% (best never-cut camera: cam3 57%; best possible from stand A 94%).
  No ground truth for ISSIA here (the annotation files are not in the folder): these are self-measured numbers.
- Teams (`studio/teams.py`): shirt colour of outfield players (detector class "player"), k-means into 2 teams
  after 150 shirts (~1 s): WHITE vs PURPLE (navy shirts look purple on these cameras; Gemini says "blue", the page
  then uses Gemini's word). Possession = team of the player nearest the ball (3 m): 41-59 / 50-50 / 47-53 per run.
  Score = confirmed goals credited to the team last on the ball (no goal in this 2-min take, so untested).
- Gemini: TWO keys now (GEMINI_API_KEY, GEMINI_API_KEY1). Requests take turns between keys; a key whose daily quota
  is used up (or a 404 model) is skipped per model; chain for MVP 3 in config studio.llm (MVP 2 defaults unchanged):
  3.5-flash-lite -> 3.5-flash -> 2.5-flash -> 3.1-flash-lite (overloaded today: 24-111 s answers). Live: key 1 was
  out for 3.5-flash-lite -> switched to key 2 at once; 12-14 looks per 2 min, delay 1.6-5 s, "story so far" summary.
- Colab specialist (user's T4, RTF 0.12-0.13): on the mosaic it fired ONCE in 2 min (players a few px tall in
  398x224). It now watches the PROGRAM (the camera on air, frames pushed by the live loop, never ahead of the
  playhead): 54 outputs in 2 min (PASS, DRIVE, HIGH PASS, HEADER, OUT). Its PASS/DRIVE signals are now logged
  (SPECIALIST_PLAY) so the page can show them. Gemini keeps the mosaic (whole pitch).
- Live fixes: program.mp4 kept only SHOWN frames (75 s file for a 120 s run -> Replay clock wrong; also MUVS):
  skipped frames now repeat the last picture; the match clock starts after the AI is ready (was 9 s missing);
  shortest shot ignores the last shot (cut by the clip end).
- Studio page rebuilt: setup panel (match, story mode, Colab URL + Test, goal watcher, Gemini keys/models),
  program with TV scoreboard bug (team colours, score, clock) + AI caption (lower third, with its delay), 3x2
  camera grid by stand with director score bars, status strip (cameras/detector/director/Gemini/Colab/goal AI),
  match card (score, possession, intensity, phase), team-coloured pitch map with every camera's view, Live feed
  (AI / Colab / map / cuts filters, emoji) + "story so far", timeline lanes (AI events, Colab model, pitch map,
  camera on air, intensity graph), "What MVP 3 proves" panel with live evidence numbers.
  Two browser limits found: 6 connections per host (6 camera videos blocked the program and API) -> half the
  cameras load from 127.0.0.1 / localhost; this Mac can't resolve cdn.tailwindcss.com -> pre-built
  dashboard/static/studio.css (CDN kept as a backup).
- Live checks in the browser (Playwright + Chrome): setup -> Test Colab -> Go live -> all six lights green,
  lag 0.02 s, ~13-19 fps shown (6 x 1080p decode + 7 MJPEG streams), run valid, appears in Replay.

### 2026-10-13: saves + goal-camera re-checks; what the Colab model really reported
- Checked against the footage (cam1, 52-76 s): ball out ~55 s, throw-in ~60 s (purple), cross ~63 s, keeper
  claims ~66 s, holds + throws out ~74 s. Colab model (program feed): OUT 55.3 s and THROW IN 60.9 s CORRECT,
  HEADER 24.4 s matches Gemini's header duel; it missed the cross and has NO "save" class. Gemini: throw-in right
  but credited to white (thrower is purple); team words unreliable. Goal-camera AI varies run to run (one false
  "goal" at ~70 s, stopped by the kick-off rule; one false "save" at 56 s).
- The page hid Colab events among ~50 PASS/DRIVE lines and left throw-in / out off the timeline: now big icons
  at event time, highlighted feed lines, a pitch-map banner; passes/runs only under the Colab filter.
- SAVE events: a goal look saying "save", or a follow-up look where the keeper holds the ball (new AfterLook
  field keeper_holds_ball), becomes a 🧤 event (goal cam). Live test (Gemini only, Colab offline): look at
  72.8 s "shot saved ~65.5 s", follow-up 84.2 s "keeper catches the ball" -> 🧤 at 65.5 s (true ~66 s).
- RE-CHECK: a story event cross / header / shot / chance / corner / free kick near a goal (or 25+ m into a half)
  asks the goal watcher for one more look at that goal 3 s after the event, even right after a routine look.
  Unit-tested; not yet seen live (Colab was offline and Gemini reported no such event in the test run).

### 2026-10-13: intensity put to work (ISSIA only; SoccerTrack/MUVS settings unchanged) + two live-mode fixes
- Danger camera: map intensity >= 0.5 with the ball in a goal area -> that goal's cameras +30. Offline (every
  frame): goal-area camera live in "danger moments" (21 s of the 2 min) 76% -> 89% (bonus 15: 89%, 45: 91%),
  ball on air 79 -> 78%, cuts 12 -> 11, shortest 3.0 s.
- Goal looks paced by intensity: near a goal, hot (>= 0.5) every 2 s, calm (<= 0.2) every 8 s, else 4 s.
  Live test: 8 looks (12 before), spread over the attacks.
- LIVE FIX 1: skipped frames dropped their detection steps (a 13-fps run kept 324 of 600 map updates). Now the
  cached detections of skipped frames still update the map (600 of 600 even at 1 fps).
- LIVE FIX 2: the 0.6 s switch wait counted decisions, not frames (1.2 s at 13 fps, 15 s at 1 fps; a starved run
  made 0 cuts). Now counted in frames. Offline results identical (every frame is decided there).
- A live run is now INVALID below 8 frames/s shown (a run starved by another app, the Antigravity IDE at ~300%
  CPU, showed 0.9 fps with no lag and was called valid). Clean live re-test still to do with the CPU free.

### 2026-10-13: live mode made smooth and as good as offline
- Light decoding: 960x540 copies of the 6 cameras (`studio.issia` makes cam*_540.mp4): every camera is read
  small (tiles, quality, shirts), only the camera on air in full HD for the program. Offline 2:00 run: 266 s ->
  76 s with the IDE busy; identical director results and teams. Replay tiles use the small copies too.
- LIVE BUG: camera cards (ball / action per camera) were rebuilt only on frames numbered like detection steps;
  live mode skips frames, so cards went stale for seconds. Now rebuilt whenever detections arrived (all three
  pipelines; offline results identical). Live (Gemini + goal watcher): 13.9 -> 22.8 fps shown, ball on air
  60 -> 77% (offline 78%), goal-area camera in danger moments 26 -> 85% (offline 89%).
- Studio: old ISSIA test runs moved to outputs/studio_archive/ (kept: 220319 + 220544 hybrid, 072441 + 082637
  Gemini); run picker = ISSIA first, invalid / no-story runs only with "all runs"; Replay fills in as it plays.

### 2026-10-13: full Hybrid live run (all fixes), run 20261013-083406-live
- Colab T4 connected (RTF 0.13, 57 inferences), Gemini 14 lines (avg delay 2.5 s) + story summary, goal watcher
  15 looks, 23.2 fps shown, lag 0.0 s, valid. 11 cuts, shortest 3.0 s, ball on air 72%, goal-area camera in
  danger moments 79%, possession 50/50 WHITE vs PURPLE.
- Colab this run: 29 PASS, 27 DRIVE, 1 BALL PLAYER BLOCK at 68.0 s (0.81) = the keeper's claim (~66 s): its
  nearest class to a save. No throw-in/out/header this time (same video, other runs had them: threshold edge).
- Goal-camera saves: 65.5 s TRUE (both follow-ups "keeper catches the ball and holds it"); 20.9 s FALSE (a duel in
  the left box, defenders clear, keeper never touches it; its follow-ups saw no keeper with the ball).
  -> rule: a 🧤 save counts only when a follow-up look sees the keeper holding the ball. The single-look "shot"
  at 20.9 s is still shown (shots are not confirmed by follow-ups).

### 2026-10-13: Alfheim = the final MVP 3 match (a full 47-min professional half, 3 real fixed cameras)
- Searched the web for full matches with several real cameras: the only public one is Alfheim (Simula, Tromsø IL,
  2013; non-commercial research use). SoccerTrack v2 = one stitched panorama; ISSIA / Fu et al. = 2-min clips;
  WorldPose / SoccerNet multi-view = short clips on request. Using only the video, not the dataset's ZXY sensors.
- Quality check (our detector, 3 x 30 s): players ~35 px tall (ISSIA ~97), ball 6-11 px (ISSIA 14-18), ball seen by
  any camera ~50% (ISSIA 85%). The 3 cameras stand together at the halfway line, so every shot is wide. All 3 share
  piece numbers + timestamps (frame-synced for 47 min). Night: strong green cast -> colour fix (ball 34 -> 38%).
- `studio/alfheim.py`: download (resumable), join 3-s pieces, colour fix, 640x480 light copies, story mosaic,
  calibration. Calibration lessons: landmarks alone put the near touchline ~3 m (100+ px) off (oblique view);
  lens straightening looked like the fix but the raw lines are straight (< 3 px bow) and it bent them (-8 px):
  removed. Fix = line hints ("these white pixels are the near touchline") + every pitch line pulled onto the
  real white lines with evenly spaced points: median gap 0.6-1.2 px. Neighbouring cameras still place the
  same player 1-1.5 m apart (+ ~2 m noise) -> merge radius 3 m for Alfheim.
- `AlfheimSource` reuses ISSIA's calibrated brain (CalibratedPipeline, goal watcher, map); director: midfield
  camera = main camera (+10), a goal camera in its third and in danger moments. Offline dress rehearsal on 90 s:
  12 cuts, shortest 3.4 s, box entries seen. Detection checkpoints every 300 frames (a run takes ~4.6 h).

### 2026-10-13 evening: live test environment (docs/MVP3_LIVE_TEST_PLAN.md), steps 1-2
- START FROM ANYWHERE (step 1): `StoryEngine(start_s=...)` / `SpecialistClient(start_s=...)` (default 0: MVP 2
  unchanged), `studio.live --start 23:40`, Studio "Start at" (click the whole-match timeline in the setup panel).
  Seeks checked frame-exact (seek vs decoding up to the frame: 0 pixel difference, ISSIA + Alfheim, small + full).
  Offline ISSIA 40 s from 0:00: director_log, map, events, cards BYTE-IDENTICAL to before. From 0:40: first story
  window 40.0-48.0 s, 0 LAG skips, first shot measured from the start (8.6 s, not 48 s), all files in match time;
  Replay plays program.mp4 with the start offset (tiles at match time, checked in Chrome).
- Alfheim sources load with missing / partial cached detections (live/info reports `cached_until`; the setup
  panel warns when "Start at" is past it). The full-half detection is running (cached up to ~2 min at 16:20).
- COLAB DETECTOR (step 2a): `colab/detector_server.py` (Roboflow football model, same output as src/detect) in its
  OWN uv environment on Colab (inference-gpu can't break the action model's torch), the main server relays WS
  /detect (one tunnel URL). NOT YET RUN ON COLAB. Local check through the relay (Mac, CoreML, 3 Alfheim frames x
  3 cams vs cached): same balls, boxes within 1 px; JPEG q85 loses 1-2 players at the 0.3 edge, q95 none.
  Size: q85 ~0.5 MB per frame for 3 cams -> ~19 Mbit/s upload at 5/s (q95 x1.6; 960x720 q90 ~0.37 MB).
- LIVE LOOP (2b/2c): `--detector colab`: answers applied when they arrive, oldest first, with their own frame
  time; > 1.5 s old or missing -> cached detection of that frame (counted; `cache_fallback: false` = pure live).
  Fake GPU (cached answers after 40 ms), Alfheim 0:30-1:30: same 7 cuts as cached mode, each 0.07-0.36 s later
  (detections applied ~0.1 s after their frame) = honest live behaviour. 143/150 frames from the detector.
  Mac cost: CLI 26.5 fps shown (cached 27.6). With Chrome open: reading the 3 full-HD detection frames IN the loop
  dropped it to 8.5 fps -> now read in the sender's thread: 16.3 fps (background Alfheim detection running too).
  OpenCV here has no hardware decoding (VideoToolbox: off). Real CoreML detector as the "Colab": 8.7 s round trip
  -> every answer late -> all from the cache, no crash.
- Fixes found on the way: Alfheim's 4:3 picture crashed program.mp4 recording (now letterboxed); detection boxes on
  4:3 tiles were drawn as if 16:9 (offset).

### 2026-10-13 evening: live test step 3 (latency) + checks
- `studio/latency.py`: every AI answer = happened -> known (Gemini looks with model + request seconds from its call
  log, Gemini events, goal-camera looks, Colab action events, the Colab detector per second + every late frame),
  latency.jsonl + live messages + run.json summary (avg / p90 / max). Older runs: derived from their logs.
  Studio: latency lane (a row per AI), delay strip, "All events" tab (filter, click = jump). Live timeline grows in
  5-minute steps (a 5-minute test on a 47-minute timeline was a sliver).
- First Alfheim run with the story layer: the mosaic clip intake took 6 min (once); the page now says "preparing".
- Test run Alfheim 1:00-1:55 (Gemini + goal watcher + FAKE GPU detector, Chrome open): Gemini avg 8.2 s / p90 17 s,
  goal AI avg 15 s, detector 0.83 s, BUT 3.5 frames/s shown -> INVALID: the Antigravity IDE used ~380% CPU, two
  other Claude sessions + the background Alfheim detection ran too. Not a fair measurement; repeat with the Mac free.
- Replay races fixed (an older run list / run answering late overwrote the newer one).
- MVP 1 quick check: same as before (ball rate 24/34/15%, 5 cuts, shortest 2.00 s). pytest: 92 passed.
- Memory: a loaded Alfheim source holds its detections: ~112 MB per 10 min cached -> ~0.5 GB for the full half. The
  dashboard now keeps ONE copy (shared by Replay, camera videos and live runs; reloaded only when the detection
  files change) and Replay sends only the run's own stretch of overlays. The background detection itself uses
  ~1.1 GB and ~150% CPU: with Chrome open a live run fell to ~4 fps while it ran (CLI: Colab mode 23 fps, cached 26).
- `python -m studio.compare_detectors --segment ... --colab-run <id> [--cached-run <id>]`: the step 2d table.
- Final checks: offline ISSIA 40 s from 0:00 still byte-identical after all changes; ISSIA from 0:30 (template
  story): first window 30.0-38.1 s, first "story so far" at 90.2 s (start + 60 s), 0 LAG. Replay tile videos no
  longer load the detections (only file paths). Colab notebook: Cloudflare tunnel by default (ngrok free = 1 GB/month,
  the detector sends ~7 GB per half). Not done: latency stats in report.md (they are in run.json).

### 2026-10-13/14: the detector on Colab (T4), first real runs
- Try 1: crash on the first convolution (CUDNN_BACKEND_API_FAILED, cuDNN 9.24): inference-gpu 1.7.4 brings torch for
  CUDA 13 + nvidia-cudnn-cu13 9.24, onnxruntime-gpu 1.21.1 is built for CUDA 12. Fix: Python 3.11 env,
  onnxruntime-gpu[cuda,cudnn] and ort.preload_dlls() BEFORE inference imports torch -> runs on the GPU.
- Plain CUDA: 535 ms per camera frame on the T4 (= the M1 with CoreML): the model is big (273 MB, 1280 input).
- TensorRT (fp16, engine cached; inference sets these options itself): 70 ms per camera frame (8x). Needs
  tensorrt-cu12-libs from pypi.nvidia.com (uv could not install the tensorrt wrapper). Engine build ~10-15 min.
  70 ms x 3 cams = 210 ms per detection step vs 200 ms needed at 5/s: just under live speed (some steps -> cache).
- Colab interrupt (■) killed the Cloudflare tunnel (error 530): it now runs in its own session.
- Notebook updated with all of this (Run all ~20-25 min). Not yet re-run from scratch.

- STEP 2d (real Colab T4 + TensorRT through the Cloudflare tunnel, Alfheim 1:00-2:00, run 20261013-221004-live):
  GPU 77 ms per camera frame, but round trip 580 ms at ~12 Mbit/s upload (0.5 MB per detection step):
  93/300 detection steps from the GPU in time, 200 from the cache (late), 25 answers ignored as too late;
  detector delay avg 1.34 s / p90 1.62 s. 4 in flight instead of 2: round trip 1,132 ms, 8.5 Mbit/s -> worse (the
  home UPLOAD is the limit, not the GPU). Parity on the 93 GPU frames vs cached (Mac CoreML, full quality):
  | cam | ball % Colab | ball % cached | same ball (10 px) | players Colab | players cached |
  | 1 | 4.3 | 7.5 | 75% | 4.6 | 4.33 |
  | 2 | 16.1 | 20.4 | 100% | 15.57 | 15.71 |
  | 3 | 54.8 | 69.9 | 96% | 17.03 | 17.14 |
  Players match; the ball is found ~20-25% less often (JPEG q85 + TensorRT fp16 push borderline balls under 0.3).
  Mac shown fps 10.5 with the IDE + 2 Claude sessions + background detection running.

## Next action (MVP 3)
Decision 2026-10-14: proof runs with the CACHED detector first (live Gemini / goal watcher / Colab action model).
1. Mac free (close the IDE, other Claude sessions; Gemini quota: one session only), `python -m studio.gemini_check`.
2. Alfheim Live, Hybrid, cached detector: 5 min from 0:00 and from 23:40; from 41:00 once the cache reaches ~46:00
   (the background detection: 38:59 cached at 2026-10-14 ~03:40, ~1.5 h left).
3. The full half (47 min), same settings. Then PROGRESS + TESTING_MVP3 0A.
Colab detector: "Later tests" A-D in docs/MVP3_LIVE_TEST_PLAN.md (pure live at the upload's rate, 960 px, q95 /
no fp16, a faster upload).
