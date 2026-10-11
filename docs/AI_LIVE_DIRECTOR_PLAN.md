# AI Live Director — Product & Validation Plan

> **Draft, living document. Last updated: 2026-10-11.**
> This is the Markdown version of `AI Vision Mixer Validation Plan (1).pdf` (October 2026), expanded with
> what we need from the AI, what is already built (MVP 1 and MVP 2), what is still missing and what is still
> undecided. **The plan will change.** Treat every "proposed" part as a direction, not a commitment.
> Items in `[brackets]` are not decided yet.

---

## Contents
1. [The problem](#1-the-problem)
2. [The idea](#2-the-idea)
3. [The output we need: a live description of every camera](#3-the-output-we-need-a-live-description-of-every-camera)
4. [How it works: four stages](#4-how-it-works-four-stages)
5. [What the AI sees](#5-what-the-ai-sees)
6. [Director logic: broadcast rules](#6-director-logic-broadcast-rules)
7. [Storytelling: a stack of small models](#7-storytelling-a-stack-of-small-models-proposed)
8. [The output: the AI commands the switcher](#8-the-output-the-ai-commands-the-switcher)
9. [Where it fits](#9-where-it-fits)
10. [Current limitations](#10-current-limitations)
11. [What is built so far](#11-what-is-built-so-far)
12. [Technical risks](#12-technical-risks)
13. [Compute plan](#13-compute-plan-proposed)
14. [Validation: datasets and the virtual stadium](#14-validation-datasets-and-the-virtual-stadium)
15. [Roadmap](#15-roadmap-draft)
16. [Success criteria](#16-success-criteria)
17. [Open decisions](#17-open-decisions)
18. [Glossary](#18-glossary)
19. [Changelog](#19-changelog)

---

## 1. The problem
**Every live cut still depends on one person.**

| | Problem | What it means |
|---|---|---|
| 01 | **Many feeds, one pair of eyes** | A director has to watch every camera at once and pick the next shot in real time. |
| 02 | **Quality rides on attention** | A late cut, a blurry feed or a missed reaction goes straight to air. |
| 03 | **Small productions go without** | Events that cannot staff a control room fall back to a single fixed camera. |

Multi-camera coverage is hard to staff, and each broadcast depends on a single seat.

## 2. The idea
**A virtual director that watches every camera and makes every cut.**

It takes in several camera feeds, analyzes them in real time and switches to the best angle automatically,
with no one at the mixer.

Example of what the director "sees" at one moment:

| Camera | Shot | State |
|---|---|---|
| CAM 1 | Wide | **ON AIR** · action tracked |
| CAM 2 | Close-up | Standby |
| CAM 3 | Crowd | VIP recognized |
| CAM 4 | Bench | Blurry · **rejected** |

---

## 3. The output we need: a live description of every camera
**This is the core of the whole system.** About once a second, for each camera, the system must know:

- **what the shot shows** (wide, close-up, crowd, bench),
- **where the ball is**,
- **how fast play is moving**,
- **who is in the shot**,
- **whether something just happened** (a goal, a foul, a shot…),
- **whether the picture is usable** (sharp, steady).

With that, choosing the best camera is straightforward: it is the director rules in section 6 applied to
these descriptions. Without it, the director is blind.

### 3.1 The per-camera description ("camera card")
One record per camera, about once per second. This is a **proposed contract**: the field names may change,
but every layer (detectors, audio, Gemini, specialist models) should write into this one shape, and the
director should read only this shape.

```jsonc
{
  "cam": "cam1",
  "time": 61.0,                 // seconds on the shared clock (when the frame was shown)
  "logged_at": 61.3,            // clock time when this description was ready (shows the delay)
  "shot_type": "wide",          // wide | medium | close_up | crowd | bench | other
  "ball": { "visible": true, "x": 0.42, "y": 0.61, "status": "detected", "confidence": 0.82 },
                                // status: detected | predicted | lost ; x/y are 0..1 of the frame
  "play_speed": 0.7,            // 0..1, how fast play is moving in this view
  "people": {
    "players": 9,
    "ids": ["white#10", "red#4"],   // jersey colour + number when readable, else track id
    "key_person": null              // e.g. "coach", "VIP": see privacy note in section 17
  },
  "event": { "type": "shot", "event_time": 60.4, "confidence": 0.7, "source": "specialist" },
                                // null when nothing happened
  "quality": { "usable": true, "blur": 0.1, "shake": 0.05 },
  "audio": { "excitement": 0.8, "whistle": false }   // shared by all cameras (one venue sound)
}
```

Rules for this record (they come from what MVP 2 taught us):
- **Causal:** it may only use video and audio up to "now", never the future.
- **Two times:** `time` (when it happened) and `logged_at` (when the AI knew). The gap is the honest delay.
- **Partial is fine:** a fast field (ball, quality) can update every second while a slow field (event, story)
  updates every few seconds. The director always uses the latest value of each field.

### 3.2 Where each field comes from today
| Field | What we have now | Status | What is missing |
|---|---|---|---|
| `ball` | MVP 1: Roboflow football model per camera + our `BallTracker` (detected / predicted / lost) | **Works, weak** | Ball found in only part of the frames (see section 10) |
| `people.players` | MVP 1: same detector (player, goalkeeper, referee) + ByteTrack IDs | **Works** | IDs change often (small, overlapping players) |
| `people.ids` (who) | Nothing | **Missing** | Jersey number reader, team colour |
| `shot_type` | Nothing (MVP 1 cams are crops of one wide shot, all "wide") | **Missing** | Shot classifier |
| `play_speed` | MVP 2 fast layer: motion intensity (OpenCV) on the whole clip | **Partly** | Per camera; motion of play vs camera pan |
| `event` | MVP 2: Gemini story lines (~3–4 s late) + Colab action-spotting model + audio cheer/whistle | **Partly, offline-tested** | Live, per camera, reliable goals |
| `quality` | Nothing | **Missing** | Blur/shake check (simple image maths, see section 6 rule 4) |
| `audio` | MVP 2 fast layer: YAMNet + loudness + whistle tone check | **Works** | Tuned on TV highlights only, not venue sound |

---

## 4. How it works: four stages
**Four stages turn raw feeds into a finished program.**

```
 01 INGEST          02 ANALYZE           03 DECIDE            04 SWITCH
 Feeds in    ──▶    AI vision     ──▶    Director logic ──▶   Clean cut
 every camera       per-camera           rule-based state     command to the
 at once            description (§3)     machine              switcher
 NDI·RTSP·SRT       detectors, audio,    broadcast rules      OBS Studio · vMix
                    story models
```

| Stage | What it does | Planned tools |
|---|---|---|
| 01 · Ingest | Every camera arrives at once, with near-zero latency. | NDI · RTSP · SRT |
| 02 · Analyze | Reads color, action and faces on every feed; produces the camera card (§3). | YOLO-family detector · pose · audio · story models |
| 03 · Decide | A rule-based state machine picks the next shot. | Broadcast rules (§6) |
| 04 · Switch | A command tells the switcher which camera to take. | OBS Studio · vMix |

### 4.1 Two speeds (important)
The plan's target is **under 100 ms** from reading a frame to commanding the switcher. That is realistic only
for the **fast loop**. Understanding the story (goals, fouls, who did what) takes **seconds**, because the
models need to see a few seconds of play. So the system runs at two speeds:

| Loop | Speed | Inputs | Decides |
|---|---|---|---|
| **Fast loop** | target < 100 ms | ball, players, play speed, feed quality, audio level | "Follow the action", "drop that blurry feed" |
| **Story loop** | seconds (today 3–4 s) | events, phase of play, story so far | "Play stopped after a goal → show the scorer / crowd" |

The director combines both: the fast loop keeps every cut on time, the story loop chooses *which kind* of
shot comes next. The 100 ms success criterion (§16) applies to the fast loop only.

## 5. What the AI sees
**The AI reads color, action and faces on every feed.**

| Capability | What it does | Planned tools |
|---|---|---|
| **Color calibration** | Matches every feed to a master camera, so cuts never jump in color. | Histogram matching · dynamic LUTs |
| **Action tracking** | Follows the ball and the players, and reads body pose to find the play. | YOLO-family detector · MediaPipe pose |
| **VIP priority** | Recognizes key people and marks their camera as high priority. | Face recognition (e.g. DeepFace), **open decision: see §17** |
| **Feed quality** | Detects blurry, shaking or out-of-focus feeds. | Sharpness + motion checks (simple image maths) |

Analysis runs on downscaled **480p/720p** copies; full-resolution feeds are kept for the output.

> Note: the plan names YOLOv8. The prototype actually uses a Roboflow football model
> (`football-players-detection-3zvbc/11`) with Ultralytics `yolo26s` as fallback. Which detector we keep is
> decided by the compute benchmark (§13), not by name.

## 6. Director logic: broadcast rules
**Four broadcast rules keep every cut clean.**

1. **Wide during high action.** Stay on wide angles while play is fast (e.g. a counter-attack or a basketball fast break).
2. **Reactions when play stops.** Cut to close-ups, the crowd or a key person as soon as the action pauses
   (goal, foul, ball out).
3. **Hold every shot.** No cut faster than the minimum shot length, so the picture never flickers.
   (Plan: 3–4 s. MVP 1 uses 2.0 s: **open decision, §17**.)
4. **Reject bad feeds.** Blurry, shaking or out-of-focus cameras are never put on air.

The rules are coded as a **state machine**, so cuts are deliberate and never jarring. Proposed states:

```
            ball in play, fast                    play stops (whistle, goal, ball out)
 IN_PLAY  ───────────────────────▶  stay wide   ─────────────────────────────▶  REACTION
   ▲   follow the ball (MVP 1 scoring:                                         close-up / crowd /
   │   ball +100, predicted +60, centered +0..30,                              scorer, held ≥ min shot
   │   players +1 each, hysteresis 0.5 s)                                           │
   └──────────────────────────── play restarts (kickoff, throw-in, free kick) ◀─────┘
 Any state: a feed with quality.usable = false can never be chosen.
```

MVP 1 already implements the IN_PLAY part (score each camera by the ball, minimum shot length, hysteresis).
The REACTION state and feed rejection need the missing fields from §3.2 (`shot_type`, `event`, `quality`).

## 7. Storytelling: a stack of small models (proposed)
**A stack of small models gives the director a sense of story.**

| Job | Model / approach | What it gives |
|---|---|---|
| **Detect events** | E2E-Spot · T-DEED (action spotting) | Goals, fouls and missed shots with a timestamp. |
| **Follow players** | ByteTrack · BoT-SORT | Tracks each player over time on top of the detector. |
| **Identify players** | Jersey-number reader | Who the moment happened to. |
| **Label each camera** | Shot classifier | Marks every feed as wide, close-up, bench or crowd. |
| **Hear the game** | Audio classifier | The whistle and the crowd roar. |
| **Choose the shot** | Sequence model | Starts as written rules (§6), then learns from human cuts. |

What MVP 2 added to this picture (tested on recorded broadcast highlights):
- A **general vision-language model (Gemini)** that reads the last ~9 s of video + audio and returns phase of
  play, events, mood, a commentary line and the on-screen score, plus a "story so far" every 60 s.
- An **action-spotting specialist** (OpenSportsLib, 12 ball-action classes) running on a Colab GPU.
- A **fusion** step that merges both, removes duplicates (a goal shown again in a replay is not counted twice)
  and keeps `event_time` + `logged_at` for every item.

It works as a proof, but it is **seconds late and not reliable enough for goals** yet (see §10).

## 8. The output: the AI commands the switcher
**The AI commands the switcher instead of replacing it.**

```
 AI DIRECTOR (Python)  ── command ──▶  SWITCHER (OBS Studio · vMix)  ── clean cut ──▶  PROGRAM OUT
 decides which camera                 executes the cut                                full resolution,
 should be live                                                                       one seamless program
```

- The AI **never encodes the final video**, so its job stays a decision. This keeps it fast.
- Each camera change is a **JSON command** over the OBS WebSocket API (vMix has a similar HTTP API).
- It builds on switchers that already exist.

Today the prototype renders its own `program.mp4` and shows a local dashboard. That is for testing only;
the real output is a switcher command.

## 9. Where it fits
**Sports streaming is the first target.**

| Moment | What the director does |
|---|---|
| **In play** | Tracks the ball and the players, and holds wide angles through fast action. |
| **Play stops** | Cuts to close-ups, crowd reactions and key people. |
| **All game** | Matches color across cameras and keeps blurry or shaking feeds off the stream. |

Validation starts on **soccer and basketball** footage. Events and live shows can reuse the same pipeline
(only the rules and the event list change).

---

## 10. Current limitations
What the field offers today, and what we measured ourselves.

| Limitation | In general | What we measured (conditions) |
|---|---|---|
| **No ready-made live describer** | No model describes live video in real time. Open scene-description models work on still frames, run a few seconds behind and are not trained on sports. | Gemini Flash Lite: story lines arrive **3.3–3.8 s** after the moment (free tier, 9 s windows, broadcast highlights). |
| **Ball tracking is weak** | The best open football model we found finds the ball in only about **4 of every 10** frames where it is visible. | Roboflow model, panoramic clip split in 3: **15–34% per camera**, **70% across any camera** (threshold 0.3). Veo follow-cam clip: 54% any camera. |
| **Event detection is research code** | Goal/foul/shot spotting exists only as research code. It runs on recorded video, not live, and needs engineering to run live. | Spotting model: on a Colab T4 **254 ms** per run, events reported ~3.7 s late; on the M1 CPU far too slow (RTF 6–11). It **never fired GOAL** on broadcast highlights (found the shots for 2 of 3 goals). |
| **No story-following solution** | Following the story of a match like a human director has no off-the-shelf solution. It has to be built and trained. | Gemini + scoreboard reading got the score right, but goal **timing** was right for only 2 of 5 goals on a 10-minute clip (free tier). |
| **Speed on small hardware** | Detection on several feeds is GPU-heavy. | MacBook M1: ~300 ms per frame per camera → about **1 fps** for 3 cameras. Far from 100 ms. |
| **Identity** | Player tracking IDs drift when players are small or overlap. | 79 track IDs for ~15 people in one 30 s camera view. |

Takeaway: the **fast fields** (ball, players, audio) work but need better models and a GPU. The **story
fields** (events, who, shot type) are the real research gap.

## 11. What is built so far
All of this is a prototype on a MacBook M1 (8 GB), using recorded clips.

### MVP 1: Camera switching (done)
- One video is split into 3 "cameras" (or 3 separate synced clips are used).
- Ball and players are detected per camera; a ball tracker marks the ball detected / predicted / lost.
- A director scores each camera by the ball and switches with a **2.0 s minimum shot** and **0.5 s hysteresis**.
- Output: `program.mp4`, `debug.mp4`, `director_log.csv`, and a live local dashboard (Live and Replay modes).
- Result on a 30 s panoramic clip: 5 cuts, all following the ball, shots 2.0–6.8 s.

### MVP 2: Live Story Reader (done)
- Plays any soccer clip as if it were live (the browser video is the clock; nothing reads beyond the playhead).
- Layers: fast audio/motion layer (every 1 s) → Gemini story (every ~8 s + on cheers) → memory (every 60 s),
  plus the optional Colab specialist.
- A Story tab shows intensity, events, mood, commentary and story so far, each with its delay.
- Best mode was **hybrid** (all layers): goals 3/3 and avg delay 3.8 s on a 3-minute highlight clip.

### How this maps to the plan
| Plan stage | Built? |
|---|---|
| 01 Ingest | Files only. No NDI/RTSP/SRT yet. |
| 02 Analyze | Ball + players per camera (MVP 1); events, audio, story (MVP 2, single feed). No shot type, quality, color, faces. |
| 03 Decide | Ball-following rules (MVP 1). No REACTION state yet. |
| 04 Switch | Renders a video. No OBS/vMix command yet. |

## 12. Technical risks
**Three technical risks shape the build.**

| Risk | Why it is hard | Design response |
|---|---|---|
| **Compute** (4+ live feeds) | Detection, face recognition and color grading on 4+ feeds are GPU-heavy. | One RTX 5090 PC in a flight case, analyzing downscaled copies. |
| **Latency** (under 100 ms) | Read, analyze and command must all finish in time for live sports. | The AI only sends a command; video encoding stays with the switcher. Story runs on the slower loop (§4.1). |
| **Context** (the game's story) | Lingering on a player after a missed shot takes more than detection. | Learn from SoccerNet, which marks where human directors cut. |

Additional risks we found while building:
- **Model reliability on real footage:** models tuned on TV broadcasts may behave differently on raw venue cameras.
- **Cloud dependence:** Gemini and Colab need internet, have rate limits and send footage off-site (§17).

## 13. Compute plan (proposed)
**One powerful unit validates the idea and travels to the venue.**

| Step | What | Measure / target |
|---|---|---|
| 1 · Baseline | **One RTX 5090 PC** (32 GB GPU) runs the AI and the switcher together. | Measure ms per feed, per model. |
| 2 · Optimize | TensorRT builds of the detector, GPU video decoding, face recognition only when play stops. | Target: **4+ feeds under 100 ms** (fast loop). |
| 3 · Deploy | The same PC travels in a flight case: connect power, the camera network and the program output. | On site: power, network, output. |

Analysis stays on 480p/720p copies; the step 1 benchmark shows whether a second GPU is needed.
A local GPU could also run the story models that today need Colab/Gemini (to evaluate).

## 14. Validation: datasets and the virtual stadium
**Open datasets replace cameras for testing.** (Dataset descriptions are as stated in the plan; check each
licence before use.)

| Dataset | What it is | Tests |
|---|---|---|
| **BASKET-Multiview** | Up to 89 synchronized views of the same basketball plays, in 1080p and 4K. | Color matching, best angle |
| **SoccerNet** | Over 3,000 multi-view videos of fouls and goals, marked where a human director cut. | Learning human cut decisions |
| **TeamTrack** | Ultra-high-resolution full-pitch views of soccer, basketball and handball. | Wide-angle ball and player tracking |
| **MultiSports** | Player bounding boxes come pre-calculated, so detection training can be skipped. | Directing logic on its own |

### Real footage we can actually get (checked 2026-10-11)
No open dataset has a **full match** filmed by **several real broadcast angles** (wide + close-up + crowd)
**with sound**. Broadcasters own those isolated camera feeds. These are the closest real sources:

| Source | What you get | Access | Fit for us |
|---|---|---|---|
| **SoccerTrack v2** ([HF](https://huggingface.co/datasets/atomscott/soccertrack-v2)) | 10 full university matches (20 halves), 4K panorama from a 3-camera rig, ball positions, **12-class ball-action events with times (incl. Goal, Shot)**, player tracks | Free, CC BY 4.0; Hugging Face login + accept terms | **Best start.** Real full matches + ground-truth events in the same 12 classes as our specialist model. Split into 3 cams like MVP 1, plus virtual close-ups cropped from 4K. No crowd/close-up cameras; audio not mentioned. |
| **Alfheim** ([Simula](https://datasets.simula.no/alfheim/)) | 2 full elite matches (Tromsø IL, 2013), **3 separate synced cameras** + panorama, player positions at 20 Hz | Free, non-commercial research only | Real separate camera files, but all from one spot at mid-field and no audio. Good for sync/ingest testing. |
| **SoccerNet-MVFouls** ([GitHub](https://github.com/SoccerNet/sn-mvfoul)) | 3,901 foul actions from 500 pro games, each with **2+ real broadcast angles** + replays | Free after signing the SoccerNet NDA | Real broadcast multi-angle, but short clips (seconds), not a full story. Good for "best angle" tests. |
| **SoccerNet-v2 broadcasts** ([devkit](https://github.com/SilvioGiancola/SoccerNetv2-DevKit)) | 500 full pro broadcasts; **camera-change and camera-type labels (13 types) for 200 games**, actions, replays | Videos need the NDA | Only the on-air program (one angle at a time), but it shows **where human directors cut** → teaches the director (§7 "choose the shot"). |
| **Soccer-ISSIA** | 2 min, 6 synced static 1080p cameras, ball annotated | Research request | Too short for a story. |

Note: the plan's "BASKET-Multiview (up to 89 views)" could not be found in a search; "BASKET" (CVPR 2025) is a
single-view skill dataset. To check before relying on it.

### The virtual stadium
**A virtual stadium proves the loop end to end, with no physical cameras.**

1. **Download:** take 3 to 4 synchronized angles of a single play.
2. **Loop:** replay the MP4 files as local NDI or RTSP streams with FFmpeg.
3. **Point:** aim the vision mixer at the local streams as if they were live.
4. **Watch:** check in OBS Studio that the AI cuts on the recorded action.

## 15. Roadmap (draft)
**Three phases lead from simulation to a live pilot.** MVP 1 and MVP 2 are the first part of Phase 1.

| Phase | Goal | Done when |
|---|---|---|
| **Phase 1 · Virtual stadium** · `[dates]` | Prove ingest, analysis, directing rules and switching on open datasets. | AI cuts match the recorded action. |
| **Phase 2 · Live cameras** · `[dates]` | Connect physical cameras and run 4+ feeds on GPU hardware. | The fast loop runs under 100 ms. |
| **Phase 3 · Pilot event** · `[dates]` | Direct a real broadcast for `[pilot partner]` with no one at the mixer. | `[pilot success measure]` |

### Proposed next steps inside Phase 1 (MVP 3 candidates)
These are proposals; order and scope are open.

| Step | What | Done when (measurable) |
|---|---|---|
| A · Camera card | Produce the §3 record for every camera, ~1/s, from existing layers. | Every camera has a record every second with `time` + `logged_at`; missing fields are `null`, never guessed. |
| B · Stream ingest | Read looped files as local RTSP/NDI streams instead of files. | 3–4 local streams read in sync (frame offset ≤ 1 frame) for a full clip. |
| C · Switcher output | Send cuts to OBS over WebSocket. | OBS program follows the director log with every cut ≤ 1 frame late. |
| D · Feed quality | Blur/shake check per camera; rejected feeds are never chosen. | On a test clip with one artificially blurred camera, it is on air 0% of the time. |
| E · Shot type | Classify each feed as wide / close-up / crowd / bench. | ≥ `[target]`% correct on a labelled sample. |
| F · Reaction state | Add REACTION to the state machine using events + shot type. | After each true goal in a test clip, a close-up/crowd shot is taken within `[N]` s. |
| G · Real multi-view data | Run A–F on a multi-view dataset (§14). | Cut log compared with human cuts (SoccerNet) and the score reported. |

Candidate tools to evaluate for the gaps (no results yet): a stronger or tiled ball detector, a jersey
number reader, a small shot classifier, a local vision-language model on the GPU PC.

## 16. Success criteria
**Four checks decide whether the idea works.**

| Check | Criterion | How we test it |
|---|---|---|
| **Cuts follow the play** | The AI switches camera when the recorded action calls for it. | Watch in OBS Studio; compare with human cuts where available. |
| **The loop is fast enough** | Reading a frame, analyzing it and commanding the switcher take **< 100 ms** (fast loop). | Timestamps per stage in the log. |
| **Shots hold steady** | No cut arrives faster than the minimum shot length (plan: 3–4 s). | Shortest shot in the director log. |
| **Only clean pictures air** | Blurry, shaking or out-of-focus cameras are never selected. | Test clip with a degraded camera (step D). |

Added from MVP 2 experience: every story item shows its delay (`logged_at − time`) and nothing ever reads
the future (enforced by a guard in code).

## 17. Open decisions
| Topic | Question | Notes |
|---|---|---|
| Minimum shot length | 3–4 s (plan) or 2.0 s (MVP 1)? | May depend on the sport and on the shot type. It is one config value. |
| Face recognition / VIP | Do we recognise people at all, and with whose consent? | Our test footage includes minors and must never be uploaded. Face recognition plus cloud models (Gemini, Colab) raises privacy and legal questions. Until decided: no face recognition, no identifying data stored. |
| Cloud vs local | Keep Gemini/Colab, or run everything on the GPU PC? | Cloud sends footage off-site and has rate limits; local needs the 5090 benchmark. |
| Detector | Which ball/player model for production? | Decided by the step 1 benchmark (§13). |
| Sports | Soccer first, basketball when? | Datasets exist for both (§14). |
| Dates, pilot partner, pilot success measure | — | `[TBD]` |

## 18. Glossary
- **Vision mixer / switcher:** the device or software (OBS, vMix) that picks which camera goes to air.
- **Program out:** the single video viewers see.
- **Cut:** switching from one camera to another.
- **Hysteresis:** waiting a short time before switching, so the picture doesn't flicker back and forth.
- **NDI / RTSP / SRT:** ways to send live video over a network.
- **Action spotting:** finding the exact moment of an event (goal, shot) in a video.
- **Causal:** using only the past and present, never the future, like a real live system.
- **RTF (real-time factor):** processing time ÷ video time. Below 1 means faster than real time.

## 19. Changelog
| Date | Change |
|---|---|
| 2026-10 | PDF "AI Vision Mixer Validation Plan" (15 slides). |
| 2026-10-11 | Markdown version created. Added: the per-camera description (§3), two-speed loop (§4.1), feed quality, current limitations with measured numbers (§10), what is built (§11), proposed next steps (§15), open decisions (§17). |
| 2026-10-11 | Added "Real footage we can actually get" (§14). MVP 3 (Directed Story) started on SoccerTrack v2: `docs/MVP3_SPEC.md`. |
