# MVP 5 — Virtual Broadcast (a full TV camera plan, rendered in 3D) — plan

Status: PLAN, waiting for the user's go (written 2026-10-14 after MVP 4's first test runs).

## 1. Why (user feedback on MVP 4)
MVP 4 mixed two worlds in one program: real Alfheim pixels (MAIN, CLOSE, BOX L/R, WIDE) and a 3D scene (GOAL L/R).
- They don't match visually: a cut from a real night picture to a bright low-poly scene breaks the broadcast.
- The real pixels can't zoom: CLOSE at 12 m is a ~6x blow-up of a 1280x960 camera (very soft).
- The goal is not "follow the ball". It is a **real TV broadcast**: smooth camera work from many angles, chosen so
  the viewer understands the game, exactly as a TV crew would film it.

So MVP 5 renders **every camera virtually** from the 3D match (the digital twin): any position, any lens, always
sharp, and the cameras move like real ones (smooth, operator-like). The virtual stadium becomes the **design tool
for the real product**: which cameras, where, how high, which lens, how they move, how the director cuts. What
works here is what we install at a real ground.

Real footage keeps two jobs: (1) the input that feeds tracking (in the real product the cameras are real), and
(2) a side-by-side reference to check the twin (players where the real ones are). It is no longer cut into the
program.

## 2. The camera plan (what a real TV crew uses; our default set = the real project's)
Pitch coordinates as in MVP 4 (metres, centre spot origin, x right, y towards the main stand; heights above grass).
The main stand is the near side (y > 34). Field of view (FOV) = horizontal angle; a wide shot ~50-60°, a close-up 3-8°.

| # | Camera | Position (x, y, height) | Type / mount | Lens (hFOV) | Moves | Used for |
|---|---|---|---|---|---|---|
| 1 | MAIN (wide follow) | (0, 60, 18) halfway, main stand, high | manned/robotic fluid head | 8-60° | pan/tilt/zoom, slow and smooth | ~65-75% of live play: the game as a whole |
| 2 | TIGHT (close follow) | (4, 60, 18) next to MAIN | manned/robotic | 2-20° | fast pan, frequent zoom | player on the ball, reactions, ~10-15% |
| 3 | HIGH 18 L | (-36, 58, 15) main stand at the left box edge | robotic | 6-45° | pan/tilt/zoom | attacks into the left box, offside line |
| 4 | HIGH 18 R | (36, 58, 15) | robotic | 6-45° | pan/tilt/zoom | same, right |
| 5 | LOW 18 L | (-34, 37, 1.6) touchline, near the left box | handheld/pedestal | 5-50° | pan/zoom | players at eye level, duels, replays |
| 6 | LOW 18 R | (34, 37, 1.6) | handheld/pedestal | 5-50° | pan/zoom | same, right |
| 7 | HIGH GOAL L | (-75, 0, 12) stand behind the left goal | robotic | 6-40° | pan/tilt/zoom | shots, goals, set pieces (live briefly + replays) |
| 8 | HIGH GOAL R | (75, 0, 12) | robotic | 6-40° | pan/tilt/zoom | same, right |
| 9 | LOW GOAL L | (-56, -6, 1.0) behind the goal line, off-centre | fixed / mini-cam | 50-70° | fixed | goal-line, replays of goals |
| 10 | LOW GOAL R | (56, 6, 1.0) | fixed / mini-cam | 50-70° | fixed | same |
| 11 | REVERSE | (0, -55, 12) opposite stand, halfway | manned | 6-50° | pan/tilt/zoom | replays only (180° rule) |
| 12 | STEADICAM | touchline, moves along (x varies, y 36, 1.7) | handheld on a track | 20-60° | walks/tracks | walk-out, celebrations, substitutions |
| 13 | SPIDERCAM | cable cam over the pitch, 15-30 m | cable rig | 20-70° | flies smoothly in 3D | build-up from above, replays, kick-off |

Tiers (same as MVP 4): T1 = 1 (+2 virtual crops); **T2 = 1, 2, 7, 8 (+3/4)** = our product default; T3 = all.
All positions, heights, lens ranges and motor/operator limits go in `config.yaml` (`studio.broadcast.cameras`);
a different stadium = a different list. A "coverage" page shows each camera's view cone on the pitch map.

## 3. How the cameras move (operators, not ball trackers)
Each camera has an operator model (Python, causal, uses a short look-ahead only within the broadcast delay):
- **Framing**: MAIN keeps the ball about one third from the edge, with **lead room** in the direction of play; its
  width shows the players involved (ball carrier, nearest defenders, the next pass options), not just the ball.
- **Smooth motion**: pan/tilt/zoom have speed, acceleration AND jerk limits (ease-in/ease-out like a fluid head):
  no instant starts or stops. Dead zone for tiny moves; zoom changes are slow and rare during play.
- **Anticipation**: aim where play is going (ball velocity + attacking direction + the likely receiver), so the
  camera is there when the pass arrives instead of chasing it.
- **Shot types** per camera: extra-wide, wide, medium, close-up, extreme close-up; TIGHT picks the player on the ball,
  a goal scorer, a goalkeeper after a save.
Measured in a test: max speed/acceleration/jerk per camera, "ball in frame" and "ball near the 1/3 lines" per shot.

## 4. Game understanding (what the director needs to know)
From the twin (all 22 players + referee + ball, with teams): possession and attacking direction, phase of play
(build-up, attack in the final third, counter, set piece, dead ball), restarts (kick-off, throw-in, goal kick,
corner, free kick, penalty: from where the ball stops and how players line up), events (shot, save, goal, foul /
injury: player down and play stopped). MVP 2's story layer can add what positions can't see.

## 5. The director (TV grammar)
- **180° rule**: live shots only from the main-stand side; the reverse angle only in replays, labelled.
- MAIN is the base; cut to TIGHT on a duel, a player on the ball in a slow phase, after a foul; to HIGH 18 for an
  attack into the box; to HIGH GOAL briefly for a shot / corner; to WIDE/SPIDERCAM for kick-off and restarts.
- **Cut on action** (on a pass or touch, not in the middle of a camera move); shot lengths like TV (main 10-30 s,
  close-ups 2-5 s); never cut to a camera that is still moving fast.
- **Replays** at dead balls after a big moment: 2-3 angles (behind goal, low 18, reverse) with a wipe, slow-motion
  (the 3D render can produce real slow motion: render at 120 fps).
- Graphics: scorebug, clock, replay wipe, credit.

## 6. Rendering (the visual quality)
Recommended: **stay in the browser (Three.js)** — the stack you know (JS) and what MVP 4 already has — but make it
look like TV:
- players: rigged 3D figures (glTF) with real animations (idle, jog, run, sprint, kick, header, celebrate) blended
  by speed and events; kits with numbers; faces not needed at broadcast distances;
- stadium: proper stands with crowd, ad boards, floodlights, real-time shadows from the floodlights, mowed-grass
  texture with wear, goal nets that move;
- camera look: anti-aliasing, a little motion blur on fast pans, depth of field on close-ups, lens FOV from the
  camera plan;
- frame-exact OFFLINE rendering at 1920x1080 (the MVP 4 recorder, extended): no live limit, so quality can be high.
- **Render only what is needed**: the program renders the on-air camera at full quality; the monitor wall renders
  all cameras small. (Everything is virtual, so the program needs no pre-recorded feeds.)
Alternative later: Unreal Engine 5 (photoreal) rendering the SAME camera paths (the director exports per-frame
camera JSON: position, look-at, FOV). Heavier, new tools, slow on an 8 GB M1: not for MVP 5.

## 7. The data the 3D match needs (the weak point today)
MVP 4's twin: home team from ZXY sensors (good), away team + referee from our detections (jittery, sometimes
doubled), ball from the detector (gaps, false balls, no height). A clean virtual broadcast needs:
- away team: one track per player (11 + keeper), smoothed (offline smoother within the broadcast delay), no doubles;
- ball: a 3D trajectory (ground rolls, passes, lofted balls with a parabola, shots), gaps filled from who has the
  ball (the player nearest the ball when it is lost);
- for a demo clip: optionally hand-checked key moments (shots, restarts) in a small correction file.
In the real product this comes from the live tracking of the real cameras (or sensors).

## 8. Steps (each: check, PROGRESS.md, commit)
0. This plan + decisions (user).
1. Camera catalogue in config + a coverage page (all 13 cameras, cones, tiers T1/T2/T3).
2. Better twin: away-team tracks, referee, 3D ball (+ check pictures against the real cameras).
3. Game-state reader (possession, direction, phases, restarts, events) from the twin; checked on 5 minutes by eye.
4. 3D scene v2: animated players, stadium, lights, shadows (check: a still from MAIN next to the real MAIN frame).
5. Camera operators v2: framing + smooth motion (speed/accel/jerk limits) for every camera type; per-frame camera
   paths `cameras.jsonl`; a test for the limits.
6. Director v3: TV grammar, cut on action, replays.
7. Offline renderer: program.mp4 (1080p) + monitor-wall mosaic; the playground page plays the result.
8. Test runs: 5 minutes, then the half; airtime per camera, shot lengths, smoothness numbers, and a viewing session.

## 9. Decisions for the user
1. Rendering engine: Three.js upgraded (recommended) or Unreal Engine 5?
2. Live or offline: is it OK that the virtual broadcast is rendered offline (higher quality), with the live page
   showing a lighter preview?
3. Demo length: one strong 5-minute clip first (with hand-checked ball/away data), or straight to the whole half?
4. Camera set to build first: T2 (MAIN, TIGHT, HIGH GOAL L/R, HIGH 18 L/R) or all 13?
