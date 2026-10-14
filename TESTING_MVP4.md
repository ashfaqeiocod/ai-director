# How to test MVP 4 — Virtual Stadium (T2 camera set)

MVP 4 simulates the camera set our product would install at a ground ("T2", `docs/MVP4_STADIUM_PLAN.md`) on the
Alfheim half (Tromsø IL – Strømsgodset, 2013-11-03, first half; Alfheim dataset, Simula, non-commercial research):

| Camera | What it is | Built from |
|---|---|---|
| MAIN | robotic camera at halfway: follows the ball, zooms by game state | the real cameras stitched into a panorama |
| CLOSE | robotic close-up next to MAIN | the panorama (soft: the sources are 1280x960) |
| WIDE | the whole pitch | the panorama, scaled |
| BOX L / BOX R | fixed, one per third | the real Alfheim cam1 / cam3 |
| GOAL L / GOAL R | behind each goal, 6 m high | a 3D scene (Three.js) from player positions |

Progress and every number: `PROGRESS.md` (MVP 4 section). Run everything from the project folder:

```bash
cd ~/Documents/ai-director
```

---

## 1. Automatic tests (~5 s)

```bash
venv/bin/python -m pytest tests/
```
All tests pass. The MVP 4 ones are in `tests/test_studio_stadium.py`: panorama geometry, camera projection,
robotic heads never faster than their motors, cards, the TV rules, the ZXY conversion.

## 2. The panorama (Step 1)

Made once (already done; ~1 h for the half):
```bash
venv/bin/python -m studio.panorama --check     # numbers + check picture only (~1 min)
venv/bin/python -m studio.panorama             # the whole half -> panorama.mp4 (own Terminal, ~1 h)
```
Open `data/alfheim/2013-11-03_H1/panorama_check.jpg`. You should see one wide picture of the whole pitch: red
pitch lines drawn from the calibration sit on the white lines; the yellow outlines are the two seams, and lines
and players run through them without a jump. Then play `data/alfheim/2013-11-03_H1/panorama.mp4`.

## 3. The digital twin (Step 3)

```bash
venv/bin/python -m studio.zxy                  # download ZXY sensor data (~87 MB), align, twin.jsonl (~1 min)
```
It prints the clock offset (~8.7 s), and the distance from each home player (sensors) to the nearest player our
detector found (median ~1.6 m). Open `data/alfheim/2013-11-03_H1/twin_check/cam2_023.7.jpg`: white circles (home,
sensors) on the red-and-white players, blue (away) on the dark-blue players, yellow on the referees.

## 4. The playground page (Steps 4 + 6)

Restart the dashboard so it has the new routes (stop the old one with Ctrl+C first):
```bash
venv/bin/python -m dashboard.server
```
Open **http://localhost:8000/static/stadium.html** (Chrome).

1. Without going live: press **Play** (3D CLOCK). The GOAL L / GOAL R tiles and the DIRECTOR view show the 3D
   stadium with the players moving. In the DIRECTOR view, drag to turn and scroll to zoom; the dropdown shows
   the scene from any camera ("from BOX L" lines up with the real BOX L picture).
2. Press **Go live** (from 23:40). Within a few seconds:
   - PROGRAM shows the broadcast with the scorebug (TIL 0-0 SIF + clock) and the LIVE badge;
   - the monitor wall shows all 7 cameras; the one on air has a red border; robotic ones show their width and speed;
   - the pitch map shows every camera's view cone; MAIN's and CLOSE's cones turn with the play; the ball is white;
   - the Log lists cuts with the director's reason ("MAIN -> GOAL L: attack in the left box, action in its view")
     and camera moves ("MAIN zooms to 28 m"). When GOAL L/R is on air, PROGRAM shows the 3D picture.
   - The status line shows fps shown and lag (the Mac shows ~6 fps and keeps up with real time).
3. **Stop** ends the live run.
4. **Record the 3D cameras**: from 23:40 for 10 s -> `outputs/stadium/alfheim-2013-11-03-H1/3d-.../cam_goall.mp4` and
   `cam_goalr.mp4` (the status says "300 of 300 frames recorded").

## 5. A directed run with every feed (Steps 2, 5, 7)

Keep the dashboard running (it records the 3D cameras through headless Chrome), then:
```bash
venv/bin/python -m studio.stadium --aim director --start 23:40 --seconds 300     # ~30 min
```
It writes `outputs/stadium/alfheim-2013-11-03-H1/<run>/`:
`cam_main.mp4 cam_close.mp4 cam_wide.mp4 cam_boxl.mp4 cam_boxr.mp4 cam_goall.mp4 cam_goalr.mp4 program.mp4
director_log.csv cameras.jsonl summary.json`. The printed summary has the airtime per camera (MAIN ~60-75%),
number of cuts, average shot, `shots_under_min: 0`, and `ball_in_main_pct` (>= 90).

Watch `program.mp4`: MAIN follows the play like a TV camera; on attacks it cuts to the behind-goal 3D view or BOX
camera for a few seconds; after the ball goes out a short CLOSE shot; no shot shorter than 2 s.

`--aim follow` (Step 2) writes only the 5 real feeds with MAIN/CLOSE simply following the ball.

## 6. MVP 1 still works

```bash
rm outputs/program.mp4 && venv/bin/python src/main.py
```
5 cuts at 2.00 / 5.32 / 10.72 / 17.20 / 23.96 s, shortest shot 2.00 s (as in `TESTING.md` section 1).

## Honest limits

- CLOSE and zoomed MAIN shots are soft: the sources are 1280x960 (a real T2 has a 4K panorama and a real robotic
  camera). Their colour is softened a little to hide the camera sensor's rainbow edges on thin lines.
- The behind-goal cameras look like a video game; away players come from the detector (less smooth than sensors,
  sometimes doubled); the ball has no real height (long fast balls get a made-up arc) and is sometimes a false
  detection of the map.
- All real-pixel cameras share one viewpoint (true for T1/T2 too). MVP 3's cam1 calibration puts the left box ~1 m
  too deep.
