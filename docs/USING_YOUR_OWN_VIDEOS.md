# Using your own videos

This guide shows how to test the AI Sports Director with **any football video**: a YouTube link, a video file
on your computer, or **3 separate camera clips**. You only change one section of `config.yaml`, called `input:`.

> Commands below use `venv/bin/python` (Mac/Linux). On Windows use `venv\Scripts\python` instead.
> New computer? Do [NEW_DEVICE_SETUP.md](NEW_DEVICE_SETUP.md) first.

---

## The 30-second version

1. Put your video(s) in the `clips/` folder (or use a YouTube link).
2. Edit the `input:` section at the top of `config.yaml` (recipes below).
3. Run:
   ```bash
   venv/bin/python src/main.py
   ```
4. Look at the results:
   - `outputs/debug.mp4`: the 3 cameras side by side, with the red LIVE border on the chosen camera
   - `outputs/program.mp4`: the final "broadcast"
   - `outputs/director_log.csv`: why each decision was made
5. Or watch it live: `venv/bin/python -m dashboard.server`, then open http://localhost:8000 and click **Replay → Start**.

**You never need to delete anything by hand.** `main.py` notices when the `input:` section (or one of your
video files) changed, removes the old cameras, detections and outputs, and starts fresh. If nothing changed,
it skips straight to the parts that are left. If a path is wrong, it stops with a clear message and **keeps
your previous results**.

---

## Two ways to give it video

| Mode | You have… | What happens |
|---|---|---|
| `split` | **ONE** wide video (a panorama, or a wide shot of the whole pitch) | It is cut into 3 vertical strips: left = cam1, middle = cam2, right = cam3 |
| `files` | **THREE** clips, one per camera (e.g. 3 phones or 3 cameras filming the same match) | Each clip becomes one camera, after they are lined up in time |

---

## Recipe A: one YouTube video (split)

```yaml
input:
  mode: split
  split:
    source: "https://www.youtube.com/watch?v=HBTFZwMdcCw"
    trim_last_seconds: 30     # only the last 30 s (keeps tests quick)
    start_seconds: 0          # ignored when trim_last_seconds is set
    max_seconds: null
```

- Needs internet and **Node.js** (YouTube requires it to download).
- The download is saved in `data/raw/downloads/` and reused, so changing only the trim settings doesn't download again.

**Choosing the part of the video**

| You want | Set |
|---|---|
| the last 30 s | `trim_last_seconds: 30` |
| 20 s starting at 1:05 | `trim_last_seconds: null`, `start_seconds: 65`, `max_seconds: 20` |
| the whole video | `trim_last_seconds: null`, `start_seconds: 0`, `max_seconds: null` |

---

## Recipe B: one video file on your computer (split)

1. Copy the file into the `clips/` folder, e.g. `clips/match.mp4`.
2. Set:
   ```yaml
   input:
     mode: split
     split:
       source: "clips/match.mp4"
       trim_last_seconds: null
       start_seconds: 0
       max_seconds: 30
   ```

Paths can be:
- relative to the project folder: `"clips/match.mp4"`
- in your home folder: `"~/Movies/match.mp4"`
- absolute: `"/Users/me/Movies/match.mp4"` or `"C:/Users/me/Videos/match.mp4"` (use forward slashes on Windows too)

Your files are only **read**: never moved, changed or deleted.

---

## Recipe C: three camera clips (files)

1. Put the 3 clips in `clips/`, e.g. `left.mp4`, `middle.mp4`, `right.mp4`.
2. Set:
   ```yaml
   input:
     mode: files
     files:
       cams:
         - { source: "clips/left.mp4",   start_seconds: 0 }
         - { source: "clips/middle.mp4", start_seconds: 0 }
         - { source: "clips/right.mp4",  start_seconds: 0 }
       audio_from: 2          # use the middle clip's sound in program.mp4
       max_seconds: 30        # null = as long as all 3 clips overlap
       size: [1280, 720]      # every camera is resized to this
       fps: 25                # every camera is converted to this
   ```
3. Run `venv/bin/python src/main.py`.

What it does for you:
- **Different sizes or shapes?** Every clip is resized to `size` without stretching (black bars fill the gaps).
- **Different frame rates (25, 30, 60…)?** All are converted to `fps`.
- **Different lengths?** All are cut to the same number of frames: the part where all 3 clips overlap.
- **Mixed sources?** Each `source` can be a file or a YouTube link.

Rules:
- There must be **exactly 3** clips, in the order left, middle, right (cam1, cam2, cam3).
- The 3 clips must show **the same moments**. See the next section.

### Lining the clips up (`start_seconds`)

If one camera started recording later than the others, the clips are out of sync. Then the director compares
cam1 at 0:05 with cam3 at 0:08, and its decisions are wrong.

How to fix it:
1. Find **one moment you can see (or hear) in all 3 clips**: the kick-off, a whistle, a clap, the ball crossing
   the halfway line.
2. Note the time of that moment in each clip, e.g. left 0:07, middle 0:05, right 0:12.
3. Subtract the **smallest** time from each one, and use the result as `start_seconds`:
   ```yaml
   cams:
     - { source: "clips/left.mp4",   start_seconds: 2 }   # 7 - 5
     - { source: "clips/middle.mp4", start_seconds: 0 }   # 5 - 5
     - { source: "clips/right.mp4",  start_seconds: 7 }   # 12 - 5
   ```
4. Run `venv/bin/python src/main.py`, open `outputs/debug.mp4`, and check that something happening across two
   cameras (a long pass, a player running across) appears at the **same moment** in both.
   If not, adjust `start_seconds` by a fraction (e.g. `2.4`) and run again.

---

## What gets redone automatically

| You change… | What `main.py` does | Time |
|---|---|---|
| anything in `input:` (or replace one of the video files) | prepares the cameras again, detects again, renders again | detection: about **1 s per video second** per camera on an M1 |
| `detection.ball_confidence` / `player_confidence` | nothing to redo: run `rm outputs/program.mp4 && venv/bin/python src/main.py` | seconds |
| the `director:` section (scores, min shot, hysteresis) | same as above: delete `outputs/program.mp4` and run again | seconds |
| `detection.model_id` (a different model) | run `venv/bin/python src/detect.py --force`, then the line above | slow |
| you want everything from scratch | `venv/bin/python src/main.py --force` | slow |

**Tip:** for a first test, use `max_seconds: 10`. Detection on a 10-second clip takes about 3–4 minutes on an M1.

---

## What makes a good test video

The model was trained on **broadcast-style football**, so the best results come from:
- a **wide view** of the pitch, from a high position (a stand, a tower)
- daylight or a well-lit stadium
- a camera that **doesn't move** (the "follow the ball" cameras keep the ball in the middle, so the director
  never needs to cut; that was our first test clip)
- 720p or 1080p (bigger is scaled down to 1080p's size automatically)

Expect weaker ball detection with phone footage at pitch level, night games, or a very far away camera.

---

## When something goes wrong

All problems are reported as `[prepare] PROBLEM: …` with what to change. The common ones:

| Message | Fix |
|---|---|
| `Video file not found: …` | Check the path and spelling in `config.yaml`. Relative paths start at the project folder. |
| `Could not download …` | Check the link, your internet, and `node --version`. |
| `input.files.cams must list exactly 3 clips` | Add or remove clips in `cams:`. |
| `start_seconds (…) must be between 0 and the clip length` | That clip is shorter than its `start_seconds`. |
| `After lining up the clips only …s overlap` | The clips don't cover the same moments, or `start_seconds` is too large. |
| `note: camN's clip has no sound` | Not an error: `program.mp4` will be silent. Pick another clip in `audio_from`. |
| Dashboard says "The input in config.yaml changed" | Run `venv/bin/python src/main.py` (or `src/prepare.py`), then press Start again. |

To check the whole setup at once (packages, ffmpeg, key, accelerator, and that your input files exist):
```bash
venv/bin/python src/check_setup.py
```

---

## Privacy

Videos (in `clips/`, `data/` and `outputs/`) are **never** uploaded to git. Only the code and the docs are.
Footage of people, especially minors, should stay on your computer.
