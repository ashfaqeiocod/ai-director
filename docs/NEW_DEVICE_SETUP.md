# Setting up on a new computer

Works on **macOS (Apple Silicon or Intel), Windows 10/11 and Linux**. Takes about 10–15 minutes,
mostly downloading Python packages (about 2 GB).

> Mac/Linux commands use `venv/bin/python`. On **Windows** use `venv\Scripts\python` everywhere instead.

---

## 1. Install the basics (once per computer)

| What | Why | How |
|---|---|---|
| **Git** | to get the code | https://git-scm.com/downloads (Mac: already there, or `xcode-select --install`) |
| **Python 3.10 – 3.13** | runs everything (3.14 is **not** supported yet) | https://www.python.org/downloads/ (Windows: tick "Add python.exe to PATH") |
| **Node.js LTS** | *only* needed to download YouTube links | https://nodejs.org |

You do **not** need to install ffmpeg: it comes inside the Python packages.

Check:
```bash
python3 --version     # Windows: python --version   → 3.10 to 3.13
git --version
node --version        # optional
```

## 2. Get the code and install the packages

```bash
git clone <your repo URL> ai-director
cd ai-director
python3 -m venv venv                              # Windows: python -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

`venv` is a private folder of Python packages just for this project. Deleting the project folder removes everything.

## 3. Add your Roboflow key

The ball/player model is downloaded from Roboflow, which needs a free API key.

1. Make a free account at https://app.roboflow.com, then **Settings → API Keys**, and copy the key.
2. Create the `.env` file:
   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   ```
3. Open `.env` and paste the key: `ROBOFLOW_API_KEY=abc123…`

`.env` is never uploaded to git. Each person uses their own key.

## 4. Tell it where to run the model (the speed setting)

In `config.yaml`, `detection.onnx_providers` picks the chip that runs the model. The first one this computer
has is used; anything it doesn't have is skipped automatically.

| Computer | Setting | Speed (per camera frame) |
|---|---|---|
| Mac with Apple Silicon (M1/M2/M3/M4) | `[CoreMLExecutionProvider, CPUExecutionProvider]` (default) | ~0.3 s on M1 |
| PC with an NVIDIA GPU | `[CUDAExecutionProvider, CPUExecutionProvider]`, plus run `venv/bin/python -m pip install onnxruntime-gpu` | fastest |
| Anything else | `[CPUExecutionProvider]` | slow: several seconds |

(The YOLO backup model uses `detection.device`: `"mps"` on Mac, `"cuda"` with NVIDIA, otherwise `"cpu"`.)

## 5. Check everything

```bash
venv/bin/python src/check_setup.py --online
```

You want to see `Ready.` at the end. Every `FAIL` line comes with the fix (`→ …`).
`--online` also tests your Roboflow key. Leave it out to check without internet.

## 6. Run it

```bash
venv/bin/python src/main.py               # prepares the video, detects, directs, renders
venv/bin/python -m dashboard.server       # dashboard → open http://localhost:8000
```

The first run downloads the model and takes longer. To test your own videos, see
[USING_YOUR_OWN_VIDEOS.md](USING_YOUR_OWN_VIDEOS.md).

Opening the results:

| | Mac | Windows | Linux |
|---|---|---|---|
| a video | `open outputs/debug.mp4` | `start outputs\debug.mp4` | `xdg-open outputs/debug.mp4` |

---

## Moving the project between computers

- **git push / pull** moves the code, `config.yaml` and the docs. That's all a new computer needs.
- Videos, detections and outputs are **not** in git. On the new computer, `main.py` downloads or prepares them
  again from the `input:` section.
- If `input:` points to files in `clips/`, copy those video files over yourself (USB, AirDrop, cloud drive…),
  because they aren't in git either.
- Live mode speed depends on the computer. Tune the `live:` section; see [LIVE_MODE.md](LIVE_MODE.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on `inference` | Python is too new or too old: use 3.10 – 3.13 and recreate the venv |
| `ImportError: libGL.so.1` (Linux) | `sudo apt install libgl1` |
| Detection takes seconds per frame | check_setup shows "CPU only": fix step 4 |
| `address already in use` when starting the dashboard | another dashboard is running: close it, or change `dashboard.port` in config.yaml |
| Download from YouTube fails | install Node.js, check the internet connection, or use a video file instead |
| Roboflow error / 403 | wrong key in `.env`: run `src/check_setup.py --online` |
