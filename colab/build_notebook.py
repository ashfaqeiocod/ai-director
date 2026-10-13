"""build_notebook.py — writes colab/specialist_server.ipynb from colab/specialist_server.py.

The notebook must be self-contained (you upload just the .ipynb to Colab), so the server code is
copied into a %%writefile cell. Edit specialist_server.py, then run:
    venv/bin/python colab/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8765
DETECTOR_PORT = 8766

HOW_TO = """# Story Reader — specialist server (Colab GPU)

This notebook runs TWO models on a free Colab GPU behind ONE public URL:
- the **football detector** (Roboflow `football-players-detection-3zvbc/11`: players, goalkeepers, referees, ball)
  for MVP 3's real live test: 3 cameras x 5 frames per second while the match plays (the Mac is too slow for that);
- the **ball action spotting** model
[`OpenSportsLab/OSL-loc-snbas-2025-e2e`](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e)
(12 classes: PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK,
PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL) on a free Colab GPU, and gives you a public URL that the
dashboard on your Mac connects to.

## How to run (5 steps)
1. **Turn on the GPU:** menu *Runtime → Change runtime type → T4 GPU → Save*.
2. **Roboflow key (for the detector):** click the 🔑 *Secrets* icon (left bar) → *Add new secret* → name
   `ROBOFLOW_API_KEY` (exactly), value = your Roboflow API key, switch *Notebook access* on. Without it the
   action model still runs, only the detector is off.
3. *(Optional, only for the action model alone; the detector needs far more than ngrok's free 1 GB/month)* **ngrok token:** make a free account at https://dashboard.ngrok.com, copy your
   authtoken, then in Colab click the 🔑 *Secrets* icon (left bar) → *Add new secret* → name
   `NGROK_AUTHTOKEN`, value = your token, and switch *Notebook access* on.
   By default the notebook uses a free Cloudflare quick tunnel (no account, no data cap); ngrok only with TUNNEL = 'ngrok' in step 5.
4. **Run everything:** menu *Runtime → Run all*. It takes ~20–25 minutes (installing, downloading the models, and
   building the detector's TensorRT engine, which makes it 8x faster). Leave it alone until the URL box appears.
5. **Copy the URL** printed in big letters at the end (it looks like `https://xxxx.ngrok-free.app`
   or `https://xxxx.trycloudflare.com`).
6. **Paste it** into the dashboard: MVP 3 http://localhost:8000/studio → ● Live → *Colab URL* → *Test*
   (both the action model and the detector should show), or MVP 2 http://localhost:8000/story → *Connect*.

Keep this tab open while you use the dashboard. Colab stops the server after ~90 minutes of no activity
(free tier); then just *Run all* again and paste the new URL.

**License note:** the model is AGPL-3.0 (research / non-commercial use)."""


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    server_code = (HERE / "specialist_server.py").read_text()
    detector_code = (HERE / "detector_server.py").read_text()
    cells = [
        markdown(HOW_TO),
        markdown("## 1. Check the GPU and install packages"),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU: Runtime -> Change runtime type -> T4 GPU'\n"
             "!pip install -q opensportslib==0.3.1 fastapi uvicorn pyngrok websockets huggingface_hub pyyaml\n"
             "import torch; print('CUDA available:', torch.cuda.is_available())"),
        markdown("## 2. The server code (same as `colab/specialist_server.py` in the project)"),
        code("%%writefile specialist_server.py\n" + server_code),
        markdown("## 2b. The football detector in its own Python environment\n"
                 "Roboflow's `inference-gpu` pins many packages; in its own environment it can't break the action "
                 "model's torch. It runs as a second local server (port 8766); the main server relays `/detect` to it."),
        code("!pip install -q uv\n"
             "!uv venv -q --seed --python 3.11 /content/detenv\n"
             "# onnxruntime-gpu[cuda,cudnn]: the CUDA + cuDNN versions this onnxruntime was built for, in its own environment\n"
             "!uv pip install -q --python /content/detenv/bin/python inference-gpu 'onnxruntime-gpu[cuda,cudnn]' fastapi uvicorn websockets\n"
             "# TensorRT (half precision on the T4's tensor cores): plain CUDA took 535 ms per camera frame, TensorRT 70 ms.\n"
             "# Its library files come from NVIDIA's own package index (uv can't install the tensorrt wrapper package).\n"
             "!/content/detenv/bin/python -m pip install -q 'tensorrt-cu12-libs==10.8.0.43' --extra-index-url https://pypi.nvidia.com\n"
             "!/content/detenv/bin/python -c \"import onnxruntime as o; print('detector env OK, onnxruntime', o.__version__, o.get_available_providers())\""),
        code("%%writefile detector_server.py\n" + detector_code),
        code("import glob, os, subprocess, site, time, requests\n"
             "try:\n"
             "    from google.colab import userdata\n"
             "    roboflow_key = userdata.get('ROBOFLOW_API_KEY')      # never printed\n"
             "except Exception:\n"
             "    roboflow_key = None\n"
             "if not roboflow_key:\n"
             "    print('No ROBOFLOW_API_KEY secret (name it exactly so, Notebook access on): the detector stays off.')\n"
             "else:\n"
             "    # Try 1: the CUDA/cuDNN that came with the detector's own onnxruntime. Try 2: torch's GPU libraries.\n"
             "    torch_libs = [d for sp in site.getsitepackages() for d in glob.glob(sp + '/nvidia/*/lib')]\n"
             "    trt_libs = glob.glob('/content/detenv/lib/python3*/site-packages/tensorrt_libs')\n"
             "    print('TensorRT libraries:', trt_libs or 'MISSING (the detector will be ~8x slower)')\n"
             "    h = None\n"
             "    for attempt, (own, libs) in enumerate([('1', trt_libs), ('0', trt_libs + torch_libs)], 1):\n"
             "        env = dict(os.environ, ROBOFLOW_API_KEY=roboflow_key, DETECTOR_OWN_CUDA=own,\n"
             "                   ONNXRUNTIME_EXECUTION_PROVIDERS='[TensorrtExecutionProvider,CUDAExecutionProvider]')\n"
             "        # the first start builds the TensorRT engine: ~10-15 min on a T4 (later starts reuse it)\n"
             "        env['LD_LIBRARY_PATH'] = ':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')])\n"
             f"        detector_proc = subprocess.Popen(['/content/detenv/bin/python', 'detector_server.py', '--port', '{DETECTOR_PORT}'],\n"
             "                                         env=env, stdout=open(f'detector{attempt}.log', 'w'), stderr=subprocess.STDOUT)\n"
             "        for _ in range(600):                                # model + TensorRT engine build: up to ~20 min\n"
             "            time.sleep(2)\n"
             "            try:\n"
             f"                h = requests.get('http://127.0.0.1:{DETECTOR_PORT}/health', timeout=2).json(); break\n"
             "            except Exception:\n"
             "                if detector_proc.poll() is not None: break\n"
             "        if h and detector_proc.poll() is None:\n"
             "            print(f'detector ready (try {attempt}):', h); break\n"
             "        print(f'detector try {attempt} FAILED. Last lines of detector{attempt}.log:')\n"
             "        print(''.join(open(f'detector{attempt}.log').readlines()[-12:]))\n"
             "        detector_proc.kill(); h = None\n"
             "    if h:\n"
             "        if 'CUDAExecutionProvider' not in h.get('providers', []):\n"
             "            print('WARNING: no CUDA for onnxruntime: the detector runs on the CPU (too slow for live). See detector.log')"),
        markdown("## 3. Load the model, warm it up, and self-test\n"
                 "Downloads `config.yaml` + `model.pt` (150 MB) from Hugging Face. The self-test runs the model on "
                 "fake frames and checks the speed: the **real-time factor** (inference time ÷ 2 s stride) must be < 1."),
        code("import time, numpy as np, cv2\n"
             "from specialist_server import Spotter, Session, make_app\n"
             "spotter = Spotter()\n"
             "print(spotter.health())\n"
             "print('warm-up (first run is slow):', round(spotter.warm_up(), 2), 's')\n"
             "times = [spotter.warm_up() for _ in range(3)]\n"
             "print(f'inference on a {spotter.clip_len}-frame window: {np.mean(times)*1000:.0f} ms '\n"
             "      f'-> real-time factor {np.mean(times)/2.0:.3f} (must be < 1)')\n"
             "# Self-test of the live session logic with synthetic JPEG frames\n"
             "session = Session(spotter, {'min_confidence': 0.3, 'stride_s': 2.0})\n"
             "ok, jpg = cv2.imencode('.jpg', np.random.randint(0, 255, (224, 398, 3), np.uint8))\n"
             "for i in range(30):\n"
             "    due = session.add(i / spotter.fps, jpg.tobytes())\n"
             "result = session.infer()\n"
             "print('self-test OK:', {k: result[k] for k in ('frames', 'infer_ms', 'rtf', 'window')})"),
        markdown("## 4. Start the server (in the background)"),
        code("import threading, uvicorn, requests\n"
             f"PORT = {PORT}\n"
             "server = uvicorn.Server(uvicorn.Config(make_app(spotter), host='0.0.0.0', port=PORT, log_level='warning'))\n"
             "threading.Thread(target=server.run, daemon=True).start()\n"
             "time.sleep(3)\n"
             "print('local /health:', requests.get(f'http://127.0.0.1:{PORT}/health').json()['model'])"),
        markdown("## 4b. Detector self-test (through the main server, like the Mac will use it)\n"
                 "30 frames x 3 cameras at 1280x960, sent as the Mac sends them. Live needs 15 camera frames per "
                 "second, so the GPU time must be **well under 66 ms per camera frame**."),
        code("import json, struct, asyncio, numpy as np, cv2, websockets\n"
             "async def detector_selftest(n=30, cams=3):\n"
             f"    async with websockets.connect('ws://127.0.0.1:{PORT}/detect', max_size=2**24) as ws:\n"
             "        ok, jpg = cv2.imencode('.jpg', np.random.randint(0, 255, (960, 1280, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])\n"
             "        jpg = jpg.tobytes(); gpu = []; t0 = time.time()\n"
             "        for i in range(n):\n"
             "            head = json.dumps({'frame': i, 'sent': time.time(), 'items': [{'cam': c, 'size': len(jpg)} for c in range(1, cams + 1)]}).encode()\n"
             "            await ws.send(struct.pack('<I', len(head)) + head + jpg * cams)\n"
             "            r = json.loads(await ws.recv())\n"
             "            if r.get('type') == 'error': print(r); return\n"
             "            gpu.append(r['infer_ms'] / cams)\n"
             "        wall = (time.time() - t0) * 1000 / (n * cams)\n"
             "        print(f'detector: {np.mean(gpu[3:]):.0f} ms per camera frame on the GPU, {wall:.0f} ms including JPEG decode '\n"
             "              f'-> {\"OK for live\" if wall < 66 else \"TOO SLOW for 15 frames/s\"}')\n"
             "if roboflow_key: await detector_selftest()\n"
             "else: print('detector off (no ROBOFLOW_API_KEY secret)')"),
        markdown("## 5. Open a public tunnel and print the URL\n"
                 "Default: a free Cloudflare quick tunnel (no account, no data cap). The detector sends ~19 Mbit/s "
                 "(~7 GB per half), far over ngrok's free 1 GB/month: set `TUNNEL = 'ngrok'` only for the action model "
                 "alone (it uses the `NGROK_AUTHTOKEN` secret)."),
        code("import re, subprocess\n"
             "from IPython.display import HTML, display\n"
             "TUNNEL = 'cloudflare'   # or 'ngrok'\n"
             "public_url = None\n"
             "try:\n"
             "    from google.colab import userdata\n"
             "    token = userdata.get('NGROK_AUTHTOKEN') if TUNNEL == 'ngrok' else None\n"
             "except Exception:\n"
             "    token = None\n"
             "if token:\n"
             "    from pyngrok import ngrok, conf\n"
             "    conf.get_default().auth_token = token\n"
             "    public_url = ngrok.connect(PORT, 'http').public_url.replace('http://', 'https://')\n"
             "else:\n"
             "    print('Cloudflare quick tunnel (no account needed).')\n"
             "    !wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared\n"
             "    # start_new_session: stopping another cell (■) must not kill the tunnel\n"
             "    tunnel = subprocess.Popen(['./cloudflared', 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'],\n"
             "                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)\n"
             "    for line in tunnel.stdout:\n"
             "        found = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)\n"
             "        if found:\n"
             "            public_url = found.group(0)\n"
             "            break\n"
             "    threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()  # keep reading its log\n"
             "    time.sleep(5)\n"
             "for attempt in range(10):   # check /health THROUGH the tunnel\n"
             "    try:\n"
             "        r = requests.get(public_url + '/health', headers={'ngrok-skip-browser-warning': '1'}, timeout=10)\n"
             "        print('/health through the tunnel:', r.status_code, r.json()['gpu']); break\n"
             "    except Exception as error:\n"
             "        print('waiting for the tunnel...', type(error).__name__); time.sleep(3)\n"
             "display(HTML(f'<div style=\"font-size:28px;font-weight:bold;padding:16px;border:3px solid #22c55e;'\n"
             "             f'border-radius:8px\">Paste this into the dashboard:<br><code>{public_url}</code></div>'))\n"
             "print(public_url)"),
        markdown("## 6. Keep this running\nThis cell prints a heartbeat every minute so you can see the server is alive. "
                 "Stop it with the ■ button when you are done."),
        code("while True:\n"
             "    print(time.strftime('%H:%M:%S'), 'server alive at', public_url)\n"
             "    time.sleep(60)"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


LIGHT_NOTE = """# LIGHT version: the action model only (~5 minutes)

Use this when the live run reads the football detector's CACHED detections (computed on the Mac before the run):
it skips the detector (its own environment + TensorRT engine build, ~15 extra minutes).
For the Colab detector (the "Later tests" in docs/MVP3_LIVE_TEST_PLAN.md) use `specialist_server.ipynb`.
No Roboflow key needed here. Steps: Runtime -> T4 GPU -> Run all -> paste the URL into Studio -> Live -> Test."""


def light(nb: dict) -> dict:
    """The same notebook without the detector cells (2b, 4b): starts in ~5 minutes."""
    def is_detector(cell: dict) -> bool:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if src.startswith("%%writefile specialist_server.py"):
            return False                     # the main server (it only MENTIONS the detector in a comment)
        return src.startswith(("%%writefile detector_server.py", "## 2b.", "## 4b.")) or \
            any(k in src for k in ("detenv", "detector_selftest"))
    cells = [markdown(LIGHT_NOTE)] + [c for c in nb["cells"] if not is_detector(c)]
    return {**nb, "cells": cells}


if __name__ == "__main__":
    full = build()
    for name, nb in (("specialist_server.ipynb", full), ("specialist_light.ipynb", light(full))):
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1))
        print(f"wrote {out} ({len(nb['cells'])} cells)")
