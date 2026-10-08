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

HOW_TO = """# Story Reader — specialist server (Colab GPU)

This notebook runs the **ball action spotting** model
[`OpenSportsLab/OSL-loc-snbas-2025-e2e`](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e)
(12 classes: PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK,
PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL) on a free Colab GPU, and gives you a public URL that the
dashboard on your Mac connects to.

## How to run (5 steps)
1. **Turn on the GPU:** menu *Runtime → Change runtime type → T4 GPU → Save*.
2. *(Optional, recommended)* **ngrok token:** make a free account at https://dashboard.ngrok.com, copy your
   authtoken, then in Colab click the 🔑 *Secrets* icon (left bar) → *Add new secret* → name
   `NGROK_AUTHTOKEN`, value = your token, and switch *Notebook access* on.
   Without it, the notebook uses a free Cloudflare tunnel instead (no account needed).
3. **Run everything:** menu *Runtime → Run all*. The first run takes ~3–5 minutes (installing + downloading the model).
4. **Copy the URL** printed in big letters at the end (it looks like `https://xxxx.ngrok-free.app`
   or `https://xxxx.trycloudflare.com`).
5. **Paste it** into the dashboard (http://localhost:8000/story → *Colab specialist* box → *Connect*),
   pick mode *fast + specialist* or *hybrid*, and press **Go Live**.

Keep this tab open while you use the dashboard. Colab stops the server after ~90 minutes of no activity
(free tier); then just *Run all* again and paste the new URL.

**License note:** the model is AGPL-3.0 (research / non-commercial use)."""


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    server_code = (HERE / "specialist_server.py").read_text()
    cells = [
        markdown(HOW_TO),
        markdown("## 1. Check the GPU and install packages"),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU: Runtime -> Change runtime type -> T4 GPU'\n"
             "!pip install -q opensportslib==0.3.1 fastapi uvicorn pyngrok websockets huggingface_hub pyyaml\n"
             "import torch; print('CUDA available:', torch.cuda.is_available())"),
        markdown("## 2. The server code (same as `colab/specialist_server.py` in the project)"),
        code("%%writefile specialist_server.py\n" + server_code),
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
        markdown("## 5. Open a public tunnel and print the URL\n"
                 "Uses ngrok if the `NGROK_AUTHTOKEN` secret exists, otherwise a free Cloudflare quick tunnel."),
        code("import re, subprocess\n"
             "from IPython.display import HTML, display\n"
             "public_url = None\n"
             "try:\n"
             "    from google.colab import userdata\n"
             "    token = userdata.get('NGROK_AUTHTOKEN')\n"
             "except Exception:\n"
             "    token = None\n"
             "if token:\n"
             "    from pyngrok import ngrok, conf\n"
             "    conf.get_default().auth_token = token\n"
             "    public_url = ngrok.connect(PORT, 'http').public_url.replace('http://', 'https://')\n"
             "else:\n"
             "    print('No NGROK_AUTHTOKEN secret: using a Cloudflare quick tunnel instead.')\n"
             "    !wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared\n"
             "    tunnel = subprocess.Popen(['./cloudflared', 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'],\n"
             "                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
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


if __name__ == "__main__":
    out = HERE / "specialist_server.ipynb"
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out}")
