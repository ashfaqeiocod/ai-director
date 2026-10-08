"""check_setup.py — "is this computer ready?" Run it first on any new device.

    venv/bin/python src/check_setup.py            (Windows: venv\\Scripts\\python src\\check_setup.py)
    venv/bin/python src/check_setup.py --online   also tests your Roboflow key (needs internet)

It checks Python, the packages, ffmpeg, Node.js, your .env key, where the model will run
(GPU / Neural Engine / CPU) and whether the videos in config.yaml can be found.
Every problem comes with the exact thing to do. Nothing is changed on your computer.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
results: list[str] = []


def report(status: str, what: str, fix: str = "") -> None:
    results.append(status)
    print(f"  [{status}] {what}")
    if fix:
        print(f"         → {fix}")


def python_cmd() -> str:
    return r"venv\Scripts\python" if os.name == "nt" else "venv/bin/python"


def check_python() -> None:
    v = sys.version_info
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if (3, 10) <= (v.major, v.minor) <= (3, 13):
        report(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        report(FAIL, f"Python {v.major}.{v.minor} (needs 3.10 – 3.13)",
               "Install Python 3.12 or 3.13, then recreate the venv (see docs/NEW_DEVICE_SETUP.md)")
    if not in_venv:
        report(WARN, "Not running inside the project's venv", f"Run it as: {python_cmd()} src/check_setup.py")


def check_packages() -> bool:
    missing = []
    for module in ("yaml", "dotenv", "numpy", "cv2", "supervision", "inference", "onnxruntime",
                   "fastapi", "uvicorn", "yt_dlp", "imageio_ffmpeg", "av"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        report(FAIL, f"Missing Python packages: {', '.join(missing)}",
               f"{python_cmd()} -m pip install -r requirements.txt")
        return False
    report(OK, "Python packages installed")
    return True


def check_ffmpeg() -> None:
    from src.common import ffmpeg_exe

    try:
        exe = ffmpeg_exe()
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "libx264" in out:
            report(OK, f"ffmpeg with H.264 ({Path(exe).name})")
        else:
            report(FAIL, "ffmpeg found but without the H.264 encoder (libx264)",
                   f"{python_cmd()} -m pip install --force-reinstall imageio-ffmpeg")
    except Exception as error:
        report(FAIL, f"ffmpeg not usable: {error}", f"{python_cmd()} -m pip install imageio-ffmpeg")


def check_node(cfg: dict) -> None:
    from src.prepare import _sources
    from src.download import is_url

    needs_node = any(is_url(s) for s in _sources(cfg))
    if shutil.which("node"):
        version = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        report(OK, f"Node.js {version} (used to download YouTube links)")
    elif needs_node:
        report(FAIL, "Node.js not found, but config.yaml uses a YouTube link",
               "Install Node.js from https://nodejs.org (LTS), or use a video file instead of a link")
    else:
        report(OK, "Node.js not installed (only needed for YouTube links; your input uses files)")


def check_key(online: bool) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not Path(".env").exists():
        report(FAIL, ".env file missing", "Copy .env.example to .env and put your Roboflow API key in it")
        return
    if not key or key == "PASTE_KEY_HERE" or key == "your_key_here":
        report(FAIL, "ROBOFLOW_API_KEY in .env is empty or a placeholder",
               "Get a free key at https://app.roboflow.com → Settings → API Keys, paste it into .env")
        return
    if not online:
        report(OK, "Roboflow key present in .env (run with --online to test it)")
        return
    try:
        import requests

        r = requests.get("https://api.roboflow.com/", params={"api_key": key}, timeout=15)
        if r.status_code == 200 and "workspace" in r.text:
            report(OK, "Roboflow key works")
        else:
            report(FAIL, f"Roboflow rejected the key (HTTP {r.status_code})", "Check the key in .env")
    except Exception as error:
        report(WARN, f"Could not reach Roboflow ({type(error).__name__})", "Check your internet connection")


def check_accelerator(cfg: dict) -> None:
    import onnxruntime

    from src.common import usable_onnx_providers

    wanted = cfg["detection"].get("onnx_providers") or []
    usable = usable_onnx_providers(wanted)
    available = onnxruntime.get_available_providers()
    fast = [p for p in usable if p != "CPUExecutionProvider"]
    if fast:
        report(OK, f"Detection will run on: {fast[0].replace('ExecutionProvider', '')} (fast)")
    else:
        hint = {"Darwin": "On a Mac this should be CoreML: keep CoreMLExecutionProvider in detection.onnx_providers",
                "Windows": "With an NVIDIA GPU: pip install onnxruntime-gpu and use CUDAExecutionProvider",
                "Linux": "With an NVIDIA GPU: pip install onnxruntime-gpu and use CUDAExecutionProvider"}
        report(WARN, f"Detection will run on the CPU only (slow: seconds per frame). "
                     f"Wanted {wanted}; this machine has {available}",
               hint.get(platform.system(), "See docs/NEW_DEVICE_SETUP.md"))


def check_input(cfg: dict) -> None:
    from src.download import is_url, resolve_path
    from src.prepare import _sources, is_up_to_date

    mode = cfg["input"].get("mode")
    if mode not in ("split", "files"):
        report(FAIL, f'input.mode is "{mode}"', 'Set input.mode to "split" or "files" in config.yaml')
        return
    sources = _sources(cfg)
    problems = [s for s in sources if s and not is_url(s) and not resolve_path(s).is_file()]
    if mode == "files" and len(sources) != cfg["video"]["num_cams"]:
        report(FAIL, f"input.files.cams has {len(sources)} clips (needs 3)", "List exactly 3 clips in config.yaml")
    for s in problems:
        report(FAIL, f"Video file not found: {resolve_path(s)}", "Fix the path in config.yaml (input section)")
    if not problems:
        report(OK, f"Input: {mode} mode, {len(sources)} source(s) found")
    if is_up_to_date(cfg):
        report(OK, "Cameras are prepared for this input")
    else:
        report(WARN, "Cameras not prepared for this input yet",
               f"{python_cmd()} src/main.py   (or {python_cmd()} src/prepare.py)")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent.parent)
    print(f"\nChecking this computer ({platform.system()} {platform.machine()}) …\n")
    check_python()
    if check_packages():
        from src.common import load_config

        cfg = load_config()
        check_ffmpeg()
        check_node(cfg)
        check_key("--online" in sys.argv)
        check_accelerator(cfg)
        check_input(cfg)
    fails, warns = results.count(FAIL), results.count(WARN)
    print()
    if fails:
        print(f"{fails} problem(s) to fix (see → lines above).")
        sys.exit(1)
    print(f"Ready{' (with ' + str(warns) + ' warning(s))' if warns else ''}. Next: {python_cmd()} src/main.py")


if __name__ == "__main__":
    main()
