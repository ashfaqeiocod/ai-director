"""demo_server.py — the dashboard on its own port (8010) for the Arm demo: venv/bin/python -m arm.demo_server

Same app as `python -m dashboard.server` (all tabs, /arm included), just a separate process and port, so the
football session's dashboard on 8000 can be restarted without stopping the client demo (and the other way round).
"""
import uvicorn

from dashboard.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="warning")
