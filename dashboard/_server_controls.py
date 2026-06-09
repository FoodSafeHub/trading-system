"""
Server controls — restart the API server from within the Streamlit dashboard.

Finds the uvicorn process by cmdline signature, kills it, relaunches it with
the same command that start.bat uses, then polls /health until it's back up.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st

_ROOT      = Path(__file__).resolve().parent.parent   # trading-system/
_UVICORN   = _ROOT / ".venv" / "Scripts" / "uvicorn.exe"
_API_CMD   = [
    str(_UVICORN),
    "app.main:app",
    "--host", "127.0.0.1",
    "--port", "8001",
    "--ssl-keyfile=key.pem",
    "--ssl-certfile=cert.pem",
]
_HEALTH_URL = "https://127.0.0.1:8001/health"
_POLL_INTERVAL = 1.5   # seconds between health checks
_MAX_WAIT      = 30    # seconds to wait for server to come back up


def _find_uvicorn_pids() -> list[int]:
    """Return PIDs of all running uvicorn processes bound to port 8001."""
    try:
        import psutil
        pids = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "uvicorn" in cmd and "8001" in cmd:
                    pids.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids
    except ImportError:
        return []


def _kill_uvicorn() -> int:
    """Kill all uvicorn processes on port 8001. Returns count killed."""
    try:
        import psutil
        killed = 0
        for pid in _find_uvicorn_pids():
            try:
                p = psutil.Process(pid)
                p.terminate()
                p.wait(timeout=5)
                killed += 1
            except Exception:
                try:
                    psutil.Process(pid).kill()
                    killed += 1
                except Exception:
                    pass
        return killed
    except ImportError:
        return 0


def _launch_uvicorn() -> subprocess.Popen:
    """Start uvicorn as a detached background process (new console window)."""
    return subprocess.Popen(
        _API_CMD,
        cwd=str(_ROOT),
        creationflags=subprocess.CREATE_NEW_CONSOLE,  # Windows: own terminal window
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_health(max_wait: float = _MAX_WAIT) -> bool:
    """Poll /health until 200 or timeout. Returns True if server came up."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(_HEALTH_URL, verify=False, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(_POLL_INTERVAL)
    return False


def restart_api_server() -> tuple[bool, str]:
    """
    Kill the running uvicorn, relaunch it, wait for /health.
    Returns (success, message).
    """
    if not _UVICORN.exists():
        return False, f"uvicorn not found at {_UVICORN}"

    killed = _kill_uvicorn()
    if killed == 0:
        # Nothing to kill — just launch fresh
        pass

    time.sleep(1.0)  # brief pause to let the port free up
    _launch_uvicorn()

    came_up = _wait_for_health()
    if came_up:
        return True, f"API server restarted successfully (killed {killed} process(es))."
    return False, f"API server launched but did not respond within {_MAX_WAIT}s — check the API window."


def render_restart_button(location=None, key: str = "restart_api") -> None:
    """
    Render a Restart API button in `location` (defaults to st).
    Shows a spinner while restarting and a success/error message after.
    """
    container = location or st

    if container.button(
        "↺ Restart API server",
        key=key,
        help=(
            "Kills the running uvicorn process and relaunches it. "
            "Use this after code changes without closing the terminal window."
        ),
        type="secondary",
        use_container_width=True,
    ):
        with st.spinner("Restarting API server…"):
            ok, msg = restart_api_server()
        if ok:
            st.toast(msg, icon="✅")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(msg, icon="❌")
