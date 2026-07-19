from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def port_available(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verified dashboard process cutover")
    parser.add_argument("--old-pid", required=True, type=int)
    parser.add_argument("--preflight-url", required=True)
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--backtest", default="data/models/backtest_no_odds_v8.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=10001, type=int)
    parser.add_argument("--log", default="data/logs/web_dashboard.log", type=Path)
    args = parser.parse_args()

    if not healthy(args.preflight_url):
        raise SystemExit(f"preflight server is not healthy: {args.preflight_url}")

    cmdline_path = Path(f"/proc/{args.old_pid}/cmdline")
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        raise SystemExit(f"old process does not exist: {args.old_pid}")
    if "boatrace_ai.web_dashboard" not in cmdline or str(args.port) not in cmdline:
        raise SystemExit(f"refusing to stop unexpected process: {cmdline}")

    os.kill(args.old_pid, signal.SIGTERM)
    if not wait_for_exit(args.old_pid, 5.0):
        raise SystemExit("old dashboard did not stop; leaving it running")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not port_available("0.0.0.0", args.port):
        time.sleep(0.05)
    if not port_available("0.0.0.0", args.port):
        raise SystemExit(f"port {args.port} did not become available")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "boatrace_ai.web_dashboard",
            "--db",
            args.db,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--backtest",
            args.backtest,
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    health_url = f"http://127.0.0.1:{args.port}/reports/models"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"replacement exited with code {process.returncode}")
        if healthy(health_url):
            print(f"dashboard cutover complete: pid={process.pid} url={health_url}")
            return 0
        time.sleep(0.1)
    process.terminate()
    raise SystemExit("replacement did not become healthy within 30 seconds")


if __name__ == "__main__":
    raise SystemExit(main())
