"""Private channel between `pixi run sim` and its clients (MujocoDriver, the virtual
camera, sim-ctl): JSON lines over a local Unix socket. Not a public interface.

Requests carry an "op". Requests with an "id" get exactly one reply {"id", "ok", ...};
servo commands have no id and get no reply. A client that sends "subscribe" also
receives {"type": "state", ...} lines at the requested rate.
"""

from __future__ import annotations

import json
import os
import pathlib

VERSION = 1


def runtime_dir() -> pathlib.Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/airbot-sim-{os.getuid()}"
    path = pathlib.Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AIRBOT_SIM_SOCKET", runtime_dir() / "airbot_sim.sock"))


def lock_path() -> pathlib.Path:
    return socket_path().with_suffix(".lock")


def encode(msg: dict) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict:
    return json.loads(line)
