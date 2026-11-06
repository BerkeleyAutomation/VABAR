"""A private NVIDIA MPS daemon for one ``vabar bench`` run (``runtime.mps = true``).

Several worker processes on one GPU normally time-slice it: each CUDA context
gets the whole GPU in turn, so physics-bound cells (cable routing, cable
insertion) gain almost nothing from a second process. Under MPS the contexts
share the GPU concurrently. Measured on cable_route/port_easy, 6 instances on 3
workers: 1,410 s without MPS, 547 s with it (1 worker: about 1,250 s for 5).

The daemon is private to the run. It has its own pipe directory, and only
processes that inherit ``CUDA_MPS_PIPE_DIRECTORY`` (this run's pool workers and
the tool servers they spawn) connect to it. Other jobs on the same GPU keep
using it directly. It serves one GPU, the first one ``CUDA_VISIBLE_DEVICES``
leaves the run, and the run's processes name that GPU by UUID. The daemon
numbers the one card it sees 0, so an index such as 3 would be refused ("not a
valid GPU ID in the MPS visible set").
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

CONTROL = "nvidia-cuda-mps-control"
PIPE_ENV = "CUDA_MPS_PIPE_DIRECTORY"
LOG_ENV = "CUDA_MPS_LOG_DIRECTORY"
_ENV_KEYS = (PIPE_ENV, LOG_ENV, "CUDA_VISIBLE_DEVICES")


def gpu_uuid(visible: str) -> str | None:
    """The UUID of the GPU that ``visible`` (an index or a UUID prefix, as CUDA_VISIBLE_DEVICES names it) means."""
    if visible.startswith(("GPU-", "MIG-")):
        return visible
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", visible or "0", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0].strip() if out else None


QUIT_TIMEOUT_S = 10
"""Seconds ``quit -t`` waits on the daemon's clients before it stops the MPS servers anyway.

A plain ``quit`` waits for every client to detach. On 2026-11-05 a `vabar bench`
interrupted while one worker hung in the renderer ran its cleanup, but ``quit``
waited on that worker (still an MPS client) past this module's 30 s subprocess
timeout and the daemon was left running. RoboSimStudio's pool now stops its
workers on any early exit; this covers clients it does not own, such as tool
servers."""


def _terminate(pipe: str) -> bool:
    """SIGTERM the control daemon whose pid file is in ``pipe``, when ``quit`` could not stop it.

    Only a process that is still ``nvidia-cuda-mps-control`` is signalled, so a
    recycled pid is left alone.
    """
    try:
        pid = int(Path(pipe, "nvidia-cuda-mps-control.pid").read_text().split()[0])
        if Path(f"/proc/{pid}/comm").read_text().strip() != CONTROL[:15]:  # comm is cut to 15 chars
            return False
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, IndexError):
        return False
    return True


def _control(command: str, env: dict[str, str]) -> bool:
    try:
        subprocess.run([CONTROL], input=command + "\n", env=env, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


@contextlib.contextmanager
def daemon(enabled: bool) -> Iterator[dict[str, str]]:
    """Run the body under a private MPS daemon when ``enabled``. Yields what provenance records.

    Without an MPS binary, with ``CUDA_MPS_PIPE_DIRECTORY`` already set (an
    outer daemon the caller chose), or when the daemon will not start, the body
    runs without one and the yielded record says why.
    """
    if not enabled:
        yield {"mps": "off"}
        return
    if os.environ.get(PIPE_ENV):
        yield {"mps": f"inherited ({PIPE_ENV}={os.environ[PIPE_ENV]})"}
        return
    if shutil.which(CONTROL) is None:
        print(f"[bench] runtime.mps: {CONTROL} not found; running without MPS", file=sys.stderr)
        yield {"mps": f"unavailable: no {CONTROL}"}
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    uuid = gpu_uuid(visible)
    if uuid is None:
        print("[bench] runtime.mps: cannot resolve the GPU's UUID; running without MPS", file=sys.stderr)
        yield {"mps": "unavailable: no GPU UUID"}
        return
    # A unix socket path is limited to ~108 bytes, so the directory sits in /tmp, not in the run's output.
    root = tempfile.mkdtemp(prefix="vabar-mps-", dir="/tmp")
    pipe, log = os.path.join(root, "pipe"), os.path.join(root, "log")
    os.makedirs(pipe)
    os.makedirs(log)
    env = {**os.environ, PIPE_ENV: pipe, LOG_ENV: log, "CUDA_VISIBLE_DEVICES": uuid}
    try:
        subprocess.run([CONTROL, "-d"], env=env, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[bench] runtime.mps: the daemon did not start ({exc}); running without MPS", file=sys.stderr)
        shutil.rmtree(root, ignore_errors=True)
        yield {"mps": f"unavailable: {exc}"}
        return
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ.update({PIPE_ENV: pipe, LOG_ENV: log, "CUDA_VISIBLE_DEVICES": uuid})
    print(f"[bench] runtime.mps: workers share {uuid} through a private MPS daemon ({pipe})")
    try:
        yield {"mps": "on", "gpu": uuid, "pipe": pipe}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if _control(f"quit -t {QUIT_TIMEOUT_S}", env) or _terminate(pipe):
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"[bench] runtime.mps: could not stop the daemon; `echo quit | {PIPE_ENV}={pipe} {CONTROL}`")
