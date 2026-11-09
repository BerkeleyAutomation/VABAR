"""One SAM 3 and one GroundingDINO per GPU for a multi-worker run (``runtime.share_tool_servers``).

Each graph worker otherwise starts its own tool servers, about 6 GB of SAM 3 and
3 GB of GroundingDINO, and those copies are what limit how many workers fit on
a GPU. graph-as-policy's tool-bundle manager shares a bundle between processes
when ``GAP_TOOL_SHARE_DIR`` names a directory (``gap_core.rpc.broker``): the
first worker to need SAM 3 starts a broker that owns the one server, and every
worker reaches it through a relay. Only stateless, GPU-heavy bundles are shared
(sam3, grounding-dino by default); the VLM bundle's seconds-long remote calls
would queue behind each other, so each worker keeps its own.

This context gives the run a private directory, makes this process the brokers'
owner (they exit when it does), and stops them when the run ends. A
graph-as-policy without the feature ignores the variable, so the run is then
exactly as before.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import tempfile
from collections.abc import Iterator
from pathlib import Path

DIR_ENV = "GAP_TOOL_SHARE_DIR"
OWNER_ENV = "GAP_TOOL_SHARE_OWNER"


@contextlib.contextmanager
def shared_tool_servers(enabled: bool) -> Iterator[dict[str, str]]:
    """Share the GPU-heavy tool servers among the run's workers when ``enabled``. Yields provenance."""
    if not enabled:
        yield {"tool_servers": "per worker"}
        return
    if os.environ.get(DIR_ENV):
        yield {"tool_servers": f"shared, inherited ({DIR_ENV}={os.environ[DIR_ENV]})"}
        return
    root = tempfile.mkdtemp(prefix="vabar-tools-", dir="/tmp")  # short: Unix socket paths are ~108 bytes
    saved = {key: os.environ.get(key) for key in (DIR_ENV, OWNER_ENV)}
    os.environ[DIR_ENV] = root
    os.environ[OWNER_ENV] = str(os.getpid())
    try:
        yield {"tool_servers": "shared", "dir": root}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        stop_brokers(root)
        shutil.rmtree(root, ignore_errors=True)


def stop_brokers(root: str) -> int:
    """SIGTERM every broker that wrote a pid file under ``root`` (each stops its server). Returns how many."""
    stopped = 0
    for pid_file in Path(root).glob("*.sock.pid"):
        try:
            os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
            stopped += 1
        except (OSError, ValueError):
            pass
    return stopped
