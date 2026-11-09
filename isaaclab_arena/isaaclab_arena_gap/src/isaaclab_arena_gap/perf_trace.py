"""Opt-in perf tracing sidecar for the GaP<->Arena bridge.

Enabled only when ``CAP_PERF_TRACE_DIR`` is set; the default eval path is
untouched. Each record is one JSON line.

Contract: the tracer must NEVER perturb or fail the traced run. Every
public method swallows its own errors and disables the tracer on the
first write failure (a full disk must not masquerade as a bridge
failure). The stream is line-buffered and an ``atexit`` hook closes it,
so records survive abnormal termination — the dominant failure mode a
perf trace exists to diagnose — up to the OS buffer of the final line.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any


class PerfTrace:
    def __init__(self, name: str):
        directory = os.environ.get("CAP_PERF_TRACE_DIR", "")
        self.enabled = bool(directory)
        self._stream = None
        if not self.enabled:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{name}-{os.getpid()}.jsonl")
            # buffering=1: line-buffered, so every record reaches the OS
            # without an explicit flush call on the hot path.
            self._stream = open(path, "a", encoding="utf-8", buffering=1)
            atexit.register(self.close)
            self.record("trace_open", unix=time.time())
        except Exception:  # noqa: BLE001 — tracing must never fail the run
            self.enabled = False
            self._stream = None

    def record(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._stream is None:
            return
        try:
            fields["ev"] = event
            fields["t"] = time.perf_counter()
            self._stream.write(json.dumps(fields) + "\n")
        except Exception:  # noqa: BLE001 — e.g. ENOSPC: disable, don't raise
            self.enabled = False
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self.record("trace_close", unix=time.time())
            self._stream.flush()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
        self._stream = None
        self.enabled = False
