"""Terminate and reap a spawned process group without orphaning children."""

from __future__ import annotations

import os
import signal


def terminate_process_group(proc, pgid=None, timeout_s: float = 10.0) -> None:
    """Terminate+reap the WHOLE spawned process group so uv/Python children can't
    orphan. Signals the group by the STORED ``pgid`` (the leader pid from
    start_new_session) — NOT os.getpgid(pid), which raises once the leader exits.
    Does NOT early-return when the launcher (leader) has already exited:
    descendants may still own the group. Falls back to proc.terminate()/kill()
    when there is no pgid (fake/external procs in tests)."""
    # A legitimately spawned leader (start_new_session) can never be pid 0 or 1,
    # and never this process's own group. killpg(1, sig) is kill(-1, sig) —
    # signal EVERYTHING this user owns (it takes down the whole desktop
    # session) — so degenerate pgids are refused outright and handled via the
    # per-process fallback below.
    try:
        _own_pgid = os.getpgid(0)
    except Exception:  # noqa: BLE001
        _own_pgid = -1
    if not isinstance(pgid, int) or pgid <= 1 or pgid == _own_pgid:
        pgid = None

    def _killpg(sig) -> bool:
        if pgid is None:
            return False
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass  # group already gone — fine
        except Exception:
            return False
        return True

    if _killpg(signal.SIGTERM):
        # Group path: grace-wait the leader, then SIGKILL the group to sweep any
        # lingering descendants even if the leader already exited.
        if proc is not None:
            try:
                proc.wait(timeout=timeout_s)
            except Exception:
                pass
        _killpg(signal.SIGKILL)
        if proc is not None:
            try:
                proc.wait(timeout=timeout_s)
            except Exception:
                pass
        return

    # No pgid (fake/external proc): use the handle directly.
    if proc is None:
        return
    poll = getattr(proc, "poll", None)
    if callable(poll) and poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_s)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=timeout_s)
        except Exception:
            pass

