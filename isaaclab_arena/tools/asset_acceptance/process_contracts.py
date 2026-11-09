"""Shared subprocess lifecycle contract for live Isaac acceptance runs."""

from __future__ import annotations

import signal
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

# Kit may crash in its own teardown after a run has persisted complete evidence.
RETRYABLE_ISAAC_CRASH_CODES = frozenset(
    {
        128 + signal.SIGSEGV,
        -signal.SIGSEGV,
        128 + signal.SIGABRT,
        -signal.SIGABRT,
    }
)
ACCEPTED_ISAAC_TEARDOWN_CODES = frozenset({0, *RETRYABLE_ISAAC_CRASH_CODES})


def is_accepted_isaac_teardown(returncode: int) -> bool:
    return returncode in ACCEPTED_ISAAC_TEARDOWN_CODES


@dataclass(frozen=True)
class FailedAttempt:
    """One fresh-process attempt that did not leave complete, valid evidence."""

    attempt: int
    process_returncode: int
    validator_returncode: int | None
    """None without a result file, 0 for valid evidence, 1 for invalid evidence."""
    will_retry: bool

    @property
    def reason(self) -> str:
        if self.validator_returncode != 0:
            return "incomplete or invalid evidence"
        return f"unexpected process return code {self.process_returncode}"


def run_fresh_process_with_retries(
    *,
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str],
    stale_outputs: Iterable[Path],
    result_path: Path,
    validate: Callable[[], None],
    label: str,
    on_failure: Callable[[FailedAttempt], None],
    max_attempts: int = 3,
) -> bool:
    """Run ``command`` in fresh processes until it leaves validated evidence.

    Each attempt first removes ``stale_outputs``. A Kit crash without complete
    evidence is retried up to ``max_attempts``; ``on_failure`` sees every
    failed attempt before the next one starts, so callers can preserve it.
    """
    stale_outputs = tuple(stale_outputs)
    for attempt in range(1, max_attempts + 1):
        for path in stale_outputs:
            path.unlink(missing_ok=True)
        completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
        validator_returncode = None
        if result_path.is_file():
            try:
                validate()
            except Exception as error:
                print(f"{label} evidence is invalid: {error!r}", file=sys.stderr)
                validator_returncode = 1
            else:
                validator_returncode = 0
        if validator_returncode == 0 and is_accepted_isaac_teardown(
            completed.returncode
        ):
            return True
        will_retry = (
            completed.returncode in RETRYABLE_ISAAC_CRASH_CODES
            and attempt < max_attempts
        )
        on_failure(
            FailedAttempt(
                attempt=attempt,
                process_returncode=completed.returncode,
                validator_returncode=validator_returncode,
                will_retry=will_retry,
            )
        )
        if not will_retry:
            return False
        print(
            f"{label} crashed without complete evidence; "
            f"retrying fresh process ({attempt}/{max_attempts}).",
            flush=True,
        )
    raise AssertionError("fresh-process retry loop exhausted without a result")
