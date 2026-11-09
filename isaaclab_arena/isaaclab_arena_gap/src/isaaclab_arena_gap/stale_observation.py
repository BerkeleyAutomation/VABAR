"""Freshness tracking for the Arena observation bridge."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


STALE_OBSERVATION_TIMEOUT_ENV = "CAP_STALE_OBSERVATION_TIMEOUT_S"
DEFAULT_STALE_OBSERVATION_TIMEOUT_S = 60.0
STARTUP_OBSERVATION_TIMEOUT_ENV = "CAP_STARTUP_OBSERVATION_TIMEOUT_S"
DEFAULT_STARTUP_OBSERVATION_TIMEOUT_S = 180.0


class BridgeIntegrationError(RuntimeError):
    """A sticky bridge failure that invalidates the running graph."""

    failure_class = "runtime_error"
    failure_domain = "infrastructure"
    retryable = True


class StaleObservationError(BridgeIntegrationError):
    """The established Arena observation stream stopped advancing."""

    failure_class = "stale_observation"

    def __init__(self, *, age_s: float, timeout_s: float, last_timestamp: float):
        self.age_s = float(age_s)
        self.timeout_s = float(timeout_s)
        self.last_timestamp = float(last_timestamp)
        super().__init__(
            "Arena observation stream did not advance for "
            f"{self.age_s:.3f}s (limit {self.timeout_s:.3f}s; "
            f"last wire timestamp {self.last_timestamp:.6f})"
        )


def _resolve_timeout(
    name: str,
    default: float,
    environ: Mapping[str, str] | None = None,
) -> float:
    source = os.environ if environ is None else environ
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return timeout_s


def resolve_stale_observation_timeout(
    environ: Mapping[str, str] | None = None,
) -> float:
    """Return the steady-state stale-stream timeout."""
    return _resolve_timeout(
        STALE_OBSERVATION_TIMEOUT_ENV,
        DEFAULT_STALE_OBSERVATION_TIMEOUT_S,
        environ,
    )


def resolve_startup_observation_timeout(
    environ: Mapping[str, str] | None = None,
) -> float:
    """Return the first simulator-step observation timeout."""
    return _resolve_timeout(
        STARTUP_OBSERVATION_TIMEOUT_ENV,
        DEFAULT_STARTUP_OBSERVATION_TIMEOUT_S,
        environ,
    )


@dataclass(frozen=True)
class _FreshnessState:
    wire_timestamp: float
    observed_at_s: float
    has_advanced: bool


class ObservationFreshnessGuard:
    """Track stream progress with wire identity and a local monotonic clock."""

    def __init__(self, timeout_s: float, startup_timeout_s: float | None = None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = float(timeout_s)
        if startup_timeout_s is None:
            startup_timeout_s = timeout_s
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self.startup_timeout_s = float(startup_timeout_s)
        self._state: _FreshnessState | None = None
        self._failure: StaleObservationError | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _timestamp(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return timestamp if math.isfinite(timestamp) else None

    def observe(
        self,
        wire_timestamp: Any,
        *,
        now_s: float,
        stream_ready: bool,
        terminating: bool = False,
    ) -> StaleObservationError | None:
        """Observe one wire snapshot and return a sticky failure when stale."""
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        timestamp = self._timestamp(wire_timestamp)
        with self._lock:
            if terminating:
                return None
            if self._failure is not None:
                return self._failure
            if self._state is None:
                if stream_ready and timestamp is not None:
                    self._state = _FreshnessState(timestamp, float(now_s), False)
                return None
            if timestamp is not None and timestamp != self._state.wire_timestamp:
                self._state = _FreshnessState(
                    timestamp,
                    float(now_s),
                    True,
                )
                return None

            age_s = max(0.0, float(now_s) - self._state.observed_at_s)
            active_timeout_s = (
                self.startup_timeout_s
                if not self._state.has_advanced
                else self.timeout_s
            )
            if age_s < active_timeout_s:
                return None
            self._failure = StaleObservationError(
                age_s=age_s,
                timeout_s=active_timeout_s,
                last_timestamp=self._state.wire_timestamp,
            )
            return self._failure

    def failure(self) -> StaleObservationError | None:
        with self._lock:
            return self._failure
