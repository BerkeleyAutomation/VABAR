"""Durable sidecar contracts for the GaP-to-Arena bridge."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GRAPH_RESULT_SCHEMA = "vabar.graph-result.v1"
BRIDGE_STATUS_SCHEMA = "vabar.bridge-status.v2"


class ContractError(ValueError):
    """Raised when a bridge sidecar does not satisfy its public contract."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_graph_result(
    result: Any,
    *,
    campaign_id: str,
    plan_sha256: str,
    image_ref: str,
    sample_id: str,
    attempt_id: str,
) -> str:
    """Validate an identity-bound GaP result and return its outcome."""
    expected_identity = {
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "image_ref": image_ref,
        "sample_id": sample_id,
        "attempt_id": attempt_id,
    }
    if not all(
        isinstance(value, str) and value for value in expected_identity.values()
    ):
        raise ContractError("Expected graph result identity is incomplete")
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != GRAPH_RESULT_SCHEMA
    ):
        raise ContractError("Graph result has an unsupported schema")
    for key, expected in expected_identity.items():
        if result.get(key) != expected:
            raise ContractError(f"Graph result has mismatched {key}")
    if not isinstance(result.get("timestamp"), str) or not result["timestamp"]:
        raise ContractError("Graph result has an invalid timestamp")

    outcome = result.get("outcome")
    if outcome not in {"started", "success", "failure", "runtime_error"}:
        raise ContractError("Graph result has an invalid outcome")
    success = result.get("success")
    if outcome == "started":
        if success is not None:
            raise ContractError("Started graph result must have success=null")
    elif not isinstance(success, bool):
        raise ContractError("Final graph result has no boolean success")
    if result.get("exit_status") not in {"success", "failure", None}:
        raise ContractError("Graph result has an invalid exit_status")
    duration_s = result.get("duration_s")
    if (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or not math.isfinite(duration_s)
        or duration_s < 0
    ):
        raise ContractError("Graph result has an invalid duration_s")
    if result.get("trace_path") is not None and not isinstance(
        result["trace_path"], str
    ):
        raise ContractError("Graph result has an invalid trace_path")

    error_type = result.get("error_type")
    error = result.get("error")
    if error_type is not None and not isinstance(error_type, str):
        raise ContractError("Graph result has an invalid error_type")
    if error is not None and not isinstance(error, str):
        raise ContractError("Graph result has an invalid error")
    failure_class = result.get("failure_class")
    if failure_class not in {None, "other", "runtime_error", "stale_observation"}:
        raise ContractError("Graph result has an invalid failure_class")

    if outcome == "started":
        valid = (
            result["exit_status"] is None
            and error_type is None
            and error is None
            and failure_class is None
        )
    elif outcome == "success":
        valid = (
            success
            and result["exit_status"] == "success"
            and error_type is None
            and error is None
            and failure_class is None
        )
    elif outcome == "failure":
        valid = (
            not success
            and result["exit_status"] == "failure"
            and failure_class == "other"
        )
    else:
        valid = (
            not success
            and result["exit_status"] != "success"
            and isinstance(error_type, str)
            and bool(error_type)
            and isinstance(error, str)
            and bool(error)
            and failure_class in {"runtime_error", "stale_observation"}
        )
        if failure_class == "stale_observation":
            valid = valid and error_type == "StaleObservationError"
    if not valid:
        raise ContractError(f"Graph result fields contradict outcome={outcome!r}")
    return outcome


def graph_result_reference(
    result: dict[str, Any], *, relative_path: str
) -> dict[str, str]:
    """Reference a result by relative path and canonical JSON digest."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ContractError("Graph result reference path must be relative")
    return {
        "schema_version": GRAPH_RESULT_SCHEMA,
        "relative_path": relative_path,
        "canonical_sha256": _canonical_sha256(result),
    }


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically publish a world-readable JSON sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


def _graph_result_identity() -> dict[str, str | None]:
    identity = {
        "campaign_id": os.environ.get("CAP_CAMPAIGN_ID"),
        "plan_sha256": os.environ.get("CAP_PLAN_SHA256"),
        "image_ref": os.environ.get("CAP_IMAGE_REF"),
        "sample_id": os.environ.get("CAP_SAMPLE_ID"),
        "attempt_id": os.environ.get("CAP_ATTEMPT_ID"),
    }
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise RuntimeError("graph result identity is incomplete: " + ", ".join(missing))
    return identity


def _graph_result(
    outcome: str,
    *,
    success: bool | None,
    exit_status: str | None = None,
    failure_class: str | None = None,
    error: BaseException | None = None,
    duration_s: float = 0.0,
    trace_path: str | None = None,
) -> dict[str, Any]:
    """One graph-result sidecar payload (``GRAPH_RESULT_SCHEMA``)."""
    return {
        "schema_version": GRAPH_RESULT_SCHEMA,
        "timestamp": utc_now(),
        **_graph_result_identity(),
        "outcome": outcome,
        "success": success,
        "exit_status": exit_status,
        "failure_class": failure_class,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error)[:2000] if error is not None else None,
        "duration_s": max(0.0, duration_s),
        "trace_path": trace_path,
    }


def started_graph_result() -> dict[str, Any]:
    """The sidecar a GaP process writes before it drives the robot."""
    return _graph_result("started", success=None)


def final_graph_result(result: Any) -> dict[str, Any]:
    """Translate GaP's ExecutionResult into the final sidecar without guessing a failure cause."""
    exit_status = getattr(result, "exit_status", None)
    error = getattr(result, "error", None)
    if exit_status == "success" and bool(getattr(result, "success", False)):
        outcome = "success"
        failure_class = None
    elif exit_status == "failure":
        # GaP may attach PipelineError to an explicit failure terminal. The
        # terminal is authoritative; the exception remains diagnostic only.
        outcome = "failure"
        failure_class = "other"
    else:
        outcome = "runtime_error"
        failure_class = "runtime_error"
        if error is None:
            error = RuntimeError("GaP returned without a terminal exit status")
    trace_path = getattr(result, "trace_path", None)
    return _graph_result(
        outcome,
        success=outcome == "success",
        exit_status=exit_status,
        failure_class=failure_class,
        error=error,
        duration_s=float(getattr(result, "duration_s", 0.0)),
        trace_path=str(trace_path) if trace_path is not None else None,
    )
