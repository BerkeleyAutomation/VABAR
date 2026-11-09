"""The RSS adapter's graph-result sidecars satisfy the bridge contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_arena_gap.bridge_contract import (
    final_graph_result,
    started_graph_result,
    validate_graph_result,
)

IDENTITY = {
    "campaign_id": "campaign",
    "plan_sha256": "0" * 64,
    "image_ref": "image",
    "sample_id": "sample",
    "attempt_id": "attempt",
}


@pytest.fixture(autouse=True)
def _identity(monkeypatch) -> None:
    for key, value in IDENTITY.items():
        monkeypatch.setenv(f"CAP_{key.upper()}", value)


def test_started_result_is_a_valid_nonterminal_sidecar() -> None:
    assert validate_graph_result(started_graph_result(), **IDENTITY) == "started"


@pytest.mark.parametrize(
    ("result", "outcome", "failure_class"),
    [
        (SimpleNamespace(success=True, exit_status="success", error=None, duration_s=1.5), "success", None),
        (SimpleNamespace(success=False, exit_status="failure", error=RuntimeError("x")), "failure", "other"),
        (SimpleNamespace(success=True, exit_status=None, error=None), "runtime_error", "runtime_error"),
        # The graph died before returning an ExecutionResult.
        (None, "runtime_error", "runtime_error"),
    ],
)
def test_final_result_translates_without_guessing_a_cause(result, outcome, failure_class) -> None:
    payload = final_graph_result(result)
    assert validate_graph_result(payload, **IDENTITY) == outcome
    assert payload["success"] is (outcome == "success")
    assert payload["failure_class"] == failure_class
    if outcome == "runtime_error":
        assert payload["error_type"]


def test_results_require_the_attempt_identity(monkeypatch) -> None:
    monkeypatch.delenv("CAP_SAMPLE_ID")
    with pytest.raises(RuntimeError, match="sample_id"):
        started_graph_result()
