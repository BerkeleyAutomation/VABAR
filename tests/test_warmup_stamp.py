# SPDX-License-Identifier: Apache-2.0
"""A cell's first graph run on an install compiles its scenes, once. No GPU: the warmup itself is a stand-in."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vabar import benchmark, warmup
from vabar import config as cfg


@pytest.fixture
def stamped(monkeypatch, tmp_path):
    """The stamp in a temporary directory, and a list of the warmups asked for."""
    calls: list[list[str]] = []
    monkeypatch.setattr(warmup, "stamp_path", lambda cell, config: tmp_path / "vabar_warm" / "stamp")
    monkeypatch.setattr(warmup, "main", lambda argv: calls.append(argv) or 0)
    return calls, tmp_path / "vabar_warm" / "stamp"


def test_a_cell_is_warmed_once(stamped) -> None:
    calls, stamp = stamped
    config = cfg.resolve("fast", [])
    warmup.ensure_warm("tool_hanging/L1", config)
    warmup.ensure_warm("tool_hanging/L1", config)
    assert len(calls) == 1
    assert calls[0][:4] == ["--cell", "tool_hanging/L1", "--config", config.ref]
    assert stamp.exists()


def test_a_failed_warmup_is_tried_again(stamped, monkeypatch) -> None:
    _, stamp = stamped
    monkeypatch.setattr(warmup, "main", lambda argv: 1)
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []))
    assert not stamp.exists()


@pytest.mark.parametrize("extra", [["--renderer", "tiled"], ["--renderer=gl"]])
def test_another_renderer_is_not_warmed(stamped, extra) -> None:
    calls, _ = stamped
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []), extra)
    assert calls == []


def test_an_install_without_rtx_is_not_warmed(stamped, monkeypatch) -> None:
    calls, _ = stamped
    monkeypatch.setattr(warmup, "stamp_path", lambda cell, config: None)
    warmup.ensure_warm("tool_hanging/L1", cfg.resolve("fast", []))
    assert calls == []


@pytest.mark.parametrize(
    ("argv", "warmed"),
    [
        ([], True),
        (["--no-warmup"], False),
        (["--solve", "idle"], False),
    ],
)
def test_bench_warms_only_a_graph_run(monkeypatch, argv, warmed) -> None:
    asked: list[str] = []
    monkeypatch.setattr(warmup, "ensure_warm", lambda cell, config, extra=(): asked.append(cell))
    monkeypatch.setattr(benchmark, "plan_cell", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(benchmark, "run_plan", lambda plan, flags: 0)
    assert benchmark.main(["--cell", "tool_hanging/L1", "--config", "fast", *argv]) == 0
    assert asked == (["tool_hanging/L1"] if warmed else [])
