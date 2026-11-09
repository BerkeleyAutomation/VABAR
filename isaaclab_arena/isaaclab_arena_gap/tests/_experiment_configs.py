"""Shared loader for the single-run Arena experiment configs."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "experiment_configs"


def load_run(path: Path | str) -> tuple[str, dict]:
    """Return ``(name, run)`` for the one run in an experiment config."""
    path = CONFIG_ROOT / path
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(document) == {"runs"}, path
    assert isinstance(document["runs"], dict) and len(document["runs"]) == 1, path
    return next(iter(document["runs"].items()))


def policy_config(path: Path | str) -> dict:
    """Return the run's policy keyword arguments without its ``type``."""
    policy = dict(load_run(path)[1]["policy"])
    policy.pop("type")
    return policy
