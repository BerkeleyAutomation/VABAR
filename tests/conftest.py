# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: run a test under a config without leaking it into the next one."""

from __future__ import annotations

import pytest

from vabar import config as cfg


@pytest.fixture
def config_sandbox(monkeypatch):
    """Undo whatever a test activates: the exported variables, the active config, the rewritten suites."""
    for name in (cfg.ENV_CONFIG, cfg.ENV_SET):
        # setenv first so the undo restores the original (or its absence)
        # even after the code under test writes the variable itself.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    yield
    if cfg._APPLIED:
        cfg.activate(cfg.load("paper"), export=False)
    cfg._ACTIVE = None
