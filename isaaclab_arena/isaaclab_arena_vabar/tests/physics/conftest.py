"""Shared fixtures for the asset drop harness tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def arena_manager(monkeypatch):
    """Stub Arena's Kit-only preset module and return its manager subclass."""
    pytest.importorskip("isaaclab_arena")
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

    # Arena's preset module needs Kit, so mirror its preset: a Newton solver
    # whose class_type is Arena's own manager subclass.
    preset_path = (
        Path(__file__).resolve().parents[3]
        / "external/IsaacLab-Arena/isaaclab_arena/environments"
        / "isaaclab_arena_manager_based_env_cfg.py"
    )
    assert "class_type=NewtonArenaMJWarpManager" in preset_path.read_text()

    class NewtonArenaMJWarpManager(NewtonMJWarpManager):
        pass

    preset_module = types.ModuleType("arena_env_cfg_stub")
    preset_module.NewtonArenaMJWarpManager = NewtonArenaMJWarpManager
    preset_module.ArenaPhysicsCfg = type(
        "ArenaPhysicsCfg",
        (),
        {
            "newton": NewtonCfg(
                solver_cfg=MJWarpSolverCfg(class_type=NewtonArenaMJWarpManager)
            )
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg",
        preset_module,
    )
    return NewtonArenaMJWarpManager
