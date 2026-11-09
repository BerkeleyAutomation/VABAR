"""Process-wide defaults, shared fixtures and dependency skips for industrial tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

import pytest


# Newton 1.5.2's USD parser can crash while USD resolves this asset's heavily
# instanced relationship graph in parallel. This test-only setting matches the
# existing asset verification contract and must be set before importing pxr.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")


# Simulator packages that test bodies import lazily. A bare interpreter (for
# example the GaP venv running ``pytest`` from isaaclab_arena_vabar/) lacks
# them, so those tests skip there instead of failing; the root venv runs them.
_SIMULATION_PACKAGES = frozenset(
    {
        "isaaclab",
        "isaaclab_arena",
        "isaaclab_newton",
        "isaaclab_physx",
        "isaacsim",
        "newton",
        "omni",
        "pxr",
        "scipy",
        "torch",
        "warp",
    }
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except ModuleNotFoundError as exc:
        package = (exc.name or "").partition(".")[0]
        if (
            package in _SIMULATION_PACKAGES
            and importlib.util.find_spec(package) is None
        ):
            pytest.skip(f"simulation dependency {package!r} is not installed")
        raise


@pytest.fixture
def stub_scene_module(monkeypatch):
    """Replace Arena's Scene container, whose runtime import requires Kit.

    Only the container is substituted; task and embodiment configs stay real.
    """
    module = ModuleType("isaaclab_arena.scene.scene")

    class Scene:
        def __init__(self, assets):
            self.assets = {asset.name: asset for asset in assets}

    module.Scene = Scene
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module
