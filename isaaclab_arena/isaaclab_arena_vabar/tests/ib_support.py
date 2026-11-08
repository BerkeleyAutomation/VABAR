"""Shared paths and loaders for the industrial benchmark tests.

Fixtures live in ``conftest.py``; plain helpers live here so test modules can
use them at import time (for example in ``pytest.mark.parametrize``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from isaaclab_arena_gap.task_sources import autolab_root

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar"
ENVIRONMENTS_ROOT = PACKAGE_ROOT / "environments"
ASSETS_ROOT = PACKAGE_ROOT / "assets"
TOOLS_ROOT = REPO_ROOT / "tools"
AUTOLAB_ROOT = autolab_root()

# Benchmark scene YAMLs, excluding the shared ``*_env_config.yaml`` physics files.
ENVIRONMENT_YAMLS = tuple(
    path.relative_to(ENVIRONMENTS_ROOT).as_posix()
    for path in sorted(ENVIRONMENTS_ROOT.glob("*/*.yaml"))
    if not path.name.endswith("_env_config.yaml")
)


def load_scene(relative_yaml: str | Path) -> dict[str, Any]:
    """Load one environment YAML relative to the package ``environments`` dir."""
    return yaml.safe_load((ENVIRONMENTS_ROOT / relative_yaml).read_text())


def scene_params(scene: dict[str, Any]) -> dict[str, Any]:
    """Return the params of a scene's single task subtask."""
    return scene["task"]["subtasks"][0]["params"]


def relations_by(scene: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index a scene's relations by ``(subject, kind)``."""
    return {
        (relation.get("subject"), relation["kind"]): relation
        for relation in scene["relations"]
    }


def load_tool(name: str) -> ModuleType:
    """Import one repository ``tools/<name>.py`` script as a module."""
    spec = importlib.util.spec_from_file_location(name, TOOLS_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_registered_environment(name: str, **cfg_kwargs):
    """Build one registered benchmark environment config (no simulator).

    Callers must stub ``isaaclab_arena.scene.scene`` (``stub_scene_module``).
    """
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    register_components()
    registry = EnvironmentRegistry()
    factory = registry.get_component_by_name(name)
    cfg = registry.get_environment_cfg_type(factory)(**cfg_kwargs)
    return factory().build(cfg)
