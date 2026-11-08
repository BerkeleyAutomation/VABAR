"""Deferred registration for policy-independent benchmark components."""

from __future__ import annotations

from importlib import import_module

_registered = False


def register_components() -> None:
    """Register benchmark components once Arena is available."""

    global _registered
    if _registered:
        return

    assets = import_module("isaaclab_arena_vabar.assets.registry")
    assets.register_assets()
    embodiments = import_module("isaaclab_arena_vabar.embodiments.registry")
    embodiments.register_embodiments()
    tasks = import_module("isaaclab_arena_vabar.tasks.registry")
    tasks.register_tasks()
    environments = import_module("isaaclab_arena_vabar.environments.registry")
    environments.register_environments()
    # Part of the benchmark, not of one policy's adapter: without it neighbouring
    # placement seeds solve the same layout whichever entry point builds the scene.
    import_module("isaaclab_arena_vabar.placement_seed_compat").install_placement_seed_compat()
    _registered = True
