"""Explicit, idempotent registration for benchmark environment factories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import make_dataclass
from importlib import import_module

from isaaclab_arena.assets.registries import EnvironmentRegistry

from isaaclab_arena_vabar.registry import registered_component

# One registration module per family, each exposing ENVIRONMENTS. Families that are
# parameterizations of one environment class list descriptors; families that need
# their own build logic list (factory, cfg type) pairs.
_FAMILY_REGISTRATIONS = (
    "isaaclab_arena_vabar.environments.tool_sorting.registration",
    "isaaclab_arena_vabar.environments.tool_hanging.registration",
    "isaaclab_arena_vabar.environments.syringe_disposal.registration",
    "isaaclab_arena_vabar.environments.usbc_insertion.registration",
    "isaaclab_arena_vabar.environments.gear_insertion.registration",
    "isaaclab_arena_vabar.environments.cable_routing.registration",
)


def register_generated_environments(
    descriptors: Iterable[object],
) -> None:
    """Register typed generated descriptors idempotently and reject collisions."""
    from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
        UsbcInsertNewtonEnvironment,
    )
    from isaaclab_arena_vabar.environments.tool_hanging.environment import (
        ToolHangingNewtonEnvironment,
    )
    from isaaclab_arena_vabar.environments.tool_sorting.environment import (
        ToolSortNewtonEnvironment,
    )

    families = (
        ToolHangingNewtonEnvironment,
        ToolSortNewtonEnvironment,
        UsbcInsertNewtonEnvironment,
    )
    registry = EnvironmentRegistry()
    for descriptor in descriptors:
        family = next(
            (f for f in families if isinstance(descriptor, f.descriptor_type)), None
        )
        if family is None:
            raise TypeError("generated environment descriptor has an invalid type")
        factory = family.for_descriptor(descriptor)
        existing = registered_component(registry, descriptor.environment_name)
        if existing is not None:
            if getattr(existing, "descriptor", None) != descriptor:
                raise RuntimeError(
                    "conflicting environment registration for "
                    f"{descriptor.environment_name!r}"
                )
            continue
        cfg_type = make_dataclass(
            f"{factory.__name__}Cfg", (), bases=(family.cfg_type,)
        )
        factory._legacy_argparse_cfg_type = cfg_type
        registry.register_environment(factory, cfg_type)


def register_environments() -> None:
    """Register parameterized benchmark environments, rejecting collisions."""

    registry = EnvironmentRegistry()
    for module_name in _FAMILY_REGISTRATIONS:
        entries = import_module(module_name).ENVIRONMENTS
        descriptors = [entry for entry in entries if not isinstance(entry, tuple)]
        for factory, cfg_type in (entry for entry in entries if isinstance(entry, tuple)):
            existing = registered_component(registry, factory.name)
            if existing is not None:
                if existing is not factory:
                    raise RuntimeError(
                        "conflicting environment registration for "
                        f"{factory.name!r}: {existing!r}"
                    )
                continue
            registry.register_environment(factory, cfg_type)
        register_generated_environments(descriptors)
