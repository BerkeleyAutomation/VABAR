"""Explicit, idempotent registration for industrial benchmark embodiments."""

from __future__ import annotations

from isaaclab_arena_vabar.registry import registered_component
from isaaclab_arena.assets.registries import AssetRegistry

from .bimanual_yam import IndustrialBimanualYamEmbodiment
from .fr3_robotiq_2f85 import IndustrialFr3Robotiq2f85Embodiment


def register_embodiments() -> None:
    """Register industrial benchmark embodiments, rejecting type collisions."""
    registry = AssetRegistry()
    for embodiment in (
        IndustrialFr3Robotiq2f85Embodiment,
        IndustrialBimanualYamEmbodiment,
    ):
        name = embodiment.name
        existing = registered_component(registry, name)
        if existing is not None:
            if existing is not embodiment:
                raise RuntimeError(
                    f"conflicting embodiment registration for {name!r}: {existing!r}"
                )
            continue
        registry.register(embodiment, name)
