"""Register the package-local HDR floor anchor."""

from isaaclab_arena.assets.registries import AssetRegistry

from isaaclab_arena_vabar.assets.hdr_shadow_receiver import IndustrialHdrShadowReceiver
from isaaclab_arena_vabar.registry import registered_component


def register_assets() -> None:
    """Register the floor anchor idempotently and reject conflicting definitions."""

    registry = AssetRegistry()
    existing = registered_component(registry, IndustrialHdrShadowReceiver.name)
    if existing is not None:
        if existing is not IndustrialHdrShadowReceiver:
            raise RuntimeError(
                f"conflicting asset registration for {IndustrialHdrShadowReceiver.name!r}"
            )
        return
    registry.register(IndustrialHdrShadowReceiver, IndustrialHdrShadowReceiver.name)
