"""Register the benchmark's shared empty-warehouse lighting preset."""

from isaaclab_arena.assets.object_library import DomeLight
from isaaclab_arena.assets.registries import AssetRegistry

from isaaclab_arena_vabar.assets.empty_warehouse_hdr import (
    empty_warehouse_dome_light_cfg,
)
from isaaclab_arena_vabar.registry import registered_component


class IndustrialEmptyWarehouseDomeLight(DomeLight):
    name = "vabar__empty_warehouse_dome_light"

    def __init__(self, **kwargs):
        super().__init__(spawner_cfg=empty_warehouse_dome_light_cfg(), **kwargs)


def register_assets() -> None:
    registry = AssetRegistry()
    name = IndustrialEmptyWarehouseDomeLight.name
    existing = registered_component(registry, name)
    if existing is not None:
        if existing is not IndustrialEmptyWarehouseDomeLight:
            raise RuntimeError(f"conflicting asset registration for {name!r}")
        return
    registry.register(IndustrialEmptyWarehouseDomeLight, name)
