"""Register AUTOLab Gear Mesh assets."""

from isaaclab_arena.assets.registries import AssetRegistry

from isaaclab_arena_vabar.assets.gear_insertion import (
    make_gear_mesh_16t,
    make_gear_mesh_20t,
    make_gear_mesh_24t,
    make_gear_mesh_board_16,
    make_gear_mesh_board_20,
    make_gear_mesh_board_24,
    make_gear_mesh_mat,
)
from isaaclab_arena_vabar.registry import registered_component

ASSET_ENTRY_POINTS = {
    "vabar__gear_mesh_16t": make_gear_mesh_16t,
    "vabar__gear_mesh_20t": make_gear_mesh_20t,
    "vabar__gear_mesh_24t": make_gear_mesh_24t,
    "vabar__gear_mesh_board_16": make_gear_mesh_board_16,
    "vabar__gear_mesh_board_20": make_gear_mesh_board_20,
    "vabar__gear_mesh_board_24": make_gear_mesh_board_24,
    "vabar__gear_mesh_mat": make_gear_mesh_mat,
}


def register_assets() -> None:
    registry = AssetRegistry()
    for name, entry_point in ASSET_ENTRY_POINTS.items():
        existing = registered_component(registry, name)
        if existing is not None:
            if existing is not entry_point:
                raise RuntimeError(f"conflicting asset registration for {name!r}")
            continue
        registry.register(entry_point, name)
