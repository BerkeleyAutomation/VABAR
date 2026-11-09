"""Package-owned task-frame adaptations shared by USB-C environments."""

from pathlib import Path
from typing import Any

from isaaclab_arena.assets.registries import AssetRegistry

from isaaclab_arena_vabar.assets.registry import (
    ConvertedAssetSpec,
    register_converted_assets,
)
from isaaclab_arena_vabar.registry import registered_component

_ASSET_ROOT = Path(__file__).resolve().parents[1]

ASSETS = (
    ConvertedAssetSpec(
        registry_name="vabar__usbc_medium_plug",
        role="object",
        relative_usd_path="vabar__usbc_easy_plug/medium_plug.usda",
        object_min_z=None,
    ),
)


def usbc_easy_port(instance_name: str = "port", **kwargs: Any):
    """AUTOLab's x2 chamfered receptacle, fixed to the work surface."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_newton.sim.schemas import NewtonRigidBodyPropertiesCfg

    initial_pose = kwargs.pop("initial_pose", None)
    if isinstance(initial_pose, dict):
        initial_pose = Pose(
            position_xyz=tuple(initial_pose["position_xyz"]),
            rotation_xyzw=tuple(initial_pose["rotation_xyzw"]),
        )
    return Object(
        name=instance_name,
        object_type=ObjectType.RIGID,
        usd_path=str(
            _ASSET_ROOT
            / "vabar__usbc_easy_port/vabar__usbc_easy_port.usda"
        ),
        initial_pose=initial_pose,
        collision_mode="mesh",
        spawn_cfg_addon={
            "rigid_props": NewtonRigidBodyPropertiesCfg(kinematic_enabled=True),
        },
        **kwargs,
    )


usbc_easy_port.name = "vabar__usbc_easy_port"
usbc_easy_port.tags = ()


def usbc_easy_plug(instance_name: str = "plug", **kwargs: Any):
    """AUTOLab's exact scaled plug mesh and source contact material."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    return Object(
        name=instance_name,
        object_type=ObjectType.RIGID,
        usd_path=str(
            _ASSET_ROOT
            / "vabar__usbc_easy_plug/vabar__usbc_easy_plug.usda"
        ),
        collision_mode="mesh",
        **kwargs,
    )


usbc_easy_plug.name = "vabar__usbc_easy_plug"
usbc_easy_plug.tags = ()


def register_assets() -> None:
    register_converted_assets(ASSETS, module_file=__file__)
    registry = AssetRegistry()
    for entry in (usbc_easy_port, usbc_easy_plug):
        existing = registered_component(registry, entry.name)
        if existing is None:
            registry.register(entry, entry.name)
        elif existing is not entry:
            raise RuntimeError(f"conflicting {entry.name} registration")
