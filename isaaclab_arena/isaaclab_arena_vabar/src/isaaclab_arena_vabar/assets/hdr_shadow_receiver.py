"""Invisible floor anchor used with HDR dome scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose

RECEIVER_VISUAL_CLEARANCE_M = 0.0005
"""Offset above the collision ground that prevents coplanar z-fighting."""

HDR_SHADOW_RECEIVER_USD_PATH = str(
    Path(__file__).resolve().parent
    / "vabar__hdr_shadow_receiver"
    / "vabar__hdr_shadow_receiver.usda"
)


def hdr_shadow_receiver_cfg(*, prim_path: str, ground_z: float = 0.0) -> AssetBaseCfg:
    """Build the invisible, collision-free floor anchor."""

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, ground_z + RECEIVER_VISUAL_CLEARANCE_M)
        ),
        spawn=UsdFileCfg(
            usd_path=HDR_SHADOW_RECEIVER_USD_PATH,
            copy_from_source=False,
            visible=False,
        ),
        collision_group=-1,
    )


class IndustrialHdrShadowReceiver(Object):
    """Package-local HDR floor anchor for benchmark graph scenes."""

    name = "vabar__hdr_shadow_receiver"
    tags: ClassVar[list[str]] = ["floor", "visual"]

    def __init__(
        self,
        instance_name: str = "hdr_shadow_receiver",
        ground_z: float = 0.0,
        **kwargs: Any,
    ) -> None:
        initial_pose = kwargs.pop("initial_pose", None)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", {}) or {})
        spawn_cfg_addon["visible"] = False
        if initial_pose is None:
            initial_pose = Pose(
                position_xyz=(0.0, 0.0, ground_z + RECEIVER_VISUAL_CLEARANCE_M)
            )
        super().__init__(
            name=instance_name,
            usd_path=HDR_SHADOW_RECEIVER_USD_PATH,
            object_type=ObjectType.BASE,
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
