# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Package-local assets for the AUTOLab Gear Mesh benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

_ASSET_CATALOG_ROOT = Path(__file__).resolve().parent
GEAR_MESH_ASSET_PATHS = {
    name: _ASSET_CATALOG_ROOT
    / f"vabar__gear_mesh_{name}"
    / f"vabar__gear_mesh_{name}.usda"
    for name in ("16t", "20t", "24t", "board", "mat")
}


class GearInsertionRigidObject(Object):
    """Spawn a committed gear USD without enabling unused contact sensors."""

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=False),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)


class GearMeshBoardObject(Object):
    """Spawn the board articulation and retain its USD-authored joint drives."""

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        # Imported here: the Newton backend loads only after the app starts.
        from isaaclab_arena_vabar.assets.fixed_base_articulation import (
            PlacedFixedBaseArticulation,
        )

        return self._add_initial_pose_to_cfg(
            ArticulationCfg(
                class_type=PlacedFixedBaseArticulation,
                prim_path=self.prim_path,
                spawn=self._get_spawn_cfg(activate_contact_sensors=False),
                actuators={
                    "pinion_drive": ImplicitActuatorCfg(
                        joint_names_expr=["pinion_joint"],
                        stiffness=0.0,
                        damping=4.0,
                        effort_limit_sim=1.5,
                        armature=3.0e-4,
                    ),
                    "button_spring": ImplicitActuatorCfg(
                        joint_names_expr=["button_joint"],
                        stiffness=400.0,
                        damping=4.0,
                        armature=1.0e-3,
                    ),
                },
                **self.asset_cfg_addon,
            )
        )


def _normalize_initial_pose(
    initial_pose: Pose | Mapping[str, Sequence[float]] | None,
) -> Pose | None:
    if initial_pose is None or isinstance(initial_pose, Pose):
        return initial_pose
    return Pose(
        position_xyz=tuple(float(value) for value in initial_pose["position_xyz"]),
        rotation_xyzw=tuple(float(value) for value in initial_pose["rotation_xyzw"]),
    )


def _make_gear_mesh_gear(
    teeth: int,
    instance_name: str,
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
) -> Object:
    gear = GearInsertionRigidObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.RIGID,
        usd_path=str(GEAR_MESH_ASSET_PATHS[f"{teeth}t"]),
        initial_pose=_normalize_initial_pose(initial_pose),
    )
    gear.disable_reset_pose()
    return gear


def _make_gear_mesh_board(
    teeth: int,
    instance_name: str,
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
) -> Object:
    board = GearMeshBoardObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.ARTICULATION,
        usd_path=str(GEAR_MESH_ASSET_PATHS["board"]),
        initial_pose=_normalize_initial_pose(initial_pose),
        spawn_cfg_addon={"variants": {"layout": f"board_{teeth}"}},
    )
    board.disable_reset_pose()
    return board


def make_gear_mesh_16t(
    instance_name: str = "gear_a", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_gear(16, instance_name, initial_pose)


def make_gear_mesh_20t(
    instance_name: str = "gear_a", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_gear(20, instance_name, initial_pose)


def make_gear_mesh_24t(
    instance_name: str = "gear_a", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_gear(24, instance_name, initial_pose)


def make_gear_mesh_board_16(
    instance_name: str = "board", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_board(16, instance_name, initial_pose)


def make_gear_mesh_board_20(
    instance_name: str = "board", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_board(20, instance_name, initial_pose)


def make_gear_mesh_board_24(
    instance_name: str = "board", initial_pose=None, **_ignored: Any
) -> Object:
    return _make_gear_mesh_board(24, instance_name, initial_pose)


def make_gear_mesh_mat(
    instance_name: str = "gear_mat",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    mat = GearInsertionRigidObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.RIGID,
        usd_path=str(GEAR_MESH_ASSET_PATHS["mat"]),
        initial_pose=_normalize_initial_pose(initial_pose),
    )
    mat.disable_reset_pose()
    return mat
