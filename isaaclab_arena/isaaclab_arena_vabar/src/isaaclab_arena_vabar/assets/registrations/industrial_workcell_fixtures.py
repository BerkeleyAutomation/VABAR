"""Register the canonical workcell tables and hand-authored fixed fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.utils.pose import Pose

from isaaclab_arena_vabar.assets.registry import (
    ConvertedAssetSpec,
    register_converted_assets,
)
from isaaclab_arena_vabar.registry import registered_component

_ASSET_ROOT = Path(__file__).resolve().parents[1]

_TABLES = (
    ConvertedAssetSpec(
        registry_name="vabar__fr3_workcell_table",
        role="background",
        relative_usd_path=(
            "vabar__fr3_workcell_table/vabar__fr3_workcell_table.usda"
        ),
        object_min_z=0.0,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__yam_workcell_table",
        role="background",
        relative_usd_path=(
            "vabar__yam_workcell_table/vabar__yam_workcell_table.usda"
        ),
        object_min_z=0.0,
    ),
)

_FIXTURE_PATHS = {
    "vabar__screwdriver_box": "vabar__screwdriver_box/box.usda",
    "vabar__pegboard_rack": "vabar__pegboard_rack/vabar__pegboard_rack.usda",
    "vabar__wrench_hook_1": "vabar__wrench_hooks/wrench_hook_1.usda",
    "vabar__wrench_hook_2": "vabar__wrench_hooks/wrench_hook_2.usda",
    "vabar__scissor_hook_1": "vabar__scissor_hooks/scissor_hook_1.usda",
    "vabar__scissor_hook_2": "vabar__scissor_hooks/scissor_hook_2.usda",
    "vabar__plier_support_left": "vabar__plier_supports/plier_support_left.usda",
    "vabar__plier_support_right": "vabar__plier_supports/plier_support_right.usda",
    "vabar__screwdriver_rack": "vabar__screwdriver_rack/rack.usda",
    "vabar__hammer_holder_1": "vabar__hammer_holders/hammer_holder_1.usda",
    "vabar__hammer_holder_2": "vabar__hammer_holders/hammer_holder_2.usda",
    "vabar__yam_usbc_bench": "vabar__yam_usbc_fixtures/bench.usda",
    "vabar__yam_usbc_cradle_front": "vabar__yam_usbc_fixtures/cradle_front.usda",
    "vabar__yam_usbc_cradle_rear": "vabar__yam_usbc_fixtures/cradle_rear.usda",
}

_BIN_APPEARANCES = frozenset(
    {"default", "bench", "electrical", "new", "power", "wiring", "syringe"}
)


def _normalize_pose(
    initial_pose: Pose | Mapping[str, Sequence[float]] | None,
) -> Pose | None:
    if initial_pose is None or isinstance(initial_pose, Pose):
        return initial_pose
    return Pose(
        position_xyz=tuple(float(value) for value in initial_pose["position_xyz"]),
        rotation_xyzw=tuple(float(value) for value in initial_pose["rotation_xyzw"]),
    )


class _WorkcellFixture(Object):
    """Benchmark fixtures may be mounted at a small pitch as well as yaw."""

    def get_world_bounding_box(self):
        initial_pose = self.get_initial_pose()
        if self.is_anchor and isinstance(initial_pose, Pose):
            return (
                self.get_bounding_box()
                .enclosing_after_rotation(initial_pose.rotation_xyzw)
                .translated(initial_pose.position_xyz)
            )
        return super().get_world_bounding_box()


def _kinematic_fixture(
    registry_name: str,
    relative_path: str,
    *,
    instance_name: str | None = None,
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    collision_mode: str | None = "mesh",
):
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_newton.sim.schemas import NewtonRigidBodyPropertiesCfg

    return _WorkcellFixture(
        name=instance_name or registry_name,
        object_type=ObjectType.RIGID,
        usd_path=str(_ASSET_ROOT / relative_path),
        initial_pose=_normalize_pose(initial_pose),
        collision_mode=collision_mode,
        spawn_cfg_addon={
            "rigid_props": NewtonRigidBodyPropertiesCfg(kinematic_enabled=True),
        },
    )


def tool_sort_bin(
    instance_name: str = "tool_sort_bin",
    side: str = "destination",
    appearance: str = "default",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
):
    """Return one kinematic sorting bin with its task-selected appearance."""

    if side not in {"source", "destination"}:
        raise ValueError(f"invalid tool-sort bin side: {side!r}")
    if appearance not in _BIN_APPEARANCES:
        raise ValueError(f"invalid tool-sort bin appearance: {appearance!r}")
    leaf = "bin1.usda" if side == "source" else f"bin2_{appearance}.usda"
    return _kinematic_fixture(
        "vabar__tool_sort_bin",
        f"vabar__tool_sort_bin/{leaf}",
        instance_name=instance_name,
        initial_pose=initial_pose,
        collision_mode="mesh",
    )


def syringe_sort_bin(
    instance_name: str = "sharps_container",
    side: str = "destination",
    appearance: str = "syringe",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
):
    """Return the benchmark-owned syringe sharps container."""

    if side != "destination":
        raise ValueError(f"invalid syringe-sort bin side: {side!r}")
    if appearance != "syringe":
        raise ValueError(f"invalid syringe-sort bin appearance: {appearance!r}")
    return _kinematic_fixture(
        "vabar__syringe_sort_bin",
        "vabar__tool_sort_bin/bin2_syringe.usda",
        instance_name=instance_name,
        initial_pose=initial_pose,
        collision_mode="mesh",
    )


def _fixture_entry_point(registry_name: str, relative_path: str):
    def entry_point(
        instance_name: str | None = None,
        initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
        collision_mode: str | None = "mesh",
        geometry_profile: str = "default",
        **_ignored: Any,
    ):
        asset_path = relative_path
        if geometry_profile != "default":
            if (
                registry_name != "vabar__screwdriver_box"
                or geometry_profile != "upstream"
            ):
                raise ValueError(
                    f"Unsupported {registry_name} geometry profile: {geometry_profile!r}"
                )
            asset_path = "vabar__screwdriver_box/box_upstream.usda"
        return _kinematic_fixture(
            registry_name,
            asset_path,
            instance_name=instance_name,
            initial_pose=initial_pose,
            collision_mode=collision_mode,
        )

    entry_point.__name__ = registry_name.removeprefix("vabar__")
    return entry_point


_ASSET_ENTRY_POINTS = {
    "vabar__tool_sort_bin": tool_sort_bin,
    "vabar__syringe_sort_bin": syringe_sort_bin,
    **{
        registry_name: _fixture_entry_point(registry_name, relative_path)
        for registry_name, relative_path in _FIXTURE_PATHS.items()
    },
}

for _registry_name, _entry_point in _ASSET_ENTRY_POINTS.items():
    _entry_point.name = _registry_name
    _entry_point.tags = ()
del _registry_name, _entry_point


def register_assets() -> None:
    """Register the two table backgrounds and all kinematic fixture factories."""

    register_converted_assets(_TABLES, module_file=__file__)
    registry = AssetRegistry()
    for name, entry_point in _ASSET_ENTRY_POINTS.items():
        existing = registered_component(registry, name)
        if existing is not None:
            if existing is not entry_point:
                raise RuntimeError(
                    f"conflicting workcell asset registration for {name!r}"
                )
            continue
        registry.register(entry_point, name)
