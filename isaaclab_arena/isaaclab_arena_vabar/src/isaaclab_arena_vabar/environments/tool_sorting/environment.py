"""Shared Newton runtime wrapper for generated current tool-sort graphs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

from isaaclab_arena_vabar.environments.graph_environment import (
    DescriptorEnvironmentFactory,
    build_graph_environment,
)
from isaaclab_arena_vabar.environments.syringe_disposal.tray_layouts import (
    LAYOUT_FILES,
    apply_tray_layouts,
)
from isaaclab_arena_vabar.physics import append_env_cfg_callback

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )


INSTRUMENT_TRAY_RIM_TO_FLOOR_M = 0.009
"""How far ``vabar__instrument_tray``'s cavity floor sits below its rim [m].

The tray is 12 mm tall with a 3 mm floor; see its ``CollisionProxy``.
"""


def seat_in_tray_cavity(arena_env: IsaacLabArenaEnvironment) -> None:
    """Place everything ``on`` the instrument tray on its cavity floor rather than its rim.

    ``on`` stands an object on its support's bounding box, whose top is the tray's
    rim. Each occupant's placement box is raised by the rim-to-floor depth instead,
    so the solve stands it that much lower. Raising the occupant rather than
    lowering the tray's top keeps the occupant's box where the solver's table
    clearance has always seen it. This affects placement only, not USD or sim
    contacts.
    """
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    tray = arena_env.scene.assets["instrument_tray"]
    for asset in arena_env.scene.assets.values():
        if not any(isinstance(rel, On) and rel.parent is tray for rel in getattr(asset, "relations", ())):
            continue
        box = asset.get_bounding_box()
        lift = box.min_point.new_tensor((0.0, 0.0, INSTRUMENT_TRAY_RIM_TO_FLOOR_M))
        asset.bounding_box = AxisAlignedBoundingBox(box.min_point + lift, box.max_point + lift)


@dataclass(frozen=True)
class ToolSortEnvironmentDescriptor:
    environment_name: str
    relative_graph_path: str
    random_yaw_init: bool = False
    inset_source_bin: bool = False
    upstream_pad_friction: bool = False
    autolab_tray_layouts: str | None = None
    """AUTOLab trial file the syringes are laid out from (see ``syringe_disposal/tray_layouts.py``)."""

    def __post_init__(self) -> None:
        path = Path(self.relative_graph_path)
        if (
            re.fullmatch(r"[a-z0-9_]+", self.environment_name) is None
            or path.is_absolute()
            or path.parent not in (Path("tool_sorting"), Path("syringe_disposal"))
            or path.suffix != ".yaml"
            or not isinstance(self.random_yaw_init, bool)
            or not isinstance(self.inset_source_bin, bool)
            or not isinstance(self.upstream_pad_friction, bool)
            or self.autolab_tray_layouts not in (None, *LAYOUT_FILES)
        ):
            raise ValueError("tool-sort environment descriptor is invalid")


def finalize_tool_sort_env_cfg(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    replicate_physics: bool | None = None,
    physics_yaml: Path | None = None,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply benchmark-only physics fields and runtime scene options."""
    from isaaclab_arena_vabar.camera_contract import configure_camera_contract
    from isaaclab_arena_vabar.physics import (
        install_benchmark_physics,
        load_physics_extensions,
    )
    from isaaclab_newton.physics import NewtonCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise TypeError("tool-sort graph YAML must select the Newton backend")
    path = physics_yaml or Path(__file__).with_name("tool_sorting_env_config.yaml")
    install_benchmark_physics(env_cfg.sim.physics, load_physics_extensions(path))
    configure_camera_contract(env_cfg)
    if replicate_physics is not None and not isinstance(replicate_physics, bool):
        raise TypeError("replicate_physics must be a boolean or null")
    if replicate_physics is not None:
        env_cfg.scene.replicate_physics = replicate_physics
    return env_cfg


@dataclass
class ToolSortNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Runtime options for generated current tool-sort environments."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False
    replicate_physics: bool | None = None


class ToolSortNewtonEnvironment(
    DescriptorEnvironmentFactory[ToolSortNewtonEnvironmentCfg]
):
    descriptor_type = ToolSortEnvironmentDescriptor
    cfg_type = ToolSortNewtonEnvironmentCfg
    family = "tool-sort"
    generated_class_prefix = "GeneratedToolSortEnvironment_"
    descriptor: ToolSortEnvironmentDescriptor | None = None

    def build(self, cfg: ToolSortNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        graph_path = self.graph_path()
        descriptor = self.descriptor
        if not isinstance(cfg.use_tiled_cameras, bool):
            raise TypeError("use_tiled_cameras must be a boolean")
        if not isinstance(cfg.use_instanceable_meshes, bool):
            raise TypeError("use_instanceable_meshes must be a boolean")
        if cfg.replicate_physics is not None and not isinstance(
            cfg.replicate_physics, bool
        ):
            raise TypeError("replicate_physics must be a boolean or null")
        arena_env = build_graph_environment(graph_path, cfg)
        arena_env.placer_params.random_yaw_init = descriptor.random_yaw_init
        if descriptor.upstream_pad_friction:
            # AUTOLab's tool_sort pad friction, authored over the selected FR3 variant.
            spawn = arena_env.embodiment.scene_config.robot.spawn
            suffix = "_instanceable" if cfg.use_instanceable_meshes else ""
            spawn.usd_path = str(Path(spawn.usd_path).with_name(f"tool_sorting{suffix}.usda"))
        if "instrument_tray" in arena_env.scene.assets:
            arena_env.placer_params.allow_best_loss_fallbacks = False
            seat_in_tray_cavity(arena_env)
        if descriptor.inset_source_bin:
            # Never admit a fallback layout that violates the bin-edge margin.
            arena_env.placer_params.allow_best_loss_fallbacks = False
            from isaaclab_arena.relations.collision_mode import CollisionMode
            from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

            # Place on the existing bin's interior floor, preserving its underside
            # for on(table). This affects placement only, not USD or sim contacts.
            source_bin = arena_env.scene.assets["bin1"]
            source_bin.bounding_box = AxisAlignedBoundingBox(
                (-0.18, -0.23, -0.02), (0.18, 0.23, 0.02)
            )
            source_bin.collision_mode = CollisionMode.BBOX

        if descriptor.autolab_tray_layouts is not None:
            syringes = tuple(
                sorted(name for name in arena_env.scene.assets if name.startswith("syringe_"))
            )
            append_env_cfg_callback(
                arena_env,
                partial(
                    apply_tray_layouts,
                    file_name=descriptor.autolab_tray_layouts,
                    syringe_names=syringes,
                    placer_params=arena_env.placer_params,
                ),
            )
        append_env_cfg_callback(
            arena_env,
            partial(
                finalize_tool_sort_env_cfg,
                replicate_physics=cfg.replicate_physics,
                physics_yaml=graph_path,
            ),
        )
        return arena_env
