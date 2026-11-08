# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered Newton environments for the gear-insertion tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )


_EASY_SCENE_SPEC = Path(__file__).with_name("easy.yaml")
_EASY_PAIR_SCENE_SPEC = Path(__file__).with_name("easy_pair.yaml")
_MEDIUM_TRAIN_SCENE_SPEC = Path(__file__).with_name("medium_train.yaml")


def _configure_gear_mesh_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Materialize benchmark-only gear controls from the graph YAML."""
    from isaaclab_arena_vabar.physics import (
        install_benchmark_physics,
        load_physics_extensions,
    )
    from isaaclab_newton.physics import NewtonCfg

    physics = env_cfg.sim.physics
    if not isinstance(physics, NewtonCfg):
        raise TypeError("gear graph YAML must select the Newton backend")
    install_benchmark_physics(
        physics,
        load_physics_extensions(Path(__file__).with_name("gear_env_config.yaml")),
        collision=True,
    )
    return env_cfg


def _placement_seed(arena_env) -> int | None:
    """The run's placement seed, applied by the builder before build-time variations."""
    return arena_env.placer_params.placement_seed


def _configure_gear_mesh_placement(arena_env):
    """Configure strict source-bounded layout sampling."""
    # On owns support Z/containment, box limits bound X/Y, and reset pose ranges
    # draw the board's narrow yaw band and the gear's full clock angle.
    arena_env.placer_params.min_unique_layouts_per_env = 5
    arena_env.placer_params.allow_best_loss_fallbacks = False
    # The loose gear is supported through the thin mat.  A global 1 cm
    # inter-object clearance would incorrectly reject it against the table
    # underneath that support chain.
    arena_env.placer_params.solver_params.clearance_m = 0.0
    return arena_env


@dataclass
class GearMeshNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the registered gear-insertion environment."""

    enable_cameras: bool = False
    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False
    episode_length_s: float | None = None


@dataclass
class GearInsertionEasyNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the registered easy gear-insertion environment."""


@dataclass
class GearMeshPairNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the two-station generated Gear Mesh family."""

    layout_names: list[str] = field(default_factory=list)


@dataclass
class GearMeshTrainNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the three-station generated Gear Mesh family."""

    layout_names: list[str] = field(default_factory=list)


class GearMeshNewtonEnvironment(ArenaEnvironmentFactory[GearMeshNewtonEnvironmentCfg]):
    """Build gear insertion from its graph and task-owned Newton profile."""

    _legacy_argparse_cfg_type = GearMeshNewtonEnvironmentCfg
    scene_spec: Path
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(self, cfg: GearMeshNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena_vabar.environments.graph_environment import (
            build_graph_environment,
        )

        arena_env = build_graph_environment(self.scene_spec, cfg)
        if cfg.episode_length_s is not None:
            if cfg.episode_length_s <= 0:
                raise ValueError("episode_length_s must be positive")
            arena_env.task.episode_length_s = cfg.episode_length_s
        from isaaclab_arena_vabar.physics import append_env_cfg_callback

        append_env_cfg_callback(arena_env, partial(self.env_cfg_callback))
        return arena_env


class GearInsertionEasyNewtonEnvironment(GearMeshNewtonEnvironment):
    """Build AUTOLab gearmesh-easy-single-01."""

    name = "vabar_gear_insertion__easy"
    relative_graph_path = "gear_insertion/easy.yaml"
    _legacy_argparse_cfg_type = GearInsertionEasyNewtonEnvironmentCfg
    scene_spec = _EASY_SCENE_SPEC
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(
        self, cfg: GearInsertionEasyNewtonEnvironmentCfg
    ) -> IsaacLabArenaEnvironment:
        arena_env = super().build(cfg)
        from isaaclab_arena_vabar.tasks.gear_insertion.variation import (
            GearFamilyVariation,
        )

        board = arena_env.scene.assets["board"]
        gear = arena_env.scene.assets["gear_a"]
        board.add_variation(
            GearFamilyVariation(
                board,
                gear,
                arena_env.task,
                seed_source=partial(_placement_seed, arena_env),
            )
        )
        return _configure_gear_mesh_placement(arena_env)


class _GearMeshLayoutNewtonEnvironment(GearMeshNewtonEnvironment):
    """Build one exact-row generated Gear Mesh family."""

    family: str
    gear_names: tuple[str, ...]
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(self, cfg: GearMeshNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        arena_env = super().build(cfg)
        from isaaclab_arena_vabar.tasks.gear_insertion.parallel_layouts import (
            apply_selected_layout,
            configure_parallel_layouts,
        )
        from isaaclab_arena_vabar.tasks.gear_insertion.variation import (
            GearLayoutVariation,
            GearLayoutVariationCfg,
        )

        board = arena_env.scene.assets["board"]
        gears = tuple(arena_env.scene.assets[name] for name in self.gear_names)
        variation = None
        if not cfg.layout_names:
            variation = GearLayoutVariation(
                board,
                gears,
                arena_env.task,
                GearLayoutVariationCfg(family=self.family),
                seed_source=partial(_placement_seed, arena_env),
            )
            board.add_variation(variation)
        _configure_gear_mesh_placement(arena_env)
        # Each RSS row has a fixed start. Scalar rebuilds select a new row;
        # explicit parallel assignments keep one row per slot across resets.
        arena_env.placer_params.min_unique_layouts_per_env = 1
        # The source generator already enforces circular tip clearances. Arena's
        # generic box validators conservatively reject some valid three-gear
        # rows, so only support is required: relations solve support Z while the
        # exact recorded X/Y/yaw remain authoritative. A candidate whose gears sink
        # into the mat is thrown out of the scene on the first physics step.
        arena_env.placer_params.required_checks = {"on_relation"}

        previous_callback = arena_env.env_cfg_callback

        def configure(env_cfg):
            env_cfg = previous_callback(env_cfg)
            if cfg.layout_names:
                return configure_parallel_layouts(
                    env_cfg,
                    family=self.family,
                    gear_names=self.gear_names,
                    layout_names=cfg.layout_names,
                )
            return apply_selected_layout(
                env_cfg,
                layout_name=variation.selected_variant,
                gear_names=self.gear_names,
            )

        arena_env.env_cfg_callback = configure
        return arena_env


class GearMeshPairNewtonEnvironment(_GearMeshLayoutNewtonEnvironment):
    """Build AUTOLab's generated gearmesh-easy-pair family."""

    name = "vabar_gear_insertion__easy_pair"
    relative_graph_path = "gear_insertion/easy_pair.yaml"
    _legacy_argparse_cfg_type = GearMeshPairNewtonEnvironmentCfg
    scene_spec = _EASY_PAIR_SCENE_SPEC
    family = "pair"
    gear_names = ("gear_a", "gear_b")


class GearMeshTrainNewtonEnvironment(_GearMeshLayoutNewtonEnvironment):
    """Build AUTOLab's generated gearmesh-medium-train family."""

    name = "vabar_gear_insertion__medium_train"
    relative_graph_path = "gear_insertion/medium_train.yaml"
    _legacy_argparse_cfg_type = GearMeshTrainNewtonEnvironmentCfg
    scene_spec = _MEDIUM_TRAIN_SCENE_SPEC
    family = "train"
    gear_names = ("gear_a", "gear_b", "gear_c")
