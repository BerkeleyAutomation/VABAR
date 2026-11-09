# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-environment Newton/VBD build of the bimanual YAM cable-routing task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import warp as wp
import yaml
from isaaclab_arena_vabar.environments.cable_routing.distribution import (
    anchor_pose_from_cable,
    sample_cable_routing_layout,
)
from isaaclab_arena_vabar.tasks.cable_routing.terminations import TerminatedCableGoal
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

if TYPE_CHECKING:
    import torch
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )


_ASSET_DIRECTORY = Path(__file__).resolve().parent / "cable_routing_assets"
_ASSET_CATALOG = Path(__file__).resolve().parents[2] / "assets"
_SCENE_ROOT = Path(__file__).resolve().parents[1] / "cable_routing"
_MEDIUM_SCENE_SPEC = _SCENE_ROOT / "medium.yaml"
_EASY_SCENE_SPEC = _SCENE_ROOT / "easy.yaml"
_TABLE_USD_PATH = (
    _ASSET_CATALOG
    / "vabar__yam_workcell_table"
    / "vabar__yam_workcell_table.usda"
)
_ROUND_PEG_USD_PATH = (
    _ASSET_CATALOG / "vabar__cable_routing_peg" / "round_peg.usdc"
)
_EASY_ROUND_PEG_USD_PATH = _ROUND_PEG_USD_PATH.with_name("orange_round_peg.usda")
_TERMINAL_ASSET_DIRECTORY = _ASSET_CATALOG / "vabar__cable_routing_terminals"
_ANCHOR_USD_PATH = _TERMINAL_ASSET_DIRECTORY / "anchor.usda"
_PORT_USD_PATH = _TERMINAL_ASSET_DIRECTORY / "port.usda"

_SCENE_LAYOUT = yaml.safe_load(_MEDIUM_SCENE_SPEC.read_text())
_EASY_SCENE_LAYOUT = yaml.safe_load(_EASY_SCENE_SPEC.read_text())
_PHYSICS_SETTINGS = yaml.safe_load(Path(__file__).with_name("cable_env_config.yaml").read_text())
_EMBODIMENT_MIDPOINT = tuple(
    float(value) for value in _SCENE_LAYOUT["embodiment_frame"]["midpoint_position_xyz"]
)
_FIXTURE_LAYOUT = _SCENE_LAYOUT["fixtures"]
_TABLE_FRAME_POSITION = tuple(
    float(value) for value in _FIXTURE_LAYOUT["table"]["position_xyz"]
)
_TABLE_FRAME_ROTATION = tuple(
    float(value) for value in _FIXTURE_LAYOUT["table"]["rotation_xyzw"]
)

_TABLE_TOP_Z = float(_FIXTURE_LAYOUT["table"]["tabletop_z"])
_TABLE_CENTER_X = _EMBODIMENT_MIDPOINT[0] + 0.3475
_BOARD_THICKNESS = 0.00635
_BOARD_TOP_Z = _TABLE_TOP_Z + _BOARD_THICKNESS
_PEG_HEIGHT = 0.0235

_YAM_BASE_Z = _EMBODIMENT_MIDPOINT[2]
_YAM_FRONT_X = _EMBODIMENT_MIDPOINT[0]
# AUTOLab's cable workcell stands the two bases 0.62 m apart.
_YAM_HALF_SEPARATION = 0.31

_CABLE_COUPLER_MANAGER_TYPE: type | None = None
_CABLE_INITIAL_BODY_POSES: dict[str, tuple[float, ...]] = {}


@dataclass(frozen=True)
class _CablePhysics:
    """Physical parameters for one procedural cable variant."""

    radius: float
    density: float
    stretch_stiffness: float
    stretch_damping: float
    bend_stiffness: float
    bend_damping: float
    cable_friction: float
    fixture_friction: float
    table_friction: float
    contact_stiffness: float
    contact_damping: float
    contact_gap: float
    color: tuple[float, float, float]
    body_frame_origin: str = "start"


@dataclass(frozen=True)
class _CableRoutingVariant:
    """One package-owned cable layout and route objective."""

    cable_local_positions: tuple[tuple[float, float, float], ...]
    peg_positions: tuple[tuple[float, float, float], ...]
    task_description: str
    physics: _CablePhysics
    table_collision_center: tuple[float, float, float]
    table_collision_size: tuple[float, float, float]
    anchor_position: tuple[float, float, float]
    anchor_rotation_xyzw: tuple[float, float, float, float]
    port_position: tuple[float, float, float]
    port_rotation_xyzw: tuple[float, float, float, float]
    terminated_goal: TerminatedCableGoal
    episode_length_s: float

    @property
    def cable_num_segments(self) -> int:
        """Return the number of rigid segments in this cable layout."""
        return len(self.cable_local_positions) - 1


def _surface_position(
    xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map an AUTOLab work-surface-frame point into this Arena scene."""
    return (
        _TABLE_CENTER_X + xyz[0],
        xyz[1],
        _TABLE_TOP_Z + xyz[2],
    )


def _yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation around world Z."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _make_sampled_cable_positions(
    cable: dict,
) -> tuple[tuple[float, float, float], ...]:
    """Reproduce a fixed AUTOLab sample's bowed cable at its declared length."""
    length = float(cable["length_m"])
    segments = int(cable["segments"])
    start = tuple(float(value) for value in cable["start_surface_xyz"])
    shape = cable["shape"]

    def sampled_value(name: str, range_name: str) -> float:
        low, high = (float(value) for value in shape[range_name])
        fraction = float(shape[name])
        return 0.5 * (low + high) + 0.5 * (high - low) * fraction

    bow = sampled_value("bow_fraction", "bow_range_m")
    wave = sampled_value("wave_fraction", "wave_range_m")
    shift = sampled_value("shift_fraction", "shift_range_m")
    yaw = math.radians(sampled_value("yaw_fraction", "yaw_range_deg"))

    points = []
    for index in range(segments + 1):
        t = index / segments
        lateral = (
            bow * math.sin(math.pi * t) + wave * math.sin(2.0 * math.pi * t) + shift
        )
        points.append(
            [
                start[0] - lateral,
                start[1] + length * t,
                start[2],
            ]
        )

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    for point in points:
        rel_x, rel_y = point[0] - center_x, point[1] - center_y
        point[0] = center_x + rel_x * cos_yaw - rel_y * sin_yaw
        point[1] = center_y + rel_x * sin_yaw + rel_y * cos_yaw

    # Builder-local positions are translated back into this scene below.
    return tuple((point[0], point[1], point[2]) for point in points)


def _terminated_variant(
    scene_layout: dict, layout_seed: int | None = None
) -> _CableRoutingVariant:
    distribution = scene_layout["distribution"]
    if layout_seed is None:
        layout_seed = int(distribution["default_seed"])
    sampled = sample_cable_routing_layout(distribution, layout_seed)
    fixtures = scene_layout["fixtures"]
    table = fixtures["table"]
    peg_positions = tuple(
        _surface_position((x, y, 0.5 * _PEG_HEIGHT))
        for x, y in sampled.peg_positions_xy
    )
    source_cable = scene_layout["cable"]
    cable = {
        **source_cable,
        "shape": {
            **source_cable["shape"],
            "bow_fraction": sampled.bow_fraction,
            "shift_fraction": sampled.shift_fraction,
            "yaw_fraction": sampled.yaw_fraction,
            "wave_fraction": sampled.wave_fraction,
        },
    }
    cable_positions = _make_sampled_cable_positions(cable)
    radius = float(cable["radius_m"])
    physical = _PHYSICS_SETTINGS["physical"]
    reference_pitch = physical["reference_length_m"] / physical["reference_segments"]
    pitch = float(cable["length_m"]) / int(cable["segments"])
    bend_scale = (radius / physical["reference_radius"]) ** 4 * (reference_pitch / pitch)
    physics = _CablePhysics(
        radius=radius,
        density=physical["density"],
        stretch_stiffness=physical["stretch_stiffness"],
        stretch_damping=physical["stretch_damping"],
        bend_stiffness=physical["bend_stiffness_at_reference"] * bend_scale,
        bend_damping=physical["bend_damping_at_reference"] * bend_scale,
        cable_friction=physical["cable_friction"],
        fixture_friction=physical["fixture_friction"],
        table_friction=physical["table_friction"],
        contact_stiffness=physical["contact_stiffness"],
        contact_damping=physical["contact_damping"],
        contact_gap=physical["contact_gap"],
        color=tuple(physical["color"]),
        body_frame_origin=physical["body_frame_origin"],
    )
    anchor_x, anchor_y, anchor_yaw = anchor_pose_from_cable(
        cable_positions, float(distribution["anchor_mouth_inset_m"])
    )
    port_x, port_y, port_yaw = sampled.port_pose_xyyaw
    goal = scene_layout["goal"]
    return _CableRoutingVariant(
        cable_local_positions=cable_positions,
        peg_positions=peg_positions,
        task_description=str(scene_layout["task_description"]),
        episode_length_s=float(scene_layout.get("max_episode_steps", 13000)) / 60.0,
        physics=physics,
        table_collision_center=_surface_position(
            tuple(float(value) for value in table["collision_center_surface_xyz"])
        ),
        table_collision_size=tuple(
            float(value) for value in table["collision_size_xyz"]
        ),
        anchor_position=_surface_position((anchor_x, anchor_y, 0.0)),
        anchor_rotation_xyzw=_yaw_xyzw(anchor_yaw),
        port_position=_surface_position((port_x, port_y, 0.0)),
        port_rotation_xyzw=_yaw_xyzw(port_yaw),
        terminated_goal=TerminatedCableGoal(
            seat_regions=tuple(
                tuple(float(value) for value in region)
                for region in goal["seat_regions_xy"]
            ),
            seat_min_fractions=tuple(
                float(value) for value in goal["seat_min_fractions"]
            ),
            port_region=tuple(float(value) for value in goal["port_region_xy"]),
            port_min_fraction=float(goal["port_min_fraction"]),
            min_tcp_distance=float(goal["min_tcp_distance_m"]),
            max_mean_speed=float(goal["max_mean_speed_mps"]),
        ),
    )


def _easy_variant(layout_seed: int | None = None) -> _CableRoutingVariant:
    return _terminated_variant(_EASY_SCENE_LAYOUT, layout_seed)


def _medium_variant(layout_seed: int | None = None) -> _CableRoutingVariant:
    return _terminated_variant(_SCENE_LAYOUT, layout_seed)


def _rotate_vector_xyzw(
    quaternion: tuple[float, float, float, float] | list[float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate one vector by an XYZW quaternion."""
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _create_cable_segment_visuals(
    world_index: int,
    positions: list[tuple[float, float, float]],
    *,
    radius: float,
    color: tuple[float, float, float],
    body_frame_origin: str,
) -> list[str]:
    """Create one render-only USD capsule under each Newton cable body path."""
    from isaaclab.sim import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    cable_root_path = f"/World/envs/env_{world_index}/Cable"
    cable_root = UsdGeom.Xform.Define(stage, cable_root_path)
    from .appearance import bind_cable_material

    bind_cable_material(stage, cable_root.GetPrim(), color)

    segment_paths = []
    for segment_index, (start, end) in enumerate(zip(positions, positions[1:])):
        segment_path = f"{cable_root_path}/Segment_{segment_index:03d}"
        segment_paths.append(segment_path)
        UsdGeom.Xform.Define(stage, segment_path)

        capsule = UsdGeom.Capsule.Define(stage, f"{segment_path}/Visual")
        capsule.CreateAxisAttr("Z")
        segment_length = math.dist(start, end)
        capsule.CreateHeightAttr(segment_length)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        if body_frame_origin == "start":
            UsdGeom.Xformable(capsule.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, 0.5 * segment_length)
            )

    return segment_paths


def _add_cable_to_newton_builder(
    builder,
    world_index: int,
    environment_position: list[float],
    environment_rotation: list[float] | tuple[float, float, float, float],
    *,
    variant: _CableRoutingVariant,
) -> None:
    """Add the segmented cable to one Newton world after USD assets are imported."""
    cable_root_path = f"/World/envs/env_{world_index}/Cable"
    if any(
        isinstance(label, str) and label.startswith(cable_root_path)
        for label in builder.body_label
    ):
        return

    physics = variant.physics
    positions: list[tuple[float, float, float]] = []
    for x, y, z in variant.cable_local_positions:
        rotated = _rotate_vector_xyzw(
            environment_rotation,
            (x + _TABLE_CENTER_X, y, z + _TABLE_TOP_Z),
        )
        positions.append(
            (
                environment_position[0] + rotated[0],
                environment_position[1] + rotated[1],
                environment_position[2] + rotated[2],
            )
        )

    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = physics.density
    shape_cfg.ke = physics.contact_stiffness
    shape_cfg.kd = physics.contact_damping
    shape_cfg.mu = physics.cable_friction
    shape_cfg.margin = 0.0
    shape_cfg.gap = physics.contact_gap
    shape_cfg.collision_filter_parent = True

    shape_start = builder.shape_count
    articulation_start = builder.articulation_count
    body_indices, joint_indices = builder.add_rod(
        positions=positions,
        radius=physics.radius,
        body_frame_origin=physics.body_frame_origin,
        cfg=shape_cfg,
        stretch_stiffness=physics.stretch_stiffness,
        stretch_damping=physics.stretch_damping,
        bend_stiffness=physics.bend_stiffness,
        bend_damping=physics.bend_damping,
        label=cable_root_path,
        color=wp.vec3(*physics.color),
    )
    rod_shape_end = builder.shape_count
    segment_count = variant.cable_num_segments
    assert len(body_indices) == segment_count, (
        f"Expected {segment_count} cable bodies, got {len(body_indices)}."
    )
    assert len(joint_indices) == segment_count - 1, (
        f"Expected {segment_count - 1} cable joints, got {len(joint_indices)}."
    )

    collision_shape_indices = list(range(shape_start, rod_shape_end))
    pose = builder.body_q[body_indices[0]]
    anchor_joint_index = builder.add_joint_fixed(
        parent=-1,
        child=body_indices[0],
        parent_xform=wp.transform(
            wp.vec3(*[float(value) for value in pose[:3]]),
            wp.quat(*[float(value) for value in pose[3:7]]),
        ),
        child_xform=wp.transform_identity(),
        label=f"{cable_root_path}/Anchor",
    )

    segment_paths = _create_cable_segment_visuals(
        world_index,
        positions,
        radius=physics.radius,
        color=physics.color,
        body_frame_origin=physics.body_frame_origin,
    )
    for segment_path, body_index in zip(segment_paths, body_indices):
        builder.body_label[body_index] = segment_path
        _CABLE_INITIAL_BODY_POSES[segment_path] = tuple(builder.body_q[body_index])
    for segment_index, shape_index in enumerate(collision_shape_indices):
        builder.shape_label[shape_index] = f"{segment_paths[segment_index]}/Collision"
    for joint_index, cable_joint_index in enumerate(joint_indices):
        builder.joint_label[cable_joint_index] = (
            f"{cable_root_path}/Joint_{joint_index:03d}"
        )
    builder.joint_label[anchor_joint_index] = f"{cable_root_path}/Anchor"
    for articulation_index in range(articulation_start, builder.articulation_count):
        builder.articulation_label[articulation_index] = cable_root_path


def _finalize_cable_newton_builder(builder) -> None:
    """Color the complete dynamic model for the VBD solve."""
    if not any(
        isinstance(label, str) and "/Cable/Segment_" in label
        for label in builder.shape_label
    ):
        return
    builder.color()


def _is_cable_builder_hook(hook) -> bool:
    return getattr(hook, "func", None) is _add_cable_to_newton_builder


def _install_cable_builder_hooks(variant: _CableRoutingVariant) -> None:
    """Install scoped Newton builder hooks for one procedural cable variant."""
    from isaaclab_newton.physics import NewtonManager

    _CABLE_INITIAL_BODY_POSES.clear()
    if not hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = []
    if not hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = []
    NewtonManager._per_world_builder_hooks = [
        hook
        for hook in NewtonManager._per_world_builder_hooks
        if not _is_cable_builder_hook(hook)
    ]
    NewtonManager._per_world_builder_hooks.append(
        partial(_add_cable_to_newton_builder, variant=variant)
    )
    if _finalize_cable_newton_builder not in NewtonManager._post_replicate_hooks:
        NewtonManager._post_replicate_hooks.append(_finalize_cable_newton_builder)


def _clear_cable_builder_hooks() -> None:
    """Remove only the builder hooks owned by this environment."""
    from isaaclab_newton.physics import NewtonManager

    if hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = [
            hook
            for hook in NewtonManager._per_world_builder_hooks
            if not _is_cable_builder_hook(hook)
        ]
    if hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = [
            hook
            for hook in NewtonManager._post_replicate_hooks
            if hook is not _finalize_cable_newton_builder
        ]
    _CABLE_INITIAL_BODY_POSES.clear()


def _get_cable_body_state_cache(
    manager_type: type,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached ordered cable body indices and their initial poses."""
    import torch

    cached_state = getattr(manager_type, "_cable_body_state_cache", None)
    if cached_state is not None:
        return cached_state

    model = manager_type.get_model()
    cable_entries = sorted(
        (
            (body_label, body_index)
            for body_index, body_label in enumerate(model.body_label)
            if isinstance(body_label, str) and "/Cable/Segment_" in body_label
        ),
        key=lambda entry: entry[0],
    )
    expected_segments = manager_type.cable_variant.cable_num_segments
    assert len(cable_entries) == expected_segments, (
        f"Expected {expected_segments} labeled cable bodies, got {len(cable_entries)}."
    )
    missing_initial_poses = [
        label for label, _ in cable_entries if label not in _CABLE_INITIAL_BODY_POSES
    ]
    assert not missing_initial_poses, (
        f"Missing initial poses for cable bodies: {missing_initial_poses}."
    )

    device = wp.to_torch(manager_type.get_state_0().body_q).device
    body_ids = torch.tensor(
        [body_index for _, body_index in cable_entries], dtype=torch.long, device=device
    )
    body_poses = torch.tensor(
        [_CABLE_INITIAL_BODY_POSES[label] for label, _ in cable_entries],
        dtype=torch.float32,
        device=device,
    )
    manager_type._cable_body_state_cache = (body_ids, body_poses)
    return body_ids, body_poses


def _restore_cable_initial_body_state(manager_type: type) -> None:
    """Restore the curved rod pose erased by Newton 1.2's unmasked startup FK."""
    body_ids, body_poses = _get_cable_body_state_cache(manager_type)

    for state in (manager_type.get_state_0(), manager_type.get_state_1()):
        if state is None:
            continue
        wp.to_torch(state.body_q)[body_ids] = body_poses
        wp.to_torch(state.body_qd)[body_ids] = 0.0
    manager_type._mark_transforms_dirty()


def _reset_cable_to_initial_state(env, env_ids) -> None:
    """Restore the procedural cable, which is not an Isaac Lab scene asset."""
    del env_ids
    assert env.num_envs == 1, (
        "The cable reset compatibility path supports one environment only."
    )
    physics_manager = env.sim.physics_manager
    _restore_cable_initial_body_state(physics_manager)
    physics_manager.forward()


def _cable_coupler_manager_type(variant: _CableRoutingVariant) -> type:
    """Return a rigid/VBD proxy manager that owns one variant's cable hooks."""
    global _CABLE_COUPLER_MANAGER_TYPE
    if _CABLE_COUPLER_MANAGER_TYPE is None:
        from isaaclab_contrib.coupling import NewtonCouplerManager

        class NewtonCableRoutingCouplerManager(NewtonCouplerManager):
            """Inject the procedural cable into the standard proxy-coupled solve."""

            cable_variant: _CableRoutingVariant

            @classmethod
            def initialize(cls, sim_context) -> None:
                cls._cable_body_state_cache = None
                _install_cable_builder_hooks(cls.cable_variant)
                super().initialize(sim_context)

            @classmethod
            def cable_body_state(cls) -> tuple[torch.Tensor, torch.Tensor]:
                return _get_cable_body_state_cache(cls)

            @classmethod
            def start_simulation(cls) -> None:
                super().start_simulation()
                _restore_cable_initial_body_state(cls)

            @classmethod
            def _solver_specific_clear(cls) -> None:
                cls._cable_body_state_cache = None
                _clear_cable_builder_hooks()
                super()._solver_specific_clear()

        _CABLE_COUPLER_MANAGER_TYPE = NewtonCableRoutingCouplerManager
    base = _CABLE_COUPLER_MANAGER_TYPE
    return type(
        base.__name__,
        (base,),
        {
            "__module__": base.__module__,
            "__qualname__": base.__qualname__,
            "cable_variant": variant,
        },
    )


def _configure_cable_routing_newton_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    variant: _CableRoutingVariant,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure RSS timing with the pinned Newton VBD path."""
    from isaaclab_arena_vabar.physics import (
        IndustrialNewtonCollisionPipelineCfg,
        SparseIndustrialMJWarpSolverCfg,
        register_kinematic_contact_filter,
    )
    from isaaclab.utils.configclass import configclass
    from isaaclab_contrib.coupling import (
        CouplerEntryCfg,
        CouplerProxyCfg,
        CouplerProxyMappingCfg,
    )
    from isaaclab_newton.physics import (
        NewtonCfg,
        NewtonShapeCfg,
        VBDSolverCfg,
    )
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    assert env_cfg.scene.num_envs == 1, (
        "The initial cable-routing compatibility environment supports one env only."
    )
    physics = variant.physics
    settings = _PHYSICS_SETTINGS["environment"]
    body_paths = _PHYSICS_SETTINGS["robot"]
    _register_cable_fingertip_contact()
    register_kinematic_contact_filter()

    @configclass
    class CableRoutingShapeCfg(NewtonShapeCfg):
        """Forward the RSS contact material values to Newton's ShapeConfig."""

        ke: float = physics.contact_stiffness
        kd: float = physics.contact_damping
        mu: float = physics.fixture_friction

    @configclass
    class CableRoutingVBDSolverCfg(VBDSolverCfg):
        """Expose AUTOLab's VBD contact controls on the pinned API."""

        rigid_avbd_beta: float = 0.0
        rigid_contact_k_start: float = 0.0
        rigid_contact_history: bool = False
        rigid_body_contact_buffer_size: int = 0

    env_cfg.num_rerenders_on_reset = settings["num_rerenders_on_reset"]
    env_cfg.sim.dt = settings["dt"]
    env_cfg.sim.render_interval = settings["render_interval"]
    env_cfg.sim.use_newton_actuators = settings["use_newton_actuators"]
    env_cfg.sim.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=physics.fixture_friction,
        dynamic_friction=physics.fixture_friction,
        restitution=_PHYSICS_SETTINGS["physical"]["material_restitution"],
        contact_stiffness=physics.contact_stiffness,
        contact_damping=physics.contact_damping,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            class_type=_cable_coupler_manager_type(variant),
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=SparseIndustrialMJWarpSolverCfg(
                        **settings["rigid_solver"]
                    ),
                    bodies=[
                        *body_paths["robot_bodies"],
                        *settings["rigid_fixture_bodies"],
                    ],
                    include_body_shapes=False,
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=CableRoutingVBDSolverCfg(**settings["vbd_solver"]),
                    bodies=settings["cable_bodies"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[
                        *body_paths["robot_proxies"],
                        *settings["rigid_fixture_bodies"],
                    ],
                    collision_pipeline=IndustrialNewtonCollisionPipelineCfg(
                        **settings["collision"]
                    ),
                    **settings["proxy"],
                )
            ],
            iterations=settings["coupler_iterations"],
        ),
        default_shape_cfg=CableRoutingShapeCfg(
            margin=settings["shape_margin"], gap=settings["shape_gap"]
        ),
        collision_cfg=IndustrialNewtonCollisionPipelineCfg(**settings["collision"]),
        num_substeps=settings["num_substeps"],
        use_cuda_graph=settings["use_cuda_graph"],
        debug_mode=settings["debug_mode"],
        deterministic_mode=settings["deterministic_mode"],
    )
    env_cfg.decimation = settings["decimation"]
    env_cfg.scene.replicate_physics = settings["replicate_physics"]
    return env_cfg


def _make_cable_routing_embodiment(
    enable_cameras: bool = False,
    *,
    use_tiled_cameras: bool = False,
    use_instanceable_meshes: bool = False,
):
    """Stand the shared YAM pair at the cable workcell's two mounts."""
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )

    robot = _PHYSICS_SETTINGS["robot"]
    embodiment = IndustrialBimanualYamEmbodiment(
        enable_cameras=enable_cameras,
        use_tiled_cameras=use_tiled_cameras,
        use_instanceable_meshes=use_instanceable_meshes,
        left_mount_position=(_YAM_FRONT_X, _YAM_HALF_SEPARATION, _YAM_BASE_Z),
        right_mount_position=(_YAM_FRONT_X, -_YAM_HALF_SEPARATION, _YAM_BASE_Z),
        gripper=robot["gripper"],
        left_initial_joint_pose=robot["home_joints"],
        right_initial_joint_pose=robot["home_joints"],
        **robot["gains"],
    )
    if embodiment.camera_config is not None and robot["gripper"] == "linear_4310":
        from isaaclab_arena_vabar.embodiments.bimanual_yam.cameras import (
            YamCableCameraCfg,
        )

        cameras = YamCableCameraCfg()
        cameras.use_tiled_camera = embodiment.camera_config.use_tiled_camera
        cameras.set_robot_mount_positions(
            embodiment.scene_config.left_robot.init_state.pos,
            embodiment.scene_config.right_robot.init_state.pos,
        )
        embodiment.camera_config = cameras
    return embodiment


def _register_cable_fingertip_contact() -> None:
    """Give both YAMs the cable's fingertip contact material."""
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import YAM_HANDS
    from isaaclab_arena_vabar.embodiments.bimanual_yam.contacts import (
        FingertipContact,
        register_fingertip_contact,
    )

    robot = _PHYSICS_SETTINGS["robot"]
    register_fingertip_contact(
        FingertipContact(**robot["fingertip_contact"]),
        name="vabar_cable_routing_fingertip_contact",
        hand=YAM_HANDS[robot["gripper"]],
    )


def _make_cable_routing_task(
    variant: _CableRoutingVariant, *, use_instanceable_meshes: bool = False
):
    """Create the minimal scene, reset, route-success, and timeout task."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena_vabar.assets.hdr_shadow_receiver import hdr_shadow_receiver_cfg
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import (
        END_EFFECTOR_BODY_NAME,
        GRASP_SITE_OFFSET,
    )
    from isaaclab_arena_vabar.mesh_instancing import configure_usd_asset_instancing
    from isaaclab_arena_vabar.tasks.cable_routing.task import CableRoutingTask
    from isaaclab_arena_vabar.tasks.cable_routing.terminations import CableWorkHand
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.sim.schemas import (
        NewtonCollisionPropertiesCfg,
        NewtonMaterialPropertiesCfg,
    )

    physics = variant.physics

    def fixture_material(
        friction: float = physics.fixture_friction,
    ) -> NewtonMaterialPropertiesCfg:
        return NewtonMaterialPropertiesCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
            contact_stiffness=physics.contact_stiffness,
            contact_damping=physics.contact_damping,
        )

    def peg_cfg(name: str, position: tuple[float, float, float]) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_EASY_ROUND_PEG_USD_PATH),
                copy_from_source=False,
                physics_material=fixture_material(),
                rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=NewtonCollisionPropertiesCfg(
                    contact_margin=0.0, contact_gap=physics.contact_gap
                ),
            ),
        )

    def terminal_cfg(
        name: str,
        path: Path,
        position: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float],
    ) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation_xyzw),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(path),
                copy_from_source=False,
                physics_material=fixture_material(),
                rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=NewtonCollisionPropertiesCfg(
                    contact_margin=0.0, contact_gap=physics.contact_gap
                ),
            ),
        )

    @configclass
    class CableRoutingSceneCfg:
        # The canonical table is authored as static collision geometry, not a
        # rigid body. VBD imports static shapes directly, so it must remain an
        # AssetBase instead of being bound through RigidObject's root-body
        # lookup. Pegs and terminals remain kinematic rigid fixtures.
        table: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=_TABLE_FRAME_POSITION,
                rot=_TABLE_FRAME_ROTATION,
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_TABLE_USD_PATH),
                visible=False,
                copy_from_source=False,
                make_uninstanceable=True,
                physics_material=fixture_material(physics.table_friction),
                collision_props=NewtonCollisionPropertiesCfg(collision_enabled=False),
            ),
        )
        table_collider: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableCollider",
            init_state=AssetBaseCfg.InitialStateCfg(pos=variant.table_collision_center),
            spawn=sim_utils.CuboidCfg(
                size=variant.table_collision_size,
                visible=False,
                physics_material=fixture_material(physics.table_friction),
                collision_props=NewtonCollisionPropertiesCfg(
                    contact_margin=0.0,
                    contact_gap=physics.contact_gap,
                ),
            ),
        )
        peg_0: RigidObjectCfg = peg_cfg("Peg0", variant.peg_positions[0])
        peg_1: RigidObjectCfg = peg_cfg("Peg1", variant.peg_positions[1])
        peg_2: RigidObjectCfg | None = (
            peg_cfg("Peg2", variant.peg_positions[2])
            if len(variant.peg_positions) >= 3
            else None
        )
        peg_3: RigidObjectCfg | None = (
            peg_cfg("Peg3", variant.peg_positions[3])
            if len(variant.peg_positions) >= 4
            else None
        )
        anchor: RigidObjectCfg = terminal_cfg(
            "Anchor",
            _ANCHOR_USD_PATH,
            variant.anchor_position,
            variant.anchor_rotation_xyzw,
        )
        port: RigidObjectCfg = terminal_cfg(
            "Port",
            _PORT_USD_PATH,
            variant.port_position,
            variant.port_rotation_xyzw,
        )
        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(
                visible=False,
                color=(0.20, 0.20, 0.20),
                physics_material=fixture_material(),
            ),
            collision_group=-1,
        )
        ground_visual: AssetBaseCfg = hdr_shadow_receiver_cfg(
            prim_path="/World/GroundVisual"
        )
        native_appearance: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/CableNativeAppearance",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5875, 0.0, 0.017)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_ASSET_DIRECTORY / "native_appearance/room.usda"),
                copy_from_source=False,
            ),
        )

        def __post_init__(self) -> None:
            for asset_cfg in (
                self.table,
                self.peg_0,
                self.peg_1,
                self.peg_2,
                self.peg_3,
                self.anchor,
                self.port,
            ):
                if asset_cfg is not None:
                    configure_usd_asset_instancing(
                        asset_cfg.spawn, use_instanceable_meshes
                    )

    @configclass
    class CableRoutingEventsCfg:
        reset_scene = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        reset_cable = EventTerm(func=_reset_cable_to_initial_state, mode="reset")

    positions = variant.cable_local_positions
    return CableRoutingTask(
        scene_cfg=CableRoutingSceneCfg(),
        events_cfg=CableRoutingEventsCfg(),
        peg_names=tuple(f"peg_{index}" for index in range(len(variant.peg_positions))),
        port_name="port",
        goal=variant.terminated_goal,
        segment_half_lengths=tuple(
            0.5 * math.dist(start, end)
            for start, end in zip(positions[:-1], positions[1:])
        ),
        seat_frame_z=-0.5 * _PEG_HEIGHT,
        # The scorer measures the right hand's grasp site against the cable.
        work_hand=CableWorkHand(
            robot_name="right_robot",
            body_name=END_EFFECTOR_BODY_NAME,
            tcp_offset=GRASP_SITE_OFFSET,
        ),
        episode_length_s=variant.episode_length_s,
        task_description=variant.task_description,
        viewer_eye=(1.25, -1.10, 1.55),
        viewer_lookat=(_TABLE_CENTER_X, 0.0, _BOARD_TOP_Z),
    )


@dataclass
class CableRoutingMediumEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure one seeded RSS Medium terminated-weave layout."""

    layout_seed: int = int(_SCENE_LAYOUT["distribution"]["default_seed"])
    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


@dataclass
class CableRoutingEasyEnvironmentCfg(CableRoutingMediumEnvironmentCfg):
    """Configure one seeded RSS Easy terminated-weave layout."""

    layout_seed: int = int(_EASY_SCENE_LAYOUT["distribution"]["default_seed"])
    """Seed for the model-changing cable, anchor, and two-peg layout."""


class CableRoutingMediumEnvironment(
    ArenaEnvironmentFactory[CableRoutingMediumEnvironmentCfg]
):
    """Build one layout from AUTOLab's four-guide Medium cable distribution."""

    name = "vabar_cable_routing__medium"
    relative_graph_path = "cable_routing/medium.yaml"
    _legacy_argparse_cfg_type = CableRoutingMediumEnvironmentCfg

    def _variant_for_cfg(
        self, cfg: CableRoutingMediumEnvironmentCfg
    ) -> _CableRoutingVariant:
        return _medium_variant(cfg.layout_seed)

    def build(self, cfg: CableRoutingMediumEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment using the locally bundled RSS assets."""
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene

        variant = self._variant_for_cfg(cfg)

        asset_paths = (
            _TABLE_USD_PATH,
            _ROUND_PEG_USD_PATH,
            _EASY_ROUND_PEG_USD_PATH,
            _ANCHOR_USD_PATH,
            _PORT_USD_PATH,
        )
        for asset_path in asset_paths:
            assert asset_path.is_file(), f"Cable-routing asset not found: {asset_path}"

        return IsaacLabArenaEnvironment(
            name=self.name,
            default_physics_backend="newton",
            embodiment=_make_cable_routing_embodiment(
                enable_cameras=cfg.enable_cameras,
                use_tiled_cameras=cfg.use_tiled_cameras,
                use_instanceable_meshes=cfg.use_instanceable_meshes,
            ),
            scene=Scene(assets=[]),
            task=_make_cable_routing_task(
                variant,
                use_instanceable_meshes=cfg.use_instanceable_meshes,
            ),
            env_cfg_callback=partial(
                _configure_cable_routing_newton_physics, variant=variant
            ),
        )


class CableRoutingEasyEnvironment(CableRoutingMediumEnvironment):
    """Build one layout from AUTOLab's Easy cable-routing distribution."""

    name = "vabar_cable_routing__easy"
    relative_graph_path = "cable_routing/easy.yaml"
    _legacy_argparse_cfg_type = CableRoutingEasyEnvironmentCfg

    def _variant_for_cfg(
        self, cfg: CableRoutingMediumEnvironmentCfg
    ) -> _CableRoutingVariant:
        assert isinstance(cfg, CableRoutingEasyEnvironmentCfg)
        return _easy_variant(cfg.layout_seed)
