# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding for AUTOLab's USB-C insertion tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

from isaaclab_arena_vabar.environments.graph_environment import (
    DescriptorEnvironmentFactory,
    build_graph_environment,
)
from isaaclab_arena_vabar.physics import load_physics_extensions


@dataclass(frozen=True)
class UsbcInsertEnvironmentDescriptor:
    """USB-C scene and cable configuration."""

    environment_name: str
    relative_graph_path: str
    cable_spec: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.environment_name or not self.relative_graph_path.endswith(".yaml"):
            raise ValueError("USB-C descriptor name or graph path is invalid")


@dataclass
class UsbcInsertNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Runtime options shared by generated USB-C insertion environments."""

    enable_cameras: bool = False

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


class UsbcInsertNewtonEnvironment(
    DescriptorEnvironmentFactory[UsbcInsertNewtonEnvironmentCfg]
):
    """Build the converted graph and attach runtime cable behavior."""

    descriptor_type = UsbcInsertEnvironmentDescriptor
    cfg_type = UsbcInsertNewtonEnvironmentCfg
    family = "USB-C"
    generated_class_prefix = "GeneratedUsbcInsertEnvironment_"
    descriptor: UsbcInsertEnvironmentDescriptor | None = None

    def build(self, cfg: UsbcInsertNewtonEnvironmentCfg) -> Any:
        from isaaclab_arena_vabar.physics import append_env_cfg_callback

        graph_path = self.graph_path()
        descriptor = self.descriptor
        arena_env = build_graph_environment(graph_path, cfg)
        _use_usbc_station_cameras(arena_env.embodiment)
        _configure_usbc_placement(arena_env)
        append_env_cfg_callback(
            arena_env,
            partial(
                _configure_env_cfg,
                cable_links=sum(
                    int(v["links"])
                    for v in descriptor.cable_spec.values()
                    if isinstance(v, Mapping)
                ),
            ),
        )
        _register_grip_friction_override(arena_env.embodiment.hand)
        _register_cables(descriptor.cable_spec)
        return arena_env


def _use_usbc_station_cameras(embodiment: Any) -> None:
    """Swap the shared YAM views for AUTOLab's USB-C station (``yam_station_usbc@v1``)."""
    if embodiment.camera_config is None:
        return
    from isaaclab_arena_vabar.embodiments.bimanual_yam.cameras import YamUsbcCameraCfg

    shared = embodiment.camera_config
    cameras = YamUsbcCameraCfg()
    cameras.use_tiled_camera = shared.use_tiled_camera
    # The wrist D405s ride whichever hand the embodiment selected.
    cameras.left_wrist_camera = shared.left_wrist_camera
    cameras.right_wrist_camera = shared.right_wrist_camera
    cameras.set_robot_mount_positions(
        embodiment.scene_config.left_robot.init_state.pos,
        embodiment.scene_config.right_robot.init_state.pos,
    )
    embodiment.camera_config = cameras


def _configure_usbc_placement(arena_env: Any) -> Any:
    """Require the authored support relations without best-loss fallbacks."""
    arena_env.placer_params.allow_best_loss_fallbacks = False
    # Generic overlap checks approximate these intentional support contacts as collisions.
    arena_env.placer_params.required_checks = {"on_relation"}
    arena_env.placer_params.solver_params.clearance_m = 0.0
    arena_env.placer_params.solver_params.lr = 0.001
    return arena_env


_GRIP_SHAPE_TOKENS = ("plug", "port", "bulkhead")
"""Shape-label substrings stiffened by AUTOLab's contact rig."""


def _register_grip_friction_override(hand) -> None:
    """Install AUTOLab's USB-C contact-rig callbacks once per process, for the fitted ``hand``."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    NewtonManager.register_callback(
        partial(_apply_full_hand_contacts, hand=hand),
        PhysicsEvent.MODEL_INIT,
        name="industrial_usbc_full_hand_contacts",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _apply_contact_stiffening,
        PhysicsEvent.MODEL_INIT,
        name="industrial_usbc_contact_stiffening",
        wrap_weak_ref=False,
    )
    NewtonManager.register_callback(
        _apply_source_friction,
        PhysicsEvent.MODEL_INIT,
        name="industrial_usbc_source_friction",
        wrap_weak_ref=False,
    )


_USBC_PHYSICS = load_physics_extensions(Path(__file__).with_name("usbc_env_config.yaml"))
_CONTACTS = _USBC_PHYSICS["contacts"]
_CABLE_PHYSICS = _USBC_PHYSICS["cable"]

SOLREF = tuple(_CONTACTS["solref"])
"""MuJoCo solref (timeconst, dampratio) the reference sets on the fit contacts."""

SOLIMP = tuple(_CONTACTS["solimp"])
FINGER_MU = _CONTACTS["finger_mu"]
FINGER_MU_TORSIONAL = _CONTACTS["finger_mu_torsional"]
FINGER_MU_ROLLING = _CONTACTS["finger_mu_rolling"]
CONTACT_GAP = _CONTACTS["gap"]
CONTACT_CONDIM = _CONTACTS["condim"]

CONTACT_KE, CONTACT_KD = _CONTACTS["stiffness"], _CONTACTS["damping"]
"""SOLREF converted to Newton stiffness/damping units."""


def _set_mujoco_shape_attributes(
    builder, shapes: tuple[int, ...], *, condim: int | None = None
) -> None:
    """Apply the source full-hand contact model."""
    import warp as wp
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
    from newton._src.solvers.mujoco.solver_mujoco import vec5

    for name, value in (
        ("mujoco:solref", wp.vec2(*SOLREF)),
        ("mujoco:solref_mode", SOLREF_MODE_RAW),
        ("mujoco:geom_solimp", vec5(*SOLIMP)),
    ):
        attr = builder.custom_attributes[name]
        if attr.values is None:
            attr.values = {}
        for index in shapes:
            attr.values[index] = value

    if condim is not None:
        attr = builder.custom_attributes["mujoco:condim"]
        if attr.values is None:
            attr.values = {}
        for index in shapes:
            attr.values[index] = condim


def _stiffen_finger_coupling(builder) -> None:
    """Keep the passive jaw coupled under the source's high closing forces.

    The canonical hand has one drive; the RSS hand drives both jaws. Default 20 ms
    equality compliance lets the passive jaw yield and unload the active drive.
    A 4 ms time constant retains the mechanical coupling during insertion.
    """
    import warp as wp

    eq_joints = builder.custom_attributes["mujoco:equality_constraint_joint1"].values
    eq_solref = builder.custom_attributes["mujoco:eq_solref"]
    if eq_solref.values is None:
        eq_solref.values = {}
    count = 0
    for index, joint in enumerate(eq_joints):
        if (
            int(joint) >= 0
            and "finger" in str(builder.joint_label[int(joint)]).casefold()
        ):
            eq_solref.values[index] = wp.vec2(*_CONTACTS["finger_equality_solref"])
            count += 1
    if not count:
        raise RuntimeError("USB-C contact rig found no YAM finger equalities")


def _apply_full_hand_contacts(_event_payload=None, *, hand) -> None:
    """Retain full finger geometry and, where the hand names one, enable housing collision, as bf985d0."""
    import newton
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None
    # Only a hand with a mimicked follower jaw has a coupling to stiffen; the RSS hand drives both.
    if hand.passive_joints:
        _stiffen_finger_coupling(builder)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    fingers = []
    housings = []
    for robot_token in ("leftrobot", "rightrobot"):
        finger_bodies = {
            i
            for i, name in enumerate(builder.body_label)
            if robot_token in str(name).casefold()
            and any(token in str(name).casefold() for token in hand.finger_body_tokens)
        }
        shapes = [
            i for i, body in enumerate(builder.shape_body) if int(body) in finger_bodies
        ]
        references = [i for i in shapes if int(builder.shape_flags[i]) & collide]
        if not references:
            raise RuntimeError(
                f"USB-C full hand found no colliding fingers for {robot_token}"
            )
        reference = references[0]
        excluded = {
            other
            for pair in builder._shape_collision_filter_pairs
            if reference in pair
            for other in pair
            if other != reference
        }
        for i, body in enumerate(builder.shape_body):
            if hand.housing_body is None:
                break
            if int(body) < 0:
                continue
            label = str(builder.body_label[int(body)]).casefold()
            if robot_token not in label or label.rsplit("/", 1)[-1] != hand.housing_body:
                continue
            if builder.shape_source[i] is None:
                continue
            builder.shape_flags[i] |= collide
            builder.shape_collision_group[i] = builder.shape_collision_group[reference]
            for other in {reference, *excluded}:
                if other != i:
                    builder.add_shape_collision_filter_pair(i, other)
            housings.append(i)
        fingers.extend(shapes)
    for i in fingers:
        builder.shape_material_mu[i] = FINGER_MU
        builder.shape_material_mu_torsional[i] = FINGER_MU_TORSIONAL
        builder.shape_material_mu_rolling[i] = FINGER_MU_ROLLING
        builder.shape_gap[i] = CONTACT_GAP
        # AUTOLab adds each CoACD fingertip piece as a triangle mesh
        # (RSS's ``_decompose_colliders``), not as a convex hull. Against the
        # parts' SDFs Newton then samples the whole pad face: 22 and 10 contacts
        # on the bulkhead flange where the same pieces as hulls give 5 and 3, and
        # a grip that few points hold pivots, so the held bulkhead pitched off the
        # flange and the plug turned in the jaws at the mouth.
        if builder.shape_type[i] == newton.GeoType.CONVEX_MESH and int(builder.shape_flags[i]) & collide:
            builder.shape_type[i] = newton.GeoType.MESH
    _set_mujoco_shape_attributes(
        builder, tuple(fingers + housings), condim=CONTACT_CONDIM
    )
    for i in housings:
        builder.custom_attributes["mujoco:condim"].values[i] = _CONTACTS[
            "housing_condim"
        ]


def _stiffen_contact_shapes(
    builder, shape_tokens, minimum_matches: int
) -> tuple[int, ...]:
    """Apply the production USB-C stiffness to selected Newton shapes."""

    matched = tuple(
        index
        for index in range(builder.shape_count)
        if any(
            token in str(builder.shape_label[index]).casefold()
            for token in shape_tokens
        )
    )
    assert len(matched) >= minimum_matches, (
        f"contact stiffening matched {len(matched)} shape(s) for {shape_tokens}; "
        f"expected at least {minimum_matches}"
    )
    for index in matched:
        builder.shape_material_ke[index] = CONTACT_KE
        builder.shape_material_kd[index] = CONTACT_KD
    return matched


def _apply_contact_stiffening(_event_payload=None) -> None:
    """Stiffen the contacts that carry the grasp and the fit."""

    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"

    matched = _stiffen_contact_shapes(builder, _GRIP_SHAPE_TOKENS, 2)
    _set_mujoco_shape_attributes(builder, matched)


def _apply_source_friction(_event_payload=None) -> None:
    """Apply the material coefficients authored by ``usbc_contact_rig``."""
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
    counts = {"plug": 0, "port": 0, "bulkhead": 0, "bench": 0, "table": 0}
    for index, raw_label in enumerate(builder.shape_label):
        label = str(raw_label).casefold()
        if "plug" in label:
            builder.shape_material_mu[index] = _CONTACTS["friction"]["plug"]
            counts["plug"] += 1
        elif "bulkhead" in label:
            builder.shape_material_mu[index] = _CONTACTS["friction"]["bulkhead"]
            counts["bulkhead"] += 1
        elif "port" in label:
            builder.shape_material_mu[index] = _CONTACTS["friction"]["port"]
            counts["port"] += 1
        elif "bench" in label:
            builder.shape_material_mu[index] = _CONTACTS["friction"]["bench"]
            counts["bench"] += 1
        elif label.endswith("table") or "/table/" in label:
            builder.shape_material_mu[index] = _CONTACTS["friction"]["table"]
            counts["table"] += 1
    if counts["plug"] < 1 or counts["port"] + counts["bulkhead"] < 1:
        raise RuntimeError(
            f"USB-C source friction did not find its contact shapes: {counts}"
        )


def _configure_env_cfg(env_cfg, *, cable_links: int = 16):
    """Install unexposed physics fields and the cable-reset event."""
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena_vabar.physics import install_benchmark_physics

    install_benchmark_physics(env_cfg.sim.physics, _USBC_PHYSICS, collision=True)
    # The spawn spread (graph_environment) runs before this, so cables re-root on the moved parts.
    env_cfg.events.reset_usbc_cables = EventTermCfg(
        func=_reset_cables, mode="reset", params={"expected_links": cable_links}
    )
    return env_cfg


# Cable physics follows the RSS Easy chain; rendering remains smooth.
# Lengths, radius, density, link counts and root transforms remain generated
# descriptor data; keeping them there means a source re-import cannot be hidden
# by runtime literals.
CABLE_MU = _CABLE_PHYSICS["mu"]
CABLE_ARMATURE = _CABLE_PHYSICS["armature"]
CABLE_JOINT_FRICTION = _CABLE_PHYSICS["joint_friction"]
CABLE_CONTACT_GAP = _CABLE_PHYSICS["contact_gap"]
CABLE_COLOR = (0.90, 0.90, 0.90)
CABLE_KE = _CABLE_PHYSICS["ke"]
CABLE_KD = _CABLE_PHYSICS["kd"]


def _body_index(builder, name: str, world_id: int | None = None) -> int:
    """Resolve one connector body without silently welding a cable to air."""
    labels = [str(label) for label in builder.body_label]
    hits = [
        index
        for index, label in enumerate(labels)
        if name.casefold() in label.casefold()
        and (world_id is None or builder.body_world[index] == world_id)
    ]
    if len(hits) != 1:
        raise ValueError(
            f"cable root {name!r} matched {len(hits)} bodies, expected exactly one: "
            f"{[labels[index] for index in hits] if hits else labels[:12]}"
        )
    return hits[0]


def _author_cable_visual(
    root_path: str, links: int, radius: float
) -> tuple[list[str], str]:
    """Author one render curve driven by the physical Newton capsules.

    RTX does not reliably discover geometry added below late-authored rigid-body
    prims.  The pinned Newton manager has a dedicated BasisCurves synchronizer:
    it reconstructs every point from the matching physical capsule endpoints on
    each render.  Physics remains authored exactly once on the Newton builder.
    """
    from isaaclab.sim import get_current_stage
    from isaaclab.sim.schemas import define_deformable_curve_properties
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, root_path)
    curve_path = f"{root_path}/geometry/mesh"
    curve = UsdGeom.BasisCurves.Define(stage, curve_path)
    curve.CreatePointsAttr([Gf.Vec3f()] * (links + 1))
    curve.CreateCurveVertexCountsAttr([links + 1])
    curve.CreateTypeAttr(UsdGeom.Tokens.cubic)
    curve.CreateBasisAttr(UsdGeom.Tokens.bspline)
    curve.CreateWrapAttr(UsdGeom.Tokens.pinned)
    curve.CreateWidthsAttr([float(2.0 * radius)])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*CABLE_COLOR)])
    define_deformable_curve_properties(curve_path, stage=stage)
    curve.GetPrim().SetCustomDataByKey("industrial:visualOnly", True)

    body_paths = [f"{root_path}/physics/link{index}" for index in range(links)]
    return body_paths, curve_path


def _cable_chain(
    builder,
    *,
    parent: int,
    root_xform,
    root_anchor,
    links: int,
    seg_half: float,
    radius: float,
    density: float,
    tag: str,
    environment_path: str | None = None,
):
    """Build one BDDL-style articulated capsule chain."""
    import warp as wp

    previous_world = builder._current_world
    builder._current_world = int(builder.body_world[parent])

    cfg = builder.default_shape_cfg.copy()
    cfg.mu = CABLE_MU
    cfg.gap = CABLE_CONTACT_GAP
    cfg.density = float(density)
    cfg.mu_torsional = 0.0
    cfg.mu_rolling = 0.0
    cfg.ke = CABLE_KE
    cfg.kd = CABLE_KD

    import re

    match = re.match(
        r"^(/World/envs/env_[0-9]+)(?:/|$)", str(builder.body_label[parent])
    )
    if match is None:
        raise ValueError(
            f"USB-C cable parent has no environment scope: {builder.body_label[parent]}"
        )
    environment_path = environment_path or match.group(1)
    root_path = f"{environment_path}/vabar_usbc_insert_{tag}"
    body_paths, curve_path = _author_cable_visual(root_path, links, radius)
    segment = 2.0 * seg_half
    joints = []
    try:
        for index in range(links):
            xform = wp.transform_multiply(
                root_xform,
                wp.transform(
                    wp.vec3(0.0, 0.0, float(seg_half + index * segment)),
                    wp.quat_identity(),
                ),
            )
            body = builder.add_link(xform=xform, label=body_paths[index])
            builder.add_shape_capsule(
                body,
                radius=float(radius),
                half_height=float(seg_half),
                cfg=cfg,
                color=CABLE_COLOR,
                label=f"{curve_path}_edge_capsule_{index}",
            )
            parent_xform = (
                root_anchor
                if index == 0
                else wp.transform(
                    wp.vec3(0.0, 0.0, float(seg_half)), wp.quat_identity()
                )
            )
            axis = (1.0, 0.0, 0.0) if index % 2 == 0 else (0.0, 1.0, 0.0)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    parent_xform=parent_xform,
                    child_xform=wp.transform(
                        wp.vec3(0.0, 0.0, float(-seg_half)), wp.quat_identity()
                    ),
                    axis=axis,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                    armature=CABLE_ARMATURE,
                    friction=CABLE_JOINT_FRICTION,
                    label=f"{tag}{index}_bend",
                )
            )
            parent = body
        builder.add_articulation(joints, label=tag)
    finally:
        builder._current_world = previous_world


def _build_cables(
    spec: Mapping[str, Any], builder, world_id: int, position, quaternion
) -> None:
    """Append cables while their Newton world is still being constructed."""
    import math

    import warp as wp

    environment_path = f"/World/envs/env_{world_id}"
    scale = float(spec["scale"])

    # Flat stage imports use Newton global world -1, while the hook receives
    # logical environment 0. Replicated builders use explicit world IDs.
    physics_world = world_id if builder.world_count else -1
    plug = spec["plug"]
    plug_links = int(plug["links"])
    plug_body = _body_index(builder, str(plug["object"]), world_id=physics_world)
    plug_spawn = builder.body_q[plug_body]
    plug_half = 0.5 * (float(plug["length"]) * scale) / plug_links
    rear = float(plug["rear_z"])
    flip = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    plug_anchor = wp.transform(wp.vec3(0.0, 0.0, rear), flip)
    plug_root = wp.transform_multiply(plug_spawn, plug_anchor)
    _cable_chain(
        builder,
        parent=plug_body,
        root_xform=plug_root,
        root_anchor=plug_anchor,
        links=plug_links,
        seg_half=plug_half,
        radius=float(plug["radius"]) * scale,
        density=float(plug["density"]),
        tag="cable",
        environment_path=environment_path,
    )

    if "bulkhead" in spec:
        rear = spec["bulkhead"]
        body = _body_index(builder, str(rear["object"]), world_id=physics_world)
        # Source computes the anchor from the spawned body and nominal world boot pose.
        root = wp.transform(
            wp.vec3(0.44, 0.0763, 0.8234),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2),
        )
        reference_body = _body_index(builder, str(rear["object"]), world_id=0 if builder.world_count else -1)
        anchor = wp.transform_multiply(
            wp.transform_inverse(builder.body_q[reference_body]), root
        )
        if body != reference_body:
            root = wp.transform_multiply(builder.body_q[body], anchor)
        _cable_chain(
            builder,
            parent=body,
            root_xform=root,
            root_anchor=anchor,
            links=int(rear["links"]),
            seg_half=0.5 * float(rear["length"]) * scale / int(rear["links"]),
            radius=float(rear["radius"]) * scale,
            density=float(rear["density"]),
            tag="pcable",
            environment_path=environment_path,
        )


def _reset_cables(env, env_ids, expected_links: int = 16) -> None:
    """Restore BDDL's straight bend-coordinate state on every episode reset."""

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    if model is None:
        raise RuntimeError("USB-C cable reset ran before Newton finalized its model")
    joint_labels = [str(label) for label in model.joint_label]
    joint_indices = [
        index for index, label in enumerate(joint_labels) if "cable" in label.casefold()
    ]
    num_envs = int(env.num_envs)
    if len(joint_indices) != expected_links * num_envs:
        raise RuntimeError(
            f"USB-C reset found {len(joint_indices)} cable joints, expected {expected_links * num_envs}: "
            f"{[joint_labels[index] for index in joint_indices[:8]]}"
        )

    selected = set(range(num_envs)) if env_ids is None else {int(i) for i in env_ids}
    child_bodies = model.joint_child.numpy()
    by_env = {i: [] for i in range(num_envs)}
    for joint_index in joint_indices:
        label = str(model.body_label[int(child_bodies[joint_index])])
        matches = [i for i in by_env if label.startswith(f"/World/envs/env_{i}/")]
        if len(matches) != 1:
            raise RuntimeError(f"USB-C cable joint has ambiguous environment: {label}")
        by_env[matches[0]].append(joint_index)
    if any(len(indices) != expected_links for indices in by_env.values()):
        raise RuntimeError("USB-C cable joint count differs between environments")
    joint_indices = [joint for i in sorted(selected) for joint in by_env[i]]

    q_starts = model.joint_q_start.numpy()
    qd_starts = model.joint_qd_start.numpy()
    states = []
    for state in (NewtonManager._state_0, NewtonManager._state_1):
        if state is not None and all(state is not existing for existing in states):
            states.append(state)
    for state in states:
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        for joint_index in joint_indices:
            q_start = int(q_starts[joint_index])
            q_end = int(q_starts[joint_index + 1])
            qd_start = int(qd_starts[joint_index])
            qd_end = int(qd_starts[joint_index + 1])
            q[q_start:q_end] = 0.0
            qd[qd_start:qd_end] = 0.0
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
    NewtonManager.invalidate_fk()


def _register_cables(spec: Mapping[str, Any]) -> None:
    """Build cables inside each world, before Newton finalizes world ordering."""
    from isaaclab_newton.physics import NewtonManager

    hooks = NewtonManager._per_world_builder_hooks
    hooks[:] = [
        hook for hook in hooks if getattr(hook, "func", None) is not _build_cables
    ]
    hooks.append(partial(_build_cables, spec))
