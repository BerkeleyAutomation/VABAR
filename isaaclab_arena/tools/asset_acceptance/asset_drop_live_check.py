"""Fresh-process CUDA/Newton drop acceptance and media capture for movable assets."""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import signal
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(
    os.environ.get("ASSET_SOURCE_ROOT", Path(__file__).resolve().parents[2])
)
_ASSET_ROOT = _SOURCE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
_ENVIRONMENT_ROOT = (
    _SOURCE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"
)
_TABLETOP_Z = 0.75


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def _usd_path(case):
    if case.relative_usd_path.startswith("@ISAAC_NUCLEUS_DIR@/"):
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

        source = case.relative_usd_path.replace(
            "@ISAAC_NUCLEUS_DIR@", ISAAC_NUCLEUS_DIR, 1
        )
        return Path(retrieve_file_path(source, force_download=False))
    return _ASSET_ROOT / case.relative_usd_path


def _support_path(case):
    leaf = case.support_asset.removeprefix("vabar__")
    return _ASSET_ROOT / case.support_asset / f"vabar__{leaf}.usda"


def _composed_bounds(
    path: Path, *, collision_only: bool, prim_suffix: str | None = None
):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.guide,
        ],
    )
    ranges = []
    for prim in stage.Traverse():
        if prim_suffix is not None and not str(prim.GetPath()).endswith(prim_suffix):
            continue
        if collision_only:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not bound.IsEmpty():
            ranges.append(bound)
    if not ranges:
        raise RuntimeError(f"USD has no matching bounded prims: {path}")
    combined = Gf.Range3d()
    for value in ranges:
        combined.UnionWith(value)
    return tuple(combined.GetMin()), tuple(combined.GetMax()), len(ranges)


def _static_preflight(path: Path):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    dynamic_bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        if (
            api.GetRigidBodyEnabledAttr().Get() is not False
            and api.GetKinematicEnabledAttr().Get() is not True
        ):
            dynamic_bodies.append(str(prim.GetPath()))
    if len(dynamic_bodies) != 1:
        raise RuntimeError(
            f"expected one enabled dynamic rigid body in {path}, got {dynamic_bodies}"
        )
    return _composed_bounds(path, collision_only=True)


def _composed_collision_points(path: Path):
    """Return collider surface points in the asset's rigid-body frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.guide,
        ],
    )
    bodies = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is not True
    ]
    if len(bodies) != 1:
        raise RuntimeError(f"expected one enabled dynamic rigid body in {path}")
    # The simulated body pose carries no scale, so scale above the body stays in the points.
    stage_to_body = (
        xforms.GetLocalToWorldTransform(bodies[0]).RemoveScaleShear().GetInverse()
    )
    points = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        transform = xforms.GetLocalToWorldTransform(prim)
        if prim.IsA(UsdGeom.Mesh):
            local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or ()
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2
            local_points = tuple(
                Gf.Vec3d(x, y, z)
                for x in (-half, half)
                for y in (-half, half)
                for z in (-half, half)
            )
        else:
            local_range = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
            if local_range.IsEmpty():
                continue
            lower, upper = local_range.GetMin(), local_range.GetMax()
            local_points = tuple(
                Gf.Vec3d(x, y, z)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            )
        points.extend(
            tuple(stage_to_body.Transform(transform.Transform(Gf.Vec3d(point))))
            for point in local_points
        )
    if not points:
        raise RuntimeError(f"USD has no enabled collider points: {path}")
    return tuple(tuple(float(value) for value in point) for point in points)


def _point_bounds(points):
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def _dynamic_body_path(root_path: str) -> str:
    from isaaclab.sim import get_current_stage
    from pxr import Usd, UsdPhysics

    root = get_current_stage().GetPrimAtPath(root_path)
    bodies = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is not True
    ]
    if len(bodies) != 1:
        raise RuntimeError(
            f"expected one spawned dynamic body below {root_path}: {bodies}"
        )
    return bodies[0]


def _format_values(values: set[float], *, prefix: str = "") -> str:
    return (
        ", ".join(f"{prefix}{value:.6g}" for value in sorted(values))
        if values
        else "not authored"
    )


def _physical_metadata(
    path: Path | None,
    runtime_mass_kg: float,
    *,
    prim_root: str | None = None,
) -> tuple[float, str, str, str]:
    """Describe the composed active collider and material opinions used by the run."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    if prim_root is None:
        stage = Usd.Stage.Open(str(path))
        prims = stage.Traverse() if stage is not None else ()
    else:
        from isaaclab.sim import get_current_stage

        stage = get_current_stage()
        root = stage.GetPrimAtPath(prim_root)
        prims = Usd.PrimRange(root) if root.IsValid() else ()
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    colliders = []
    material_paths = set()
    prims = tuple(prims)
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if prim.IsA(UsdGeom.Mesh):
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"
            )
            kind = f"Mesh[{approximation}]"
        else:
            kind = prim.GetTypeName() or "CollisionAPI"
        colliders.append(kind)
        binding = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding.ComputeBoundMaterial("physics")
        if material:
            material_paths.add(str(material.GetPath()))
    if not material_paths:
        material_paths.update(
            str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.MaterialAPI)
        )
    static_friction, dynamic_friction, restitution = set(), set(), set()
    for material_path in material_paths:
        prim = stage.GetPrimAtPath(material_path)
        if not prim.IsValid():
            continue
        api = UsdPhysics.MaterialAPI(prim)
        for target, attribute in (
            (static_friction, api.GetStaticFrictionAttr()),
            (dynamic_friction, api.GetDynamicFrictionAttr()),
            (restitution, api.GetRestitutionAttr()),
        ):
            value = attribute.Get()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                target.add(float(value))
    counts = Counter(colliders)
    collider_text = ", ".join(
        f"{kind} x{count}" for kind, count in sorted(counts.items())
    )
    friction_text = f"static={_format_values(static_friction)}, dynamic={_format_values(dynamic_friction)}"
    return runtime_mass_kg, friction_text, _format_values(restitution), collider_text


def _canonicalize_principal_axes(prim_path: str) -> None:
    """Author equivalent quaternion signs that the USD physics importer handles safely."""
    from isaaclab.sim import get_current_stage
    from pxr import Gf, Usd, UsdPhysics

    root = get_current_stage().GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"spawned asset prim is missing: {prim_path}")
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        attribute = UsdPhysics.MassAPI(prim).GetPrincipalAxesAttr()
        quaternion = attribute.Get()
        if quaternion is None or float(quaternion.GetReal()) >= 0:
            continue
        imaginary = quaternion.GetImaginary()
        attribute.Set(
            Gf.Quatf(
                -float(quaternion.GetReal()),
                Gf.Vec3f(
                    -float(imaginary[0]),
                    -float(imaginary[1]),
                    -float(imaginary[2]),
                ),
            )
        )


def _quat_matrix_xyzw(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_bounds(local_min, local_max, pose):
    matrix = _quat_matrix_xyzw(pose[3:7])
    corners = []
    for x in (local_min[0], local_max[0]):
        for y in (local_min[1], local_max[1]):
            for z in (local_min[2], local_max[2]):
                corners.append(
                    tuple(
                        float(pose[row])
                        + matrix[row][0] * x
                        + matrix[row][1] * y
                        + matrix[row][2] * z
                        for row in range(3)
                    )
                )
    return (
        tuple(min(corner[axis] for corner in corners) for axis in range(3)),
        tuple(max(corner[axis] for corner in corners) for axis in range(3)),
    )


def _world_point_bounds(local_points, pose):
    matrix = _quat_matrix_xyzw(pose[3:7])
    world_points = tuple(
        tuple(
            float(
                float(pose[row])
                + matrix[row][0] * point[0]
                + matrix[row][1] * point[1]
                + matrix[row][2] * point[2]
            )
            for row in range(3)
        )
        for point in local_points
    )
    return _point_bounds(world_points)


def _authored_world_bounds(local_min, local_max, position, rotation_xyzw):
    return _world_bounds(local_min, local_max, (*position, *rotation_xyzw))


@dataclass
class _SimOnlyEnvCfg:
    """The ``sim`` slice of an Arena env cfg, for Arena's ``apply_env_cfg_override``."""

    sim: Any


def _physics_cfg(profile: str):
    """Return the production family's ``(dt, NewtonCfg)`` for one asset profile."""
    import copy

    from isaaclab_arena_vabar.physics import (
        IndustrialMJWarpSolverCfg,
        SparseIndustrialMJWarpSolverCfg,
        install_benchmark_physics,
    )
    from isaaclab.sim import SimulationCfg
    from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import (
        load_env_graph_spec_dict,
    )
    from isaaclab_arena.environment_spec.env_cfg_override import (
        apply_env_cfg_override,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        ArenaPhysicsCfg,
    )

    from asset_acceptance.asset_drop_cases import PRODUCTION_PHYSICS_YAML

    if profile not in PRODUCTION_PHYSICS_YAML:
        raise ValueError(f"unknown physics profile: {profile}")

    document = load_env_graph_spec_dict(
        _ENVIRONMENT_ROOT / PRODUCTION_PHYSICS_YAML[profile]
    )
    # Arena's own override step keeps the preset's NewtonArenaMJWarpManager for
    # merge-only solver overrides and resyncs the manager only when the YAML
    # replaces solver_cfg through ``_target_``.
    env_cfg = _SimOnlyEnvCfg(
        SimulationCfg(physics=copy.deepcopy(ArenaPhysicsCfg().newton))
    )
    apply_env_cfg_override(env_cfg, {"sim": document["env_cfg_override"]["sim"]})
    sim_cfg = env_cfg.sim
    physics = sim_cfg.physics
    extensions = document["benchmark_physics"]
    solver_type = (
        SparseIndustrialMJWarpSolverCfg
        if "jacobian" in extensions["solver_cfg"]
        else IndustrialMJWarpSolverCfg
    )
    # Like the production benchmark callbacks, replace solver_cfg after the
    # override without touching the manager class chosen above.
    install_benchmark_physics(
        physics,
        extensions,
        solver_type=solver_type,
        collision="collision_cfg" in extensions,
    )
    return sim_cfg.dt, physics


def _drop_physics_cfg(profile: str):
    """Return ``_physics_cfg`` plus the drop-acceptance-only solver overrides."""
    from asset_acceptance.asset_drop_cases import DROP_ONLY_SOLVER_OVERRIDES

    dt, physics = _physics_cfg(profile)
    for key, value in DROP_ONLY_SOLVER_OVERRIDES.get(profile, {}).items():
        setattr(physics.solver_cfg, key, value)
    return dt, physics


def _register_standalone_usbc_contact_stiffening() -> None:
    from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
        _stiffen_contact_shapes,
    )
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    def stiffen_connectors(_event_payload=None):
        builder = NewtonManager._builder
        assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
        _stiffen_contact_shapes(builder, ("plug", "bulkhead"), 1)

    NewtonManager.register_callback(
        stiffen_connectors,
        PhysicsEvent.MODEL_INIT,
        name="asset_drop_usbc_contact_stiffening",
        wrap_weak_ref=False,
    )


def _project(point, eye, target, width=1280, height=720):
    import numpy as np

    eye, target, point = np.asarray(eye), np.asarray(target), np.asarray(point)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = point - eye
    depth = float(np.dot(relative, forward))
    focal_pixels = 28.0 / 20.955 * width
    return (
        width / 2 + focal_pixels * float(np.dot(relative, right)) / depth,
        height / 2 - focal_pixels * float(np.dot(relative, up)) / depth,
    )


def _save_thumbnail(rgb, path: Path, pivot, rotation, eye, target, axis_length):
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    pivot_pixel = _project(pivot, eye, target)
    matrix = _quat_matrix_xyzw(rotation)
    for column, (label, color) in enumerate(
        (("X", "#ff3333"), ("Y", "#33dd55"), ("Z", "#3388ff"))
    ):
        endpoint = tuple(
            pivot[row] + matrix[row][column] * axis_length for row in range(3)
        )
        endpoint_pixel = _project(endpoint, eye, target)
        draw.line((pivot_pixel, endpoint_pixel), fill=color, width=6)
        draw.text(
            (endpoint_pixel[0] + 4, endpoint_pixel[1] + 4),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    radius = 7
    draw.ellipse(
        (
            pivot_pixel[0] - radius,
            pivot_pixel[1] - radius,
            pivot_pixel[0] + radius,
            pivot_pixel[1] + radius,
        ),
        outline="white",
        width=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _drop_initial_state(case, position):
    """Spawn pose for the drop; the drop height and verdict assume this same orientation.

    The pinned Isaac Lab takes ``InitialStateCfg.rot`` as (x, y, z, w), the order the
    case already uses, so the rotation passes through unconverted.
    """
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg.InitialStateCfg(
        pos=tuple(position), rot=tuple(case.rotation_xyzw)
    )


def _write_result_payload(result, case, result_dir: Path) -> None:
    payload = {
        "asset": case.registry_name,
        "physics_profile": case.physics_profile,
        "passed": result.passed,
        "result": result.to_dict(),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{case.registry_name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print("ASSET_DROP_ACCEPTANCE " + json.dumps(payload, sort_keys=True), flush=True)


def run_asset_drop(case, record_dir: Path, result_dir: Path, *, record: bool = True):
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab_arena_vabar.assets.empty_warehouse_hdr import (
        empty_warehouse_dome_light_cfg,
    )
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    from isaaclab_newton.assets import RigidObject
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from asset_acceptance.asset_drop_cases import DROP_HEIGHT_M
    from asset_acceptance.asset_drop_harness import (
        AssetDropObservation,
        AssetDropResult,
        evaluate_asset_drop,
        settling_time,
    )

    assert torch.cuda.is_available(), "live drop test requires a visible CUDA device"
    if record:
        record_dir.mkdir(parents=True, exist_ok=True)
    asset_prim_path = f"/World/Drop_0/{case.registry_name}"
    dt, newton_cfg = _drop_physics_cfg(case.physics_profile)
    if case.physics_profile == "usbc_native":
        _register_standalone_usbc_contact_stiffening()
    sim_cfg = SimulationCfg(
        dt=dt,
        render_interval=1,
        gravity=(0, 0, -9.81),
        device="cuda:0",
        physics=newton_cfg,
    )
    _static_preflight(_usd_path(case))
    collision_points = _composed_collision_points(_usd_path(case))
    local_min, local_max = _point_bounds(collision_points)
    table_min, table_max, _ = _composed_bounds(
        _support_path(case), collision_only=False, prim_suffix="/placement_surface"
    )
    table_translation = (
        -(table_min[0] + table_max[0]) / 2,
        -(table_min[1] + table_max[1]) / 2,
        _TABLETOP_Z - table_max[2],
    )
    rotated_min, _ = _world_point_bounds(
        collision_points, (0, 0, 0, *case.rotation_xyzw)
    )
    object_position = (0, 0, _TABLETOP_Z + DROP_HEIGHT_M - rotated_min[2])
    footprint = (
        table_min[0] + table_translation[0],
        table_max[0] + table_translation[0],
        table_min[1] + table_translation[1],
        table_max[1] + table_translation[1],
    )
    objects, cameras, writers, camera_poses = [], [], [], []
    with build_simulation_context(device="cuda:0", sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        if record:
            dome_cfg = empty_warehouse_dome_light_cfg()
            dome_cfg.func("/World/DropTestDomeLight", dome_cfg)
        table_cfg = sim_utils.UsdFileCfg(usd_path=str(_support_path(case)))
        table_cfg.func("/World/Drop_0/Table", table_cfg, translation=table_translation)
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(_usd_path(case)),
            activate_contact_sensors=True,
        )
        cfg = RigidObjectCfg(
            prim_path=asset_prim_path,
            spawn=spawn_cfg,
            init_state=_drop_initial_state(case, object_position),
        )
        objects.append(RigidObject(cfg=cfg))
        contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=_dynamic_body_path(asset_prim_path),
                update_period=0.0,
                history_length=1,
            )
        )
        # Newton's USD importer requires principal-axis quaternions to use a
        # non-negative real component even though q and -q encode one rotation.
        _canonicalize_principal_axes(asset_prim_path)
        if record:
            eye, target = ((0.48, -0.62, 1.16), (0, 0, 0.82))
            rotation = tuple(
                quat_from_matrix(
                    create_rotation_matrix_from_view(
                        torch.tensor([eye]), torch.tensor([target]), up_axis="Z"
                    )
                )[0].tolist()
            )
            cameras.append(
                Camera(
                    CameraCfg(
                        prim_path="/World/Drop_0/Camera",
                        update_period=0,
                        height=720,
                        width=1280,
                        data_types=["rgb"],
                        offset=CameraCfg.OffsetCfg(
                            pos=eye, rot=rotation, convention="opengl"
                        ),
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=28,
                            horizontal_aperture=20.955,
                            clipping_range=(0.01, 100),
                        ),
                        renderer_cfg=IsaacRtxRendererCfg(
                            enable_scene_partitioning=False
                        ),
                    )
                )
            )
            camera_poses.append((eye, target))
        sim.reset()
        if record:
            for _ in range(4):
                sim.render()
        for camera, (eye, target) in zip(cameras, camera_poses, strict=True):
            camera.update(0, force_recompute=True)
            rgb = _torch(camera.data.output["rgb"])[0, ..., :3].detach().cpu().numpy()
            extent = max(local_max[axis] - local_min[axis] for axis in range(3))
            _save_thumbnail(
                rgb,
                record_dir / f"{case.registry_name}.thumbnail.png",
                object_position,
                case.rotation_xyzw,
                eye,
                target,
                max(0.04, extent * 0.55),
            )
            writers.append(
                imageio.get_writer(
                    record_dir / f"{case.registry_name}.mp4",
                    fps=30,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                )
            )

        trajectories = [([], 0, True) for _ in objects]
        masses = []
        for obj in objects:
            mass = _torch(obj.data.body_mass)
            inertia = _torch(obj.data.body_inertia)
            if (
                obj.num_bodies != 1
                or not bool(torch.isfinite(mass).all())
                or not bool((mass > 0).all())
                or not bool(torch.isfinite(inertia).all())
            ):
                raise RuntimeError(
                    f"invalid runtime mass or inertia for {case.registry_name}"
                )
            masses.append(float(mass.sum().item()))

        next_capture_time = 0.0
        for step in range(round(10 / dt)):
            simulation_time = (step + 1) * dt
            capture = record and simulation_time + 1e-9 >= next_capture_time
            sim.step(render=capture)
            for obj in objects:
                obj.update(dt)
            if capture:
                for camera, writer in zip(cameras, writers, strict=True):
                    camera.update(dt, force_recompute=True)
                    writer.append_data(
                        _torch(camera.data.output["rgb"])[0, ..., :3]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                next_capture_time += 1 / 30
            for index, (obj, data) in enumerate(
                zip(objects, trajectories, strict=True)
            ):
                poses, contacts, finite = data
                pose = _torch(obj.data.root_link_pose_w)[0].detach().cpu()
                velocity = _torch(obj.data.root_com_vel_w)[0].detach().cpu()
                poses.append(tuple(float(value) for value in pose))
                contact_sensor.update(dt, force_recompute=True)
                force = _torch(contact_sensor.data.net_forces_w)
                contacts += int(
                    force is not None
                    and float(torch.linalg.vector_norm(force).max()) > 1.0e-4
                )
                finite = (
                    finite
                    and bool(torch.isfinite(pose).all())
                    and bool(torch.isfinite(velocity).all())
                )
                trajectories[index] = (poses, contacts, finite)

        results = []
        tail_samples = max(1, round(1 / dt))
        for index in range(len(objects)):
            poses, contacts, finite = trajectories[index]
            pose = _torch(objects[index].data.root_link_pose_w)[0].detach().cpu()
            final_min, final_max = _world_point_bounds(collision_points, pose)
            inside = (
                final_min[0] >= footprint[0] - 0.002
                and final_max[0] <= footprint[1] + 0.002
                and final_min[1] >= footprint[2] - 0.002
                and final_max[1] <= footprint[3] + 0.002
            )
            observation = AssetDropObservation(
                finite,
                contacts,
                final_min[2],
                _TABLETOP_Z,
                inside,
                *AssetDropObservation.tail_from_poses(poses[-tail_samples:]),
            )
            verdict = evaluate_asset_drop(observation)
            mass, friction, restitution, collider = _physical_metadata(
                None,
                masses[index],
                prim_root=asset_prim_path,
            )
            results.append(
                AssetDropResult(
                    case.registry_name,
                    case.physics_profile,
                    verdict.passed,
                    verdict.failures,
                    observation.final_lowest_point_z,
                    observation.tabletop_z,
                    max(observation.tail_translations_m),
                    max(observation.tail_rotations_rad),
                    contacts,
                    settling_time(poses, dt),
                    mass,
                    friction,
                    restitution,
                    collider,
                )
            )
        for writer in writers:
            writer.close()
        # Persist before Kit teardown so a completed run remains recoverable if
        # the simulator driver fails during process shutdown.
        _write_result_payload(results[0], case, result_dir)
    return results[0]


def main() -> None:
    from isaaclab.app import AppLauncher
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run identical physics acceptance without cameras or media generation.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = not args.no_record

    with SimulationAppContext(args):
        # Asset registration imports Arena/USD types. Keep it after Kit starts;
        # importing it before SimulationApp preloads PXR wrappers and can break
        # schema extensions and remote asset resolution during startup.
        from asset_acceptance.asset_drop_cases import case_for_asset

        run_asset_drop(
            case_for_asset(args.asset),
            args.record_dir,
            args.result_dir,
            record=not args.no_record,
        )


if __name__ == "__main__":
    main()
