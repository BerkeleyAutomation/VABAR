"""Shared physics defaults for industrial benchmark environments."""

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg


@configclass
class IndustrialMJWarpSolverCfg(MJWarpSolverCfg):
    """MuJoCo-Warp options used by the benchmark but not yet exposed by Lab."""

    enable_multiccd: bool = False


@configclass
class IndustrialNewtonCollisionPipelineCfg(NewtonCollisionPipelineCfg):
    """Newton contact-matching options used by contact-rich benchmarks."""

    contact_matching: Literal["disabled", "latest", "sticky"] = "disabled"


@configclass
class SparseIndustrialMJWarpSolverCfg(IndustrialMJWarpSolverCfg):
    """Expose the sparse Jacobian setting absent from Lab's solver config."""

    jacobian: str = "sparse"


def load_physics_extensions(path: Path) -> dict[str, Any]:
    """Read benchmark-only fields from the same YAML as Arena's graph override."""
    from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import (
        load_env_graph_spec_dict,
    )

    values = load_env_graph_spec_dict(path).get("benchmark_physics")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"benchmark_physics in {path} must be a nonempty mapping")
    return values


def with_unexposed_fields(
    config: Any, config_type: type, overrides: dict[str, Any] | None = None
) -> Any:
    """Preserve YAML-populated fields while adding benchmark-only config fields."""
    values = (
        {
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.init and field.name != "class_type"
        }
        if config is not None
        else {}
    )
    values.update(overrides or {})
    return config_type(**values)


def install_benchmark_physics(
    physics: Any,
    extensions: dict[str, Any],
    *,
    solver_type: type = IndustrialMJWarpSolverCfg,
    collision: bool = False,
) -> None:
    """Install a YAML's benchmark-only solver (and collision) fields in place."""
    register_kinematic_contact_filter()
    physics.solver_cfg = with_unexposed_fields(
        physics.solver_cfg, solver_type, extensions["solver_cfg"]
    )
    if collision:
        physics.collision_cfg = with_unexposed_fields(
            physics.collision_cfg,
            IndustrialNewtonCollisionPipelineCfg,
            extensions["collision_cfg"],
        )


def _filter_kinematic_contacts(_event_payload=None) -> None:
    """Exclude contacts between kinematic bodies of the same world.

    Fixtures such as tool-hanging hooks and their pegboard rack are kinematic
    Newton bodies. Newton's own broad phase skips such immovable pairs, but with
    ``use_mujoco_contacts`` MuJoCo-Warp collides them as ordinary free bodies
    (armature 1e10). An overlap between two fixtures then gives one a solver
    velocity every substep; Newton restores its pose, but a tool resting on it
    keeps the friction drag and creeps off. Filtering every shape
    pair of two kinematic bodies makes Newton emit a MuJoCo body exclude.
    """
    from isaaclab_newton.physics import NewtonManager
    from newton import BodyFlags

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
    kinematic = [
        body
        for body, flags in enumerate(builder.body_flags)
        if int(flags) & int(BodyFlags.KINEMATIC)
    ]
    for index, body_a in enumerate(kinematic):
        for body_b in kinematic[index + 1 :]:
            if builder.body_world[body_a] != builder.body_world[body_b]:
                continue
            for shape_a in builder.body_shapes[body_a]:
                for shape_b in builder.body_shapes[body_b]:
                    builder.add_shape_collision_filter_pair(shape_a, shape_b)


def register_kinematic_contact_filter() -> None:
    """Install the kinematic-pair contact filter once per process."""
    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    name = "vabar_kinematic_contact_filter"
    if any(entry[3] == name for entry in PhysicsManager._callbacks.values()):
        return
    NewtonManager.register_callback(
        _filter_kinematic_contacts,
        PhysicsEvent.MODEL_INIT,
        name=name,
        wrap_weak_ref=False,
    )


def append_env_cfg_callback(arena_env: Any, callback: Callable[[Any], Any]) -> None:
    """Run a benchmark callback after an Arena graph's native YAML override."""
    previous = arena_env.env_cfg_callback
    if previous is None:
        arena_env.env_cfg_callback = callback
        return

    def combined(env_cfg):
        return callback(previous(env_cfg))

    arena_env.env_cfg_callback = combined
