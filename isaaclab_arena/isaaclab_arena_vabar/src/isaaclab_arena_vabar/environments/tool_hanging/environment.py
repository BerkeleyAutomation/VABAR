"""Shared Newton runtime wrapper for generated tool-hanging graphs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

from isaaclab_arena_vabar.environments.graph_environment import (
    DescriptorEnvironmentFactory,
    build_graph_environment,
)

_LEFT_FINGERTIP_SHAPE_TOKENS = ("/lf_down/Sphere", "/rf_down/Sphere")
_ROBOT_TOKENS = ("/LeftRobot/", "/RightRobot/")
_FINGER_SHAPE_TOKENS = ("/lf_rot/", "/rf_rot/")
# The crank hand's finger colliders: two on each ``*_rot`` link and eleven on each
# ``*_down`` pad (six tip spheres, two boxes, three capsules), per finger.
_FINGER_SHAPES_PER_ROBOT = 26
FINGER_CONTACT_PRIORITY = 1
"""MuJoCo ``priority`` of the YAM finger colliders, for descriptors that opt in.

MuJoCo averages two geoms' ``solref``/``solimp`` unless one has the higher
priority. The hand's colliders carry AUTOLab's stiff finger contact
(``solref 0.004 1``, ``solimp 0.998 0.998 0.001``); a tool keeps the default
(``0.02 1``, ``0.9 0.95 0.001``). AUTOLab's RSS version averages them too (no finger
priority), but its Newton collision pipeline pinches with several points per pad
where Arena's MuJoCo pipeline makes one. Under the average a pinched scissors
creeps out of the closed jaws at about 0.3 mm/s and falls 32-38 s after the
grasp, before the inboard hangs release (their pre-approach times out in the RSS
run too). With the hand prioritised, finger contacts use the hand's own parameters.

A scissors-only compensation, like the plier's fingertip torsional friction:
the artist ring spanner's handle is pinched at one point per pad by about 1 N,
and that grip needs the averaged compliance -- with the hand's stiff contact the
pad contacts chatter between colliders and the spanner falls out during carry
(wrench easy 4/10 -> 0/10). Friction and ``condim`` are unchanged: MuJoCo
already took the larger, the finger's."""
_FINGER_PRIORITY_REQUIRED_SHAPE_TOKEN = "/scissors_"

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )


@dataclass(frozen=True)
class ToolHangingEnvironmentDescriptor:
    """Immutable runtime binding from a registry name to its graph."""

    environment_name: str
    relative_graph_path: str
    work_arm: str | None = None
    random_yaw_init: bool = False
    left_fingertip_torsional_friction: float | None = None
    fingertip_friction_required_shape_tokens: tuple[str, ...] = ()
    fingertip_friction_forbidden_shape_tokens: tuple[str, ...] = ()
    finger_contact_priority: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9_]+", self.environment_name) is None:
            raise ValueError("tool-hanging environment descriptor name is invalid")
        path = Path(self.relative_graph_path)
        if (
            path.is_absolute()
            or path.suffix != ".yaml"
            or path.parent != Path("tool_hanging")
            or not path.stem
        ):
            raise ValueError(
                "tool-hanging environment descriptor graph path is invalid"
            )
        if self.work_arm not in (None, "left", "right"):
            raise ValueError("tool-hanging environment descriptor work arm is invalid")
        if not isinstance(self.random_yaw_init, bool):
            raise ValueError("tool-hanging random-yaw flag is invalid")
        if not isinstance(self.finger_contact_priority, bool):
            raise ValueError("tool-hanging finger contact priority flag is invalid")
        friction = self.left_fingertip_torsional_friction
        if friction is not None and (
            isinstance(friction, bool)
            or not isinstance(friction, (int, float))
            or not isfinite(friction)
            or friction < 0.0
        ):
            raise ValueError("tool-hanging fingertip torsional friction is invalid")
        required = self.fingertip_friction_required_shape_tokens
        forbidden = self.fingertip_friction_forbidden_shape_tokens
        if friction is None and (required or forbidden):
            raise ValueError("tool-hanging fingertip friction scope requires friction")
        if friction is not None and not required:
            raise ValueError(
                "tool-hanging fingertip friction requires a positive scope"
            )
        if any(
            not isinstance(token, str) or not token for token in (*required, *forbidden)
        ):
            raise ValueError("tool-hanging fingertip friction scope is invalid")


def configure_tool_hanging_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    work_arm: str | None = None,
    replicate_physics: bool | None = None,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Install the unexposed sparse Jacobian and runtime-only options."""
    from isaaclab_arena_vabar.physics import (
        SparseIndustrialMJWarpSolverCfg,
        install_benchmark_physics,
        load_physics_extensions,
    )

    install_benchmark_physics(
        env_cfg.sim.physics,
        load_physics_extensions(
            Path(__file__).with_name("tool_hanging_env_config.yaml")
        ),
        solver_type=SparseIndustrialMJWarpSolverCfg,
    )
    if replicate_physics is not None and not isinstance(replicate_physics, bool):
        raise TypeError("replicate_physics must be a boolean or null")
    if replicate_physics is not None:
        env_cfg.scene.replicate_physics = replicate_physics
    if work_arm == "left":
        env_cfg.actions.right_arm_action = None
        env_cfg.actions.right_gripper_action = None
    elif work_arm == "right":
        env_cfg.actions.left_arm_action = None
        env_cfg.actions.left_gripper_action = None
    return env_cfg


def _apply_left_fingertip_torsional_friction(
    friction: float,
    required_shape_tokens: tuple[str, ...],
    forbidden_shape_tokens: tuple[str, ...],
    _event_payload=None,
) -> None:
    """Set torsional friction only when the active scene matches its descriptor."""
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
    labels = tuple(str(label) for label in builder.shape_label)
    if not all(
        any(token in label for label in labels) for token in required_shape_tokens
    ):
        return
    if any(any(token in label for label in labels) for token in forbidden_shape_tokens):
        return
    matched = [
        index
        for index, label in enumerate(labels)
        if "/LeftRobot/" in label
        and any(token in label for token in _LEFT_FINGERTIP_SHAPE_TOKENS)
    ]
    matched_by_env: dict[int, list[int]] = {}
    for index in matched:
        env_match = re.search(r"/env_(\d+)/", labels[index])
        assert env_match is not None, (
            f"tool-hanging fingertip shape has no environment identity: {labels[index]}"
        )
        matched_by_env.setdefault(int(env_match.group(1)), []).append(index)
    expected_env_ids = {
        int(env_match.group(1))
        for label in labels
        if (env_match := re.search(r"/env_(\d+)/", label)) is not None
    }
    invalid_counts = {
        env_id: len(matched_by_env.get(env_id, ()))
        for env_id in expected_env_ids
        if len(matched_by_env.get(env_id, ())) != 12
    }
    assert expected_env_ids and not invalid_counts, (
        "tool-hanging friction requires 12 active fingertip shapes per environment; "
        f"matched {len(matched)} total with invalid counts {invalid_counts}"
    )
    for index in matched:
        builder.shape_material_mu_torsional[index] = friction


def _register_left_fingertip_torsional_friction(
    descriptor: ToolHangingEnvironmentDescriptor,
) -> None:
    """Install a descriptor-selected Newton material override once per build."""
    friction = descriptor.left_fingertip_torsional_friction
    if friction is None:
        return

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    name = f"industrial_{descriptor.environment_name}_left_fingertip_friction"
    if any(entry[3] == name for entry in PhysicsManager._callbacks.values()):
        return
    NewtonManager.register_callback(
        partial(
            _apply_left_fingertip_torsional_friction,
            float(friction),
            descriptor.fingertip_friction_required_shape_tokens,
            descriptor.fingertip_friction_forbidden_shape_tokens,
        ),
        PhysicsEvent.MODEL_INIT,
        name=name,
        wrap_weak_ref=False,
    )


def _apply_finger_contact_priority(_event_payload=None) -> None:
    """Give every YAM finger collider MuJoCo contact priority in a scissors scene."""
    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
    labels = [str(label) for label in builder.shape_label]
    if not any(_FINGER_PRIORITY_REQUIRED_SHAPE_TOKEN in label for label in labels):
        return

    def robot_of(label: str) -> str | None:
        for token in _ROBOT_TOKENS:
            if token in label:
                return label[: label.index(token) + len(token)]
        return None

    matched = [
        index
        for index, label in enumerate(labels)
        if int(builder.shape_flags[index]) & int(ShapeFlags.COLLIDE_SHAPES)
        and robot_of(label) is not None
        and any(token in label for token in _FINGER_SHAPE_TOKENS)
    ]
    counts = Counter(robot_of(labels[index]) for index in matched)
    robots = {robot for label in labels if (robot := robot_of(label)) is not None}
    assert robots and all(
        counts[robot] == _FINGER_SHAPES_PER_ROBOT for robot in robots
    ), (
        f"expected {_FINGER_SHAPES_PER_ROBOT} finger colliders on each YAM, "
        f"matched {dict(counts)} on {sorted(robots)}"
    )
    attribute = builder.custom_attributes["mujoco:geom_priority"]
    if attribute.values is None:
        attribute.values = {}
    for index in matched:
        attribute.values[index] = FINGER_CONTACT_PRIORITY


def _register_finger_contact_priority(
    descriptor: ToolHangingEnvironmentDescriptor,
) -> None:
    """Install the finger contact priority once per build, if the descriptor opts in."""
    if not descriptor.finger_contact_priority:
        return
    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    name = "industrial_tool_hanging_finger_contact_priority"
    if any(entry[3] == name for entry in PhysicsManager._callbacks.values()):
        return
    NewtonManager.register_callback(
        _apply_finger_contact_priority,
        PhysicsEvent.MODEL_INIT,
        name=name,
        wrap_weak_ref=False,
    )


@dataclass
class ToolHangingNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Runtime options shared by generated tool-hanging environments."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False
    replicate_physics: bool | None = None


class ToolHangingNewtonEnvironment(
    DescriptorEnvironmentFactory[ToolHangingNewtonEnvironmentCfg]
):
    """Build the descriptor's graph with runtime-only options."""

    descriptor_type = ToolHangingEnvironmentDescriptor
    cfg_type = ToolHangingNewtonEnvironmentCfg
    family = "tool-hanging"
    generated_class_prefix = "GeneratedToolHangingEnvironment_"
    descriptor: ToolHangingEnvironmentDescriptor | None = None

    def build(self, cfg: ToolHangingNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena_vabar.physics import append_env_cfg_callback

        graph_path = self.graph_path()
        descriptor = self.descriptor
        arena_env = build_graph_environment(graph_path, cfg)
        arena_env.placer_params.random_yaw_init = descriptor.random_yaw_init
        _register_left_fingertip_torsional_friction(descriptor)
        _register_finger_contact_priority(descriptor)
        append_env_cfg_callback(
            arena_env,
            partial(
                configure_tool_hanging_physics,
                work_arm=descriptor.work_arm,
                replicate_physics=cfg.replicate_physics,
            ),
        )
        return arena_env
