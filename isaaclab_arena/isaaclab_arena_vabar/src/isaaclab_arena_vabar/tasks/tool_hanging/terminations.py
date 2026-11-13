"""Tool-hanging completion predicates and the released, held success term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena_vabar.numeric import as_torch

from .geometry import LoopFeature, RodFeature, _world_point, encloses, threaded

_FORCE_WINDOW_STEPS = 10
_FORCE_HISTORY_ATTR = "_tool_hanging_force_history"
_HOLD_STATE_ATTR = "_tool_hanging_hold_state"


# Isaac Lab prepares termination parameters in place, so this dataclass must stay mutable.
@dataclass
class HandTcp:
    """One arm's tool centre, as the RSS version's ``hands-clear`` reads it."""

    robot_name: str
    body_name: str
    offset_xyz: tuple[float, float, float]


@dataclass
class HangingGoalSpec:
    """One graph-resolved tool/fixture scoring relation."""

    tool_name: str
    fixture_name: str
    loop_owner_name: str
    rod_owner_name: str
    contact_sensor_name: str
    loop: LoopFeature | None = None
    rod: RodFeature | None = None
    containment_min_xyz: tuple[float, float, float] | None = None
    containment_max_xyz: tuple[float, float, float] | None = None
    containment_point_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_force_n: float = 0.2
    speed_m_s: float = 0.15
    alternative_fixture_names: tuple[str, ...] = ()
    predicate: Literal["threaded", "encloses"] = "threaded"
    alternative_loops: tuple[LoopFeature, ...] = ()
    hands_clear_m: float | None = None
    """The release clause's ``R``: every TCP must be farther than this from the release point."""
    release_point_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The point ``hands_clear_m`` is measured from, in the tool's frame (see the scene YAMLs)."""

    @property
    def fixture_names(self) -> tuple[str, ...]:
        return self.alternative_fixture_names or (self.fixture_name,)

    @property
    def tool_cfg(self) -> SceneEntityCfg:
        return SceneEntityCfg(self.tool_name)

    @property
    def fixture_cfg(self) -> SceneEntityCfg:
        return SceneEntityCfg(self.fixture_name)

    @property
    def contact_sensor_cfg(self) -> SceneEntityCfg:
        return SceneEntityCfg(self.contact_sensor_name)


def _force_magnitude(sensor) -> torch.Tensor:
    force = as_torch(sensor.data.force_matrix_w)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    return magnitude.reshape(magnitude.shape[0], -1).amax(dim=-1)


def _filtered_force_magnitude(env, sensor_name: str, sensor) -> torch.Tensor:
    """Return the RSS ten-control-step maximum without double sampling."""
    current = _force_magnitude(sensor)
    episode_steps = getattr(env, "episode_length_buf", None)
    if episode_steps is None:
        return current
    steps = as_torch(episode_steps).to(device=current.device, dtype=torch.long)
    histories = getattr(env, _FORCE_HISTORY_ATTR, None)
    if histories is None:
        histories = {}
        setattr(env, _FORCE_HISTORY_ATTR, histories)
    state = histories.get(sensor_name)
    if state is None or state[0].shape[0] != current.shape[0]:
        state = (
            torch.zeros(
                (current.shape[0], _FORCE_WINDOW_STEPS),
                device=current.device,
                dtype=current.dtype,
            ),
            torch.full_like(steps, -1),
        )
        histories[sensor_name] = state
    history, last_steps = state
    changed = steps != last_steps
    reset = steps < last_steps
    if torch.any(reset):
        history[reset] = 0
    if torch.any(changed):
        prior = history[changed].clone()
        history[changed, :-1] = prior[:, 1:]
        history[changed, -1] = current[changed]
        last_steps[changed] = steps[changed]
    return history.amax(dim=-1)


def _world_pose(asset_or_frame) -> torch.Tensor:
    """Read either a rigid asset pose or an Arena object-reference frame pose."""
    if hasattr(asset_or_frame, "data"):
        position = as_torch(asset_or_frame.data.root_link_pos_w)
        orientation = as_torch(asset_or_frame.data.root_link_quat_w)
    else:
        position, orientation = (
            as_torch(value) for value in asset_or_frame.get_world_poses()
        )
    return torch.cat((position, orientation), dim=-1)


def _finite_rows(*poses: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return which rows are finite in every pose, and the poses with the others replaced.

    A tool that the solver throws to a non-finite state has not been hung: its
    rows score as incomplete instead of failing the whole episode's scoring.
    """
    finite = torch.stack([torch.isfinite(pose).all(dim=-1) for pose in poses]).all(dim=0)
    identity = torch.zeros_like(poses[0][0])
    identity[6] = 1.0
    return finite, [torch.where(finite.unsqueeze(-1), pose, identity) for pose in poses]


def _contained_in_fixture_box(goal: HangingGoalSpec, tool_pose: torch.Tensor, fixture_pose: torch.Tensor) -> torch.Tensor:
    assert goal.containment_min_xyz is not None and goal.containment_max_xyz is not None
    # RSS region_bounds translates world-axis bounds by the fixture origin;
    # it does not rotate them. A named landmark follows the tool's orientation.
    offset = _world_point(goal.containment_point_xyz, tool_pose) - fixture_pose[:, :3]
    minimum = torch.as_tensor(goal.containment_min_xyz, device=offset.device, dtype=offset.dtype)
    maximum = torch.as_tensor(goal.containment_max_xyz, device=offset.device, dtype=offset.dtype)
    return torch.all((offset >= minimum) & (offset <= maximum), dim=-1)


def tool_hanging_goal_completion(env, *, goals: tuple[HangingGoalSpec, ...]) -> torch.Tensor:
    """Return the current same-step completion state for every hanging goal."""

    completions: list[torch.Tensor] = []
    for goal in goals:
        tool = env.scene[goal.tool_name]
        if goal.containment_min_xyz is not None:
            finite, (tool_pose, fixture_pose) = _finite_rows(
                _world_pose(tool), _world_pose(env.scene[goal.fixture_name])
            )
            geometry_complete = _contained_in_fixture_box(goal, tool_pose, fixture_pose)
        else:
            assert goal.loop is not None and goal.rod is not None
            finite, (loop_pose, rod_pose) = _finite_rows(
                _world_pose(env.scene[goal.loop_owner_name]),
                _world_pose(env.scene[goal.rod_owner_name]),
            )
            if goal.predicate == "threaded":
                predicate = threaded
            elif goal.predicate == "encloses":
                predicate = encloses
            else:
                raise ValueError(
                    f"unsupported tool-hanging predicate {goal.predicate!r}"
                )
            geometry_complete = torch.stack(
                [
                    predicate(loop, goal.rod, loop_pose, rod_pose)
                    for loop in (goal.loop, *goal.alternative_loops)
                ],
                dim=-1,
            ).any(dim=-1)
        geometry_complete = geometry_complete & finite
        sensor = env.scene[goal.contact_sensor_name]
        has_contact = (
            _filtered_force_magnitude(env, goal.contact_sensor_name, sensor)
            >= goal.contact_force_n
        )
        speed = torch.linalg.vector_norm(as_torch(tool.data.root_lin_vel_w), dim=-1)
        is_settled = speed <= goal.speed_m_s
        completions.append(geometry_complete & has_contact & is_settled)
    if not completions:
        raise ValueError("tool_hanging requires at least one goal")
    return torch.stack(completions, dim=-1)


def tool_hanging_success(env, *, goals: tuple[HangingGoalSpec, ...]) -> torch.Tensor:
    """Return success only when every tool completes all clauses in one step."""

    return torch.all(tool_hanging_goal_completion(env, goals=goals), dim=-1)


def _tcp_positions(env, hands: tuple[HandTcp, ...]) -> torch.Tensor:
    """Every hand's tool-centre position, shape ``(num_envs, len(hands), 3)``."""
    positions = []
    for hand in hands:
        robot = env.scene[hand.robot_name]
        names = list(robot.data.body_names)
        if names.count(hand.body_name) != 1:
            raise ValueError(f"{hand.robot_name} must have exactly one body named {hand.body_name!r}")
        index = names.index(hand.body_name)
        body_position = as_torch(robot.data.body_link_pos_w)[:, index]
        body_quaternion = as_torch(robot.data.body_link_quat_w)[:, index]
        offset = torch.as_tensor(hand.offset_xyz, device=body_position.device, dtype=body_position.dtype)
        positions.append(body_position + math_utils.quat_apply(body_quaternion, offset.expand_as(body_position)))
    return torch.stack(positions, dim=1)


def tool_hanging_hands_clear(
    env, *, goals: tuple[HangingGoalSpec, ...], hands: tuple[HandTcp, ...]
) -> torch.Tensor:
    """Per goal, whether no hand's TCP is within ``hands_clear_m`` of the tool's release point.

    The RSS version's ``(> (hands-clear TOOL) R)``: the distance from the tool to the nearest
    tool centre of either arm. A goal without ``hands_clear_m`` has no release clause.
    """
    tcps = _tcp_positions(env, hands) if hands else None
    clear: list[torch.Tensor] = []
    for goal in goals:
        tool_pose = _world_pose(env.scene[goal.tool_name])
        if goal.hands_clear_m is None or tcps is None:
            clear.append(torch.ones(tool_pose.shape[0], dtype=torch.bool, device=tool_pose.device))
            continue
        point = _world_point(goal.release_point_xyz, tool_pose)
        distance = torch.linalg.vector_norm(tcps - point[:, None, :], dim=-1).amin(dim=-1)
        clear.append(distance > goal.hands_clear_m)
    return torch.stack(clear, dim=-1)


def tool_hanging_released_completion(
    env, *, goals: tuple[HangingGoalSpec, ...], hands: tuple[HandTcp, ...]
) -> torch.Tensor:
    """Per goal, every clause of the RSS version's goal this step: hung, at rest, and let go."""
    return tool_hanging_goal_completion(env, goals=goals) & tool_hanging_hands_clear(
        env, goals=goals, hands=hands
    )


def tool_hanging_held_success(
    env, *, goals: tuple[HangingGoalSpec, ...], hands: tuple[HandTcp, ...], hold_s: float
) -> torch.Tensor:
    """Success once every goal, release included, has held for ``hold_s`` of simulated time.

    The RSS version wraps each tool's clause in ``(hold-for 1.0 ...)`` and ends the episode on
    success; its ``HoldFor`` counts held time from the first step the clause held. Both tools of
    a level-2 scene holding for the last second is the same as their conjunction holding for it,
    so one streak per environment serves. Evaluated once per
    control step; a repeated call within a step reads the stored streak.
    """
    complete = torch.all(tool_hanging_released_completion(env, goals=goals, hands=hands), dim=-1)
    steps = as_torch(env.episode_length_buf).to(device=complete.device, dtype=torch.long)
    state = getattr(env, _HOLD_STATE_ATTR, None)
    if state is None or state[0].shape[0] != complete.shape[0]:
        state = (torch.zeros_like(steps), torch.full_like(steps, -1))
        setattr(env, _HOLD_STATE_ATTR, state)
    streak, last_steps = state
    reset = steps < last_steps
    streak[reset] = 0
    changed = steps != last_steps
    streak[changed] = torch.where(complete[changed], streak[changed] + 1, torch.zeros_like(streak[changed]))
    last_steps[changed] = steps[changed]
    # As RSS's HoldFor: held time runs from the first step the clause held, so N steps hold (N - 1) dt.
    held_s = (streak - 1).clamp(min=0).to(torch.float64) * float(env.step_dt)
    return (streak > 0) & (held_s >= float(hold_s) - 1.0e-9)
