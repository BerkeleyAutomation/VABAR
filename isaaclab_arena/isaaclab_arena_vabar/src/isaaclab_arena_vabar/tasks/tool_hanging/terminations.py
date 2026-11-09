"""Tool-hanging completion predicates and the final-state success term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena_vabar.numeric import as_torch

from .geometry import LoopFeature, RodFeature, _world_point, encloses, threaded

_FORCE_WINDOW_STEPS = 10
_FORCE_HISTORY_ATTR = "_tool_hanging_force_history"


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
            geometry_complete = _contained_in_fixture_box(
                goal, _world_pose(tool), _world_pose(env.scene[goal.fixture_name])
            )
        else:
            assert goal.loop is not None and goal.rod is not None
            loop_pose = _world_pose(env.scene[goal.loop_owner_name])
            rod_pose = _world_pose(env.scene[goal.rod_owner_name])
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


def _episode_is_ending(env, like: torch.Tensor) -> torch.Tensor:
    """Whether this step ends the episode: the policy asked to stop, or the time limit is reached."""
    request_state = getattr(env, "external_policy_termination_requested", None)
    policy_termination = (
        torch.as_tensor(request_state(), device=like.device, dtype=torch.bool)
        if request_state is not None
        else torch.zeros_like(like)
    )
    return policy_termination | (as_torch(env.episode_length_buf) >= env.max_episode_length)


def tool_hanging_final_state_success(
    env, *, goals: tuple[HangingGoalSpec, ...]
) -> torch.Tensor:
    """Score the scene the episode ends on, exactly as AUTOLab v1.0 scores tool hanging.

    AUTOLab's suite sets ``(:terminate-on-success false)`` and evaluates its goal on the
    episode's last step: the graph's end plus its settle steps, or the step limit. A goal met
    earlier, mid-insertion with the tool still in hand, does not end the episode. As in
    AUTOLab, nothing checks release: a tool still pressed against its fixture when the step
    limit ends the episode passes if every clause holds.

    The goal is evaluated on every step so the contact-force window keeps its history.
    """
    complete = tool_hanging_success(env, goals=goals)
    return complete & _episode_is_ending(env, complete)
