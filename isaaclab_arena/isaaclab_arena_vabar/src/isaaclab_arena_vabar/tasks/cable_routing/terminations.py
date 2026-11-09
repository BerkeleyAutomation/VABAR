"""Native-equivalent scored predicate for terminated cable routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp
from isaaclab.utils import math as math_utils

from isaaclab_arena_vabar.numeric import as_torch

from .geometry import capsule_centerline


@dataclass
class TerminatedCableGoal:
    """RSS world-axis bounds relative to each fixture's origin."""

    seat_regions: tuple[tuple[float, float, float, float], ...]
    seat_min_fractions: tuple[float, ...]
    port_region: tuple[float, float, float, float]
    port_min_fraction: float
    min_tcp_distance: float
    max_mean_speed: float


@dataclass
class CableWorkHand:
    """The articulation body whose TCP must clear the routed cable."""

    robot_name: str
    body_name: str
    tcp_offset: tuple[float, float, float]


def terminated_cable_route_success(
    env,
    *,
    peg_names: tuple[str, ...],
    port_name: str,
    goal: TerminatedCableGoal,
    segment_half_lengths: tuple[float, ...],
    seat_frame_z: float,
    work_hand: CableWorkHand,
) -> torch.Tensor:
    """Score at graph completion or the RSS task's natural step horizon.

    ``env.sim.physics_manager.cable_body_state()`` returns the ordered cable
    body indices. ``seat_frame_z`` is the guide region's floor relative to each
    peg origin.
    """

    def points_in_region(points, fixture, region, *, frame_z: float = 0.0):
        fixture_position = as_torch(fixture.data.root_pos_w)
        # The RSS goal region (``goals.bind._region``) translates the declared bounds
        # by RegionHandle.origin_now; fixture orientation does not rotate them.
        local = points - fixture_position[:, None, :]
        x_min, y_min, x_max, y_max = region
        return (
            (local[..., 0] >= x_min)
            & (local[..., 0] <= x_max)
            & (local[..., 1] >= y_min)
            & (local[..., 1] <= y_max)
            & (local[..., 2] >= frame_z)
            & (local[..., 2] <= frame_z + 0.15)
        )

    physics_manager = env.sim.physics_manager
    cable_body_ids, _ = physics_manager.cable_body_state()
    cable_body_q = wp.to_torch(physics_manager.get_state_0().body_q)[
        cable_body_ids
    ].unsqueeze(0)
    cable_centers = cable_body_q[..., :3]
    half_lengths = cable_centers.new_tensor(segment_half_lengths)
    cable_points = capsule_centerline(cable_body_q, half_lengths)

    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for peg_name, region, minimum in zip(
        peg_names, goal.seat_regions, goal.seat_min_fractions
    ):
        fraction = (
            points_in_region(
                cable_points,
                env.scene[peg_name],
                region,
                frame_z=seat_frame_z,
            )
            # Native NumPy averages boolean occupancy in float64. Rounding a
            # declared decimal threshold to float32 can admit one fewer point.
            .to(torch.float64)
            .mean(dim=1)
        )
        result &= fraction >= minimum

    port = env.scene[port_name]
    in_port = points_in_region(cable_points, port, goal.port_region)
    result &= in_port.to(torch.float64).mean(dim=1) >= goal.port_min_fraction

    result &= points_in_region(cable_points[:, -1:], port, goal.port_region)[:, 0]

    tcp_offset = cable_centers.new_tensor(work_hand.tcp_offset)
    robot = env.scene[work_hand.robot_name]
    body_ids, _ = robot.find_bodies(work_hand.body_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"{work_hand.robot_name} must contain exactly one "
            f"{work_hand.body_name} body"
        )
    body_position = as_torch(robot.data.body_link_pos_w)[:, body_ids[0]]
    body_orientation = as_torch(robot.data.body_link_quat_w)[:, body_ids[0]]
    tcp_position_w = body_position + math_utils.quat_apply(
        body_orientation, tcp_offset.expand(env.num_envs, -1)
    )
    closest_tcp_distance = torch.linalg.vector_norm(
        cable_points - tcp_position_w[:, None, :], dim=-1
    ).amin(dim=1)
    result &= closest_tcp_distance > goal.min_tcp_distance

    cable_velocities = wp.to_torch(physics_manager.get_state_0().body_qd)[
        cable_body_ids, :3
    ].unsqueeze(0)
    mean_speed = torch.linalg.vector_norm(cable_velocities, dim=-1).mean(dim=1)
    result &= mean_speed <= goal.max_mean_speed
    request_state = getattr(env, "external_policy_termination_requested", None)
    policy_termination = (
        torch.as_tensor(request_state(), device=result.device, dtype=torch.bool)
        if request_state is not None
        else torch.zeros_like(result)
    )
    result &= policy_termination | (env.episode_length_buf >= env.max_episode_length)
    return result
