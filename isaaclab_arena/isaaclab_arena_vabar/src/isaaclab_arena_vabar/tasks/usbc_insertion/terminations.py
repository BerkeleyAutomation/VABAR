# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AUTOLab-equivalent scored predicates for USB-C insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.utils.math as math_utils
import torch

from isaaclab_arena_vabar.numeric import as_torch

from .geometry import Bore, measure_insertion


@dataclass
class UsbcGoalSpec:
    """Thresholds, as the source's task document states them."""

    bore: Bore
    depth_min: float
    lateral_max: float
    tilt_max: float | None
    speed_max: float
    plug_name: str
    port_name: str
    tcp_distance_min: float
    require_released: bool

    def __post_init__(self) -> None:
        if self.depth_min <= 0.0 or self.depth_min > self.bore.depth:
            raise ValueError("depth gate must be positive and inside the bore")
        for label, value in (
            ("lateral", self.lateral_max),
            ("speed", self.speed_max),
        ):
            if value <= 0.0:
                raise ValueError(f"{label} gate must be positive")
        if self.tilt_max is not None and self.tilt_max <= 0.0:
            raise ValueError("tilt gate must be positive")


@dataclass
class UsbcWorkHandSpec:
    """Work-hand geometry used by AUTOLab's release predicates."""

    robot_name: str
    tcp_body_name: str
    tcp_offset_xyz: tuple[float, float, float]
    gripper_action_name: str
    gripper_joint_name: str
    span_m: float
    open_joint_m: float
    grasp_width_m: float

    def __post_init__(self) -> None:
        for name in (
            self.robot_name,
            self.tcp_body_name,
            self.gripper_action_name,
            self.gripper_joint_name,
        ):
            if not name:
                raise ValueError("USB-C work-hand names must be non-empty")
        if len(self.tcp_offset_xyz) != 3 or not all(
            math.isfinite(value) for value in self.tcp_offset_xyz
        ):
            raise ValueError("USB-C TCP offset must contain three finite values")
        if min(self.span_m, self.open_joint_m, self.grasp_width_m) <= 0.0:
            raise ValueError("USB-C work-hand dimensions must be positive")


STALL_MARGIN_M = 2.0e-4
GAP_BAND_M = 1.5e-3


def _unique_index(names: list[str], expected: str, kind: str) -> int:
    matches = [index for index, name in enumerate(names) if name == expected]
    if len(matches) != 1:
        raise ValueError(
            f"USB-C work hand must contain exactly one {kind} named {expected!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _release_state(env, plug_position: torch.Tensor, hand: UsbcWorkHandSpec):
    """Return the RSS ``(not gripped, tcp-distance)`` measurements."""
    robot = env.scene[hand.robot_name]
    joint_index = _unique_index(
        list(robot.data.joint_names), hand.gripper_joint_name, "joint"
    )
    body_index = _unique_index(
        list(robot.data.body_names), hand.tcp_body_name, "body"
    )

    measured = as_torch(robot.data.joint_pos)[:, joint_index]
    commanded = env.action_manager.get_term(
        hand.gripper_action_name
    ).processed_actions[:, 0]
    gap = hand.span_m - 2.0 * (hand.open_joint_m - measured)
    lag = measured - commanded
    gripped = (lag > STALL_MARGIN_M) & (
        torch.abs(gap - hand.grasp_width_m) < GAP_BAND_M
    )

    body_position = as_torch(robot.data.body_link_pos_w)[:, body_index]
    body_quaternion = as_torch(robot.data.body_link_quat_w)[:, body_index]
    offset = torch.as_tensor(
        hand.tcp_offset_xyz, device=body_position.device, dtype=body_position.dtype
    ).expand_as(body_position)
    tcp_position = body_position + math_utils.quat_apply(body_quaternion, offset)
    tcp_distance = torch.linalg.vector_norm(plug_position - tcp_position, dim=-1)
    return ~gripped, tcp_distance


def usbc_insertion_seated(
    env, goal: UsbcGoalSpec, work_hand: UsbcWorkHandSpec
) -> torch.Tensor:
    """Evaluate every source goal clause against simulator ground truth."""

    plug = env.scene[goal.plug_name]
    port = env.scene[goal.port_name]
    reading = measure_insertion(
        as_torch(plug.data.root_link_pos_w),
        as_torch(plug.data.root_link_quat_w),
        as_torch(plug.data.root_lin_vel_w),
        as_torch(port.data.root_link_pos_w),
        as_torch(port.data.root_link_quat_w),
        goal.bore,
    )
    released, tcp_distance = _release_state(
        env, as_torch(plug.data.root_link_pos_w), work_hand
    )
    result = (
        (reading.depth >= goal.depth_min)
        & (reading.lateral <= goal.lateral_max)
        & (reading.speed <= goal.speed_max)
        & (tcp_distance > goal.tcp_distance_min)
    )
    if goal.tilt_max is not None:
        result &= reading.tilt <= goal.tilt_max
    return result & released if goal.require_released else result
