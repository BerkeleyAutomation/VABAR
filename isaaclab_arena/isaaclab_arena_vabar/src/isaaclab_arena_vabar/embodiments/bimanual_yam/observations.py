"""Fail-closed ordered observations for either YAM articulation."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .config import (
    ARM_JOINT_NAMES,
    END_EFFECTOR_BODY_NAME,
    GRIPPER_CLOSED_POSITION,
    GRIPPER_JOINT_NAME,
    GRIPPER_OPEN_POSITION,
)


def _robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _index(names: list[str], expected: str, kind: str, side: str) -> int:
    try:
        return names.index(expected)
    except ValueError as error:
        raise ValueError(
            f"{side} is missing required {kind} {expected!r}"
        ) from error


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return arm positions in the declared action order for one side."""

    robot = _robot(env, asset_cfg)
    indices = [
        _index(robot.data.joint_names, name, "joint", asset_cfg.name)
        for name in ARM_JOINT_NAMES
    ]
    return wp.to_torch(robot.data.joint_pos)[:, indices]


def gripper_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_position: float = GRIPPER_OPEN_POSITION,
    joint_name: str = GRIPPER_JOINT_NAME,
) -> torch.Tensor:
    """Return zero-open, one-closed state clamped to the policy range."""

    robot = _robot(env, asset_cfg)
    index = _index(robot.data.joint_names, joint_name, "joint", asset_cfg.name)
    position = wp.to_torch(robot.data.joint_pos)[:, index : index + 1]
    travel = GRIPPER_CLOSED_POSITION - open_position
    closedness = (position - open_position) / travel
    return torch.clamp(closedness, min=0.0, max=1.0)


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the selected side's link_6 world position."""

    robot = _robot(env, asset_cfg)
    index = _index(
        robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name
    )
    return wp.to_torch(robot.data.body_pos_w)[:, index, :]


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the selected side's link_6 world quaternion (w, x, y, z)."""

    robot = _robot(env, asset_cfg)
    index = _index(
        robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name
    )
    return wp.to_torch(robot.data.body_quat_w)[:, index, :]
