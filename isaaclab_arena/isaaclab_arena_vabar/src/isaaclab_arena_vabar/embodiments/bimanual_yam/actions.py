"""Deterministic joint actions for both YAM articulations."""

import torch
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

from .config import (
    ARM_JOINT_NAMES,
    CRANK_4310,
    GRIPPER_CLOSED_POSITION,
    YamHandSpec,
)


class ContinuousJointPositionZeroToOneAction(BinaryJointPositionAction):
    """Map 0=open and 1=closed continuously onto the YAM finger joint."""

    def process_actions(self, actions: torch.Tensor) -> None:
        closedness = torch.nan_to_num(
            actions, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        self._raw_actions[:] = closedness
        self._processed_actions = self._open_command + closedness * (
            self._close_command - self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class ContinuousJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    class_type: type[ContinuousJointPositionZeroToOneAction] = (
        ContinuousJointPositionZeroToOneAction
    )


class FiniteJointPositionAction(JointPositionAction):
    """Keep absolute YAM targets finite and inside the soft joint limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        super().process_actions(actions)
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.where(
            torch.isfinite(self._processed_actions),
            self._processed_actions,
            default,
        )
        self._processed_actions = torch.maximum(
            torch.minimum(target, limits[..., 1]), limits[..., 0]
        )


@configclass
class FiniteJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[FiniteJointPositionAction] = FiniteJointPositionAction


def _arm_action(asset_name: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


def gripper_action(
    asset_name: str, hand: YamHandSpec = CRANK_4310
) -> ContinuousJointPositionZeroToOneActionCfg:
    """One continuous 0=open, 1=closed command over ``hand``'s driven fingers."""
    return ContinuousJointPositionZeroToOneActionCfg(
        asset_name=asset_name,
        joint_names=list(hand.driven_joints),
        open_command_expr={name: hand.open_position for name in hand.driven_joints},
        close_command_expr={
            name: GRIPPER_CLOSED_POSITION for name in hand.driven_joints
        },
    )


@configclass
class IndustrialBimanualYamActionsCfg:
    """Left arm/gripper followed by right arm/gripper in policy order."""

    left_arm_action: ActionTermCfg = _arm_action("left_robot")
    left_gripper_action: ActionTermCfg = gripper_action("left_robot")
    right_arm_action: ActionTermCfg = _arm_action("right_robot")
    right_gripper_action: ActionTermCfg = gripper_action("right_robot")
