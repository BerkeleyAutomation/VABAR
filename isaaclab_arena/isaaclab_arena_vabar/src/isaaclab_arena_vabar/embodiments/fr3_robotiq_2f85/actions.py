"""Actions for the industrial FR3 Robotiq embodiments."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass
from .config import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSED_ANGLE,
    GRIPPER_JOINT_NAME,
)


@configclass
class IndustrialFr3RobotiqActionsCfg:
    """Seven ordered absolute arm targets and a zero-to-one gripper command."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        preserve_order=True,
        use_default_offset=False,
        scale=GRIPPER_CLOSED_ANGLE,
        offset=0.0,
    )
