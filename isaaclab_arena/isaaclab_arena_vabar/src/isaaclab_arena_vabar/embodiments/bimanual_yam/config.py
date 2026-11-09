"""Scene, actuator, reset, and observation configs for the bimanual YAM."""

from __future__ import annotations

from dataclasses import dataclass, replace

import isaaclab.sim as sim_utils
from isaaclab_arena_vabar.assets.registrations.industrial_i2rt_yam import (
    ROBOT_USD_PATH,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
END_EFFECTOR_BODY_NAME = "link_6"
# The asset's ``grasp_site``: the point between the fingertips, in the link 6 frame.
GRASP_SITE_OFFSET = (0.0, 0.0, 0.1347)
# The driven prismatic coordinate increases as the jaws open. This matches the
# YAM's physical convention while policy actions remain 0=open,
# 1=closed through the binary action term below.
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0
HOME_ARM_JOINT_POSITIONS = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class YamHandSpec:
    """One selection of the YAM USD's ``gripper`` variant set.

    ``driven_joints`` take the position action; ``passive_joints`` follow them
    through the asset's mimic constraint and are left undriven.
    """

    variant: str
    driven_joints: tuple[str, ...]
    passive_joints: tuple[str, ...]
    open_position: float
    passive_open_sign: float
    fingertip_shape_tokens: tuple[str, ...]
    fingertip_shapes_per_robot: int
    # Body-label substrings of the moving fingers, and the body carrying the housing.
    finger_body_tokens: tuple[str, ...]
    # The housing body the USB-C contact rig makes collide; None leaves it visual-only.
    housing_body: str | None


CRANK_4310 = YamHandSpec(
    variant="crank_4310",
    driven_joints=(GRIPPER_JOINT_NAME,),
    passive_joints=(PASSIVE_GRIPPER_JOINT_NAME,),
    open_position=GRIPPER_OPEN_POSITION,
    passive_open_sign=-1.0,
    # Six collision spheres per fingertip.
    fingertip_shape_tokens=("/lf_down/Sphere", "/rf_down/Sphere"),
    fingertip_shapes_per_robot=12,
    finger_body_tokens=("finger",),
    housing_body="link_6",
)
# AUTOLab's linear_4310 hand, which its cable and USB-C graphs were written
# against. Both fingers slide 0 (closed) to 47.5 mm (open); each is driven, as
# in AUTOLab's MJCF.
LINEAR_4310 = YamHandSpec(
    variant="linear_4310",
    driven_joints=("joint7", "joint8"),
    passive_joints=(),
    open_position=0.0475,
    passive_open_sign=1.0,
    # AUTOLab's CoACD decomposition: 51 hulls on the left fingertip, 52 on the right.
    fingertip_shape_tokens=("/tip_left/CableConvex_", "/tip_right/CableConvex_"),
    fingertip_shapes_per_robot=103,
    finger_body_tokens=("/tip_left", "/tip_right"),
    # Visual-only, unlike AUTOLab's colliding housing. In Newton 1.5 this
    # triangle-mesh housing reports contacts up to ~200 mm from its surface: with
    # USB-C medium's support hand on the bulkhead it pressed ~30 contacts on the
    # held part, and when the work hand reached its aim station the two housings,
    # ~280 mm apart, "touched", jerked both arms and threw the part out of the
    # grip. The pads stand 135 mm out from the wrist; nothing this task touches
    # reaches the housing.
    housing_body=None,
)
# AUTOLab's tool-hanging hand: the crank hand with each finger opening to 47.5 mm,
# as its yam_tool_hanging MJCF has it, rather than MuJoCo Menagerie's 37.5 mm.
CRANK_4310_WIDE = replace(CRANK_4310, variant="crank_4310_wide", open_position=0.0475)
YAM_HANDS = {hand.variant: hand for hand in (CRANK_4310, CRANK_4310_WIDE, LINEAR_4310)}


def _yam_actuators() -> dict[str, ImplicitActuatorCfg]:
    return {
        "arm_joints_1_3": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            stiffness=80.0,
            damping=6.0,
            effort_limit_sim=28.0,
        ),
        "arm_joint_4": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            stiffness=30.0,
            damping=2.0,
            effort_limit_sim=10.0,
        ),
        "arm_joints_5_6": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"],
            stiffness=30.0,
            damping=2.0,
            effort_limit_sim=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME],
            stiffness=200.0,
            damping=14.0,
            effort_limit_sim=20.0,
        ),
        # The imported equality mirrors the driven left finger onto this joint.
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[PASSIVE_GRIPPER_JOINT_NAME],
            stiffness=0.0,
            damping=0.0,
        ),
    }


def _yam_articulation(
    prim_path: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    joint_pos = dict(zip(ARM_JOINT_NAMES, HOME_ARM_JOINT_POSITIONS, strict=True))
    joint_pos[GRIPPER_JOINT_NAME] = GRIPPER_OPEN_POSITION
    joint_pos[PASSIVE_GRIPPER_JOINT_NAME] = -GRIPPER_OPEN_POSITION
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD_PATH),
            activate_contact_sensors=True,
            # Keep link gravity enabled. The asset's MuJoCo compensation carries
            # the links' weight under Newton.
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=rotation,
            joint_pos=joint_pos,
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=_yam_actuators(),
    )


def select_yam_hand(robot: ArticulationCfg, hand: YamHandSpec) -> None:
    """Spawn ``robot`` with ``hand``'s variant, finger joints, and finger drives."""
    if robot.spawn is None:
        raise RuntimeError("YAM scene has no robot spawner")
    robot.spawn.variants = {"gripper": hand.variant}
    joint_pos = {
        name: value
        for name, value in robot.init_state.joint_pos.items()
        if name in ARM_JOINT_NAMES
    }
    joint_pos.update({name: hand.open_position for name in hand.driven_joints})
    joint_pos.update(
        {
            name: hand.passive_open_sign * hand.open_position
            for name in hand.passive_joints
        }
    )
    robot.init_state.joint_pos = joint_pos
    drive = robot.actuators["gripper"]
    drive.joint_names_expr = list(hand.driven_joints)
    robot.actuators.pop("gripper_passive", None)
    if hand.passive_joints:
        robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=list(hand.passive_joints),
            stiffness=0.0,
            damping=0.0,
        )


# Mount pose for both bases. The tool-sort work surface sits at z = 0.80, so the
# bases stand on it; z = 0.0 would spawn the gripper below the ground plane.
MOUNT_HEIGHT_M = 0.80
MOUNT_X_M = -0.25
# Kept at the rig's real 0.50 m separation. Aligning each base with its own bin
# (left to y=+0.28) was tried and made no measurable difference to reach, so the
# faithful spacing stays.
MOUNT_HALF_SEPARATION_M = 0.25
# Mount orientation, as (w, x, y, z). Measured at spawn with gravity disabled,
# because post-reset poses are meaningless here -- the arm collapses under gravity
# before they can be read. Gripper offset from the base, identity vs candidates:
#   identity (1,0,0,0): dx=+0.242 dz=-0.393  flange up, arm hangs
#   (0,1,0,0):          dx=-0.242 dz=-0.393  still hanging, only mirrored
#   (0,0,1,0):          dx=-0.242 dz=+0.393  upright but facing away from the bins
#   (0,0,0,1):          dx=+0.242 dz=+0.393  upright and facing the work surface
MOUNT_ROTATION = (0.0, 0.0, 0.0, 1.0)


@configclass
class IndustrialBimanualYamSceneCfg:
    """Two independently addressable YAM articulations separated by 0.5 m."""

    left_robot: ArticulationCfg = _yam_articulation(
        "{ENV_REGEX_NS}/LeftRobot",
        (MOUNT_X_M, MOUNT_HALF_SEPARATION_M, MOUNT_HEIGHT_M),
        MOUNT_ROTATION,
    )
    right_robot: ArticulationCfg = _yam_articulation(
        "{ENV_REGEX_NS}/RightRobot",
        (MOUNT_X_M, -MOUNT_HALF_SEPARATION_M, MOUNT_HEIGHT_M),
        MOUNT_ROTATION,
    )


@configclass
class IndustrialBimanualYamEventCfg:
    """Reset each YAM to its independently configured joint defaults."""

    reset_left_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("left_robot"),
        },
    )
    reset_right_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("right_robot"),
        },
    )


@configclass
class IndustrialBimanualYamObservationsCfg:
    """Ordered, side-qualified policy observations for both articulations."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            from .observations import arm_joint_pos, ee_pos, ee_quat, gripper_pos

            for side in ("left", "right"):
                asset_cfg = SceneEntityCfg(f"{side}_robot")
                setattr(
                    self,
                    f"{side}_joint_pos",
                    ObsTerm(func=arm_joint_pos, params={"asset_cfg": asset_cfg}),
                )
                setattr(
                    self,
                    f"{side}_gripper_pos",
                    ObsTerm(func=gripper_pos, params={"asset_cfg": asset_cfg}),
                )
                setattr(
                    self,
                    f"{side}_eef_pos",
                    ObsTerm(func=ee_pos, params={"asset_cfg": asset_cfg}),
                )
                setattr(
                    self,
                    f"{side}_eef_quat",
                    ObsTerm(func=ee_quat, params={"asset_cfg": asset_cfg}),
                )
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
