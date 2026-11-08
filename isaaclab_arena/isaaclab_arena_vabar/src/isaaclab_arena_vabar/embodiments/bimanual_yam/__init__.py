"""Arena embodiment contract for two independently spawned I2RT YAM arms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from isaaclab_arena_vabar.assets.registrations.industrial_i2rt_yam import (
    ROBOT_INSTANCEABLE_USD_PATH,
    ROBOT_USD_PATH,
)
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

from isaaclab_arena_vabar.numeric import is_finite_real

from .actions import IndustrialBimanualYamActionsCfg, gripper_action
from .cameras import YamCameraCfg, linear_hand_wrist
from .config import (
    ARM_JOINT_NAMES,
    CRANK_4310,
    END_EFFECTOR_BODY_NAME,
    LINEAR_4310,
    YAM_HANDS,
    IndustrialBimanualYamEventCfg,
    IndustrialBimanualYamObservationsCfg,
    IndustrialBimanualYamSceneCfg,
    select_yam_hand,
)


class IndustrialBimanualYamEmbodiment(EmbodimentBase):
    """Two canonical I2RT YAM arms with one absolute-joint, ordered action vector."""

    name = "vabar__bimanual_yam"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.DUAL_ARM
    action_config_type = IndustrialBimanualYamActionsCfg

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        left_initial_joint_pose: Sequence[float] | None = None,
        right_initial_joint_pose: Sequence[float] | None = None,
        left_mount_position: Sequence[float] | None = None,
        right_mount_position: Sequence[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        use_tiled_cameras: bool = False,
        use_instanceable_meshes: bool = False,
        robot_usd_path: str | Path = ROBOT_USD_PATH,
        instanceable_robot_usd_path: str | Path = ROBOT_INSTANCEABLE_USD_PATH,
        left_mount_rotation: Sequence[float] | None = None,
        right_mount_rotation: Sequence[float] | None = None,
        arm_stiffness: float | None = None,
        arm_damping: float | None = None,
        gripper_stiffness: float = 200.0,
        gripper_damping: float = 14.0,
        gripper_effort_limit_sim: float | None = 20.0,
        soft_joint_pos_limit_factor: float = 1.0,
        gripper: str = CRANK_4310.variant,
    ):
        if initial_pose is not None:
            raise ValueError(
                "initial_pose is unsupported for the two-root bimanual YAM; "
                "configure each articulation through its scene pose"
            )
        if arm_mode not in (None, ArmMode.DUAL_ARM):
            raise ValueError("arm_mode must be ArmMode.DUAL_ARM for the bimanual YAM")

        left_pose = _validate_joint_pose(
            "left_initial_joint_pose", left_initial_joint_pose
        )
        right_pose = _validate_joint_pose(
            "right_initial_joint_pose", right_initial_joint_pose
        )
        left_rotation = _validate_rotation("left_mount_rotation", left_mount_rotation)
        right_rotation = _validate_rotation(
            "right_mount_rotation", right_mount_rotation
        )
        if arm_stiffness is not None:
            arm_stiffness = _validate_nonnegative("arm_stiffness", arm_stiffness)
        if arm_damping is not None:
            arm_damping = _validate_nonnegative("arm_damping", arm_damping)
        gripper_stiffness = _validate_nonnegative(
            "gripper_stiffness", gripper_stiffness
        )
        gripper_damping = _validate_nonnegative("gripper_damping", gripper_damping)
        if gripper_effort_limit_sim is not None:
            gripper_effort_limit_sim = _validate_nonnegative(
                "gripper_effort_limit_sim", gripper_effort_limit_sim
            )
        if gripper not in YAM_HANDS:
            raise ValueError(
                f"gripper must be one of {sorted(YAM_HANDS)}, got {gripper!r}"
            )
        hand = YAM_HANDS[gripper]
        self.hand = hand
        soft_joint_pos_limit_factor = _validate_unit_interval(
            "soft_joint_pos_limit_factor", soft_joint_pos_limit_factor
        )
        self._robot_usd_path = _validate_usd_path("robot_usd_path", robot_usd_path)
        self._instanceable_robot_usd_path = _validate_usd_path(
            "instanceable_robot_usd_path", instanceable_robot_usd_path
        )
        super().__init__(
            enable_cameras,
            None,
            concatenate_observation_terms,
            arm_mode,
        )
        self.scene_config = IndustrialBimanualYamSceneCfg()
        self.action_config = self.action_config_type()
        self.observation_config = IndustrialBimanualYamObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms
        self.event_config = IndustrialBimanualYamEventCfg()
        self.camera_config = YamCameraCfg() if enable_cameras else None
        self.set_use_tiled_cameras(use_tiled_cameras)
        self.reward_config = None
        self.mimic_env = None
        if hand is not CRANK_4310:
            self._select_hand(hand)

        for robot in (self.scene_config.left_robot, self.scene_config.right_robot):
            for group_name, actuator in robot.actuators.items():
                if not group_name.startswith("arm_"):
                    continue
                if arm_stiffness is not None:
                    actuator.stiffness = arm_stiffness
                if arm_damping is not None:
                    actuator.damping = arm_damping
            gripper = robot.actuators["gripper"]
            gripper.stiffness = gripper_stiffness
            gripper.damping = gripper_damping
            gripper.effort_limit_sim = gripper_effort_limit_sim
            robot.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor

        # Where the two bases stand. Exposed because a scene, not the embodiment,
        # decides how far apart the arms are -- which is what makes a workspace
        # partition (and so a forced handover) possible or impossible.
        if left_mount_position is not None:
            self.scene_config.left_robot.init_state.pos = _validate_mount(
                "left_mount_position", left_mount_position
            )
        if right_mount_position is not None:
            self.scene_config.right_robot.init_state.pos = _validate_mount(
                "right_mount_position", right_mount_position
            )
        if left_rotation is not None:
            self.scene_config.left_robot.init_state.rot = left_rotation
        if right_rotation is not None:
            self.scene_config.right_robot.init_state.rot = right_rotation

        self.set_use_instanceable_meshes(use_instanceable_meshes)

        if self.camera_config is not None:
            self.camera_config.set_robot_mount_positions(
                self.scene_config.left_robot.init_state.pos,
                self.scene_config.right_robot.init_state.pos,
            )

        if left_pose is not None:
            self.scene_config.left_robot.init_state.joint_pos.update(
                dict(zip(ARM_JOINT_NAMES, left_pose, strict=True))
            )
        if right_pose is not None:
            self.scene_config.right_robot.init_state.joint_pos.update(
                dict(zip(ARM_JOINT_NAMES, right_pose, strict=True))
            )

    def _select_hand(self, hand) -> None:
        """Swap both arms onto another variant of the YAM's ``gripper`` set."""
        for side in ("left", "right"):
            select_yam_hand(getattr(self.scene_config, f"{side}_robot"), hand)
            setattr(
                self.action_config,
                f"{side}_gripper_action",
                gripper_action(f"{side}_robot", hand),
            )
            term = getattr(self.observation_config.policy, f"{side}_gripper_pos")
            term.params.update(
                open_position=hand.open_position, joint_name=hand.driven_joints[0]
            )
            # The linear hand carries its D405 on its own gripper body, as AUTOLab's I2RT arm does.
            if hand is LINEAR_4310 and isinstance(self.camera_config, YamCameraCfg):
                setattr(self.camera_config, f"{side}_wrist_camera", linear_hand_wrist(side))

    def set_use_tiled_cameras(self, enabled: bool) -> None:
        """Select tiled camera sensors when cameras are enabled."""
        if not isinstance(enabled, bool):
            raise TypeError("use_tiled_cameras must be a boolean")
        if self.camera_config is not None:
            self.camera_config.use_tiled_camera = enabled

    def set_use_instanceable_meshes(self, enabled: bool) -> None:
        """Select the authored YAM render-instancing variant."""
        if not isinstance(enabled, bool):
            raise TypeError("use_instanceable_meshes must be a boolean")
        usd_path = (
            self._instanceable_robot_usd_path if enabled else self._robot_usd_path
        )
        for robot in (self.scene_config.left_robot, self.scene_config.right_robot):
            if robot.spawn is None:
                raise RuntimeError("YAM scene has no robot spawner")
            robot.spawn.usd_path = str(usd_path)

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME


def _validate_mount(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three position values")
    if not all(is_finite_real(value) for value in values):
        raise ValueError(f"{name} must contain three finite numeric values")
    return tuple(float(value) for value in values)


def _validate_joint_pose(
    name: str, values: Sequence[float] | None
) -> tuple[float, ...] | None:
    if values is None:
        return None
    if len(values) != len(ARM_JOINT_NAMES):
        raise ValueError(f"{name} must contain exactly six arm positions")
    if not all(is_finite_real(value) for value in values):
        raise ValueError(f"{name} must contain six finite numeric values")
    return tuple(float(value) for value in values)


def _validate_rotation(
    name: str, values: Sequence[float] | None
) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four quaternion values")
    if not all(is_finite_real(value) for value in values):
        raise ValueError(f"{name} must contain four finite numeric values")
    rotation = tuple(float(value) for value in values)
    if math.isclose(sum(value * value for value in rotation), 0.0):
        raise ValueError(f"{name} must not be the zero quaternion")
    return rotation


def _validate_nonnegative(name: str, value: float) -> float:
    if not is_finite_real(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _validate_unit_interval(name: str, value: float) -> float:
    value = _validate_nonnegative(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be at most one")
    return value


def _validate_usd_path(name: str, value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


__all__ = [
    "IndustrialBimanualYamActionsCfg",
    "IndustrialBimanualYamEmbodiment",
    "IndustrialBimanualYamEventCfg",
    "IndustrialBimanualYamObservationsCfg",
    "IndustrialBimanualYamSceneCfg",
]
