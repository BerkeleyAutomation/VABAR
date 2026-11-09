"""Arena observation/action adapters for the supported GaP robot profiles."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from isaaclab_arena_gap.gap_wire import GapObsAdapter, validity_flag
from isaaclab_arena_gap.msgpack_transport import GapProtocolError
from isaaclab_arena_gap.yam_description import YAM_CRANK_4310_WIDE, YAM_LINEAR_4310, YAM_CRANK_4310

_DEPTH = "distance_to_image_plane"


class RobotAdapter(GapObsAdapter):
    """Shared RGB-D and camera-frame encoding."""

    def __init__(
        self,
        *,
        camera: str,
        bridge_camera: str,
        wrist_camera: str = "",
        wrist_bridge_camera: str = "",
        aux_camera: str = "",
        aux_bridge_camera: str = "",
        extra_cameras: dict[str, str] | None = None,
    ) -> None:
        self.cameras = tuple(
            pair
            for pair in (
                (camera, bridge_camera),
                (wrist_camera, wrist_bridge_camera),
                (aux_camera, aux_bridge_camera),
                *(extra_cameras or {}).items(),
            )
            if pair[0]
        )

    @staticmethod
    def _environment_origin(unwrapped: Any, env_id: int, like: Any) -> Any:
        origins = getattr(unwrapped.scene, "env_origins", None)
        return (
            like.new_zeros((1, 3)) if origins is None else origins[env_id : env_id + 1]
        )

    @classmethod
    def _camera(cls, unwrapped: Any, name: str, env_id: int) -> dict[str, Any]:
        from isaaclab.utils.math import matrix_from_quat

        camera = unwrapped.scene[name]
        row = slice(env_id, env_id + 1)
        position = camera.data.pos_w[row] - cls._environment_origin(
            unwrapped, env_id, camera.data.pos_w
        )
        quaternion = camera.data.quat_w_ros[row]
        pose = position.new_zeros((4, 4))
        pose[3, 3] = 1.0
        pose[:3, :3] = matrix_from_quat(quaternion)[0]
        pose[:3, 3] = position[0]
        output = camera.data.output
        if _DEPTH not in output:
            raise GapProtocolError(f"camera {name!r} does not render {_DEPTH!r}")
        rgb = np.ascontiguousarray(
            output["rgb"][env_id, ..., :3].cpu().numpy(), dtype=np.uint8
        )
        depth = np.ascontiguousarray(
            np.nan_to_num(
                output[_DEPTH][env_id].squeeze(-1).cpu().numpy(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dtype=np.float32,
        )
        if rgb.shape[:2] != depth.shape:
            raise GapProtocolError(
                f"RGB and depth resolutions differ for {name!r}: rgb={rgb.shape[:2]}, depth={depth.shape}"
            )
        return {
            "images": {"rgb": rgb},
            "depth_data": depth,
            "intrinsics": {
                "left": {
                    "intrinsics_matrix": camera.data.intrinsic_matrices[env_id]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                }
            },
            "pose_mat": pose.cpu().numpy().astype(np.float32),
        }

    def arm_base_position(self, env: Any, env_id: int = 0) -> tuple[float, ...]:
        unwrapped = env.unwrapped
        robot = self._robot(unwrapped)
        row = slice(env_id, env_id + 1)
        position = robot.data.root_pos_w[row] - self._environment_origin(
            unwrapped, env_id, robot.data.root_pos_w
        )
        return tuple(float(value) for value in position[0].cpu())

    @staticmethod
    def _robot(unwrapped: Any) -> Any:
        raise NotImplementedError

    def _frame(self, env: Any, env_id: int, state: list[float]) -> dict[str, Any]:
        frame = {"timestamp": time.time(), "left": {"joint_pos": state}}
        frame.update(
            {
                bridge_name: self._camera(env.unwrapped, camera_name, env_id)
                for camera_name, bridge_name in self.cameras
            }
        )
        return frame


class Fr3Adapter(RobotAdapter):
    action_dim = 8
    _closed = 0.8

    @staticmethod
    def _robot(unwrapped: Any) -> Any:
        return unwrapped.scene["robot"]

    def __init__(self, **cameras: Any) -> None:
        super().__init__(
            camera=cameras.get("camera") or "exterior_cam",
            bridge_camera=cameras.get("bridge_camera") or "robot0_robotview",
            wrist_camera=cameras.get("wrist_camera", ""),
            wrist_bridge_camera=cameras.get("wrist_bridge_camera") or "eye_in_hand",
            aux_camera=cameras.get("aux_camera", ""),
            aux_bridge_camera=cameras.get("aux_bridge_camera") or "side",
            extra_cameras=cameras.get("extra_cameras"),
        )

    @staticmethod
    def _state(env: Any, env_id: int) -> tuple[np.ndarray, float]:
        robot = env.unwrapped.scene["robot"]
        joints = robot.data.joint_pos[env_id]
        finger = float(joints[7].cpu()) if joints.shape[0] > 7 else 0.0
        return joints[:7].cpu().numpy(), finger

    def build_obs_frame(
        self, env: Any, observation: dict[str, Any], env_id: int
    ) -> dict[str, Any]:
        del observation
        joints, finger = self._state(env, env_id)
        opened = float(np.clip(1.0 - finger / self._closed, 0.0, 1.0))
        frame = self._frame(env, env_id, [*map(float, joints), opened])
        velocity = self._robot(env.unwrapped).data.joint_vel[env_id].cpu().numpy()
        finger_velocity = float(velocity[7]) if velocity.shape[0] > 7 else 0.0
        frame["left"]["joint_vel"] = [
            *map(float, velocity[:7]),
            -finger_velocity / self._closed,
        ]
        robot = self._robot(env.unwrapped)
        frame["passive_joints"] = {
            "names": list(robot.data.joint_names[8:]),
            "positions": robot.data.joint_pos[env_id, 8:].cpu().numpy(),
            "velocities": velocity[8:].copy(),
        }
        return frame

    def decode_action(
        self, action: dict[str, Any], env: Any, env_id: int = 0
    ) -> np.ndarray:
        del env, env_id
        joints, opened = self.parse_wire_action(action)
        return np.asarray([*joints, 1.0 - opened], dtype=np.float32)

    def hold_action(self, env: Any, env_id: int = 0) -> np.ndarray:
        joints, finger = self._state(env, env_id)
        closed = float(np.clip(finger / self._closed, 0.0, 1.0))
        return np.asarray([*joints, closed], dtype=np.float32)


class YamAdapter(RobotAdapter):
    action_dim = 7
    _finger_open = YAM_CRANK_4310.open_joint_m
    _finger_joint = YAM_CRANK_4310.finger_joint_name
    _default_wrist_camera = "{arm}_wrist_camera"

    def __init__(self, **cameras: Any) -> None:
        configured_arms = cameras.get("arms")
        arms = tuple(configured_arms) if configured_arms else ("left",)
        if not 1 <= len(arms) <= 2 or any(arm not in ("left", "right") for arm in arms):
            raise ValueError(
                f"YAM arms must contain one or two of 'left'/'right', got {arms!r}"
            )
        if len(set(arms)) != len(arms):
            raise ValueError(f"YAM arms must be unique, got {arms!r}")
        self._arms = arms
        self._arm = arms[0]
        self._robot_name = f"{self._arm}_robot"
        super().__init__(
            camera=cameras.get("camera") or "top_camera",
            bridge_camera=cameras.get("bridge_camera") or "overhead",
            wrist_camera=cameras.get("wrist_camera")
            or self._default_wrist_camera.format(arm=self._arm),
            wrist_bridge_camera=cameras.get("wrist_bridge_camera")
            or "eye_in_hand_left",
            aux_camera=cameras.get("aux_camera", ""),
            aux_bridge_camera=cameras.get("aux_bridge_camera") or "side",
            extra_cameras=cameras.get("extra_cameras"),
        )

    def _robot(self, unwrapped: Any) -> Any:
        return unwrapped.scene[self._robot_name]

    @classmethod
    def _state(
        cls, env: Any, robot_name: str, env_id: int
    ) -> tuple[np.ndarray, float]:
        try:
            robot = env.unwrapped.scene[robot_name]
            indices = [
                robot.data.joint_names.index(f"joint{index}") for index in range(1, 7)
            ]
            finger_index = robot.data.joint_names.index(cls._finger_joint)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"YAM scene is missing the {robot_name!r} arm contract"
            ) from error
        return (
            robot.data.joint_pos[env_id, indices].cpu().numpy(),
            float(robot.data.joint_pos[env_id, finger_index].cpu()),
        )

    @classmethod
    def _velocity(cls, env: Any, robot_name: str, env_id: int) -> np.ndarray:
        robot = env.unwrapped.scene[robot_name]
        names = robot.data.joint_names
        indices = [names.index(f"joint{index}") for index in range(1, 7)]
        finger = names.index(cls._finger_joint)
        velocities = robot.data.joint_vel[env_id]
        return np.asarray([
            *(float(velocities[index].cpu()) for index in indices),
            float(velocities[finger].cpu()) / cls._finger_open,
        ])

    def action_dim_for(self, env: Any, env_id: int = 0) -> int:
        del env_id
        width = int(
            getattr(env.unwrapped.action_manager, "total_action_dim", self.action_dim)
        )
        if width not in (7, 14):
            raise ValueError(f"YAM action layout must have width 7 or 14, got {width}")
        return width

    def command_action_dim_for(self, env: Any, env_id: int = 0) -> int:
        del env, env_id
        return 7 * len(self._arms)

    def command_action_slices_for(self, env: Any, env_id: int = 0) -> dict[str, slice]:
        width = self.action_dim_for(env, env_id)
        wire_arms = ("left", "right")
        slices: dict[str, slice] = {}
        for wire_arm, physical_arm in zip(
            wire_arms[: len(self._arms)], self._arms, strict=True
        ):
            offset = 0 if physical_arm == "left" else 7
            if offset + 7 > width:
                raise ValueError(
                    f"the {physical_arm} YAM arm requires a bimanual action layout"
                )
            slices[wire_arm] = slice(offset, offset + 7)
        return slices

    def gripper_action_indices(self, env: Any, env_id: int = 0) -> tuple[int, ...]:
        return tuple(
            block.stop - 1
            for block in self.command_action_slices_for(env, env_id).values()
        )

    def arm_base_positions(
        self, env: Any, env_id: int = 0
    ) -> tuple[tuple[float, float, float], ...]:
        unwrapped = env.unwrapped
        bases = []
        for arm in self._arms:
            robot = unwrapped.scene[f"{arm}_robot"]
            row = slice(env_id, env_id + 1)
            position = robot.data.root_pos_w[row] - self._environment_origin(
                unwrapped, env_id, robot.data.root_pos_w
            )
            bases.append(tuple(float(value) for value in position[0].cpu()))
        return tuple(bases)

    def arm_base_position(
        self, env: Any, env_id: int = 0
    ) -> tuple[float, float, float]:
        return self.arm_base_positions(env, env_id)[0]

    def _hold(self, env: Any, robot_name: str, env_id: int) -> np.ndarray:
        joints, finger = self._state(env, robot_name, env_id)
        opened = float(np.clip(finger / self._finger_open, 0.0, 1.0))
        return np.asarray([*joints, 1.0 - opened], dtype=np.float32)

    def build_obs_frame(
        self, env: Any, observation: dict[str, Any], env_id: int
    ) -> dict[str, Any]:
        del observation
        states = []
        for arm in self._arms:
            joints, finger = self._state(env, f"{arm}_robot", env_id)
            opened = float(np.clip(finger / self._finger_open, 0.0, 1.0))
            states.append([*map(float, joints), opened])
        frame = self._frame(env, env_id, states[0])
        frame["left"]["joint_vel"] = self._velocity(
            env, f"{self._arms[0]}_robot", env_id
        )
        if len(states) == 2:
            frame["right"] = {"joint_pos": states[1]}
            frame["right"]["joint_vel"] = self._velocity(
                env, f"{self._arms[1]}_robot", env_id
            )
        return frame

    def decode_action(
        self, action: dict[str, Any], env: Any, env_id: int = 0
    ) -> np.ndarray:
        out = self.hold_action(env, env_id)
        for wire_arm, action_slice in self.command_action_slices_for(
            env, env_id
        ).items():
            block = action.get(wire_arm)
            joints, opened = self.parse_wire_action({"left": block}, expected_arm_dof=6)
            out[action_slice] = np.asarray([*joints, 1.0 - opened], dtype=np.float32)
        return out

    def hold_action(self, env: Any, env_id: int = 0) -> np.ndarray:
        names = ("left_robot", "right_robot")[: self.action_dim_for(env, env_id) // 7]
        return np.concatenate([self._hold(env, name, env_id) for name in names])


class BimanualYamAdapter(YamAdapter):
    """Expose both YAM arms with independent command validity."""

    def build_obs_frame(self, env, observation, env_id=0):
        frame = super().build_obs_frame(env, observation, env_id)
        joints, finger = self._state(env, "right_robot", env_id)
        frame["right"] = {
            "joint_pos": [
                *map(float, joints),
                float(np.clip(finger / self._finger_open, 0, 1)),
            ]
        }
        frame["right"]["joint_vel"] = self._velocity(env, "right_robot", env_id)
        return frame

    def gripper_action_indices(self, env, env_id=0):
        return (6, 13)

    def decode_action(self, action, env, env_id=0):
        return self.decode_validated_action(action, env, env_id, strict=False)

    def base_poses(self, env, env_id=0):
        poses = []
        for name in ("left_robot", "right_robot"):
            robot = env.unwrapped.scene[name]
            origin = self._environment_origin(
                env.unwrapped, env_id, robot.data.root_pos_w
            )[0]
            poses.append(
                (
                    (robot.data.root_pos_w[env_id] - origin).cpu().tolist(),
                    robot.data.root_quat_w[env_id].cpu().tolist(),
                )
            )
        return poses

    def decode_validated_action(self, action, env, env_id=0, *, strict=True):
        row = self.hold_action(env, env_id).copy()
        if row.shape != (14,):
            raise GapProtocolError("bimanual YAM requires 14 action channels")
        for i, name in enumerate(("left", "right")):
            raw = action.get(name)
            raw = dict(raw) if isinstance(raw, dict) else {}
            arm_valid = validity_flag(raw, "arm_valid", not strict)
            grip_valid = validity_flag(raw, "gripper_valid", not strict)
            offset = i * 7
            if not arm_valid:
                raw["joint_pos"] = row[offset : offset + 6]
            if not grip_valid:
                raw["gripper"] = 1 - row[offset + 6]
            joints, opened = self.parse_wire_action({"left": raw}, expected_arm_dof=6)
            row[offset : offset + 7] = [*joints, 1 - opened]
        return row


class WideBimanualYamAdapter(BimanualYamAdapter):
    """Both YAMs on the USD's ``crank_4310_wide`` gripper variant."""

    _finger_open = YAM_CRANK_4310_WIDE.open_joint_m


class LinearBimanualYamAdapter(BimanualYamAdapter):
    """Both YAMs on the USD's ``linear_4310`` gripper variant."""

    _finger_open = YAM_LINEAR_4310.open_joint_m
    _finger_joint = YAM_LINEAR_4310.finger_joint_name


def make_robot_adapter(profile: str, **cameras: Any) -> GapObsAdapter:
    adapters = {
        "fr3": Fr3Adapter,
        "yam_bimanual": BimanualYamAdapter,
        "yam_bimanual_wide": WideBimanualYamAdapter,
        "yam_bimanual_linear": LinearBimanualYamAdapter,
    }
    try:
        return adapters[profile](**cameras)
    except KeyError as error:
        raise ValueError(
            f"unsupported robot_profile {profile!r}; expected one of {tuple(adapters)}"
        ) from error
