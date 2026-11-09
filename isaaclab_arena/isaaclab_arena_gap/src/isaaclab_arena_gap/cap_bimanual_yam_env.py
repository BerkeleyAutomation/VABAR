"""Bimanual bridge transport: two YAM arms on one Arena bridge."""

from __future__ import annotations

import json
import os
import time

import numpy as np
from gap.envs.registry import EnvConfig

from isaaclab_arena_gap.cap_franka_real_env import (
    CapFrankaRealEnv,
    _camera_names_from_env,
    _control_frequency,
)
from isaaclab_arena_gap.yam_description import YAM_BIMANUAL_ARM_HOME


class CapBimanualYamEnv(CapFrankaRealEnv):
    """Keep independent targets and validity for each arm and gripper."""

    def __init__(self, *args, arm_names: tuple[str, str], **kwargs):
        if set(arm_names) != {"left", "right"}:
            raise ValueError("bimanual arm names must be left and right")
        self.arm_names = arm_names
        self._arm_targets = [np.zeros(6), np.zeros(6)]
        self._gripper_targets = [1.0, 1.0]
        self._arm_valid = [False, False]
        self._gripper_valid = [False, False]
        super().__init__(*args, **kwargs)

    def _wire_inputs_locked(self):
        return (
            [q.copy() for q in self._arm_targets],
            list(self._gripper_targets),
            list(self._arm_valid),
            list(self._gripper_valid),
        )

    def _wire_action(self, targets, grippers, arms_valid, grippers_valid):
        return {
            "timestamp": time.time(),
            **{
                name: {
                    "joint_pos": targets[i].tolist(),
                    "gripper": float(np.clip(grippers[i], 0.0, 1.0)),
                    "arm_valid": arms_valid[i],
                    "gripper_valid": grippers_valid[i],
                }
                for i, name in enumerate(self.arm_names)
            },
        }

    def _heartbeat_loop(self):
        # Freshness is checked by the inherited observation guard.
        self._republish_stop.wait()

    def set_robot_io_targets(
        self, arm_targets: np.ndarray, gripper_targets: np.ndarray
    ) -> None:
        """Publish both arms and grippers in a single wire snapshot."""
        arms = np.asarray(arm_targets, dtype=np.float64)
        grippers = np.asarray(gripper_targets, dtype=np.float64)
        if arms.shape != (2, 6) or grippers.shape != (2,):
            raise ValueError("bimanual robot I/O requires two six-joint targets")
        if not np.isfinite(arms).all() or not np.isfinite(grippers).all():
            raise ValueError("bimanual robot I/O targets must be finite")
        if np.any((grippers < 0.0) | (grippers > 1.0)):
            raise ValueError("bimanual gripper targets must be in [0, 1]")
        with self._target_lock:
            self._arm_targets = [row.copy() for row in arms]
            self._gripper_targets = [float(value) for value in grippers]
            self._arm_valid = [True, True]
            self._gripper_valid = [True, True]
            self._publish_action_locked()


def make_env(suite_name="franka_real", task_id=0, camera_names=None, **kwargs):
    poses = json.loads(os.environ["CAP_GAP_ARM_BASE_POSES"])
    if len(poses) != 2:
        raise ValueError("bimanual YAM needs two physical base poses")
    work_arm = os.environ.get("CAP_GAP_WORK_ARM", "left")
    if work_arm not in ("left", "right"):
        raise ValueError("CAP_GAP_WORK_ARM must be left or right")
    arm_names = (work_arm, "right" if work_arm == "left" else "left")
    names = camera_names or _camera_names_from_env() or ["overhead"]
    env = CapBimanualYamEnv(
        arm_names=arm_names, camera_names=names,
        port=int(os.environ.get("GAP_PORT", "9000")),
    )
    return env, EnvConfig(
        arm_dof=6,
        num_arms=2,
        action_mode="absolute_joints",
        control_freq=_control_frequency(),
        home_joints=YAM_BIMANUAL_ARM_HOME,
        arm_bases=((0.0, 0.0, 0.0),) * 2,
        default_cameras=tuple(names),
        # Simulated: `is_real` only stops Connector.go_home from moving, and graphs
        # rely on that retract to clear the overhead view before perceiving again.
        is_real=False,
    )

