"""GaP's narrow robot I/O over the existing Arena observation/action bridge.

The connector receives :class:`RobotIOEnv`; this driver alone sees the bridge
server. The bridge still owns Arena episode stepping and benchmark scoring.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    from gap.robot_io import (
        CameraSample,
        JointPositionCommand,
        RobotSample,
        validate_command,
    )
except ImportError:  # AUTOLab v1.0's pinned GaP predates gap.robot_io
    from isaaclab_arena_gap.robot_io_types import (
        CameraSample,
        JointPositionCommand,
        RobotSample,
        validate_command,
    )


def _joint_names(block: dict) -> tuple[str, ...]:
    return tuple(name.decode("utf-8") if isinstance(name, bytes) else str(name)
                 for name in _field(block, "names") or ())


def _field(mapping: Any, name: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    return mapping.get(name, mapping.get(name.encode()))


class ArenaBridgeRobotIO:
    """Named arm joints and virtual gripper positions over the Arena bridge wire."""

    def __init__(
        self,
        bridge: Any,
        *,
        control_freq: float,
        camera_names: tuple[str, ...],
        arm_dof: int = 7,
        wire_arms: tuple[str, ...] = ("left",),
        joint_prefix: str = "fr3",
        wait_timeout_s: float = 60.0,
        passive_joints: bool = False,
    ):
        self._bridge = bridge
        self._period = 1.0 / float(control_freq)
        self._wait_timeout_s = float(wait_timeout_s)
        self._camera_names = camera_names
        self.arm_dof = int(arm_dof)
        self.wire_arms = tuple(wire_arms)
        self.joint_names = tuple(
            name
            for arm in self.wire_arms
            for name in (
                *(f"{arm}/{joint_prefix}_joint{index}" for index in range(1, arm_dof + 1)),
                f"{arm}/gripper_open_fraction",
            )
        )
        self._last_wire: object | None = None
        self._sequence = -1
        self._active_joint_count = len(self.joint_names)
        self._passive_joint_names = ()
        if passive_joints:
            self.observe()  # Wait for the first joint sample before fixing the layout.
            block = _field(self._last_wire, "passive_joints")
            if not isinstance(block, dict):
                raise ValueError("Arena observation omits the passive joint state")
            self._passive_joint_names = _joint_names(block)
            self.joint_names += self._passive_joint_names

    @property
    def control_period_s(self) -> float:
        return self._period

    def get_integration_failure(self) -> Exception | None:
        return self._bridge.get_integration_failure()

    def observe(self) -> RobotSample:
        bridge = self._bridge
        deadline = time.monotonic() + self._wait_timeout_s
        while True:
            failure = bridge.get_integration_failure()
            if failure is not None:
                raise failure
            wire = bridge.server.latest_observation
            if isinstance(wire, dict):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Arena bridge has no robot observation")
            time.sleep(0.002)
        if wire is not self._last_wire:
            self._sequence += 1
            self._last_wire = wire
        position_rows = [
            np.asarray(_field(_field(wire, arm), "joint_pos"), dtype=np.float64).copy()
            for arm in self.wire_arms
        ]
        for row in position_rows:
            row[-1] = np.clip(row[-1], 0.0, 1.0)
        positions = np.concatenate(position_rows)
        velocities = np.concatenate(
            [np.asarray(_field(_field(wire, arm), "joint_vel"), dtype=np.float64)
             for arm in self.wire_arms]
        )
        if self._passive_joint_names:
            block = _field(wire, "passive_joints")
            if _joint_names(block) != self._passive_joint_names:
                raise ValueError("Arena passive joint layout changed during the episode")
            positions = np.r_[positions, _field(block, "positions")]
            velocities = np.r_[velocities, _field(block, "velocities")]
        cameras: dict[str, CameraSample] = {}
        for name in self._camera_names:
            block = _field(wire, name)
            if not isinstance(block, dict):
                continue
            images = _field(block, "images")
            rgb = _field(images, "rgb")
            depth = _field(block, "depth_data")
            calibration = _field(_field(block, "intrinsics"), "left")
            cameras[name] = CameraSample(
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth_m=None if depth is None else np.asarray(depth, dtype=np.float32),
                intrinsics=np.asarray(_field(calibration, "intrinsics_matrix")),
                pose=np.asarray(_field(block, "pose_mat")),
                timestamp_s=float(_field(wire, "timestamp")),
            )
        return RobotSample(
            sequence=self._sequence,
            timestamp_s=float(_field(wire, "timestamp")),
            joint_positions=positions,
            joint_velocities=velocities,
            cameras=cameras,
        )

    def exchange(self, command: JointPositionCommand) -> RobotSample:
        validate_command(command, len(self.joint_names))
        if command.kp is not None or command.kd is not None:
            raise NotImplementedError("Arena does not support per-command joint gains")
        target = np.asarray(command.positions, dtype=np.float64)
        bridge = self._bridge
        width = self.arm_dof + 1
        # Passive joints are measured for FK; only the arm and gripper driver
        # targets are actuated. RobotIOEnv holds the remaining sample entries.
        rows = target[:self._active_joint_count].reshape(len(self.wire_arms), width)
        # No request may land between the baseline and the publish: it would
        # carry the previous target yet be returned as this command's sample.
        with bridge.server.exchange_lock:
            previous = bridge.server.latest_observation
            if len(rows) == 1:
                bridge.set_robot_io_target(rows[0, : self.arm_dof], float(rows[0, -1]))
            else:
                bridge.set_robot_io_targets(rows[:, : self.arm_dof], rows[:, -1])
        while bridge.server.latest_observation is previous:
            failure = bridge.get_integration_failure()
            if failure is None and hasattr(bridge, "_check_observation_freshness"):
                failure = bridge._check_observation_freshness()
            if failure is not None:
                raise failure
            terminating = getattr(bridge, "_cap_terminating", None)
            if terminating is not None and terminating.is_set():
                raise RuntimeError("Arena bridge closed while waiting for a robot observation")
            time.sleep(0.002)
        bridge._sim_step_count += 1
        return self.observe()
