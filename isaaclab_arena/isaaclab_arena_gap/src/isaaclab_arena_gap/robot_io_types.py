"""The RobotIO wire types, for GaP runtimes that predate ``gap.robot_io``.

AUTOLab's v1.0 pins a GaP revision without that module, but the Arena bridge
client speaks these four names. They mirror ``gap.robot_io`` field for field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSample:
    """One camera capture with calibration in the graph's reference frame."""

    rgb: np.ndarray
    depth_m: np.ndarray | None
    intrinsics: np.ndarray
    pose: np.ndarray
    timestamp_s: float


@dataclass(frozen=True)
class RobotSample:
    """Measured state; camera captures may be older than the joint sample."""

    sequence: int
    timestamp_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    cameras: dict[str, CameraSample]


@dataclass(frozen=True)
class JointPositionCommand:
    """Absolute targets in the driver's joint order."""

    positions: np.ndarray
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None


def validate_command(command: JointPositionCommand, joint_count: int) -> None:
    """Reject ambiguous or unsafe numeric payloads before a driver writes."""
    if joint_count <= 0:
        raise ValueError("joint_count must be positive")
    for name in ("positions", "kp", "kd"):
        value = getattr(command, name)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (joint_count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {joint_count} finite values")
        if name != "positions" and np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative")
