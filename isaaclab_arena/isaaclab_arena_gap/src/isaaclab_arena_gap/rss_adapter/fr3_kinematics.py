"""The planner's model of Arena's FR3: the Panda's chain, with the FR3's joint stops.

The FR3 shares the Panda's link frames (the same DH parameters), so AUTOLab's
``panda_kinematics.urdf`` describes its chain exactly. Its joint ranges differ,
and joint 6 most: the FR3 turns [0.5445, 4.5169] rad, the Panda [-0.0175, 3.7525].
Arena simulates the FR3 stops, and ``robot.describe_arm`` reports them from the
same asset, so a planner loaded with the Panda's file would refuse configurations
the graph is told the arm can reach, and could plan into ones the arm cannot.
This writes the Panda's file with only the arm joints' ``lower``/``upper`` (and
their soft limits) replaced; frames, inertias, velocities and efforts stay the
Panda's, which keeps AUTOLab's trajectory timing.
"""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

FR3_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (math.radians(-157.20242), math.radians(157.20242)),
    (math.radians(-102.19848), math.radians(102.19848)),
    (math.radians(-166.19786), math.radians(166.19786)),
    (math.radians(-174.29948), math.radians(-8.697499)),
    (math.radians(-160.8006), math.radians(160.8006)),
    (math.radians(31.197552), math.radians(258.79932)),
    (math.radians(-172.79834), math.radians(172.79834)),
)
"""Joint 1-7 position limits [rad] of Arena's FR3 asset
(``vabar__fr3_robotiq_2f85/franka_fr3_v1/Payload/Physics.usda``, stated there in degrees)."""

PANDA_ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def with_joint_limits(
    urdf_text: str, limits: Sequence[tuple[float, float]], joints: Sequence[str] = PANDA_ARM_JOINTS
) -> str:
    """``urdf_text`` with each of ``joints`` limited to the matching ``(lower, upper)`` of ``limits``."""
    if len(limits) != len(joints):
        raise ValueError(f"{len(limits)} limits for {len(joints)} joints")
    root = ET.fromstring(urdf_text)
    by_name = {joint.get("name"): joint for joint in root.iter("joint")}
    for name, (lower, upper) in zip(joints, limits, strict=True):
        joint = by_name.get(name)
        limit = None if joint is None else joint.find("limit")
        if limit is None:
            raise ValueError(f"the URDF has no limited joint {name!r}")
        limit.set("lower", f"{lower:.6f}")
        limit.set("upper", f"{upper:.6f}")
        safety = joint.find("safety_controller")
        if safety is not None:
            safety.set("soft_lower_limit", f"{lower:.6f}")
            safety.set("soft_upper_limit", f"{upper:.6f}")
    return ET.tostring(root, encoding="unicode")


_WRITTEN: dict[Path, Path] = {}


def fr3_kinematics_urdf(panda_urdf: Path) -> Path:
    """A copy of ``panda_urdf`` limited to the FR3's joint stops, written once per process."""
    panda_urdf = Path(panda_urdf).resolve()
    if panda_urdf not in _WRITTEN:
        folder = Path(tempfile.mkdtemp(prefix="arena_fr3_kinematics_"))
        target = folder / "fr3_kinematics.urdf"
        target.write_text(with_joint_limits(panda_urdf.read_text(), FR3_JOINT_LIMITS))
        _WRITTEN[panda_urdf] = target
    return _WRITTEN[panda_urdf]
