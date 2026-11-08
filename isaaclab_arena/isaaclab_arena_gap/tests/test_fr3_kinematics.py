"""The planner's FR3 model: the Panda's chain at the joint stops Arena simulates."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaaclab_arena_gap.rss_adapter.fr3_kinematics import (
    FR3_JOINT_LIMITS,
    PANDA_ARM_JOINTS,
    fr3_kinematics_urdf,
    with_joint_limits,
)
from isaaclab_arena_gap.task_sources import fr3_robotiq_usd_path

PHYSICS = fr3_robotiq_usd_path().parent / "franka_fr3_v1/Payload/Physics.usda"

# Joint 6 after the syringe graph's wrist turn on far picks: the pitch-up
# ends a Panda-limited planner refused (> 3.7525), and the pitch-down ends the FR3 stop rejects.
PITCH_UP_J6 = (3.7537, 3.7711, 3.7923, 4.0221)
PITCH_DOWN_J6 = (0.6121, 0.6295, 0.6507)


def _asset_limits() -> list[tuple[float, float]]:
    text = PHYSICS.read_text()
    limits = []
    for i in range(1, 8):
        block = re.search(
            rf'def PhysicsRevoluteJoint "fr3_joint{i}".*?physics:lowerLimit = (\S+).*?physics:upperLimit = (\S+)',
            text,
            re.S,
        )
        assert block, f"fr3_joint{i} has no limits in {PHYSICS}"
        limits.append((math.radians(float(block[1])), math.radians(float(block[2]))))
    return limits


def _limits(urdf_text: str) -> dict[str, tuple[float, float]]:
    root = ET.fromstring(urdf_text)
    return {
        joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
        for joint in root.iter("joint")
        if joint.find("limit") is not None
    }


def _frames(urdf_text: str) -> dict[str, tuple]:
    root = ET.fromstring(urdf_text)
    return {
        joint.get("name"): (
            joint.get("type"),
            ET.tostring(joint.find("origin")) if joint.find("origin") is not None else None,
            ET.tostring(joint.find("axis")) if joint.find("axis") is not None else None,
            joint.find("parent").get("link"),
            joint.find("child").get("link"),
        )
        for joint in root.iter("joint")
    }


def test_fr3_limits_are_the_arena_assets():
    for (lower, upper), (asset_lower, asset_upper) in zip(FR3_JOINT_LIMITS, _asset_limits(), strict=True):
        assert lower == pytest.approx(asset_lower, abs=1e-9)
        assert upper == pytest.approx(asset_upper, abs=1e-9)


TWO_JOINTS = """<robot name="r">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j1" type="revolute"><origin xyz="0 0 0.3" rpy="0 0 0"/><parent link="a"/><child link="b"/>
    <axis xyz="0 0 1"/><limit effort="87" lower="-1.0" upper="1.0" velocity="2.175"/>
    <safety_controller k_position="100" k_velocity="40" soft_lower_limit="-1.0" soft_upper_limit="1.0"/></joint>
  <joint name="finger" type="prismatic"><parent link="b"/><child link="c"/>
    <limit effort="100" lower="0.0" upper="0.04" velocity="0.2"/></joint>
</robot>"""


def test_only_the_named_joints_limits_change():
    out = with_joint_limits(TWO_JOINTS, [(-0.5, 2.0)], joints=["j1"])
    assert _limits(out) == {"j1": (-0.5, 2.0), "finger": (0.0, 0.04)}
    assert _frames(out) == _frames(TWO_JOINTS)
    joint = ET.fromstring(out).find("joint")
    assert joint.find("limit").get("velocity") == "2.175"
    assert joint.find("safety_controller").get("soft_lower_limit") == "-0.500000"
    with pytest.raises(ValueError, match="no limited joint"):
        with_joint_limits(TWO_JOINTS, [(-0.5, 2.0)], joints=["missing"])


def _panda_urdf() -> Path:
    paths = pytest.importorskip("robosimstudio.paths")
    return paths.resolve_path("embodiments/panda/panda_kinematics.urdf")


def test_fr3_urdf_is_the_panda_chain_at_fr3_stops():
    panda = _panda_urdf()
    fr3 = fr3_kinematics_urdf(panda).read_text()
    assert _frames(fr3) == _frames(panda.read_text())
    limits = _limits(fr3)
    for name, (lower, upper) in zip(PANDA_ARM_JOINTS, FR3_JOINT_LIMITS, strict=True):
        assert limits[name] == pytest.approx((lower, upper), abs=1e-6)
    assert limits["panda_finger_joint1"] == _limits(panda.read_text())["panda_finger_joint1"]
    lower, upper = limits["panda_joint6"]
    assert all(lower <= j6 <= upper for j6 in PITCH_UP_J6)
    assert all(j6 - lower < 0.15 for j6 in PITCH_DOWN_J6)


def test_fr3_robotiq_plans_and_solves_at_fr3_stops():
    _panda_urdf()
    from isaaclab_arena_gap.rss_adapter.embodiments import fr3_robotiq

    spec = fr3_robotiq((-0.5, -0.1, 0.912, 0.0, 0.0, 0.0, 1.0))
    assert spec.curobo_urdf == spec.urdf
    limits = _limits(Path(spec.curobo_urdf).read_text())
    assert limits["panda_joint6"] == pytest.approx(FR3_JOINT_LIMITS[5], abs=1e-6)
    assert spec.curobo_config == "franka.yml"
