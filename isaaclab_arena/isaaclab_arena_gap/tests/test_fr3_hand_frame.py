"""The 2F-85's tool centre in Arena's ``robotiq_base`` frame.

Arena's FR3 asset roots the hand at the coupling's underside; RoboSimStudio's
registered hand states its tool centre from MuJoCo Menagerie's ``base_mount``, 7 mm
higher. The resident hand must carry that offset, or every grasp drives the
fingertips 7 mm deeper than the graph asks.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaaclab_arena_gap.task_sources import autolab_root, fr3_robotiq_usd_path

ROOT = Path(__file__).resolve().parents[2]
ARENA_HAND = fr3_robotiq_usd_path().parent / "robotiq_2f85/Payload/Geometry.usda"
AUTOLAB_HAND = autolab_root() / "assets/embodiments/robotiq_2f85/2f85.xml"


def _arena_driver_z() -> float:
    """``robotiq_left_driver``'s height in ``robotiq_base``'s frame, from Arena's USD."""
    block = re.search(
        r'def Xform "robotiq_left_driver".*?double3 xformOp:translate = \(([^)]*)\)', ARENA_HAND.read_text(), re.S
    )
    assert block, f"no robotiq_left_driver in {ARENA_HAND}"
    return float(block[1].split(",")[2])


def _menagerie_driver_z_in_base_mount() -> float:
    """``left_driver``'s height in ``base_mount``'s frame, from the MJCF RoboSimStudio registers."""
    if not AUTOLAB_HAND.is_file():
        pytest.skip(f"{AUTOLAB_HAND} is not checked out")
    root = ET.parse(AUTOLAB_HAND).getroot()
    mount = next(b for b in root.iter("body") if b.get("name") == "base_mount")
    base = next(b for b in mount.iter("body") if b.get("name") == "base")
    driver = next(b for b in base.iter("body") if b.get("name") == "left_driver")
    # base turns about z only, so heights add.
    assert base.get("quat") in (None, "1 0 0 -1")
    return float(base.get("pos").split()[2]) + float(driver.get("pos").split()[2])


def test_base_mount_origin_is_measured_off_both_hand_files():
    pytest.importorskip("robosimstudio")
    from isaaclab_arena_gap.rss_adapter import embodiments

    measured = _arena_driver_z() - _menagerie_driver_z_in_base_mount()
    assert embodiments.FR3_HAND_BASE_MOUNT_ORIGIN == pytest.approx((0.0, 0.0, measured), abs=1e-6)


def test_resident_hand_tool_centre_is_the_pinch_point_in_arenas_frame():
    pytest.importorskip("robosimstudio")
    from robosimstudio.robots.registry import get_gripper

    from isaaclab_arena_gap.rss_adapter.embodiments import (
        FR3_HAND,
        FR3_HAND_BASE_MOUNT_ORIGIN,
        FR3_HAND_LABELS,
        resident_hand,
    )

    rss_gripper = get_gripper(FR3_HAND)
    resident = get_gripper(resident_hand(FR3_HAND, FR3_HAND_LABELS, FR3_HAND_BASE_MOUNT_ORIGIN))
    assert resident.tcp_body == "robotiq_base"
    expected = tuple(a + b for a, b in zip(rss_gripper.tcp_offset, FR3_HAND_BASE_MOUNT_ORIGIN, strict=True))
    assert resident.tcp_offset == pytest.approx(expected, abs=1e-9)
    # MuJoCo Menagerie's pinch site: 145 mm along base, 3.8 mm above base_mount, 7 mm above robotiq_base.
    assert resident.tcp_offset[2] == pytest.approx(0.1558, abs=1e-6)
    assert resident.tcp_offset_rot == rss_gripper.tcp_offset_rot


def _arena_driver_constants() -> dict[str, float]:
    """The FR3 embodiment's per-finger driver gains, read from its source without importing Isaac Lab."""
    import ast

    config = ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/embodiments/fr3_robotiq_2f85/config.py"
    names = {"AUTOLAB_DRIVER_STIFFNESS", "AUTOLAB_DRIVER_DAMPING", "AUTOLAB_DRIVER_EFFORT_LIMIT", "DRIVEN_FINGERS"}
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(config.read_text()).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names
    }


def test_arena_driver_gains_are_autolabs_per_finger_gains():
    """GaP's shadow closes RoboSimStudio's 2F-85; Arena drives its fingers with the same gains."""
    pytest.importorskip("robosimstudio")
    from robosimstudio.robots.registry import get_gripper

    from isaaclab_arena_gap.rss_adapter.embodiments import FR3_HAND

    actuation = get_gripper(FR3_HAND).actuation
    arena = _arena_driver_constants()
    assert arena["DRIVEN_FINGERS"] == 2
    assert arena["AUTOLAB_DRIVER_STIFFNESS"] == pytest.approx(actuation.target_ke)
    assert arena["AUTOLAB_DRIVER_DAMPING"] == pytest.approx(actuation.target_kd)
    # RoboSimStudio states no effort for this hand, so Newton's 1e6 default stands.
    assert not actuation.effort_for(0)
    assert arena["AUTOLAB_DRIVER_EFFORT_LIMIT"] == pytest.approx(1.0e6)


def _autolab_spring_link() -> dict[str, float]:
    """The spring-link joint's spring in the MJCF RoboSimStudio registers for the 2F-85."""
    if not AUTOLAB_HAND.is_file():
        pytest.skip(f"{AUTOLAB_HAND} is not checked out")
    root = ET.parse(AUTOLAB_HAND).getroot()
    default = next(d for d in root.iter("default") if d.get("class") == "spring_link")
    joint = default.find("joint")
    return {name: float(joint.get(name)) for name in ("stiffness", "damping", "springref")}


def test_arena_spring_link_matches_autolabs_hand():
    """GaP's shadow closes RoboSimStudio's 2F-85; Arena's fingers carry the same spring preload."""
    pytest.importorskip("pxr")
    from pxr import Usd

    on_cart = (
        fr3_robotiq_usd_path().parent.parent
        / "vabar__fr3_robotiq_2f85_on_cart/vabar__fr3_robotiq_2f85_on_cart.usda"
    )
    stage = Usd.Stage.Open(str(on_cart))
    joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() in ("left_spring_link_joint", "right_spring_link_joint")
    ]
    assert len(joints) == 2
    for prim in joints:
        arena = {name: prim.GetAttribute(f"mjc:{name}").Get() for name in ("stiffness", "damping", "springref")}
        assert arena == pytest.approx(_autolab_spring_link(), abs=1e-9)
