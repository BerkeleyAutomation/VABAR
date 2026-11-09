# SPDX-License-Identifier: Apache-2.0
"""What the tool_sort suite needs that RoboSimStudio does not ship: the robots its host rows name.

``tool_sort.suite`` offers its stripped workcell to two other robots through
``(:host ...)`` rows -- a UR5e with the Menagerie 2F-85, and the same UR5e
wearing the i2rt linear_4310 -- noted "compiles; no policy". No benchmark run
builds them, but every pairing a suite declares is resolved when the Studio's
Task panel lists its robots, so both have to be registered for the suite to open
there. Transcribed from the RoboSimStudio revision the suite was written against;
the assets are VABAR's own, under ``assets/embodiments/{ur5,linear_4310}/``.

Registered when the suite is scanned (``tasks/tool_sort/staging.py`` imports this module).
"""

from __future__ import annotations

import math

from robosimstudio.robots.grippers.linear_4310 import (
    GRASP_SITE_ROT,
    GRIP_CLOSE_MEASURED,
    GRIP_LOCAL_EXT,
    GRIP_OPEN,
    TCP_OFFSET,
)
from robosimstudio.robots.registry import register_embodiment, register_gripper
from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec, GripperSpec

__all__ = ["LINEAR_4310_ATTACHED", "UR5", "UR5_HOME_Q"]

UR5_HOME_Q = (0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0)
"""Elbow up, tool down over the bench."""

UR5 = register_embodiment(
    EmbodimentSpec(
        name="ur5",
        # A composed copy of Menagerie's UR5e whose <general> actuators are
        # position servos, so joint targets drive it under Newton.
        asset=AssetSpec(path="embodiments/ur5/ur5e_arm.xml"),
        # 0.55 m behind the work surface's centre, turned a half turn so +x faces the work.
        base_offset=(-0.55, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        collapse_fixed_joints=False,
        hull_colliders=False,
        arms=(
            ArmSpec(
                name="right",
                joint_names=(
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ),
                ee_body_name="wrist_3_link",
                gripper="robotiq_2f85_menagerie",
                home_q=UR5_HOME_Q,
            ),
        ),
        # The model's own per-joint position gains are kept.
        arm_actuation=None,
        gripper_actuation=None,
        ik_iterations=8,
        # The TCP sits on wrist_3_joint's axis, so a roll correction turns the tool in place.
        honour_roll=True,
    )
)

MOUNT_XFORM = (0.0, 0.1, 0.0, 0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
"""The ``gripper`` body in a UR flange's frame (xyz, xyzw): the attachment site, then a half turn about x."""

LINEAR_4310_ATTACHED = register_gripper(
    GripperSpec(
        name="linear_4310_attached",
        asset=AssetSpec(path="embodiments/linear_4310/linear_4310.xml"),
        mount_body="wrist_3_link",
        mount_xform=MOUNT_XFORM,
        finger_names=("joint7", "joint8"),
        finger_open_q=(GRIP_OPEN, GRIP_OPEN),
        finger_closed_q=(0.0, 0.0),
        open_width=GRIP_OPEN,
        span=0.09396,
        tcp_body="gripper",
        tcp_offset=TCP_OFFSET,
        tcp_offset_rot=GRASP_SITE_ROT,
        approach_axis=tuple(float(v) for v in GRIP_LOCAL_EXT),
        close_axis=tuple(float(v) for v in GRIP_CLOSE_MEASURED),
        finger_bodies=(("tip_left",), ("tip_right",)),
        tip_reach=0.035,
        actuation=ActuationSpec(target_ke=600.0, target_kd=20.0),
    )
)
