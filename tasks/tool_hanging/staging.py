# SPDX-License-Identifier: Apache-2.0
"""The tool-hanging suite's arrangement, under two names, and one rig.

The four scripted variants use Newton contact generation. Their holders collide
through their original meshes, preserving the authored openings rather than
replacing them with convex-decomposition pieces. MuJoCo only integrates the
articulated rigid dynamics; ``use_mujoco_contacts=False`` disables its collision
generation and the composer routes every pair through a Newton pipeline. The
raised ceilings cover the decomposed moving tools.

``tool_hanging_plain_960hz@v1`` was the hammers' -- that scene approximates its
meshes to boxes, has no hull sets, and wants nothing beyond the rates -- and it
was always, field for field, the same arrangement. It stays registered under
its own name because the hammers' documents stage it, and it is the *same
object* so the two names cannot drift apart.

**The rigs are registered, not supplied.** The general rig retains its overhead
and two side views.  The Level-1 rig uses the tuned two-camera oblique pair
defined below.
"""

from __future__ import annotations

import vabar  # noqa: F401 -- the scripted-routine module RoboSimStudio leaves out; see vabar/__init__.py
import vabar.ext.tool_hanging  # noqa: F401 -- registers this suite's robot, yam_tool_hanging

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.specs import ArrangementSpec, CollisionSpec, backend
from robosimstudio_bench.taskbuild import register_arrangement, register_camera_rig

__all__ = [
    "TOOL_HANGING_LEVEL1_RIG",
    "TOOL_HANGING_PHYSICS_480",
    "TOOL_HANGING_PHYSICS",
    "TOOL_HANGING_PLAIN_PHYSICS",
    "TOOL_HANGING_RIG",
]

BOARD_X = 0.880
"""Far table edge; the pegboard wall's front plane, and what the rig frames."""

LOOK_AT = (BOARD_X, 0.0, 0.728 + 0.240)
"""The reference branch's shared aim point above the 0.728 m work surface."""

TOOL_HANGING_PHYSICS = register_arrangement(
    "tool_hanging_960hz@v1",
    ArrangementSpec(
        control_hz=60.0,
        substeps=16,
        backends=(("*", backend("mujoco", use_mujoco_contacts=False, nconmax=2048, njmax=2048)),),
        collision=CollisionSpec(rigid_contact_max=8192, rigid_gap=5.0e-4),
    ),
)

TOOL_HANGING_PHYSICS_480 = register_arrangement(
    "tool_hanging_480hz@v1",
    ArrangementSpec(
        control_hz=60.0,
        substeps=8,
        backends=(
            # A five-body Level-2 scene reached nefc=4,394 in an end-to-end
            # wrench run; keep headroom instead of truncating constraints.
            ("*", backend("mujoco", use_mujoco_contacts=False, nconmax=4096, njmax=8192)),
        ),
        collision=CollisionSpec(rigid_contact_max=8192, rigid_gap=5.0e-4),
    ),
)

TOOL_HANGING_PLAIN_PHYSICS = register_arrangement("tool_hanging_plain_960hz@v1", TOOL_HANGING_PHYSICS)

TOOL_HANGING_RIG = register_camera_rig(
    "tool_hanging_views@v1",
    (
        CameraSpec(
            name="overhead",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(LOOK_AT[0], LOOK_AT[1], LOOK_AT[2] + 0.85),
            look_at=LOOK_AT,
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="side",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(LOOK_AT[0] - 0.62, LOOK_AT[1] + 0.82, LOOK_AT[2] + 0.90),
            look_at=(LOOK_AT[0] - 0.05, LOOK_AT[1], LOOK_AT[2] - 0.02),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="side_right",
            width=640,
            height=480,
            fov_y=50.0,
            eye=(LOOK_AT[0] - 0.62, LOOK_AT[1] - 0.82, LOOK_AT[2] + 0.90),
            look_at=(LOOK_AT[0] - 0.05, LOOK_AT[1], LOOK_AT[2] - 0.02),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
    ),
)


LEVEL1_OVERHEAD = CameraSpec(
    name="overhead",
    width=640,
    height=480,
    fov_y=55.0,
    # Oblique top view: the original vertical view projected the table wrench
    # directly onto the pegboard hook.  Moving toward the robot separates the
    # horizontal work surface from the vertical fixture while retaining both
    # in one calibrated RGB-D frame.  The fixture is localized before grasp
    # and remains static. This remains the semantic-perception camera because
    # shifting it laterally degraded segmentation and held-object registration.
    eye=(LOOK_AT[0] - 0.58, LOOK_AT[1], LOOK_AT[2] + 0.70),
    look_at=(LOOK_AT[0] - 0.06, LOOK_AT[1], LOOK_AT[2] - 0.10),
    # Ninety-degree roll relative to v1: the table/fixture span is presented
    # across the landscape frame. The camera centre and look ray are unchanged.
    up=(1.0, 0.0, 0.0),
    near=0.01,
    far=4.0,
    channels=("rgb", "depth"),
)

LEVEL1_SIDE = CameraSpec(
    name="side",
    width=640,
    height=480,
    fov_y=55.0,
    eye=(LOOK_AT[0] - 0.58, LOOK_AT[1] - 0.12, LOOK_AT[2] + 0.70),
    look_at=(LOOK_AT[0] - 0.06, LOOK_AT[1], LOOK_AT[2] - 0.10),
    up=(1.0, 0.0, 0.0),
    near=0.01,
    far=4.0,
    channels=("rgb", "depth"),
)

_D405_WRIST_XFORM = (-0.0017, 0.079729, 0.066021, 0.0, -0.42304971, 0.90610647, 0.0)
"""D405 optical pose relative to a YAM link-6 frame.

The fixed camera bodies are collapsed during model import, so mounting a
CameraSpec on their labels can alias both cameras to one arm.  This composed
link-to-optical transform keeps the physical D405 offset and points the optical
+Z axis out past the gripper toward the workspace.
"""

LEVEL1_WRIST_LEFT = CameraSpec(
    name="eye_in_hand_left",
    width=320,
    height=240,
    fov_y=58.0,
    parent_frame="left",
    xform=_D405_WRIST_XFORM,
    near=0.005,
    far=1.5,
    channels=("rgb", "depth"),
)

LEVEL1_WRIST_RIGHT = CameraSpec(
    name="eye_in_hand_right",
    width=320,
    height=240,
    fov_y=58.0,
    parent_frame="right",
    xform=_D405_WRIST_XFORM,
    near=0.005,
    far=1.5,
    channels=("rgb", "depth"),
)

TOOL_HANGING_LEVEL1_RIG = register_camera_rig(
    "tool_hanging_views_level1@v2",
    (LEVEL1_OVERHEAD, LEVEL1_SIDE, LEVEL1_WRIST_LEFT, LEVEL1_WRIST_RIGHT),
)
"""Two static RGB-D views plus the original wrist cameras on both arms.

``overhead`` is the primary semantic-perception view. ``side`` supplies the
complementary RGB-D surface observations used by collision-world fusion. Both
wrists provide close-range held-feature measurements after grasping.
"""



# The YAM station: one head camera, a left-shoulder camera and a D405 on each wrist, every one 1280x720.
YAM_STATION_RIG = register_camera_rig(
    "yam_station_tool_hanging@v1",
    (
        CameraSpec(
            name="overhead",
            width=1280,
            height=720,
            fov_y=55.0,
            eye=(0.3025, 0.0, 1.668),
            look_at=(0.8225, 0.0, 0.868),
            up=(1.0, 0.0, 0.0),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="side",
            width=1280,
            height=720,
            fov_y=50.0,
            eye=(0.2625, 0.82, 1.868),
            look_at=(0.8325, 0.0, 0.948),
            near=0.01,
            far=4.0,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="eye_in_hand_left",
            width=1280,
            height=720,
            fov_y=58.0,
            parent_frame="left",
            xform=_D405_WRIST_XFORM,
            near=0.005,
            far=1.5,
            channels=("rgb", "depth"),
        ),
        CameraSpec(
            name="eye_in_hand_right",
            width=1280,
            height=720,
            fov_y=58.0,
            parent_frame="right",
            xform=_D405_WRIST_XFORM,
            near=0.005,
            far=1.5,
            channels=("rgb", "depth"),
        ),
    ),
)
