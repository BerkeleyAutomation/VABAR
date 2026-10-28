# SPDX-License-Identifier: Apache-2.0
"""What the tool_hanging suite needs that RoboSimStudio does not ship: its robot.

``yam_tool_hanging`` is the bimanual YAM on the tool-hanging cell's own mount.
It is deliberately a separate robot from upstream's ``yam_bimanual`` rather than
a parameter on it: the cell narrows the shoulder spacing and mounts the pair
50 mm further forward and 60 mm higher, which is the lift-and-rotate clearance a
hang needs, and a change every other task would have to be re-measured against.

Registered when the suite is scanned (``tasks/tool_hanging/staging.py`` imports
this module). The MJCF and the cuRobo model are VABAR's own, under
``assets/embodiments/yam/``. No ``urdf`` is named, so IK falls back to Newton's
solver, as it did when these documents were measured.
"""

from __future__ import annotations

from robosimstudio.robots.registry import register_embodiment
from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

__all__ = ["YAM_TOOL_HANGING"]

_ARM_HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
"""Per-arm home posture [rad]: the YAM's own home keyframe."""


def _arm(side: str) -> ArmSpec:
    return ArmSpec(
        name=side,
        joint_prefix=f"{side}_",
        ee_body_name=f"{side}_link_6",
        gripper="yam_linear_arena",
        home_q=_ARM_HOME,
    )


YAM_TOOL_HANGING = register_embodiment(
    EmbodimentSpec(
        name="yam_tool_hanging",
        asset=AssetSpec(path="embodiments/yam/yam_tool_hanging.xml"),
        base_offset=(-0.2975, 0.0, 0.082, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=True,
        gravcomp_restate=True,
        enable_self_collisions=False,
        arms=(_arm("left"), _arm("right")),
        arm_actuation=None,
        gripper_actuation=None,
        ik_iterations=8,
        # The planner model: a graph asking for `--motion curobo` gets a real plan.
        # The URDF is the chain the config's collision spheres were measured on.
        curobo_config="embodiments/yam/yam_curobo.yml",
        curobo_urdf="embodiments/yam/i2rt_yam/urdf/yam.urdf",
        curobo_time_scale=2.0,
    )
)
