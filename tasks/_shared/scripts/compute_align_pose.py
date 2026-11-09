"""Compute the align-pose for direct-IK grasping.

Constructs an SE(3) pose at ``(grasp_pose.x, grasp_pose.y, target_obb_top + 0.15)``
with the same rotation as the grasp pose. Used by the ``grasping-direct-ik``
skill: the gripper rotates into the grasp orientation at this align-pose
first, then descends straight down to the actual grasp pose.

It is asked for now: ``robot.describe_workspace().align_clearance_m`` is the fingers'
reach plus the housing standing above them plus the pad thickness plus a margin, off
the hand that is actually on the arm. A hand with a taller housing gets a taller
clearance instead of a collision.

``clearance`` may still be passed to pin it, which is what a workflow does when the
*held* object rather than the hand is what needs the room.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose


class Output(TypedDict):
    align_pose: Se3Pose


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
    clearance: float = 0.0,
) -> Output:
    if clearance <= 0.0:
        clearance = float(ctx.tool("robot.describe_workspace")["align_clearance_m"])
    approach_z = target_obb["center"]["z"] + target_obb["extent"]["z"] + clearance
    align_pose: Se3Pose = {
        "position": {
            "x": grasp_pose["position"]["x"],
            "y": grasp_pose["position"]["y"],
            "z": approach_z,
        },
        "rotation": grasp_pose["rotation"],
    }
    return {"align_pose": align_pose}
