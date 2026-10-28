"""Move the robot to hover aside before observation.

Goes home first for a known reference pose, then shifts by a fixed Y offset
to clear the source bin from the overhead camera's field of view.
"""

from gap import NodeContext
from gap_core.types import Se3Pose

_SIDE_OFFSET_Y = 0.35
"""Metres to shift in world +Y from the home EEF position."""


def run(ctx: NodeContext, side_offset_y: float = _SIDE_OFFSET_Y) -> None:
    ctx.tool("robot.go_home")
    obs = ctx.tool("robot.get_observation")
    home_pos = obs["arms"][0]["ee_pose"]["position"]
    home_rot = obs["arms"][0]["ee_pose"]["rotation"]
    pose: Se3Pose = {
        "position": {
            "x": float(home_pos["x"]),
            "y": float(home_pos["y"]) + float(side_offset_y),
            "z": float(home_pos["z"]),
        },
        "rotation": home_rot,
    }
    ctx.tool("robot.go_to_pose", pose=pose)
