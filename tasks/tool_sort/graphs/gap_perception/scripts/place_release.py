"""VAB-style release: open the gripper, then a LINEAR retract straight up.

Opens with a settle so the object lands inside the basket, then retracts
vertically with ``robot.go_to_pose_cartesian`` (cuRobo ``plan_directed_linear``,
VAB segment 6) instead of the free-space ``robot.go_home`` swing. Leaving the arm
at hover above the basket is fine for the loop: the next grasp's rise/XY segments
carry it to the next object, and the agentview perception sees the table clearly.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Vec3

class Output(TypedDict):
    done: bool


def run(ctx: NodeContext, place_position: Vec3, hover_z: float = -1.0) -> Output:
    # Open + hold so the released object settles in the basket BEFORE retracting.
    ctx.tool("robot.open_gripper", settle_steps=60)
    p = place_position
    # Derive retract height from the perceived placement pose. The previous
    # absolute default came from another workspace and was below this table.
    if hover_z < 0.0:
        hover_z = float(p["z"]) + 0.10
    # Preserve whichever downward yaw the transport selected as reachable.
    # Forcing a canonical yaw here turns a vertical retract into an unnecessary
    # wrist rotation and can fail after the object has already been released.
    current_rotation = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
    # Seg 6: straight-up retract to hover (Z only).
    ctx.tool(
        "robot.go_to_pose_cartesian",
        pose={"position": {"x": float(p["x"]), "y": float(p["y"]), "z": float(hover_z)},
              "rotation": current_rotation},
    )
    return {"done": True}
