"""Release inside the aperture and retract vertically."""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    released: bool


def run(ctx: NodeContext, final_pose: dict[str, Any]) -> Output:
    ctx.tool("robot.open_gripper", settle_steps=100)
    retract = {"position": dict(final_pose["position"]), "rotation": dict(final_pose["rotation"])}
    retract["position"]["z"] = float(retract["position"]["z"]) + 0.16
    ctx.tool("robot.go_to_pose_cartesian", pose=retract)
    ctx.tool("robot.wait_steps", steps=200)
    return {"released": True}
