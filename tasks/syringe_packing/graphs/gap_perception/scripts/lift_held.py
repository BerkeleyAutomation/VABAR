"""Lift the closed hand straight up, clear of the tray and its spare, before the carry."""

from typing import TypedDict

from gap import NodeContext

LIFT_M = 0.20
"""How far above the grasp the hand rises [m]. The scripted routine's own
``LIFT_UP``: high enough that the carry and the upright turn happen well above
the tray, the spare barrel and the bins."""


class Output(TypedDict):
    lifted_pose: dict


def run(ctx: NodeContext) -> Output:
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    lifted = {"position": dict(ee["position"]), "rotation": dict(ee["rotation"])}
    lifted["position"]["z"] = float(lifted["position"]["z"]) + LIFT_M
    ctx.tool("robot.go_to_pose_cartesian", pose=lifted)
    return {"lifted_pose": lifted}
