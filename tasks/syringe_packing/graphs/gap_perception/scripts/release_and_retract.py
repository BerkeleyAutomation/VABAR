"""Release inside the aperture and retract vertically."""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    released: bool


RETRACT_M = (0.16, 0.08)
"""Retract heights to try, highest first.

The retract is planned by cuRobo, not handed to the Cartesian servo. The hand
ends its insertion low over the box, with the shoulder near its limit.
``tray_clutter_t11`` (sweep s2, 2026-09-13): joint 2 was 0.096 rad from its upper
limit, the servo gave up 102 mm short of a 160 mm rise, and the episode aborted
with the syringe already released into the box."""


def run(ctx: NodeContext, final_pose: dict[str, Any]) -> Output:
    ctx.tool("robot.open_gripper", settle_steps=100)
    retract = None
    for height in RETRACT_M:
        retract = {"position": dict(final_pose["position"]), "rotation": dict(final_pose["rotation"])}
        retract["position"]["z"] = float(retract["position"]["z"]) + height
        plan = ctx.tool("motion.plan_linear", end=retract, orientation="lock", allow_start_contact=True,
                        contact_margin=0.008)
        if plan.get("planned") and (plan.get("trajectory") or {}).get("waypoints"):
            ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
            break
    else:
        # Nothing planned: the servo's own best effort. The syringe is already
        # released, so falling short here costs only clearance for the homing move.
        try:
            ctx.tool("robot.go_to_pose_cartesian", pose=retract)
        except Exception as error:  # noqa: BLE001
            print(f"[insert_sg.release] retract fell short: {error}", flush=True)
    ctx.tool("robot.wait_steps", steps=200)
    return {"released": True}
