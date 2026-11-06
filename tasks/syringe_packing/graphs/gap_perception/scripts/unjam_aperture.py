"""Tip a syringe left standing in the sharps-box aperture into the box, then re-check.

Why: in sweep s6 (2026-09-13) every failed trial ended the same way. After 3-5
posts the syringes already in the box stood or leaned under the hole, with their
tops 30-100 mm below the rim, and the next one landed on them. That syringe was
left standing in the aperture with its top 2-26 mm above the rim. The
post-release check returned ``blocked`` and the episode aborted, even with 5 of 8
syringes already in the box (t01). A syringe standing on a pile is easy to topple.

So, before giving up: close the gripper, come top-down over the hole, push the
fingertips 25 mm below the rim onto the protruding top, and sweep a little along
the box's long axis. The standing syringe tips over into the cavity. Every move
tolerates contact: a servo that stalls against the syringe has done its job. The
workflow re-runs the post-release check afterwards, and only a second ``blocked``
aborts.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    status: str
    deepest_fingertip_z: float


CLOSED_FINGERTIP_BEYOND_TCP_M = 0.0179
"""Closed Robotiq fingertips beyond the TCP (compute_grasp.FINGERTIP_BEYOND_TCP_M)."""
ABOVE_RIM_M = 0.06
"""Fingertip height above the rim to come over the hole at."""
PUSH_BELOW_RIM_M = 0.025
"""How deep into the hole the fingertips push. The closed finger band is +/-13.4 mm
wide, and it enters the 55 mm hole with the hand body well above the lid."""
SWEEP_M = 0.010
"""Lateral sweep inside the hole, along the box's long (world y) axis. It keeps the
finger band clear of the hole's edge."""


def _pose(position: np.ndarray, rotation: dict[str, float]) -> dict[str, Any]:
    return {"position": dict(zip(("x", "y", "z"), map(float, position))), "rotation": dict(rotation)}


def _move(ctx: NodeContext, pose: dict[str, Any]) -> bool:
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
        return True
    except Exception as error:  # noqa: BLE001 -- contact is expected here
        print(f"[unjam_sg.push] move stopped short: {error}", flush=True)
        return False


def run(ctx: NodeContext, aperture: dict[str, float], rim_z: float) -> Output:
    down = ctx.tool("robot.describe_arm")["home_pose"]["rotation"]
    center = np.array([aperture["x"], aperture["y"]], dtype=np.float64)
    rim = float(rim_z)

    def tcp_at(xy: np.ndarray, fingertip_z: float) -> dict[str, Any]:
        return _pose(np.array([xy[0], xy[1], fingertip_z + CLOSED_FINGERTIP_BEYOND_TCP_M]), down)

    ctx.tool("robot.set_grip", preset="close", ramp_steps=40, settle_steps=40)
    above = tcp_at(center, rim + ABOVE_RIM_M)
    plan = ctx.tool("motion.plan_to_pose", pose=above)
    if plan.get("planned") and (plan.get("trajectory") or {}).get("waypoints"):
        ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
    elif not _move(ctx, above):
        return {"status": "failed", "deepest_fingertip_z": float("nan")}

    _move(ctx, tcp_at(center, rim - PUSH_BELOW_RIM_M))
    deepest = float(ctx.tool("robot.get_ee_pose")["pose"]["position"]["z"]) - CLOSED_FINGERTIP_BEYOND_TCP_M
    for dy in (SWEEP_M, -SWEEP_M, 0.0):
        _move(ctx, tcp_at(center + np.array([0.0, dy]), rim - PUSH_BELOW_RIM_M))
    _move(ctx, above)
    ctx.tool("robot.open_gripper", settle_steps=60)
    ctx.tool("robot.wait_steps", steps=100)
    return {"status": "done", "deepest_fingertip_z": deepest}
