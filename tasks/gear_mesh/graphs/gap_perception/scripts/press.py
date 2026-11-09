"""Press the start button, found by the camera, through its stop with the closed jaws."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402


def run(ctx):
    geom = gb.board(ctx)
    if geom["button"] is None:
        return {"route": "not_started", "reason": "no button in view"}
    bx, by = geom["button"]["xy"]
    cap_top = float(geom["button"]["cap_top_z"])
    travel = float(geom["button"]["travel"])
    q = geom["base_q"]
    ctx.tool("robot.set_grip", preset="close", settle_steps=20)
    gb.let_go()
    # The closed jaws' fingertips are what press, and on a geared hand they sit lower closed than open.
    drop = gb.reach_now(ctx)
    ctx.tool("robot.go_to_pose", pose=gb.pose(bx, by, geom["carry_z"], q))
    ctx.tool("robot.wait_steps", steps=15)
    gb.leg(ctx, gb.pose(bx, by, cap_top + drop + 0.006, q), 200)
    ctx.tool("robot.wait_steps", steps=10)
    target = cap_top + drop - travel - 0.001
    down = ctx.tool("robot.move_cartesian_until_contact", pose=gb.pose(bx, by, target, q), max_steps=600, stall_steps=30)  # 10 s budget, 0.5 s stall window at 60 Hz control steps
    ctx.tool("robot.wait_steps", steps=45)
    gb.leg(ctx, gb.pose(bx, by, geom["carry_z"], q), 200)
    return {
        "route": "pressed",
        "button_xy": [round(bx, 4), round(by, 4)],
        "cap_top_z": round(cap_top, 4),
        "press": down if isinstance(down, dict) else str(down),
    }
