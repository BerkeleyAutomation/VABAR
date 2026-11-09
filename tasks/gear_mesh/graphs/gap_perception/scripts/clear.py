"""Open, lift to carry height and step off the chain, whatever happened at the peg."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402


def run(ctx, index: int):
    x, y, z = gb.tcp(ctx)
    holding, fraction = gb.holding(ctx)
    set_down = False
    if holding:
        geom = gb.board(ctx)
        if z > geom["carry_z"] - 0.03:
            # Something is still in the jaws up at carry height: set it down on the mat, off the chain, do not drop it.
            floor = geom["mat_top_z"]
            q = geom["base_q"]
            # A free patch of the mat: farthest from every gear in view and from the chain. The mat lies on the
            # arm's side of the board, which is -x of the chain in this cell.
            chain_x = min(s["seat"][0] for s in geom["stations"])
            others = [g["xy"] for g in geom["reading"]["gears"]] + [s["seat"][:2] for s in geom["stations"]]
            spots = [(chain_x - dx, yy) for dx in (0.10, 0.16) for yy in (-0.10, -0.05, 0.0, 0.05, 0.10)]
            x, y = max(spots, key=lambda sp: min([gb.distance_xy(sp, o) for o in others] or [1.0]))
            gb.leg(ctx, gb.pose(x, y, geom["carry_z"], q), 200)
            # A bounded leg aimed past the set-down height: the servo counts a target within its arrival ball as
            # reached, so the held gear's underside ends within a millimetre of the mat. Its height under the
            # tool centre is what the pick measured (geom["lift"]), not a fingertip figure.
            gb.leg(ctx, gb.pose(x, y, floor + 0.5 * gb.FACE + geom["lift"] + 0.001 - 0.0045, q), 240)
            ctx.tool("robot.wait_steps", steps=10)
            set_down = True
    ctx.tool("robot.set_grip", preset="open", settle_steps=20)
    gb.let_go()
    gb.step_aside(ctx)
    return {"route": "cleared", "index": int(index), "seated": gb.seated(ctx, index), "set_down": set_down, "gripper": fraction}
