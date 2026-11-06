"""Carry the held gear over its peg and stage it.

The gear is centred in the jaws by the pick itself (the descent aimed at the
camera's centre of the disc, and the jaws close symmetrically). The lowering
stops on contact, bounded, because the second gear's teeth can land on its
neighbour's tips above the model's height.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402


SEARCH_SHORT_M = 0.008
"""A lowering that stops this far above the staged height stopped on the peg's tip, not on the cone [m]."""
SEARCH_RING = ((0.003, 0.0), (-0.003, 0.0), (0.0, 0.003), (0.0, -0.003), (0.003, 0.003), (-0.003, -0.003), (0.003, -0.003), (-0.003, 0.003))
"""Where the bore is tried next, around the peg the camera placed [m]: a ring inside the cone's capture."""


def run(ctx, index: int):
    geom = gb.board(ctx)
    station = gb.station(geom, index)
    name = station["gear"]
    tx, ty, _tz = gb.tcp(ctx)
    holding, fraction = gb.holding(ctx)
    if not holding:
        return {"route": "lost", "index": int(index), "gear": name, "reason": "the jaws hold nothing", "gripper": fraction}
    held = gb.held_gear(ctx, geom["reading"])
    bias = None if held is None else [round(float(held["xy"][0]) - tx, 4), round(float(held["xy"][1]) - ty, 4)]
    sx, sy, _ = station["seat"]
    q = gb.station_quat(geom, index)
    ctx.tool("robot.go_to_pose", pose=gb.pose(sx, sy, geom["carry_z"], q))
    # Over the peg to within the cone's capture before the lowering, for the same reason the pick is.
    gb.leg(ctx, gb.pose(sx, sy, geom["carry_z"], q), 120, tolerance=gb.FINE_TOLERANCE)
    ctx.tool("robot.wait_steps", steps=20)
    holding, fraction = gb.holding(ctx)
    if not holding:
        return {"route": "lost", "index": int(index), "gear": name, "reason": "dropped during the carry", "gripper": fraction}
    staged = float(station["staged_tcp_z"])
    down = ctx.tool("robot.move_cartesian_until_contact", pose=gb.pose(sx, sy, staged, q), max_steps=480, stall_steps=36)  # 8 s budget, 0.6 s stall window at 60 Hz control steps
    # THE BORE HAS TO FIND THE CONE. The lowering stops on contact, and a stop well above the staged height is
    # the gear's underside on the peg's tip -- the bore missed the cone by more than the 2.3 mm it captures
    # (measured: pair-09 and pair-11 stopped 10-20 mm short, released there, and the gear fell off the board).
    # The pads centre the gear along the closing axis only, so what is left is the pick's error along the pad
    # faces plus the peg's. A short search: lift clear of the tip and come down again a few millimetres over,
    # on a ring, until the lowering reaches the staged height.
    searched = []
    for dx, dy in SEARCH_RING:
        short = float(down.get("remaining_distance", 0.0)) if isinstance(down, dict) else 0.0
        if not (isinstance(down, dict) and down.get("outcome") == "contact" and short > SEARCH_SHORT_M):
            break
        gb.leg(ctx, gb.pose(sx + dx, sy + dy, staged + 0.025, q), 60)
        down = ctx.tool(
            "robot.move_cartesian_until_contact", pose=gb.pose(sx + dx, sy + dy, staged, q), max_steps=300, stall_steps=36  # 5 s budget, 0.6 s stall window at 60 Hz control steps
        )
        searched.append([round(1000 * dx, 1), round(1000 * dy, 1), round(1000 * float(down.get("remaining_distance", 0.0)), 1) if isinstance(down, dict) else None])
    ctx.tool("robot.wait_steps", steps=20)
    return {
        "route": "carried",
        "index": int(index),
        "gear": name,
        "seen_in_hand_at_pick": held is not None,
        "camera_centre_bias_m": bias,
        "lowering": down if isinstance(down, dict) else str(down),
        "search": searched,
        "gripper": fraction,
    }
