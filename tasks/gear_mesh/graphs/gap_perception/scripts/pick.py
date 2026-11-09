"""Pick station ``index``'s gear where the camera sees it: over, descend, close, lift.

Held is judged without the simulator: the jaws stopped part-way (``robot.get_gripper``
reads part-way closed on a gear, about 0 closed on nothing, 1.0 open). The camera's
view of the held gear is recorded as a diagnostic only -- the hand hides it from the
off-axis overhead camera more often than not.

Two things are read off the hand here rather than assumed: where its pads are at
the pre-open width (``gearboard.reach_now``), and, once closed, how far the tool
centre stands over the gear it holds (``gearboard.take_hold``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gearboard as gb  # noqa: E402
import sight  # noqa: E402


def run(ctx, index: int):
    geom = gb.board(ctx)
    station = gb.station(geom, index)
    name = station["gear"]
    g = station["assigned"]
    if g is None:
        return {"route": "missed", "index": int(index), "gear": name, "reason": "no gear of that class in view"}
    gx, gy = float(g["xy"][0]), float(g["xy"][1])
    gz = float(g["top_z"]) - 0.5 * gb.FACE
    visit = int(sight._CACHE.get("visits", {}).get(int(index), 1))
    if g.get("perched"):
        # A gear left on the peg's cone: grasp it a little higher than a flat gear (it is tilted) and
        # carry on as with a mat pick; the next descent, a third of a pitch round, gets another chance at the mesh.
        gz += 0.003
    else:
        # A gear lying on the mat has its centre half a face above the mat, whatever a mask that took some
        # mat with it says its top is: read 13 mm low once (pair-02), the pads went into the gear's rim,
        # stood it on edge and carried it that way.
        gz = max(gz, float(geom["mat_top_z"]) + 0.5 * gb.FACE - 0.002)
    # A retry grasps the gear a third of a tooth pitch further round, so the descent meets the
    # neighbour's teeth at a different phase than the pass that jammed.
    q = gb.quat_yaw(gb.station_quat(geom, index), (visit - 1) * float(station["sweep"]) / 3.0)
    carry_z = geom["carry_z"]
    # Both the pick and the seat have to be standable at this heading; its half-turn twin is the same grasp.
    sx, sy, sz = station["seat"]
    flipped = False
    if not (gb.reachable(ctx, gb.pose(gx, gy, gz + 0.02, q)) and gb.reachable(ctx, gb.pose(sx, sy, sz + 0.03, q))):
        sight._CACHE["flip"] = not bool(sight._CACHE.get("flip"))
        q = gb.quat_yaw(q, 3.141592653589793)
        flipped = True
    ctx.tool("robot.set_grip", width_m=float(station["pre_open_m"]), settle_steps=20)
    ctx.tool("robot.go_to_pose", pose=gb.pose(gx, gy, carry_z, q))
    ctx.tool("robot.wait_steps", steps=10)
    # WHERE THE PADS ARE AT THIS OPENING, from the hand. On a geared hand the fingertips move along the
    # approach axis as the jaws close, so the reach is read now, at the pre-open the descent is made at,
    # and not once per episode. It is never measured by pushing the jaws onto the mat: on the 2F-85 that
    # folds the passive linkage and reads the tool centre 19 mm past where the pads stopped.
    drop = gb.reach_now(ctx)
    lift = drop + gb.RIM_CLEAR - 0.5 * gb.FACE
    gb.let_go()
    # The servo stops within 4 mm of a target. Where it actually stopped over the gear says which way it
    # leans here, and the descent aims the other way by as much, so the jaws come down centred.
    tx, ty, _ = gb.tcp(ctx)
    ax, ay = gx + (gx - tx), gy + (gy - ty)
    # Within the cone's capture, not the servo's 4 mm: what the hand misses by here is carried to the peg.
    down = gb.leg(ctx, gb.pose(ax, ay, gz + lift, q), 220, tolerance=gb.FINE_TOLERANCE)
    ctx.tool("robot.wait_steps", steps=15)
    ctx.tool("robot.set_grip", preset="close", ramp_steps=30, settle_steps=70)
    # What the jaws now hold, measured rather than modelled: the tool centre over the gear's centre, with
    # the gear still lying where the camera saw it. Every carry and seat height is built on this.
    held_lift = gb.take_hold(ctx, gz)
    gb.leg(ctx, gb.pose(gx, gy, carry_z, q), 200)
    ctx.tool("robot.wait_steps", steps=20)
    held = gb.held_gear(ctx)
    jaws_on_something, fraction = gb.holding(ctx)
    return {
        "route": "grasped" if jaws_on_something else "missed",
        "index": int(index),
        "gear": name,
        "seen_in_hand": held is not None,
        "held_teeth": None if held is None else held["teeth"],
        "gripper": fraction,
        "visit": visit,
        "tip_reach_m": round(float(drop), 4),
        "held_lift_m": round(float(held_lift), 4),
        "flipped": flipped,
        "descent": down if isinstance(down, dict) else str(down),
        "carry_z": round(float(carry_z), 4),
    }
