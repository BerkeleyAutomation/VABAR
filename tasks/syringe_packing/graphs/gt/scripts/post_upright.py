"""Stand the held tool on end over a container's hole and post it in.

A compartment and a sealed box are not the same destination. A compartment is
open from above, so releasing a tool over it works — that is what
:file:`descend_release.py` does. A sharps box is a closed shell whose lid is
solid everywhere except one 55 mm hole, and a 150 mm syringe let go flat over it
lands *on the lid*: the graph completes every node, reports success and scores
zero, which is exactly what ``syringe_packing/syringe`` did before this node existed.

The sequence is the reference's ``run_tool_sort.post_upright``, and so is the
reasoning behind it: the tool goes in the way a person does it — held upright
over the hole and let go — rather than carried down through it, because in that
pose the gripper is horizontal and would have to duck below the rim to insert
the tool by hand, which puts the arm somewhere it tracks badly.

**The upright orientation is derived from measurement, not from a convention.**
The reference rotates its own home quaternion by two fixed axis-angles, which
only works if the grasp always leaves the tool in the hand the same way. Here
the tool's long axis is read live off its OBB and the *minimal* rotation taking
that axis to world-down is composed onto the current end-effector orientation.
Because the tool is rigidly held, rotating the hand by that rotation stands the
tool on end whatever yaw the grasp happened to pick.

**The release height is measured too, for the reason the reference gives:** how
far the tool hangs below the gripper once upright depends on where along it the
fingers closed, predicting it from the grasp is a centimetre or two out, and a
centimetre is the difference between dropping cleanly through the hole and
catching its rim. So ``sim.get_object_bounds`` is read *after* the tool is
standing, not before.
"""

from typing import TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    posted: bool
    object_name: str
    hang: float
    release_z: float


POST_CLEARANCE = 0.01
"""Gap between the tool's lower end and the lid when it is let go [m]."""
RETRACT_MULTIPLE = 1.6
"""How much further than the stand-up height to lift away to, as a multiple.

The reference stood up at 0.14 m above the rim and retracted to 0.22, a ratio of
about 1.6 on a robot whose hand needed 0.14 to clear. Both were heights for *that*
hand; what survives portably is the ratio, applied to whatever clearance the hand
that arrived actually needs (``robot.describe_workspace().align_clearance_m``).
"""
TURN_STEPS = 4
"""Intermediate configurations the 90-degree stand-up is broken into.

Sliced so no single leg asks for a large reorientation at once, and done in
joint space because each ``robot.go_to_pose`` is a 40-waypoint ``plan_linear``
-- about 17 s of solver time during which the episode's step budget keeps
draining. Six Cartesian turn legs spent ~100 s of budget and got *less* far than
the one move they replaced; ``robot.solve_ik`` plus ``robot.move_to_joints``
costs a single IK solve each.

**This does not currently achieve the stand-up, and the reason is not the
slicing.** ``robot.solve_ik`` constrains position and the *approach axis*
only -- ``ArmIK.solve_q`` aligns the tool's +z and leaves wrist roll free by
design, because constraining it makes a top-down chain stall in a saddle. A roll
about the approach axis is exactly what standing a tool on end is, so the
component that matters here is the one the solver discards. The check after the
turn catches it; fixing it needs a full-orientation solve or direct wrist-joint
commands, neither of which belongs in a node script.
"""


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat(m: np.ndarray) -> dict:
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _slerp(a: dict, b: dict, t: float) -> dict:
    """Shortest-arc interpolation between two wxyz quaternions."""
    qa = np.array([float(a[k]) for k in ("w", "x", "y", "z")])
    qb = np.array([float(b[k]) for k in ("w", "x", "y", "z")])
    dot = float(np.dot(qa, qb))
    if dot < 0.0:  # take the short way round
        qb, dot = -qb, -dot
    if dot > 0.9995:  # nearly identical: lerp is stable where slerp is not
        q = qa + t * (qb - qa)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        q = (np.sin((1.0 - t) * theta) * qa + np.sin(t * theta) * qb) / sin_theta
    q = q / np.linalg.norm(q)
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


def _rotation_taking(axis: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The minimal rotation matrix taking unit ``axis`` onto unit ``target``."""
    cross = np.cross(axis, target)
    dot = float(np.dot(axis, target))
    norm = float(np.linalg.norm(cross))
    if norm < 1.0e-9:
        # Already aligned, or exactly opposed: a half turn about any
        # perpendicular. Opposed cannot happen here (the sign of the long axis
        # is chosen below to keep the angle acute) but a 180-degree case that
        # silently returned the identity would leave the tool lying flat.
        if dot > 0.0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(axis, perp)
        perp /= np.linalg.norm(perp)
        k = np.array([[0.0, -perp[2], perp[1]], [perp[2], 0.0, -perp[0]], [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2.0 * (k @ k)
    k = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - dot) / (norm * norm))


def run(ctx: NodeContext, object_name: str = "", region_name: str = "") -> Output:
    if not object_name or not region_name:
        goal = ctx.tool("sim.next_goal")
        object_name = object_name or str(goal["object_name"])
        region_name = region_name or str(goal["region_name"])

    container = ctx.tool("sim.get_container", region_name=region_name)
    aperture, rim = container["aperture"], float(container["rim_z"])
    ax, ay = float(aperture["x"]), float(aperture["y"])

    # The tool's long axis in world, as it currently sits in the fingers.
    obb = ctx.tool("sim.get_object_obb", object_name=object_name)["obb"]
    long_axis = int(ctx.tool("sim.get_object_bounds", object_name=object_name)["long_axis"])
    axis = _quat_to_mat(obb["orientation"])[:, long_axis]
    axis = axis / np.linalg.norm(axis)
    down = np.array([0.0, 0.0, -1.0])
    # Either end may go down; take the one already closer to down so the hand
    # turns as little as possible.
    if float(np.dot(axis, down)) < 0.0:
        axis = -axis

    # Standing a tool on end is a roll about the approach axis, and a solver that
    # leaves roll free cannot express it. Ask before spending the turn: the arm
    # says whether its IK honours roll, and a graph that knows it does not can
    # decline here instead of discovering it in the check below.
    if not bool(ctx.tool("robot.describe_arm")["solver"]["honours_roll"]):
        raise RuntimeError(
            f"posting {object_name} upright needs wrist-roll authority, and this arm's IK "
            f"({ctx.tool('robot.describe_arm')['solver']['name']}) leaves roll free by design. "
            "Run with a full-orientation backend (--ik pyroki) or route this goal to a "
            "compartment rather than a container."
        )

    ee = ctx.tool("robot.get_ee_pose")["pose"]
    held = ee["rotation"]
    upright = _mat_to_quat(_rotation_taking(axis, down) @ _quat_to_mat(held))

    # Translate first, holding the orientation the grasp left, so the move over
    # the box is a pure translation and nothing is asked of the pinch yet. This
    # is the one leg that stays Cartesian: it is a real traverse across the cell
    # and wants the straight-line guarantee.
    # Clear of the rim by whatever *this* hand needs to get its whole envelope
    # over the lid: fingers, plus the housing standing above them, plus a margin.
    approach_above_rim = float(ctx.tool("robot.describe_workspace")["align_clearance_m"])
    above = {"x": ax, "y": ay, "z": rim + approach_above_rim}
    ctx.tool("robot.go_to_pose", pose={"position": above, "rotation": held})

    # Then stand it up in place, a slice of the turn at a time. See TURN_STEPS.
    for i in range(1, TURN_STEPS + 1):
        _move_through(ctx, {"position": above, "rotation": _slerp(held, upright, i / TURN_STEPS)})

    # Did it actually stand up? ``long_axis`` is the world axis the tool is now
    # longest along, so 2 means on end. Checking rather than assuming, because
    # the failure is otherwise silent and looks like a bad release: a tool still
    # lying flat hangs a centimetre or two below the hand instead of half its
    # length, the release height computed from that is nonsense, and it is let
    # go horizontally over a hole it cannot fit through.
    bounds = ctx.tool("sim.get_object_bounds", object_name=object_name)
    if int(bounds["long_axis"]) != 2:
        raise RuntimeError(
            f"{object_name} is still lying flat after the stand-up "
            f"(longest along world axis {int(bounds['long_axis'])}, not z). "
            "robot.solve_ik constrains position and the approach axis only -- ArmIK.solve_q "
            "leaves wrist roll free by design -- so a roll about the approach axis, which is "
            "exactly what standing a tool on end is, cannot be commanded through it."
        )

    # Now that it is upright, measure how far it actually hangs.
    ee_z = float(ctx.tool("robot.get_ee_pose")["pose"]["position"]["z"])
    lowest = float(bounds["lowest_z"])
    hang = ee_z - lowest
    release_z = rim + POST_CLEARANCE + hang

    _move_through(ctx, {"position": {"x": ax, "y": ay, "z": release_z}, "rotation": upright})
    ctx.tool("robot.open_gripper", settle_steps=60)
    _move_through(
        ctx,
        {"position": {"x": ax, "y": ay, "z": rim + approach_above_rim * RETRACT_MULTIPLE}, "rotation": upright},
    )
    return {"posted": True, "object_name": object_name, "hang": float(hang), "release_z": float(release_z)}


def _move_through(ctx: NodeContext, pose: dict) -> None:
    """Reach a pose by solving it once and moving there in joint space.

    The cheap sibling of ``robot.go_to_pose``: one IK solve rather than a
    40-waypoint Cartesian plan. Right for the short in-place legs around the
    hole, where the straight-line guarantee buys nothing and the planning time
    comes straight out of the episode's budget.
    """
    joints = ctx.tool("robot.solve_ik", pose=pose)["joint_config"]
    ctx.tool("robot.move_to_joints", joint_config=joints)
