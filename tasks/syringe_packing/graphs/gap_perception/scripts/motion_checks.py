"""Checked trajectory execution using joint and TCP proprioception."""

import numpy as np
from scipy.spatial.transform import Rotation


def vec(p):
    return np.array([p[k] for k in "xyz"], dtype=float)


def rotation(p):
    q = p["rotation"]
    return Rotation.from_quat([q[k] for k in ("x", "y", "z", "w")])


def pose(p, r):
    q = r.as_quat()
    return {
        "position": dict(zip("xyz", map(float, p), strict=True)),
        "rotation": dict(zip(("x", "y", "z", "w"), map(float, q), strict=True)),
    }


def resample(trajectory, scale=2.0):
    rows = np.array([w["positions"] for w in trajectory["waypoints"]], dtype=float)
    src = np.linspace(0, 1, len(rows))
    dst = np.linspace(0, 1, max(2, int((len(rows) - 1) * scale) + 1))
    out = np.column_stack([np.interp(dst, src, rows[:, i]) for i in range(rows.shape[1])])
    return {"waypoints": [{"positions": row.tolist()} for row in out]}


def pose_error(actual, target):
    position = float(np.linalg.norm(vec(actual["position"]) - vec(target["position"])))
    angle = float((rotation(actual).inv() * rotation(target)).magnitude())
    return position, angle


def check_pose(actual, target, position_tolerance=0.003, angle_tolerance=0.035):
    position_error, angle_error = pose_error(actual, target)
    if position_error > position_tolerance or angle_error > angle_tolerance:
        raise RuntimeError(
            f"trajectory tracking failed: {position_error * 1000:.1f} mm, "
            f"{np.degrees(angle_error):.1f} deg; release prohibited"
        )
    return actual


def checked_execute(ctx, trajectory, target, *, scale=2.0, position_tolerance=0.003, angle_tolerance=0.035):
    if not trajectory or not trajectory.get("waypoints"):
        raise RuntimeError("No trajectory to execute")
    ctx.tool("robot.execute_trajectory", trajectory=resample(trajectory, scale), tolerance=0.002,
             max_steps_per_waypoint=180)
    ctx.tool("robot.wait_steps", steps=40)
    actual = ctx.tool("robot.get_ee_pose")["pose"]
    return check_pose(actual, target, position_tolerance, angle_tolerance)


def planned_linear(ctx, target, world=None, attachment=None, *, scale=2.0, contact=False, position_tolerance=0.003):
    result = ctx.tool("motion.plan_linear", end=target, orientation="lock", world_config=world,
                      attached_object=attachment, allow_goal_contact=contact, allow_start_contact=contact)
    if not result.get("planned"):
        raise RuntimeError("collision-aware linear motion refused")
    return checked_execute(ctx, result["trajectory"], target, scale=scale, position_tolerance=position_tolerance)
