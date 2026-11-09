"""Stand a held elongated object upright and post it through an aperture."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    posted: bool


def _rotation(pose):
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _pose(position, rotation):
    q = rotation.as_quat()
    return {"position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}}


def run(ctx: NodeContext, aperture: dict[str, float], rim_z: float,
        target_obb: dict[str, Any], long_axis_world: dict[str, float]) -> Output:
    arm = ctx.tool("robot.describe_arm")
    if not bool(arm["solver"]["honours_roll"]):
        raise RuntimeError("posting through an aperture requires full wrist orientation; use --ik pyroki")
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    current = _rotation(ee)
    axis = np.array([long_axis_world[k] for k in ("x", "y", "z")], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if float(axis @ [0.0, 0.0, -1.0]) < 0.0:
        axis = -axis
    correction, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [axis])
    upright = correction * current
    center = np.array([aperture[k] for k in ("x", "y", "z")], dtype=np.float64)
    workspace = ctx.tool("robot.describe_workspace")
    clearance = float(workspace["align_clearance_m"])
    above = center + np.array([0.0, 0.0, clearance])

    # Do not ask IK to translate half a metre and turn the held syringe upright
    # in one solve. First leave the table vertically with the current grasp,
    # then reorient in free space, and only then transit over the aperture.
    here = np.array([ee["position"][k] for k in ("x", "y", "z")], dtype=np.float64)
    lift = here.copy()
    lift[2] = max(float(lift[2] + 0.12), float(workspace["transport_z"]))
    ctx.tool("robot.go_to_pose_cartesian", pose=_pose(lift, current))
    ctx.tool("robot.go_to_pose", pose=_pose(lift, upright))
    ctx.tool("robot.go_to_pose", pose=_pose(above, upright))
    # Release with the lower end just inside the perceived rim. Object length
    # and grasp offset are both observed from the OBB.
    length = 2.0 * max(float(target_obb["extent"][k]) for k in ("x", "y", "z"))
    release = center + np.array([0.0, 0.0, max(0.03, 0.45 * length)])
    ctx.tool("robot.go_to_pose_cartesian", pose=_pose(release, upright))
    ctx.tool("robot.open_gripper", settle_steps=100)
    ctx.tool("robot.go_to_pose_cartesian", pose=_pose(above + np.array([0.0, 0.0, 0.08]), upright))
    ctx.tool("robot.wait_steps", steps=150)
    return {"posted": True}
