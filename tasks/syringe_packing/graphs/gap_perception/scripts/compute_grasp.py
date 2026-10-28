"""Top-down short-axis grasp honoring the active gripper's close axis."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    grasp_pose: dict[str, Any]
    align_pose: dict[str, Any]
    long_axis_world: dict[str, float]
    grasp_station: str
    object_width_m: float


# Grasp the observed centroid -- the centre of mass, where reorienting the
# 150 mm syringe swings no lever against the fingers. routine.py's hull survey
# (its GRASP_HANDLE_FRAC docstring) shows the drawn hull carries no *vertices*
# from -15..+45 mm, but a convex hull's surface interpolates across that band:
# from the barrel's 10.7 mm at -15 to the flange's 23.6 at +45, ~14 mm at the
# centroid. The old centroid failure was the close command, not the geometry --
# the GT routine swept to 19.6 mm (the flange-derived OBB width) and stopped in
# open air above a 14 mm surface. This graph closes to object_width_m minus a
# 5 mm squeeze (see workflow.json's robot.set_grip), i.e. ~6 mm, so the servo
# presses well past the hull surface and the grip is loaded, not swept.
OBSERVED_CENTER_TO_BODY_GRASP_M = 0.0
CAP_TO_CENTRE_OF_MASS_M = 0.043
"""Where the centre of mass sits along the barrel from the perceived needle-cap
centre, measured against ground truth over twelve episodes on 2026-09-09:
-43.0 +/- 0.2 mm in every one. The OBB centre is not that stable: the cloud
misses ~35 mm of the thin cap end (span 113-117 mm for a 150 mm syringe), so
its centre sits 14-17 mm toward the plunger, and the pinch landed 15-25 mm from
the CoM in every episode. Gravity's lever about the pinch axis -- the one axis
two pads resist only by contact torsion -- is what pivoted the syringe in the
wrist turn on trial t08 (33 deg in 10 steps, a pad momentarily unloading), and
replaying that turn six times gave one slip in six: a marginal grasp, not a
broken one. Anchoring the station to the cap removes the lever."""
BARREL_WIDTH_FALLBACK_M = 0.0107
MIN_GRASP_CLEARANCE_FROM_BOTTOM_M = 0.006


def run(ctx: NodeContext, target_obb: dict[str, Any], target_cloud: dict[str, Any],
        needle_cap_center: dict[str, float]) -> Output:
    q = target_obb["orientation"]
    frame = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([target_obb["extent"][k] for k in ("x", "y", "z")])
    vertical = int(np.argmax(np.abs(frame.T @ [0.0, 0.0, 1.0])))
    horizontal = [index for index in range(3) if index != vertical]
    short = min(horizontal, key=lambda index: extents[index])
    long = max(range(3), key=lambda index: extents[index])
    short_world = frame[:, short].copy(); short_world[2] = 0.0
    short_world /= np.linalg.norm(short_world)
    gripper = ctx.tool("robot.describe_gripper")
    close = gripper["close_axis"]

    def top_down_rotation(short_axis: np.ndarray) -> dict[str, float]:
        """Build the complete wrist frame instead of adding yaw to home.

        The latter is backend/convention dependent and was observed to leave the
        Panda nearly world-aligned.  Here the tool's -Z is explicitly downward
        and its declared jaw-closing axis is explicitly the perceived short axis.
        """
        tool_z = np.array([0.0, 0.0, -1.0])
        if abs(float(close.get("x", 0.0))) >= abs(float(close.get("y", 0.0))):
            tool_x = short_axis
            tool_y = np.cross(tool_z, tool_x)
        else:
            tool_y = short_axis
            tool_x = np.cross(tool_y, tool_z)
        quat = Rotation.from_matrix(np.column_stack((tool_x, tool_y, tool_z))).as_quat()
        return {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])}

    rotation = top_down_rotation(short_world)
    center = dict(target_obb["center"])
    points = np.asarray(target_cloud["points"], dtype=np.float64).reshape(-1, 3)

    # Register the known asymmetric mesh station to the observed cloud. PCA/
    # OBB axis signs are arbitrary, so orient the long axis away from the
    # explicitly perceived red needle cap and toward the plunger flange.
    axis = frame[:, long].copy()
    axis /= np.linalg.norm(axis)
    origin = np.median(points, axis=0)
    along = (points - origin) @ axis
    span = float(np.ptp(along))
    grasp_station = "obb_center_fallback"
    grasp_center = np.array([center[k] for k in ("x", "y", "z")], dtype=np.float64)
    if len(points) >= 30 and span >= 0.08:
        cap = np.array([needle_cap_center[k] for k in ("x", "y", "z")], dtype=np.float64)
        obb_center = np.array([center[k] for k in ("x", "y", "z")], dtype=np.float64)
        # The cap-to-centre feature line is much more stable for this thin
        # object than PCA on a sparse/transparent silhouette. Use it for both
        # the mesh station and jaw yaw; PCA only establishes the fallback.
        directed = obb_center - cap
        directed[2] = 0.0
        if float(np.linalg.norm(directed)) > 0.02:
            axis = directed / np.linalg.norm(directed)
            short_world = np.array([-axis[1], axis[0], 0.0])
            rotation = top_down_rotation(short_world)
        elif float(axis @ (obb_center - cap)) < 0.0:
            axis = -axis
        grasp_center = cap + CAP_TO_CENTRE_OF_MASS_M * axis
        grasp_center[2] = float(center["z"])
        grasp_station = "cap_anchored_centre_of_mass"

    # Keep the centered grasp vertically through the object's middle while
    # retaining the measured 6 mm table-clearance guard.
    bottom = float(target_obb["center"]["z"] - target_obb["extent"]["z"])
    grasp_center[2] = max(
        float(target_obb["center"]["z"]),
        bottom + MIN_GRASP_CLEARANCE_FROM_BOTTOM_M,
    )

    center = {k: float(v) for k, v in zip(("x", "y", "z"), grasp_center, strict=True)}
    # The IK target is the grasp/TCP site, placed on the mesh's barrel rather
    # than at the silhouette centroid or the wider plunger flange.
    grasp = {"position": center, "rotation": rotation}
    workspace = ctx.tool("robot.describe_workspace")
    clearance = max(
        float(workspace["align_clearance_m"]),
        float(workspace.get("pregrasp_standoff_m", 0.0)),
    )
    align = {
        "position": {**grasp["position"], "z": float(center["z"] + clearance)},
        "rotation": rotation,
    }
    # Estimate the jaw target from the observed cross-section. The percentile
    # rejects sparse flange/cap pixels; the fallback is the barrel's own
    # measured hull width (see the module docstring) for when transparent
    # RGB-D is too incomplete for a stable estimate.
    transverse = points - origin - np.outer((points - origin) @ axis, axis)
    observed_width = 2.0 * float(np.percentile(np.linalg.norm(transverse, axis=1), 35))
    object_width = observed_width if 0.006 <= observed_width <= 0.025 else BARREL_WIDTH_FALLBACK_M
    return {"grasp_pose": grasp, "align_pose": align,
            "long_axis_world": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "grasp_station": grasp_station, "object_width_m": object_width}
