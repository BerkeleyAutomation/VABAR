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
    preclose_width_m: float


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
FINGERTIP_BEYOND_TCP_M = (
    (0.0131, 0.0179), (0.0176, 0.0178), (0.0223, 0.0176), (0.0268, 0.0173), (0.0309, 0.0170),
    (0.0355, 0.0165), (0.0436, 0.0155), (0.0502, 0.0144), (0.0576, 0.0129), (0.0648, 0.0112),
    (0.0707, 0.0095), (0.0766, 0.0077), (0.0818, 0.0059),
)
"""(jaw inner gap, fingertip distance beyond the TCP) [m] for the Robotiq 2F-85,
measured from recorded pad poses on ``tray_clutter_t03``. The fingertips swing
down as the jaw closes: +5.9 mm beyond the TCP fully open, +17.0 mm at 30 mm.
The descent is made pre-closed, so the grasp TCP rises by the difference from
open. That keeps ``finger.reach_m`` + ``clearance_m`` off the floor.

**This table is one hand's.** It describes a four-bar linkage's arc, and
:func:`fingertip_drop` applies it only to a hand that declares itself geared."""
OPEN_FINGERTIP_BEYOND_TCP_M = 0.0059
PRECLOSE_MAX_WIDTH_M = 0.030
"""The widest descent jaw. It still leaves +/-10 mm for the barrel to enter."""
FINGER_BAND_PAST_JAW_M = 0.0085
"""How far the fingertip band reaches past each side of the jaw's inner gap
(measured: a 30.9 mm gap spans +/-23.5 mm; a 13.1 mm gap spans +/-14.8 mm)."""
PRECLOSE_NEIGHBOUR_MARGIN_M = 0.002
PRECLOSE_BARREL_MARGIN_M = 0.006
"""The narrowest descent jaw is the barrel plus this, so +/-3 mm for it to enter."""


def preclose_width(object_width: float, jaw_clearance: float | None) -> float:
    """Descent jaw width that keeps the fingertip band off the nearest neighbour."""
    fit = PRECLOSE_MAX_WIDTH_M if jaw_clearance is None else (
        2.0 * (float(jaw_clearance) - FINGER_BAND_PAST_JAW_M - PRECLOSE_NEIGHBOUR_MARGIN_M))
    return float(np.clip(fit, object_width + PRECLOSE_BARREL_MARGIN_M, PRECLOSE_MAX_WIDTH_M))


def fingertip_drop(width: float, gripper: dict[str, Any]) -> float:
    """How much further past the TCP the fingertips sit pre-closed to ``width`` than open [m].

    A four-bar hand's pads ride an arc: closing the jaw swings them *down*, so a
    descent made pre-closed puts the fingertips lower than ``finger.reach_m``
    (the open-hand figure) says, and the grasp TCP has to rise by the difference
    or the pads plough the surface. A **prismatic** jaw has no arc -- its pads
    only translate across the jaw -- so its fingertip sits at ``finger.reach_m``
    at every width and there is nothing to compensate. Which one is in the hand
    is the hand's own declaration, not this graph's guess:
    ``describe_gripper``'s ``kind`` is :attr:`GripperSpec.geared
    <robosimstudio.specs.robot.GripperSpec.geared>`, whose own docstring is
    "a 2F-85's driven revolute sweeps an arc and the pads follow it".

    Adding a linkage's arc to a prismatic hand is what stopped this graph
    measuring at all. This suite's documents declare the export's own Panda, and
    ``robot.describe_gripper`` answers ``franka_hand_robosuite``, ``kind``
    *prismatic*, ``finger.reach_m`` 9.3 mm. On ``tray_pair_both_t00``
    (2026-09-16) the table returned 11.2 mm at the 30 mm descent jaw and the
    grasp TCP was commanded to z 0.8065 instead of 0.7953: the fingertips shut
    at 0.7972, 3.4 mm *above* a barrel whose top is 0.7938. The captured tape
    has ``syringe_0`` sitting at 0.7883 for all 2728 steps while the hand lifted
    200 mm empty -- and ``hand_map_sg.map``, the only node that looked, refused
    the empty hand. 15 of 15 campaign documents died there.
    """
    if str(gripper.get("kind", "")) != "geared":
        return 0.0
    gaps, beyond = zip(*FINGERTIP_BEYOND_TCP_M, strict=True)
    return float(np.interp(width, gaps, beyond)) - OPEN_FINGERTIP_BEYOND_TCP_M
BARREL_WIDTH_FALLBACK_M = 0.0107
MIN_GRASP_CLEARANCE_FROM_BOTTOM_M = 0.006


def run(ctx: NodeContext, target_obb: dict[str, Any], target_cloud: dict[str, Any],
        needle_cap_center: dict[str, float], support_z: float | None = None,
        jaw_clearance_m: float | None = None) -> Output:
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
        # A global cap hint must never redefine the selected shaft's yaw.
        if float(axis @ (obb_center - cap)) < 0.0:
            axis = -axis
        short_world = np.array([-axis[1], axis[0], 0.0])
        short_world /= np.linalg.norm(short_world)
        rotation = top_down_rotation(short_world)
        # Project the endpoint onto this shaft before choosing a station.
        cap = origin + axis * float((cap - origin) @ axis)
        grasp_center = cap + CAP_TO_CENTRE_OF_MASS_M * axis
        grasp_center[2] = float(center["z"])
        grasp_station = "cap_anchored_centre_of_mass"

    finger = gripper.get("finger") or {}
    transverse = points - origin - np.outer((points - origin) @ axis, axis)
    observed_width = 2.0 * float(np.percentile(np.linalg.norm(transverse, axis=1), 35))
    object_width = observed_width if 0.006 <= observed_width <= 0.025 else BARREL_WIDTH_FALLBACK_M
    descent_width = preclose_width(object_width, jaw_clearance_m)
    drop = fingertip_drop(descent_width, gripper)
    if support_z is not None and finger.get("stated"):
        # Height from the perceived support surface (perceive_target's ring
        # beside the syringe), using open-robot-skills' own rule for keeping
        # the fingertip below the TCP out of the surface: support + reach +
        # clearance. The OBB-derived height this replaces followed floor pixels
        # mixed into the target cloud down to 1.5 mm above the tray floor, and
        # the pads drove syringes into it (tray_clutter_t03).
        grasp_center[2] = (
            float(support_z) + float(finger.get("reach_m", 0.0)) + float(finger.get("clearance_m", 0.0))
            + drop
        )
    else:
        # Keep the centered grasp vertically through the object's middle while
        # retaining the measured 6 mm table-clearance guard.
        bottom = float(target_obb["center"]["z"] - target_obb["extent"]["z"])
        grasp_center[2] = max(
            float(target_obb["center"]["z"]),
            bottom + MIN_GRASP_CLEARANCE_FROM_BOTTOM_M,
        ) + drop

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
    return {"grasp_pose": grasp, "align_pose": align,
            "long_axis_world": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "grasp_station": grasp_station, "object_width_m": object_width,
            "preclose_width_m": descent_width}

