"""Straight-Z placement INTO a walled container: lift (Z) → XY over the
container → LINEAR descend to inside the walls.

The generalized form of ``grocery_packing/transport_move.py`` (the reference
recipe that scores on the VAB packing suites), for any container with an
interior (basket, bin, box, tote). Lift/XY are straight-line cartesian
(``robot.go_to_pose_cartesian``); the descend uses cuRobo's AXIS-CONSTRAINED
linear move (``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``,
``orientation_mode="LOCK"``) — a guaranteed straight vertical drop. The descend
lowers the TCP to ``container_top − place_offset`` (with ``place_offset`` small
or negative to sit just above the rim so the held object clears the walls on the
way down and drops in) so the object is placed IN the container, not from a
lateral swing. For placement ONTO a surface or into a described sub-region/zone,
use the ``compute_drop_pose → waypoint_move → descend_release_linear`` path
instead — this script is the walled-container fast path.
"""

import logging
import math
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Vec3

logger = logging.getLogger(__name__)

class Output(TypedDict):
    place_position: Vec3


def _cartesian(ctx: NodeContext, x: float, y: float, z: float, rotation: dict) -> None:
    pose = {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": rotation}
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:  # noqa: BLE001
        # A Cartesian servo can hit a joint-limit basin even when another arm
        # configuration reaches the same clear-space pose. Use the reusable
        # CuRobo single-pose planner for a genuine joint-space reconfiguration.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to curobo.plan_to_pose.", x, y, z, exc,
        )
        observation = ctx.tool("robot.get_observation")
        plan = ctx.tool(
            "curobo.plan_to_pose",
            target_pose=pose,
            start_joint_position=observation["arms"][0]["joint_state"],
        )
        if not plan.get("success") or not plan.get("trajectory"):
            raise RuntimeError("CuRobo could not plan the clear-space transport leg")
        ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"], max_steps_per_waypoint=60)
    reached = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    error = math.sqrt(
        (float(reached["x"]) - float(x)) ** 2
        + (float(reached["y"]) - float(y)) ** 2
        + (float(reached["z"]) - float(z)) ** 2
    )
    if error > 0.03:
        raise RuntimeError(
            f"transport leg did not reach its requested pose: error={error:.3f} m, "
            f"requested=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"reached=({reached['x']:.3f}, {reached['y']:.3f}, {reached['z']:.3f})"
        )


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    # FINGERTIP-frame heights. Distance is from_z - target_z, NOT
    # get_ee_pose().z - target_z: get_ee_pose returns the panda_hand link
    # (~0.10 m above the fingertip), which overshoots the descent by the TCP
    # offset and rams the object into the basket floor.
    dist = float(from_z) - float(target_z)
    if dist <= 0.002:
        return
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool("robot.execute_trajectory", trajectory=res["trajectory"], max_steps_per_waypoint=60)
    else:
        ee_pose = ctx.tool("robot.get_ee_pose")["pose"]
        ee = ee_pose["position"]
        _cartesian(ctx, ee["x"], ee["y"], target_z, ee_pose["rotation"])


def _plan_reachable_interior(
    ctx: NodeContext,
    center: dict,
    extent: dict,
    z: float,
    preferred_x: float,
    preferred_y: float,
    rotation: dict,
) -> tuple[float, float]:
    """Plan to a reachable point inside the perceived destination footprint."""
    margin = 0.015
    xr = max(0.0, float(extent["x"]) - margin)
    yr = max(0.0, float(extent["y"]) - margin)
    cx, cy = float(center["x"]), float(center["y"])
    candidates = [
        (preferred_x, preferred_y),
        (cx - xr, cy - yr), (cx - xr, cy + yr),
        (cx + xr, cy - yr), (cx + xr, cy + yr),
        (cx, cy),
    ]
    seen: set[tuple[float, float]] = set()
    for x, y in candidates:
        key = (round(x, 5), round(y, 5))
        if key in seen:
            continue
        seen.add(key)
        try:
            _cartesian(ctx, x, y, z, rotation)
            return x, y
        except Exception:  # noqa: BLE001
            logger.debug("destination candidate (%.3f, %.3f) unreachable", x, y)
    raise RuntimeError("CuRobo found no reachable point inside the destination OBB")


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = -1.0,
    lift_z: float = -1.0,
    place_offset: float = 0.06,
) -> Output:
    c = container_obb["center"]
    e = container_obb["extent"]
    place_z = float(c["z"]) + float(e["z"]) - float(place_offset)
    current_pose = ctx.tool("robot.get_ee_pose")["pose"]
    cur = current_pose["position"]
    # Preserve the current grasp orientation throughout lift, transfer, and
    # release; do not add a task-irrelevant yaw over a destination box.
    transport_rotation = current_pose["rotation"]
    # Prefer the compartment center. Perceived masks can slightly overestimate
    # a compartment footprint; clamping the current hand position into that
    # footprint biases release toward a divider and can look like a label swap.
    # _plan_reachable_interior still tries alternative interior points if this
    # center pose is genuinely unreachable.
    margin = 0.015
    x_radius = max(0.0, float(e["x"]) - margin)
    y_radius = max(0.0, float(e["y"]) - margin)
    bx = float(c["x"])
    by = float(c["y"])
    # Derive safe heights from perceived geometry unless the graph supplies
    # explicit world-frame heights. The previous defaults belonged to another
    # workspace and were below this task's table.
    if lift_z < 0.0:
        # Do not add clearance blindly above an already-clear grasp pose. That
        # pushed the fixed-down wrist to the upper workspace boundary in this
        # scene. Seven centimetres above the release pose clears the bin rim;
        # otherwise retain the current height so the lateral leg never descends.
        lift_z = max(place_z + 0.07, min(float(cur["z"]), place_z + 0.08))
    if transport_z < 0.0:
        # The release pose can itself be above the rim (negative place_offset),
        # so clearance must be measured from place_z rather than added twice.
        # Carry two centimetres higher than the previous trajectory so held
        # tools have a little more clearance over the source-bin and dividers.
        transport_z = place_z + 0.08
    # First lift the grasped item straight up so it clears the table before the
    # lateral move (avoids dragging it across the scene). lift_z is kept BELOW
    # transport_z on purpose, but even a MODEST in-place lift can be infeasible
    # at a far-edge grasp XY (x~0.8): the arm grasps low there yet has almost no
    # vertical room, so BOTH the straight-line and the planned solve fail. That
    # is exactly why this leg was dropped before. So the lift is BEST-EFFORT: if
    # it can't be solved we skip it and go straight up-and-over -- Seg 4 lifts
    # the item anyway as it moves toward the central, reachable basket. The lift
    # must never abort the place; only Segs 4/5 are essential.
    try:
        _cartesian(ctx, cur["x"], cur["y"], lift_z, transport_rotation)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[transport_move] in-place lift to z=%.3f infeasible at (%.3f, %.3f) "
            "(%s); skipping lift, going straight up-and-over.",
            lift_z, cur["x"], cur["y"], exc,
        )
    bx, by = _plan_reachable_interior(
        ctx, c, e, transport_z, bx, by, transport_rotation,
    )                                                   # Seg 4: up & over the basket
    _descend_linear(ctx, place_z, transport_z)         # Seg 5: descend INTO basket (constrained Z)
    return {"place_position": {"x": bx, "y": by, "z": place_z}}
