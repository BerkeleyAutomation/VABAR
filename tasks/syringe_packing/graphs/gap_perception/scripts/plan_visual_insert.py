"""Reobserve the held syringe and plan a straight feature-to-aperture insertion."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    placement_plan: dict[str, Any]


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    matrix[:3, 3] = [p["x"], p["y"], p["z"]]
    return matrix


def _visible_tip(ctx: NodeContext, observation: dict[str, Any], object_description: str,
                 direction_marker_description: str) -> np.ndarray | None:
    """Return the best wrist-observed physical syringe endpoint."""
    best = None
    for camera in observation.get("cameras", []):
        if "eye_in_hand" not in camera.get("name", ""):
            continue
        syringe = ctx.tool("sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3)
        cap = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=direction_marker_description, max_results=3
        )
        score = float((syringe.get("scores") or [0.0])[0])
        if not syringe.get("masks") or not cap.get("masks") or score < 0.04:
            continue
        cap_score = float((cap.get("scores") or [0.0])[0])
        if cap_score < 0.04 or (best is not None and score + cap_score <= best[0]):
            continue
        best = (score + cap_score, camera, syringe["masks"][0], cap["masks"][0])
    if best is None:
        return None
    _, camera, syringe_mask, cap_mask = best
    def points(mask):
        cloud = ctx.tool(
            "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
        )["points"]
        return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    syringe_points, cap_points = points(syringe_mask), points(cap_mask)
    if len(syringe_points) < 20 or len(cap_points) < 6:
        return None
    center = np.median(syringe_points, axis=0)
    _, _, basis = np.linalg.svd(syringe_points - center, full_matrices=False)
    axis = basis[0]
    if float(axis @ (np.median(cap_points, axis=0) - center)) < 0.0:
        axis = -axis
    along = (syringe_points - center) @ axis
    return np.median(syringe_points[along >= np.percentile(along, 98)], axis=0)


HELD_MAX_DISTANCE_M = 0.15
"""Farthest a syringe cloud may sit from the hand and still be the held one [m]."""


def _visible_leading_end(ctx: NodeContext, observation: dict[str, Any], object_description: str,
                         hand: np.ndarray, axis: np.ndarray) -> np.ndarray | None:
    """The held syringe's end that leads along ``axis``, seen from a fixed side camera.

    The perception twin of the ground-truth graph's ``lowest_z``: the barrel
    hangs vertically below the hand at the transit pose, and a side view sees
    its whole length, so the points farthest along the fixture axis are the
    end that will meet the hole. Aimed directly, this replaces a guess about
    where the flange end sits relative to the tip.
    """
    best = None
    for camera in observation.get("cameras", []):
        if camera.get("name") != "agentview" or "depth" not in camera:
            continue
        detected = ctx.tool("sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3)
        for mask, score in zip(detected.get("masks") or [], detected.get("scores") or []):
            if float(score) < 0.04:
                continue
            cloud = ctx.tool(
                "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
                depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
            )["points"]
            points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(points) < 20:
                continue
            distance = float(np.linalg.norm(np.median(points, axis=0) - hand))
            if distance <= HELD_MAX_DISTANCE_M and (best is None or distance < best[0]):
                best = (distance, points)
    if best is None:
        return None
    # The mask's cloud can carry background pixels metres away; only points
    # within a barrel's reach of the hand can be the held syringe, and only
    # an end within that reach is believed.
    points = best[1]
    points = points[np.linalg.norm(points - hand, axis=1) <= HELD_MAX_DISTANCE_M]
    if len(points) < 20:
        return None
    along = points @ axis
    end = np.median(points[along >= np.percentile(along, 95)], axis=0)
    if float(np.linalg.norm(end - hand)) > HELD_MAX_DISTANCE_M:
        return None
    return end


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    insertion_in_hand: dict[str, Any],
    aperture: dict[str, float],
    fixture_axis: dict[str, float],
    rim_z: float,
    world_config: dict[str, Any],
    attached_object: dict[str, Any],
    relation: str = "shaft_into_aperture",
    object_description: str = "syringe",
    direction_marker_description: str = "red needle cap of syringe",
) -> Output:
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    ee_matrix = _matrix(ee)
    hand_feature = _matrix(insertion_in_hand)
    rotation = ee_matrix[:3, :3]

    # The fixture stage projected the aperture mask onto the measured lid
    # plane. Preserve that feature instead of snapping back to the box center.
    center = np.array([aperture["x"], aperture["y"], rim_z], dtype=np.float64)
    axis = np.array([fixture_axis[k] for k in ("x", "y", "z")], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)

    # Aim whichever end of the barrel is pointing down the fixture axis. The
    # wrist-only stand-up pitches the hand the way its range allows, which
    # may leave the needle end up; the syringe goes through the hole either
    # way, but the hand has to be placed for the end that leads. The grasp is
    # at the barrel's centre, so the far end is the tip's offset mirrored
    # through the TCP. The red cap refines the aim only when the tip leads.
    tip_hand = hand_feature[:3, 3]
    tip_leads = float((rotation @ tip_hand) @ axis) >= float((rotation @ -tip_hand) @ axis)
    feature_hand = tip_hand if tip_leads else -tip_hand
    aim_source = "tip offset" if tip_leads else "tip offset mirrored"
    # Best: the leading end seen from the side, whichever end it is. Next: the
    # red cap from the wrist, when the tip leads. Last: the offset above.
    leading = _visible_leading_end(ctx, observation, object_description, ee_matrix[:3, 3], axis)
    if leading is not None:
        feature_hand = rotation.T @ (leading - ee_matrix[:3, 3])
        aim_source = "side view"
    elif tip_leads:
        visible_tip = _visible_tip(ctx, observation, object_description, direction_marker_description)
        if visible_tip is not None:
            feature_hand = rotation.T @ (visible_tip - ee_matrix[:3, 3])
            aim_source = "wrist cap"

    def hand_for(feature_target: np.ndarray) -> dict[str, float]:
        position = feature_target - rotation @ feature_hand
        return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}

    # The clearance-motion skill already translated to the staging pose above
    # the aperture. This plan contains only local approach and engagement.
    targets = [center - 0.020 * axis, center + 0.035 * axis]
    waypoints = []
    for index, target in enumerate(targets):
        waypoint = {
            "pose": {"position": hand_for(target), "rotation": dict(ee["rotation"])},
            # Both legs are straight, orientation-locked strokes along the
            # fixture axis: the hand is already directly above the aperture, so
            # the approach is a 60 mm descent and nothing else. As a free
            # joint-space goal it was measured re-solving the arm into another
            # elbow branch for that descent -- joints 1, 3 and 5 swinging
            # 0.8-1.2 rad, the wrist turning about the pinch and the syringe
            # pivoting 40 degrees in the fingers before the stroke began.
            "cartesian": True,
            "mode": "planned_linear",
        }
        if index == len(targets) - 1:
            waypoint["allow_goal_contact"] = True
        waypoints.append(waypoint)
    return {
        "placement_plan": {
            "waypoints": waypoints,
            "world_config": world_config,
            "attached_object": attached_object,
            "aim_source": aim_source,
            "leading_end_in_hand": {"x": float(feature_hand[0]), "y": float(feature_hand[1]), "z": float(feature_hand[2])},
        }
    }
