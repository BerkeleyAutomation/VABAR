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
    feature_hand = hand_feature[:3, 3]
    rotation = ee_matrix[:3, :3]
    visible_tip = _visible_tip(ctx, observation, object_description, direction_marker_description)
    if visible_tip is not None:
        feature_hand = rotation.T @ (visible_tip - ee_matrix[:3, 3])

    # The fixture stage projected the aperture mask onto the measured lid
    # plane. Preserve that feature instead of snapping back to the box center.
    center = np.array([aperture["x"], aperture["y"], rim_z], dtype=np.float64)
    axis = np.array([fixture_axis[k] for k in ("x", "y", "z")], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)

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
            # Both poses outside the aperture are collision-aware free-space
            # goals. Only crossing the fixture plane is an insertion stroke.
            "cartesian": index == len(targets) - 1,
            "mode": "planned_linear" if index == len(targets) - 1 else "planned_joint",
        }
        if waypoint["cartesian"]:
            waypoint["allow_goal_contact"] = True
        waypoints.append(waypoint)
    return {
        "placement_plan": {
            "waypoints": waypoints,
            "world_config": world_config,
            "attached_object": attached_object,
        }
    }
