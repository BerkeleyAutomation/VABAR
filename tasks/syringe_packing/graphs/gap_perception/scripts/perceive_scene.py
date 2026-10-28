"""Perceive a syringe and the aperture of a sharps container from RGB-D."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    needle_cap_center: dict[str, float]
    insertion_pose: dict[str, Any]
    box_mask: np.ndarray
    aperture: dict[str, float]
    rim_z: float


def _camera(observation, name):
    return next(camera for camera in observation["cameras"] if camera.get("name") == name)


def _mask(ctx, image, query, threshold):
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        raise ValueError(f"could not perceive {query!r}")
    return np.asarray(result["masks"][0], dtype=np.uint8)


def _optional_mask(ctx, image, query, threshold):
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        return None
    return np.asarray(result["masks"][0], dtype=np.uint8)


def _narrow_endpoint(points):
    """Return the syringe endpoint opposite its visibly wider flange."""
    origin = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
    axis = basis[0]
    along = (points - origin) @ axis
    low_cut, high_cut = np.percentile(along, [15.0, 85.0])
    tails = [points[along <= low_cut], points[along >= high_cut]]
    widths = []
    for tail in tails:
        centered = tail - np.median(tail, axis=0)
        perpendicular = centered - np.outer(centered @ axis, axis)
        widths.append(float(np.percentile(np.linalg.norm(perpendicular, axis=1), 90.0)))
    # The cap/needle end is narrow; the plunger flange is the wide endpoint.
    return np.median(tails[int(np.argmin(widths))], axis=0)


def _points(ctx, camera, mask):
    cloud = ctx.tool(
        "geometry.mask_to_world_points", mask=mask, depth=camera["depth"],
        intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def run(ctx: NodeContext, observation: dict[str, Any]) -> Output:
    overhead = _camera(observation, "overhead")
    syringe_mask = _mask(ctx, overhead["rgb"], "syringe", 0.08)
    syringe_points = _points(ctx, overhead, syringe_mask)
    if len(syringe_points) < 30:
        raise ValueError("syringe mask contains too few depth points")
    target_obb = ctx.tool(
        "geometry.filter_and_compute_obb", points={"points": syringe_points.astype(np.float32)}
    )["obb"]

    # The red needle cap is a small but highly distinctive directional
    # feature.  It resolves the arbitrary PCA/OBB long-axis sign so downstream
    # grasping always moves toward the opposite, plunger-flange end.
    cap_mask = _optional_mask(ctx, overhead["rgb"], "red needle cap of syringe", 0.06)
    cap_points = _points(ctx, overhead, cap_mask) if cap_mask is not None else np.empty((0, 3))
    cap_center = np.median(cap_points, axis=0) if len(cap_points) >= 8 else _narrow_endpoint(syringe_points)
    object_center = np.array([target_obb["center"][k] for k in ("x", "y", "z")])
    insertion_axis = cap_center - object_center
    insertion_axis /= max(float(np.linalg.norm(insertion_axis)), 1.0e-12)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ insertion_axis)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, insertion_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(insertion_axis, x_axis)
    feature_q = Rotation.from_matrix(np.column_stack((x_axis, y_axis, insertion_axis))).as_quat()

    box_mask = _mask(ctx, overhead["rgb"], "yellow sharps disposal box", 0.08)
    box_points = _points(ctx, overhead, box_mask)
    box_obb = ctx.tool(
        "geometry.filter_and_compute_obb", points={"points": box_points.astype(np.float32)}
    )["obb"]
    rim_z = float(box_obb["center"]["z"] + box_obb["extent"]["z"])

    hole_mask = _mask(ctx, overhead["rgb"], "hole in top of yellow sharps box", 0.04)
    hole_points = _points(ctx, overhead, hole_mask)
    if len(hole_points) < 8:
        raise ValueError("aperture mask contains too few depth points")
    # With the selected near-vertical task camera, the robust interior-depth XY
    # centre is the validated insertion target; lid height comes from the box.
    # The supplied box mesh places the circular opening at the lid centre.
    # Register that known feature through the perceived box OBB rather than
    # using interior hole depth, whose XY changes with occlusion/parallax.
    center = np.array([box_obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    return {
        "target_obb": target_obb,
        "target_mask": syringe_mask,
        "target_cloud": {"points": syringe_points.astype(np.float32)},
        "needle_cap_center": {
            "x": float(cap_center[0]), "y": float(cap_center[1]), "z": float(cap_center[2]),
        },
        "insertion_pose": {
            "position": {"x": float(cap_center[0]), "y": float(cap_center[1]), "z": float(cap_center[2])},
            "rotation": {"w": float(feature_q[3]), "x": float(feature_q[0]), "y": float(feature_q[1]), "z": float(feature_q[2])},
        },
        "box_mask": box_mask,
        "aperture": {"x": float(center[0]), "y": float(center[1]), "z": rim_z},
        "rim_z": rim_z,
    }
