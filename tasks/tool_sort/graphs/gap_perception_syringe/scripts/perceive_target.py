"""Localize the syringe and its directed insertion feature from overhead RGB-D."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    insertion_pose: dict[str, Any]
    cap_center: dict[str, float]
    functional_feature: dict[str, Any]


def _overhead(observation):
    return next(c for c in observation["cameras"] if c.get("name") == "overhead")


def _mask(ctx, image, query, threshold, optional=False):
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=3)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        if optional:
            return None
        raise ValueError(f"could not perceive {query!r}")
    return np.asarray(result["masks"][0], dtype=np.uint8)


def _points(ctx, camera, mask):
    cloud = ctx.tool(
        "geometry.mask_to_world_points", mask=mask, depth=camera["depth"],
        intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _narrow_endpoint(points):
    origin = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
    axis = basis[0]
    along = (points - origin) @ axis
    tails = [points[along <= np.percentile(along, 15)], points[along >= np.percentile(along, 85)]]
    widths = []
    for tail in tails:
        centered = tail - np.median(tail, axis=0)
        perpendicular = centered - np.outer(centered @ axis, axis)
        widths.append(float(np.percentile(np.linalg.norm(perpendicular, axis=1), 90)))
    return np.median(tails[int(np.argmin(widths))], axis=0)


def run(ctx: NodeContext, observation: dict[str, Any],
        object_description: str = "syringe",
        direction_marker_description: str = "red needle cap of syringe",
        feature_type: str = "tip") -> Output:
    camera = _overhead(observation)
    target_mask = _mask(ctx, camera["rgb"], object_description, 0.08)
    points = _points(ctx, camera, target_mask)
    if len(points) < 30:
        raise ValueError("syringe mask contains too few depth points")
    obb = ctx.tool(
        "geometry.filter_and_compute_obb", points={"points": points.astype(np.float32)}
    )["obb"]
    cap_mask = _mask(ctx, camera["rgb"], direction_marker_description, 0.06, optional=True)
    cap_points = _points(ctx, camera, cap_mask) if cap_mask is not None else np.empty((0, 3))
    cap = np.median(cap_points, axis=0) if len(cap_points) >= 8 else _narrow_endpoint(points)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    axis = cap - center
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    along = (points - center) @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, axis))).as_quat()
    insertion_pose = {
        "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
        "rotation": {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])},
    }
    centered = points - tip
    radial = centered - np.outer(centered @ axis, axis)
    radius = float(np.percentile(np.linalg.norm(radial, axis=1), 35))
    return {
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": {"points": points.astype(np.float32)},
        "cap_center": {"x": float(cap[0]), "y": float(cap[1]), "z": float(cap[2])},
        "insertion_pose": insertion_pose,
        "functional_feature": {"kind": feature_type, "pose": insertion_pose,
            "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "radius_outer": max(radius, 1.0e-4), "confidence": 1.0,
            "description": direction_marker_description},
    }
