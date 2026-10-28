"""Express the perceived syringe insertion feature and collision body in TCP frame."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    insertion_in_hand: dict[str, Any]
    attached_object: dict[str, Any]


def _pose_matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    matrix[:3, 3] = [p["x"], p["y"], p["z"]]
    return matrix


def _matrix_pose(matrix: np.ndarray) -> dict[str, Any]:
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]), "z": float(matrix[2, 3])},
        "rotation": {"w": float(quat[3]), "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2])},
    }


def _attachment_spheres(world_points: Any, world_hand: np.ndarray) -> dict[str, Any]:
    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 8:
        raise ValueError("syringe attachment requires at least eight RGB-D points")
    hand_points = (
        np.linalg.inv(world_hand)
        @ np.column_stack((points, np.ones(len(points)))).T
    ).T[:, :3]
    import trimesh
    from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

    # MORPHIT optimizes sphere coverage on a deterministic, watertight hull.
    # The small radius contraction keeps the collision model usable in tight
    # fixture clearances while retaining the measured object extent.
    mesh = trimesh.points.PointCloud(hand_points).convex_hull
    result = fit_spheres_to_mesh(
        mesh, num_spheres=64, surface_radius=0.002,
        fit_type=SphereFitType.MORPHIT, iterations=200, compute_metrics=True,
    )
    centers = result.centers.detach().cpu().numpy().reshape(-1, 3)
    radii = result.radii.detach().cpu().numpy().reshape(-1)
    shrink = 0.002
    radii = np.maximum(radii - shrink, 0.001)
    valid = np.isfinite(centers).all(axis=1) & np.isfinite(radii) & (radii > 0.0)
    if not np.any(valid):
        raise RuntimeError("CuRobo MORPHIT fitting returned no syringe spheres")
    return {
        "frame": "tcp",
        "sphere_fit_type": SphereFitType.MORPHIT.value,
        "sphere_radius_shrink_m": shrink,
        "spheres": [
            {
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "radius": float(sphere_radius),
            }
            for center, sphere_radius in zip(centers[valid], radii[valid])
        ],
    }


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    fallback_insertion_pose: dict[str, Any],
    grasp_pose: dict[str, Any],
    target_cloud: dict[str, Any],
) -> Output:
    """Use wrist RGB-D when visible, with the overhead feature as a measured fallback."""
    # Before the object moves, its measured feature relative to the commanded
    # grasp frame is known.  Preserve that rigid transform as the fallback.
    # Transforming the old world-frame tip with the *post-lift* hand pose made a
    # failed wrist detection look like a large, false in-hand displacement.
    fallback_in_hand = np.linalg.inv(_pose_matrix(grasp_pose)) @ _pose_matrix(fallback_insertion_pose)
    feature_world = None
    attachment_points = np.asarray(target_cloud["points"], dtype=np.float64).reshape(-1, 3)
    attachment_in_grasp = (
        np.linalg.inv(_pose_matrix(grasp_pose))
        @ np.column_stack((attachment_points, np.ones(len(attachment_points)))).T
    ).T[:, :3]
    wrist_attachment = False

    wrist = None
    syringe = None
    cap = None
    best_score = -1.0
    # As in tool hanging, inspect every eye-in-hand view and use the strongest
    # usable feature observation instead of depending on camera list order.
    for candidate in observation.get("cameras", []):
        if "eye_in_hand" not in candidate.get("name", ""):
            continue
        candidate_syringe = ctx.tool(
            "sam3.segment_text", image=candidate["rgb"], query="syringe", max_results=3
        )
        score = float((candidate_syringe.get("scores") or [0.0])[0])
        if not candidate_syringe.get("masks") or score < 0.04 or score <= best_score:
            continue
        wrist, syringe, best_score = candidate, candidate_syringe, score
        cap = ctx.tool(
            "sam3.segment_text", image=candidate["rgb"],
            query="red needle cap of syringe", max_results=3,
        )
    if wrist is not None:
        assert syringe is not None and cap is not None
        if syringe.get("masks") and syringe.get("scores") and float(syringe["scores"][0]) >= 0.04:
            cloud = ctx.tool(
                "geometry.mask_to_world_points", mask=np.asarray(syringe["masks"][0], dtype=np.uint8),
                depth=wrist["depth"], intrinsics=wrist["intrinsics"], camera_pose=wrist["pose"],
            )["points"]
            points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(points) >= 20:
                attachment_points = points
                wrist_attachment = True
                if cap.get("masks") and cap.get("scores") and float(cap["scores"][0]) >= 0.04:
                    cap_cloud = ctx.tool(
                        "geometry.mask_to_world_points", mask=np.asarray(cap["masks"][0], dtype=np.uint8),
                        depth=wrist["depth"], intrinsics=wrist["intrinsics"], camera_pose=wrist["pose"],
                    )["points"]
                    cap_points = np.asarray(cap_cloud["points"], dtype=np.float64).reshape(-1, 3)
                    if len(cap_points) >= 6:
                        centered = points - np.median(points, axis=0)
                        _, _, basis = np.linalg.svd(centered, full_matrices=False)
                        axis = basis[0]
                        cap_center = np.median(cap_points, axis=0)
                        if float(axis @ (cap_center - np.median(points, axis=0))) < 0.0:
                            axis = -axis
                        along = centered @ axis
                        tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
                        reference = np.array([0.0, 0.0, 1.0])
                        if abs(float(reference @ axis)) > 0.9:
                            reference = np.array([1.0, 0.0, 0.0])
                        x_axis = np.cross(reference, axis); x_axis /= np.linalg.norm(x_axis)
                        y_axis = np.cross(axis, x_axis)
                        feature_world = np.eye(4)
                        feature_world[:3, :3] = np.column_stack((x_axis, y_axis, axis))
                        feature_world[:3, 3] = tip

    ee = ctx.tool("robot.get_ee_pose")["pose"]
    world_hand = _pose_matrix(ee)
    if not wrist_attachment:
        # Move the pre-grasp cloud with the same rigid grasp transform.  Leaving
        # it at its old table coordinates corrupts TCP-frame collision spheres
        # after the verification lift.
        attachment_points = (
            world_hand
            @ np.column_stack((attachment_in_grasp, np.ones(len(attachment_in_grasp)))).T
        ).T[:, :3]
    insertion_in_hand = fallback_in_hand if feature_world is None else np.linalg.inv(world_hand) @ feature_world
    return {
        "insertion_in_hand": _matrix_pose(insertion_in_hand),
        "attached_object": _attachment_spheres(attachment_points, world_hand),
    }
