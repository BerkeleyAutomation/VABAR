"""Express the perceived syringe insertion feature and collision body in TCP frame."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    insertion_in_hand: dict[str, Any]
    attached_object: dict[str, Any]


CAP_SCORE_MIN = 0.04
"""SAM-3 confidence floor for the cap in a wrist view. Deliberately the overhead
gate, which a wrist view essentially never clears: admitted at 0.008 the cap
was found at the *wrong end* (the flange) and the tip direction flipped, so the
tip's direction stays the overhead measurement's and the wrist cloud refines
only its sideways position -- see the lateral refinement below."""

HELD_EXTENT_M = 0.10
"""Farthest a point of the held barrel can be from the cloud's centre [m]: a
150 mm syringe's half-length with room for the flange and the cap."""

SNAP_TO_JAW_AXIS_MAX_DEG = 15.0
"""A top-down pinch across a cylinder puts the barrel along one jaw axis by
construction, and the record bears it out: over 22 episodes (2026-09-09) the
true barrel sat 0.3-1.5 deg from the hand's y axis. The overhead silhouette's
SVD, which this node carried as the barrel direction, pitched it by up to
4.3 deg (a -0.088 z-component in seven of the 22), and the wrist turn stood
*that* axis up: those seven episodes carried the syringe 7-8 deg off vertical
and one (t20) struck the lid on the way in. So the direction is snapped to
the nearest hand axis when the estimate is within this many degrees of one;
an estimate further off than that is left alone rather than trusted."""


def _snap_to_jaw_axis(feature_in_hand: np.ndarray) -> np.ndarray:
    """Rotate ``feature_in_hand`` so its z (the barrel) lies exactly on the nearest hand axis."""
    z = feature_in_hand[:3, 2] / max(float(np.linalg.norm(feature_in_hand[:3, 2])), 1.0e-12)
    axes = [np.array(v, dtype=np.float64) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
    target = max(axes, key=lambda a: float(a @ z))
    cos_angle = float(np.clip(target @ z, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if np.degrees(angle) > SNAP_TO_JAW_AXIS_MAX_DEG or angle < 1.0e-6:
        return feature_in_hand
    pivot = np.cross(z, target)
    pivot /= max(float(np.linalg.norm(pivot)), 1.0e-12)
    snapped = feature_in_hand.copy()
    snapped[:3, :3] = Rotation.from_rotvec(pivot * angle).as_matrix() @ feature_in_hand[:3, :3]
    return snapped


HELD_MAX_DISTANCE_M = 0.12
"""Farthest a cloud may sit from the TCP and still be the barrel in the fingers [m]:
the held syringe is within a few centimetres, the tray's spare tens."""


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([float(value[k]) for k in ("x", "y", "z")], dtype=np.float64)


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
    held_points = None
    # The held syringe is the one in the fingers, not the one SAM scores
    # highest: with a spare in the tray 90 mm away the two score within 0.01
    # of each other and the top mask is a coin flip. Back-project every
    # candidate and keep the cloud nearest the hand -- the held barrel sits a
    # couple of centimetres from the TCP, the spare tens.
    hand_now = _vec(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    best_distance = HELD_MAX_DISTANCE_M
    for candidate in observation.get("cameras", []):
        if "eye_in_hand" not in candidate.get("name", ""):
            continue
        detected = ctx.tool("sam3.segment_text", image=candidate["rgb"], query="syringe", max_results=3)
        for mask, score in zip(detected.get("masks") or [], detected.get("scores") or []):
            if float(score) < 0.04:
                continue
            cloud = ctx.tool(
                "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
                depth=candidate["depth"], intrinsics=candidate["intrinsics"], camera_pose=candidate["pose"],
            )["points"]
            points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(points) < 20:
                continue
            distance = float(np.linalg.norm(np.median(points, axis=0) - hand_now))
            if distance < best_distance:
                wrist, held_points, best_distance = candidate, points, distance
    if wrist is not None:
        # The mask's cloud carries what the mask covered *behind* the barrel --
        # tray floor seen past it -- and a convex hull over those far points is
        # what the sphere fit would cover. Keep what lies within a barrel's
        # length of the cloud's robust centre and nothing further.
        held_center = np.median(held_points, axis=0)
        points = held_points[np.linalg.norm(held_points - held_center, axis=1) <= HELD_EXTENT_M]
        held_center = np.median(points, axis=0)
        attachment_points = points
        wrist_attachment = True
        cap = ctx.tool(
            "sam3.segment_text", image=wrist["rgb"], query="red needle cap of syringe", max_results=3,
        )
        # The cap that belongs to *this* barrel: nearest its cloud, same rule.
        cap_points = np.empty((0, 3))
        cap_distance = HELD_MAX_DISTANCE_M
        for mask, score in zip(cap.get("masks") or [], cap.get("scores") or []):
            # A cap is a few dozen wrist pixels; recorded true positives score
            # ~0.01, and the nearest-to-barrel rule below does the choosing.
            if float(score) < CAP_SCORE_MIN:
                continue
            cap_cloud = ctx.tool(
                "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
                depth=wrist["depth"], intrinsics=wrist["intrinsics"], camera_pose=wrist["pose"],
            )["points"]
            candidate_points = np.asarray(cap_cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(candidate_points) < 6:
                continue
            distance = float(np.linalg.norm(np.median(candidate_points, axis=0) - held_center))
            if distance < cap_distance:
                cap_points, cap_distance = candidate_points, distance
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
    if feature_world is None and wrist_attachment and len(points) >= 20:
        # No cap seen from the wrist, so the tip's *direction* stays the
        # overhead measurement's -- but its sideways position need not: the
        # barrel's line is right here in the wrist cloud. Slide the fallback
        # tip onto that line, keeping its distance along it. The lateral
        # error this removes is what put the lower end on the lid: the grasp
        # never closes exactly where it was planned, and a few millimetres of
        # barrel offset become the whole margin the aperture allows.
        centered = points - held_center
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        line = basis[0] / np.linalg.norm(basis[0])
        fallback_world = world_hand @ fallback_in_hand
        tip = fallback_world[:3, 3]
        tip_on_line = held_center + line * float((tip - held_center) @ line)
        fallback_world[:3, 3] = tip_on_line
        feature_world = fallback_world
    insertion_in_hand = fallback_in_hand if feature_world is None else np.linalg.inv(world_hand) @ feature_world
    insertion_in_hand = _snap_to_jaw_axis(insertion_in_hand)
    return {
        "insertion_in_hand": _matrix_pose(insertion_in_hand),
        "attached_object": _attachment_spheres(attachment_points, world_hand),
    }
