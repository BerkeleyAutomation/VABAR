"""Fit the artist plug's orange shell, with independent depth support at its gray tip."""

import numpy as np


def fit_plug_shell(points, shape, axis_hint, up_hint, scene_points):
    """Return the shoulder frame; local +Z points from the shell into the tip.

    Keep the RSS surface-registration gates and update law. The observed
    shell-front edge adds the axial constraint absent from a side-only cloud.
    Require visible tip depth at the resulting pose to reject an occlusion edge
    mistaken for the shoulder. No robot or simulated object pose is consumed.
    """
    import trimesh
    from scipy.spatial.transform import Rotation

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 150:
        return None
    up = np.array(up_hint, dtype=float, copy=True)
    axis = np.array(axis_hint, dtype=float, copy=True)
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        return None
    up /= np.linalg.norm(up)
    axis -= up * (axis @ up)
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        return None
    axis /= np.linalg.norm(axis)
    rotation = np.column_stack((up, np.cross(axis, up), axis))
    vertices = np.asarray(shape["surface_vertices"])
    faces = np.asarray(shape["surface_triangles"])
    shell_faces = faces[np.max(vertices[faces, 2], axis=1) <= 1e-7]
    shell = trimesh.Trimesh(vertices, shell_faces, process=False)
    shell.remove_unreferenced_vertices()

    top = np.percentile(points @ up, 90)
    face = points[np.abs(points @ up - top) < 0.0007]
    if len(face) < 150:
        return None
    front = np.percentile(face @ axis, 99)
    edge = face[face @ axis >= front - 0.0007]
    if len(edge) < 8 or np.ptp(edge @ rotation[:, 1]) < 0.5 * shell.extents[1]:
        return None
    translation = rotation @ np.array(
        [
            top - shell.bounds[1, 0],
            np.mean(np.percentile(face @ rotation[:, 1], [2, 98])),
            front,
        ]
    )
    initial = translation.copy()
    points = points[:: max(1, len(points) // 2000)]
    for iteration in range(35):
        local = (points - translation) @ rotation
        closest, distance, triangles = trimesh.proximity.closest_point(shell, local)
        take = distance < max(0.001, 0.010 * 0.88**iteration)
        if take.sum() < 150:
            return None
        x, normal = local[take], shell.face_normals[triangles[take]]
        residual = np.sum((closest[take] - x) * normal, axis=1)
        jacobian = np.column_stack((np.cross(x, normal), normal))
        local_edge = (edge - translation) @ rotation
        edge_normal = np.tile([0.0, 0.0, 1.0], (len(edge), 1))
        jacobian = np.vstack(
            [
                jacobian,
                np.column_stack((np.cross(local_edge, edge_normal), edge_normal)),
            ]
        )
        residual = np.r_[residual, -local_edge[:, 2]]
        step = np.linalg.lstsq(jacobian, residual, rcond=1e-6)[0]
        translation -= rotation @ step[3:]
        rotation = rotation @ Rotation.from_rotvec(-step[:3]).as_matrix()
    rms = float(np.sqrt(np.mean(residual**2)))
    rank = np.linalg.matrix_rank(
        jacobian * np.array([1 / 0.03] * 3 + [1.0] * 3), tol=1e-3
    )
    if (
        rank < 6
        or rms > 0.00035
        or take.mean() < 0.7
        or np.linalg.norm(translation - initial) > 0.010
    ):
        return None

    tip_faces = faces[np.min(vertices[faces, 2], axis=1) >= -1e-7]
    tip = trimesh.Trimesh(vertices, tip_faces, process=False)
    tip.remove_unreferenced_vertices()
    scene = np.asarray(scene_points, dtype=float).reshape(-1, 3)
    local_scene = (scene[np.isfinite(scene).all(axis=1)] - translation) @ rotation
    # Check the visible upper tip face, away from its shoulder and end edges.
    candidates = local_scene[
        (local_scene[:, 2] > 0.15 * tip.bounds[1, 2])
        & (local_scene[:, 2] < 0.85 * tip.bounds[1, 2])
        & (np.abs(local_scene[:, 1]) < tip.bounds[1, 1] + 0.001)
        & (local_scene[:, 0] > 0.6 * tip.bounds[1, 0])
        & (local_scene[:, 0] < tip.bounds[1, 0] + 0.001)
    ]
    if len(candidates) < 40:
        return None
    _, distance, _ = trimesh.proximity.closest_point(tip, candidates)
    support = distance < 0.001
    if support.sum() < 40 or support.mean() < 0.7:
        return None
    return {
        "point": translation.tolist(),
        "axis": rotation[:, 2].tolist(),
        "up": rotation[:, 0].tolist(),
        "residual_m": rms,
        "inlier_count": int(take.sum()),
        "tip_support_count": int(support.sum()),
    }
