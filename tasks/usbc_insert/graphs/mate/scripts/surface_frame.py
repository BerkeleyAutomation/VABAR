"""Register current RGB-D surfaces to the connectors' known local geometry."""

import numpy as np


def _register(points, shape, rotation, translation):
    import trimesh
    from scipy.spatial.transform import Rotation

    if len(points) < 150:
        return None
    points = points[:: max(1, len(points) // 2000)]
    mesh = trimesh.Trimesh(shape["surface_vertices"], shape["surface_triangles"], process=False)
    rotation, translation = rotation.copy(), translation.copy()
    for iteration in range(35):
        local = (points - translation) @ rotation
        closest, distance, triangles = trimesh.proximity.closest_point(mesh, local)
        take = distance < max(0.001, 0.010 * 0.88**iteration)
        if take.sum() < 150:
            return None
        x, normal = local[take], mesh.face_normals[triangles[take]]
        residual = np.sum((closest[take] - x) * normal, axis=1)
        jacobian = np.column_stack((np.cross(x, normal), normal))
        step = np.linalg.lstsq(jacobian, residual, rcond=1.0e-6)[0]
        # Moving the model induces the inverse displacement on observed points.
        translation -= rotation @ step[3:]
        rotation = rotation @ Rotation.from_rotvec(-step[:3]).as_matrix()
    rms = float(np.sqrt(np.mean(residual**2)))
    # A flat patch can have zero residual without locating the mouth. Require
    # translations and both axis directions; the barrel's roll may be ambiguous.
    rank = np.linalg.matrix_rank(jacobian * np.array([1 / 0.03] * 3 + [1.0] * 3), tol=1.0e-3)
    if rank < 5 or rms > 0.00035 or take.mean() < 0.7:
        return None
    return {
        "rotation": rotation,
        "translation": translation,
        "residual_m": rms,
        "inlier_count": int(take.sum()),
        "rank": rank,
    }


def fit_receptacle(points, shape, mouth_hint, axis_hint):
    """Fit a fresh socket frame; use the visible extent to recover from stale height."""
    mouth_hint = np.asarray(mouth_hint, dtype=float)
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.linalg.norm(points - mouth_hint, axis=1) < 0.080]
    if len(points) < 150:
        return None
    axis = np.array(axis_hint, dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    up -= axis * (up @ axis)
    up /= np.linalg.norm(up)
    across = np.cross(axis, up)
    # The bulkhead's local +x is insertion and +y is the slot's wide axis.
    rotation = np.column_stack((axis, up, across))
    local_mouth = np.asarray(shape["mouth_local"])
    # The silhouette is only an initialization. The complete 3D surface fit
    # determines the result, including pitch, flange steps and mouth offset.
    rough = (
        np.percentile(points @ axis, 2.0) * axis
        + np.mean(np.percentile(points @ across, [2.0, 98.0])) * across
        + (np.percentile(points @ up, 99.0) - shape["flange_radius"]) * up
    )
    candidates = []
    for initial in (mouth_hint, rough):
        fit = _register(points, shape, rotation, initial - rotation @ local_mouth)
        if fit is None:
            continue
        mouth = fit["translation"] + fit["rotation"] @ local_mouth
        direction = fit["rotation"][:, 0]
        turn = np.degrees(np.arccos(np.clip(direction @ axis, -1.0, 1.0)))
        if np.linalg.norm(mouth - mouth_hint) <= 0.060 and turn <= 35.0:
            candidates.append(
                {
                    "mouth": mouth.tolist(),
                    "axis": direction.tolist(),
                    "residual_m": fit["residual_m"],
                    "inlier_count": fit["inlier_count"],
                }
            )
            if fit["residual_m"] < 0.00008:
                break
    return min(candidates, key=lambda fit: fit["residual_m"]) if candidates else None


def fit_plug(points, shape, point_hint, axis_hint, up_hint):
    """Locate the shoulder and keyed attitude without a pitched silhouette bias."""
    axis = np.array(axis_hint, dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    up = np.array(up_hint, dtype=float, copy=True)
    up -= axis * (up @ axis)
    up /= np.linalg.norm(up)
    rotation = np.column_stack((up, np.cross(axis, up), axis))
    point_hint = np.asarray(point_hint, dtype=float)
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.linalg.norm(points - point_hint, axis=1) < 0.060]
    fit = _register(points, shape, rotation, point_hint)
    if fit is None or fit["rank"] < 6 or np.linalg.norm(fit["translation"] - point_hint) > 0.010:
        return None
    return {
        "point": fit["translation"].tolist(),
        "axis": fit["rotation"][:, 2].tolist(),
        "up": fit["rotation"][:, 0].tolist(),
        "residual_m": fit["residual_m"],
        "inlier_count": fit["inlier_count"],
    }
