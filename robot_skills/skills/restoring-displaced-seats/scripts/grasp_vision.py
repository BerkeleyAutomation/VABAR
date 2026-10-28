"""Refine visible bench cable points with the cable RGB-D centreline.

The fixed third-person camera supplies topology and refines the position
of visible local runs without replacing occluded parts of that model. Uses the
known white jacket and calibrated camera, with no simulator object state.
"""
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .perceive_rod import _camera, _frame
from .fixtures_cv import bench_z


def visible_cable_mask(rgb, depth, K, T, surface_z):
    """Known white jacket near the measured bench; excludes raised robot links."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth).squeeze()
    mask = (rgb.min(axis=2) > 140) & (np.ptp(rgb.astype(float), axis=2) < 25)
    mask &= np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    d = depth[v, u]
    xyz = (T @ np.column_stack([(u-K[0,2])*d/K[0,0],
                                (v-K[1,2])*d/K[1,1], d, np.ones(len(d))]).T).T[:,:3]
    valid = (xyz[:,2] > surface_z+.0005) & (xyz[:,2] < surface_z+.030)
    mask[:] = False
    mask[v[valid], u[valid]] = True
    return mask


def refine_grasp_curve(curve, observation, surface_z=None):
    points = np.asarray(curve, dtype=float)
    result = dict(refined=0, max_shift_mm=0.)
    block = _camera(observation, 'cable') if observation.get('cameras') else None
    frame = _frame(block) if block else None
    if frame is None or len(points) < 2:
        return points.tolist(), result
    rgb, depth, K, T = frame
    surface_z = bench_z(depth, K, T) if surface_z is None else float(surface_z)
    rgb = np.asarray(rgb); depth = np.asarray(depth).squeeze()
    mask = visible_cable_mask(rgb, depth, K, T, surface_z)
    v, u = np.nonzero(skeletonize(mask, method='lee'))
    if len(v) < 9:
        return points.tolist(), result
    d = depth[v, u]
    cloud = (T @ np.column_stack([(u-K[0,2])*d/K[0,0],
                                  (v-K[1,2])*d/K[1,1], d, np.ones(len(d))]).T).T[:,:3]
    ray = cloud-T[:3,3]
    cloud += .003 * ray / np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-9)
    tree = cKDTree(cloud[:,:2]); refined = points.copy(); shifts=[]
    for i, point in enumerate(points):
        if not surface_z-.005 <= point[2] <= surface_z+.035:
            continue
        dist, ids = tree.query(point[:2], k=9)
        local = cloud[ids[dist < .016]]
        if len(local) < 5:
            continue
        centre = local.mean(axis=0)
        _, values, axes = np.linalg.svd(local-centre, full_matrices=False)
        if values[0] < 1e-4 or values[1] > .4*values[0]:
            continue
        axis = axes[0]
        projected = centre+np.clip((point-centre)@axis,
                                    np.min((local-centre)@axis),
                                    np.max((local-centre)@axis))*axis
        shift = float(np.linalg.norm(projected[:2]-point[:2]))
        if shift > .012:
            continue
        refined[i] = projected
        shifts.append(shift)
    result.update(refined=len(shifts), max_shift_mm=1000*max(shifts, default=0.))
    return refined.tolist(), result
