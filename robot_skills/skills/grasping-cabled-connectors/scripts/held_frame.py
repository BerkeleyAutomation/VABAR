"""Estimate the held plug's top plane and insertion direction from RGB-D points."""

import numpy as np


def fit_top_frame(points, axis_hint):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 400:
        return None
    rng = np.random.default_rng(0)
    best = np.empty(0, dtype=int)
    for _ in range(160):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        size = np.linalg.norm(normal)
        if size < 1e-10:
            continue
        normal /= size
        if abs(normal[2]) < .9:
            continue
        inliers = np.flatnonzero(abs((points - sample[0]) @ normal) < .00035)
        if len(inliers) > len(best):
            best = inliers
    if len(best) < max(120, len(points) * .25):
        return None
    face = points[best]
    centre = face.mean(axis=0)
    _, singular, axes = np.linalg.svd(face - centre, full_matrices=False)
    if singular[1] < .025 or singular[2] / max(singular[1], 1e-12) > .1:
        return None
    normal = axes[-1] * np.sign(axes[-1, 2])
    # The fingers can hide most of the body, leaving a short top patch whose
    # principal axis is ambiguous. The lower tab is still visible. Project
    # both surfaces onto the fitted plane, excluding background depth at RTX
    # silhouette pixels and points outside the body's measured lateral span.
    hint = np.asarray(axis_hint, dtype=float)
    hint = hint - normal * (hint @ normal)
    hint /= max(np.linalg.norm(hint), 1.e-12)
    across = np.cross(normal, hint)
    lo, hi = np.percentile((face - centre) @ across, [2., 98.])
    local = points - centre
    heights, lateral = local @ normal, local @ across
    supported = (heights >= -.007) & (heights <= .0007) & (lateral >= lo-.001) & (lateral <= hi+.001)
    projected = local[supported] - heights[supported, None] * normal
    if len(projected) < 120:
        return None
    _, spread, directions = np.linalg.svd(projected - projected.mean(0), full_matrices=False)
    if spread[0] < 1.5 * spread[1]:
        return None
    axis = directions[0]
    if axis @ np.asarray(axis_hint) < 0:
        axis = -axis
    if np.ptp(np.percentile(projected @ axis, [2., 98.])) < .016:
        return None
    return dict(axis=axis.tolist(), up=normal.tolist(), point=centre.tolist(),
                inlier_count=len(best), residual_m=float(np.sqrt(np.mean(((face-centre) @ normal)**2))))


def frame_matrix(axis, up):
    axis = np.array(axis, dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    up = np.array(up, dtype=float, copy=True)
    up -= axis * (up @ axis)
    up /= np.linalg.norm(up)
    return np.column_stack((up, np.cross(axis, up), axis))


def alignment_delta(plug_axis, plug_up, bore_axis):
    """Rotation that aligns a measured plug frame with the receptacle frame."""
    return frame_matrix(bore_axis, [0., 0., 1.]) @ frame_matrix(plug_axis, plug_up).T
