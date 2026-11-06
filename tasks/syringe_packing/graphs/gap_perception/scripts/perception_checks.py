"""RGB-D evidence helpers. All inputs are observations and robot proprioception."""

import numpy as np
from scipy.spatial.transform import Rotation

PROMPT = "medical syringe"


def xyz(p):
    return np.array([p[k] for k in ("x", "y", "z")], dtype=np.float64)


def matrix(pose):
    m = np.eye(4)
    q = pose["rotation"]
    m[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    m[:3, 3] = xyz(pose["position"])
    return m


def crop_for_world_box(camera, lower, upper, padding=16):
    corners = np.array(
        [[x, y, z] for x in [lower[0], upper[0]] for y in [lower[1], upper[1]] for z in [lower[2], upper[2]]]
    )
    c = (np.linalg.inv(matrix(camera["pose"])) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    c = c[c[:, 2] > 0.01]
    if not len(c):
        raise ValueError("inspection region is behind the camera")
    uv = c @ np.asarray(camera["intrinsics"]).T
    uv = uv[:, :2] / uv[:, 2:]
    h, w = np.asarray(camera["rgb"]).shape[:2]
    x0, y0 = np.maximum(0, np.floor(uv.min(axis=0) - padding)).astype(int)
    x1, y1 = np.minimum([w, h], np.ceil(uv.max(axis=0) + padding)).astype(int)
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise ValueError("inspection region is outside the camera image")
    return (int(x0), int(y0), int(x1), int(y1))


def detect(ctx, camera, *, crop=None, prompt=PROMPT, floor=0.01):
    """Restore cropped masks into the original calibrated image before backprojection."""
    rgb = np.asarray(camera["rgb"])
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if crop is None else crop
    result = ctx.tool("sam3.segment_text", image=rgb[y0:y1, x0:x1], query=prompt, max_results=0)
    detections = []
    for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
        if float(score) < floor:
            continue
        local = np.asarray(mask, dtype=np.uint8)
        if local.shape != (y1 - y0, x1 - x0):
            raise ValueError("segmentation mask dimensions do not match image crop")
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = local
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=full,
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 12:
            continue
        center = np.median(points, axis=0)
        points = points[np.linalg.norm(points - center, axis=1) < 0.22]
        if len(points) < 12:
            continue
        center = np.median(points, axis=0)
        _, _, basis = np.linalg.svd(points - center, full_matrices=False)
        projection = (points - center) @ basis.T
        extent = np.percentile(projection, 95, axis=0) - np.percentile(projection, 5, axis=0)
        yy, xx = np.nonzero(full)
        detections.append(
            {
                "score": float(score),
                "center": center,
                "points": points,
                "axis": basis[0],
                "extent": extent,
                "camera": camera["name"],
                "crop": crop,
                "bounds": (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)),
            }
        )
    return detections


def rod_like(d):
    e = d["extent"]
    return 0.025 <= e[0] <= 0.22 and e[1] <= 0.045 and e[0] >= 2.0 * max(e[1], 0.003)


def summary(d):
    return {
        "score": d["score"],
        "center": dict(zip("xyz", map(float, d["center"]), strict=True)),
        "camera": d["camera"],
        "crop": d["crop"],
        "extent": d["extent"].tolist(),
    }


def distinct(items, distance=0.02):
    result = []
    for d in sorted(items, key=lambda x: x["score"], reverse=True):
        if not any(np.linalg.norm(d["center"] - x["center"]) < distance for x in result):
            result.append(d)
    return result


def expanded_bounds(items, camera, padding=50):
    h, w = np.asarray(camera["rgb"]).shape[:2]
    bounds = np.array([d["bounds"] for d in items])
    return (
        max(0, int(bounds[:, 0].min()) - padding),
        max(0, int(bounds[:, 1].min()) - padding),
        min(w, int(bounds[:, 2].max()) + padding),
        min(h, int(bounds[:, 3].max()) + padding),
    )

