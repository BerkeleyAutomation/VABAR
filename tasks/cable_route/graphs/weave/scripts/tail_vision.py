"""Current RGB-D tip and grasp observations for the known white cable scene.

This terminal helper uses the +y free-end ordering of the authored boards.
Colour and depth isolate visible cable surfaces; no simulated cable state or
tracked centreline is used. It refuses a clipped tip or an occluded grasp band.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perceive_rod import _camera, _frame
from fixtures_cv import bench_z


def measure_tail(rgb, depth, K, T, *, surface_z=None):
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    surface = bench_z(depth, K, T) if surface_z is None else float(surface_z)
    mask = ((rgb.min(axis=2) > 170) & (np.ptp(rgb.astype(float), axis=2) < 25)
            & np.isfinite(depth) & (depth > 0))
    v, u = np.nonzero(mask)
    z = depth[v, u]
    cam = np.column_stack(((u-K[0, 2])*z/K[0, 0],
                           (v-K[1, 2])*z/K[1, 1], z, np.ones(len(z))))
    world = (T @ cam.T).T[:, :3]
    valid = (world[:, 2] > surface + .0005) & (world[:, 2] < surface + .012)
    world, u, v = world[valid], u[valid], v[valid]
    if len(world) < 30:
        return {"found": False, "reason": "too few visible cable pixels"}
    tip_y = float(np.percentile(world[:, 1], 99.7))
    tip_mask = world[:, 1] > tip_y - .003
    if np.count_nonzero(tip_mask) < 6:
        return {"found": False, "reason": "too few tip pixels"}
    if np.any((u[tip_mask] < 3) | (u[tip_mask] >= rgb.shape[1]-3)
              | (v[tip_mask] < 3) | (v[tip_mask] >= rgb.shape[0]-3)):
        return {"found": False, "reason": "visible tip meets the image boundary"}
    end = np.median(world[tip_mask], axis=0)
    band = world[np.abs(world[:, 1] - (tip_y - .020)) < .003]
    grasp = None
    if len(band) >= 8:
        point = np.median(band, axis=0)
        # Known 7 mm cable: the camera sees its near surface, not its axis.
        ray = point - T[:3, 3]
        point += .0028 * ray / max(float(np.linalg.norm(ray)), 1e-9)
        grasp = [float(x) for x in point[:2]]
    return {"found": True, "endpoint": [float(x) for x in end],
            "grasp_xy": grasp, "grasp_pixels": len(band), "surface_z": surface}


def observe_tail(ctx):
    observation = ctx.tool("robot.get_observation")
    first_tip = None
    for camera in ("cable",):
        block = _camera(observation, camera)
        frame = _frame(block) if block else None
        if frame is None:
            continue
        result = measure_tail(*frame)
        result["camera"] = camera
        if not result.get("found"):
            continue
        if result.get("grasp_xy") is not None:
            return result
        if first_tip is None:
            first_tip = result
    return first_tip or {"found": False, "reason": "no unclipped visible cable tip"}
