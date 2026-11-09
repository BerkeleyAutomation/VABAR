"""Artist wrench geometry supplied to the RSS CAD-registration graph."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

_CENTER = np.array([0.1279999316, 0.2214786708, 0.004249974])
_RING = np.array([0.024331, 0.2214774, 0.00425])
# The RSS held-wrench registration expects +X toward the ring and +Y normal.
_HELD_FRAME = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])


def _usd_mesh(relative_path, prim_path):
    import trimesh
    from pxr import Usd, UsdGeom

    asset = (
        Path(__file__).resolve().parents[3]
        / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
        / relative_path
    )
    stage = Usd.Stage.Open(str(asset))
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.array([transform.Transform(p) for p in mesh.GetPointsAttr().Get()])
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    triangles, offset = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = indices[offset:offset + count]
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        offset += count
    return trimesh.Trimesh(vertices, triangles, process=False)


@lru_cache(maxsize=1)
def _cad_models():
    import trimesh

    mesh = _usd_mesh(
        "vabar__wrench_ring_19/vabar__wrench_ring_19.usda",
        "/Asset/Geometry/base_01_obj_00/base_01_mesh_00",
    )
    vertices = mesh.vertices - _CENTER
    directory = TemporaryDirectory(prefix="vabar-wrench-cad-")
    planar = Path(directory.name) / "planar.obj"
    held = Path(directory.name) / "held.obj"
    trimesh.Trimesh(vertices, mesh.faces, process=False).export(planar)
    trimesh.Trimesh(vertices @ _HELD_FRAME.T, mesh.faces, process=False).export(held)
    return directory, planar, held


def artist_wrench_profiles(profiles):
    """Keep the RSS registration strategies, with CAD and landmarks from the rendered asset."""
    _, planar, held = _cad_models()
    result = deepcopy(profiles)
    for profile in result:
        if profile["kind"] != "wrench":
            continue
        profile.update(
            object_query="wrench",
            feature_query="closed ring end of wrench",
            strategy="planar_cad_loop",
            model_path=str(planar),
            feature_landmarks_xy=[(_RING - _CENTER)[:2].tolist()],
            radius_inner=0.0102,
        )
        profile["registration_profile"].update(
            query="wrench",
            model_path=str(held),
            model_scale=1.0,
            feature_local=(_HELD_FRAME @ (_RING - _CENTER)).tolist(),
            normal_local=[0.0, 1.0, 0.0],
            normal_axis_index=1,
        )
    return result


@lru_cache(maxsize=1)
def _hook_geometry():
    import trimesh

    mesh = _usd_mesh("vabar__wrench_hooks/wrench_hook_2.usda", "/Asset/fixture/Geometry")
    points, _ = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    distal = mesh.vertices[mesh.vertices[:, 0] > mesh.bounds[1, 0] - 0.01]
    tip = 0.5 * (distal.min(axis=0) + distal.max(axis=0))
    tip[0] = mesh.bounds[1, 0]
    radius = 0.5 * (mesh.bounds[1, 1] - mesh.bounds[0, 1])
    return points, tip, float(radius)


def refine_artist_hook(result):
    """Fit the known upright hook to RGB-D, avoiding face-to-center heuristics."""
    from scipy.spatial import cKDTree

    observed = np.asarray(result["fixture_cloud"]["points"], dtype=np.float64)
    observed = observed[np.isfinite(observed).all(axis=1)]
    if len(observed) < 40:
        raise ValueError("hook CAD registration needs at least 40 observed points")
    points, local_tip, radius = _hook_geometry()
    axis = np.array([result["fixture_axis"][key] for key in "xyz"])
    axis /= np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    up -= axis * (up @ axis)
    up /= np.linalg.norm(up)
    rotation = np.column_stack((axis, np.cross(up, axis), up))
    model = points @ rotation.T
    tree = cKDTree(model)
    initial = np.array([result["hook_tip"][key] for key in "xyz"]) - rotation @ local_tip
    best = None
    # Search across the possible visible faces, then fit translation only.
    # The board and gravity determine orientation independently of this fit.
    for height in np.arange(-0.02, 0.0201, 0.001):
        translation = initial + up * height
        for _ in range(20):
            distances, indices = tree.query(observed - translation)
            keep = distances < min(0.008, float(np.quantile(distances, 0.85)))
            if keep.sum() < 30:
                break
            delta = np.median(observed[keep] - model[indices[keep]] - translation, axis=0)
            translation += delta
            if np.linalg.norm(delta) < 1e-6:
                break
        error = float(np.sqrt(np.mean(np.minimum(tree.query(observed - translation)[0], 0.01) ** 2)))
        if best is None or error < best[0]:
            best = error, translation
    error, translation = best
    if error > 0.003 or np.linalg.norm(translation - initial) > 0.025:
        raise ValueError(f"hook CAD registration rejected: residual {error:.4f} m")
    refined = deepcopy(result)
    tip = dict(zip("xyz", map(float, translation + rotation @ local_tip), strict=True))
    refined["hook_tip"] = tip
    refined["fixture_axis"] = dict(zip("xyz", map(float, axis), strict=True))
    refined["fixture_feature"]["axis"] = dict(refined["fixture_axis"])
    refined["fixture_feature"]["pose"]["position"] = dict(tip)
    refined["fixture_feature"]["radius_outer"] = radius
    return refined
