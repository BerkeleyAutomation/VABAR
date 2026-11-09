"""Render deterministic thumbnails of the enabled collider geometry for assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    WORKSPACE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
)


def _triangulate(counts, indices) -> list[tuple[int, int, int]]:
    triangles = []
    offset = 0
    for count in counts:
        count = int(count)
        face = [int(value) for value in indices[offset : offset + count]]
        triangles.extend(
            (face[0], face[index], face[index + 1])
            for index in range(1, len(face) - 1)
        )
        offset += count
    if offset != len(indices):
        raise ValueError("mesh face counts do not consume every face index")
    return triangles


def _geometry_prims(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    selected = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        candidates = (
            (prim,)
            if prim.IsA(UsdGeom.Gprim)
            else Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        )
        for candidate in candidates:
            if candidate.IsA(UsdGeom.Gprim):
                selected[str(candidate.GetPath())] = candidate
    if not selected:
        raise RuntimeError("asset has no enabled collider geometry")
    return tuple(selected[path] for path in sorted(selected))


def _mesh_triangles(prim, xforms):
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull, QhullError

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or ()
    transform = xforms.GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [
            tuple(transform.Transform(Gf.Vec3d(point)))
            for point in points
        ],
        dtype=float,
    )
    approximation = (
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"
    )
    if approximation == "convexHull" and len(vertices) >= 4:
        try:
            faces = ConvexHull(vertices).simplices
        except QhullError:
            faces = _triangulate(
                mesh.GetFaceVertexCountsAttr().Get() or (),
                mesh.GetFaceVertexIndicesAttr().Get() or (),
            )
    else:
        faces = _triangulate(
            mesh.GetFaceVertexCountsAttr().Get() or (),
            mesh.GetFaceVertexIndicesAttr().Get() or (),
        )
    return vertices, np.asarray(faces, dtype=int)


def _cube_triangles(prim, xforms):
    import numpy as np
    from pxr import Gf, UsdGeom

    half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
    local = tuple(
        Gf.Vec3d(x, y, z)
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
    )
    transform = xforms.GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [tuple(transform.Transform(point)) for point in local], dtype=float
    )
    faces = np.asarray(
        (
            (0, 1, 3), (0, 3, 2),
            (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
        ),
        dtype=int,
    )
    return vertices, faces


def _collision_triangles(asset_path: Path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {asset_path}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    surfaces = []
    for prim in _geometry_prims(stage):
        if prim.IsA(UsdGeom.Mesh):
            vertices, faces = _mesh_triangles(prim, xforms)
        elif prim.IsA(UsdGeom.Cube):
            vertices, faces = _cube_triangles(prim, xforms)
        else:
            raise TypeError(
                f"unsupported collider geometry {prim.GetTypeName()} at {prim.GetPath()}"
            )
        if len(vertices) and len(faces):
            surfaces.append(vertices[faces])
    if not surfaces:
        raise RuntimeError(f"asset has no renderable collider geometry: {asset_path}")
    return surfaces


def render_collision_thumbnail(asset_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    surfaces = _collision_triangles(asset_path)
    points = np.concatenate([surface.reshape(-1, 3) for surface in surfaces])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) * 0.58, 1e-6)

    figure = plt.figure(figsize=(6.4, 3.6), dpi=100, facecolor="#20252b")
    axes = figure.add_subplot(111, projection="3d", facecolor="#20252b")
    for surface in surfaces:
        axes.add_collection3d(
            Poly3DCollection(
                surface,
                facecolor="#36c5e5",
                edgecolor="#123d48",
                linewidth=0.18,
                alpha=0.82,
            )
        )
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1, 1, 1))
    axes.view_init(elev=24, azim=-52)
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)


def _asset_cases(names: set[str] | None):
    from asset_acceptance.asset_drop_cases import ASSET_DROP_CASES

    cases = tuple(
        case
        for case in ASSET_DROP_CASES
        if names is None or case.registry_name in names
    )
    if names is not None:
        missing = sorted(names - {case.registry_name for case in cases})
        if missing:
            raise ValueError(f"unknown movable assets: {missing}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", action="append", default=[])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    cases = _asset_cases(set(args.asset) if args.asset else None)
    for case in cases:
        relative_path = case.relative_usd_path
        if relative_path.startswith("@ISAAC_NUCLEUS_DIR@/"):
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

            source = relative_path.replace(
                "@ISAAC_NUCLEUS_DIR@", ISAAC_NUCLEUS_DIR, 1
            )
            asset_path = Path(retrieve_file_path(source, force_download=False))
        else:
            asset_path = ASSET_ROOT / relative_path
        render_collision_thumbnail(
            asset_path, output / f"{case.registry_name}.collision.png"
        )
        print(f"Rendered collider: {case.registry_name}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vabar-matplotlib")
    raise SystemExit(main())
