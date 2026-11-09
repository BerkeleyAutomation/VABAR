"""Build a deterministic stored CoACD collision payload from one USD mesh."""

from __future__ import annotations

import argparse
from pathlib import Path


def _triangles(counts, indices):
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
    return triangles


def build(
    source: Path,
    mesh_path: str,
    output: Path,
    *,
    threshold: float = 0.05,
    merge: bool = False,
    max_output_hulls: int = 64,
) -> None:
    import coacd
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise ValueError(f"could not open source USD: {source}")
    mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(mesh_path))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    triangles = np.asarray(
        _triangles(
            list(mesh.GetFaceVertexCountsAttr().Get()),
            list(mesh.GetFaceVertexIndicesAttr().Get()),
        ),
        dtype=np.int32,
    )
    if points.size == 0 or triangles.size == 0:
        raise ValueError(f"source prim is not a bounded polygon mesh: {mesh_path}")

    hulls = coacd.run_coacd(
        coacd.Mesh(points, triangles),
        threshold=threshold,
        max_convex_hull=64,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=5,
        mcts_max_depth=1,
        merge=merge,
        decimate=False,
        max_ch_vertex=256,
        seed=0,
    )
    if not hulls:
        raise ValueError("CoACD produced no hulls")
    if len(hulls) > max_output_hulls:
        raise ValueError(
            f"CoACD produced {len(hulls)} hulls, exceeding the admitted "
            f"maximum of {max_output_hulls}"
        )

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    for index, (vertices, faces) in enumerate(hulls):
        hull = UsdGeom.Mesh.Define(stage, f"/Asset/Hull_{index:03d}")
        hull.CreatePointsAttr(np.asarray(vertices, dtype=np.float64).tolist())
        face_array = np.asarray(faces, dtype=np.int32)
        hull.CreateFaceVertexCountsAttr([3] * len(face_array))
        hull.CreateFaceVertexIndicesAttr(face_array.reshape(-1).tolist())
        hull.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        hull.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
        hull.GetPrim().CreateAttribute(
            "asset:collisionHullIndex", Sdf.ValueTypeNames.Int
        ).Set(index)
    stage.GetRootLayer().Save()
    print(f"{output}: stored {len(hulls)} convex hulls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("mesh_path")
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--max-output-hulls", type=int, default=64)
    args = parser.parse_args()
    build(
        args.source,
        args.mesh_path,
        args.output,
        threshold=args.threshold,
        merge=args.merge,
        max_output_hulls=args.max_output_hulls,
    )


if __name__ == "__main__":
    main()
