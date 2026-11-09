"""Collide the tool-sort objects with their visible shape instead of a box.

The RSS MJCF declares a single oriented box as the collision geom for most
tool-sort objects and carries the real mesh alongside it as visual-only geometry
(`contype="0" conaffinity="0"`). The box overshoots the drill's true volume by
3.4x and the clamp's by 8.2x, so objects collide as blocks while looking like
tools.

This withdraws collision from the box, hides it, and applies collision to the
visual mesh already in the asset, tagged `convexDecomposition`. Newton maps that token to CoACD and
decomposes at load (`newton/_src/utils/import_usd.py`), so no geometry is stored:
the mesh is one that ships anyway, and the parts are derived from it.

The visual prim is authored `instanceable`, which forbids per-instance edits, so
it is de-instanced first. These assets carry one instance each, so nothing is
lost by that.

Usage:
    python collide_with_visual_mesh.py <asset_name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
)


def apply(asset: str) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    base = ASSET_ROOT / asset / "payloads" / "base.usda"
    stage = Usd.Stage.Open(str(base))

    primitives = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Mesh)
    ]
    if len(primitives) != 1:
        raise SystemExit(f"{asset}: expected one primitive collider, found {len(primitives)}")
    primitive = primitives[0]

    visuals = [
        prim
        for prim in primitive.GetParent().GetChildren()
        if prim.GetName().endswith("_visual")
    ]
    if len(visuals) != 1:
        raise SystemExit(f"{asset}: expected one visual prim, found {len(visuals)}")
    visual = visuals[0]
    visual.SetInstanceable(False)

    meshes = [child for child in visual.GetChildren() if child.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise SystemExit(f"{asset}: expected one visual mesh, found {len(meshes)}")
    mesh = meshes[0]

    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexDecomposition")
    primitive.CreateAttribute(
        "physics:collisionEnabled", Sdf.ValueTypeNames.Bool
    ).Set(False)
    # The box is authored dark red and, once it stops colliding, is a proxy
    # nothing reads. It still renders over the mesh it was standing in for, so
    # the cameras see red blocks where the tools are.
    UsdGeom.Imageable(primitive).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    stage.GetRootLayer().Save()
    print(f"{asset}: {primitive.GetName()} disabled, {mesh.GetName()} collides ({len(points)} verts)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset")
    apply(parser.parse_args().asset)


if __name__ == "__main__":
    main()
