"""Visual-material contract for the shared tool-hanging pegboard."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from ib_support import ASSETS_ROOT, ENVIRONMENT_YAMLS, load_scene

PEGBOARD_USD = ASSETS_ROOT / "vabar__pegboard_rack/vabar__pegboard_rack.usda"


def test_shared_pegboard_panels_resolve_the_rss_brown_matte_material() -> None:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(PEGBOARD_USD))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    for mesh in meshes:
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        assert material.GetPath() == "/Asset/pegboard_material", mesh.GetPath()

    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        (0.6, 0.372549, 0.184314)
    )
    assert shader.GetInput("metallic").Get() == 0
    assert shader.GetInput("roughness").Get() == pytest.approx(0.85)


def test_all_tool_hanging_pegboards_use_the_shared_materialized_asset() -> None:
    tool_hanging = [path for path in ENVIRONMENT_YAMLS if path.startswith("tool_hanging/")]
    assert tool_hanging
    for path in tool_hanging:
        pegboards = [
            obj
            for obj in load_scene(path).get("objects", ())
            if obj.get("id") == "pegboard_rack"
        ]
        assert len(pegboards) == 1, PurePosixPath(path).stem
        assert pegboards[0]["registry_name"] == "vabar__pegboard_rack"
