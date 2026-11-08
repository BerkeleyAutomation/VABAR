"""AUTOLab v1.0 appearance contract for the slim tool-hanging screwdriver."""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from ib_support import ASSETS_ROOT

SLIM_USD = ASSETS_ROOT / "vabar__screwdriver_slim/vabar__screwdriver_slim.usda"
MESH = "/Asset/Geometry/Visual_obj_00/Visual_mesh_00"


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _diffuse(prim) -> tuple[float, ...]:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert material, prim.GetPath()
    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("metallic").Get() == 0
    assert shader.GetInput("roughness").Get() == pytest.approx(0.6)
    return tuple(shader.GetInput("diffuseColor").Get())


def test_slim_screwdriver_draws_autolab_mtl_colours_as_path_tracer_albedo() -> None:
    stage = Usd.Stage.Open(str(SLIM_USD))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(MESH))
    subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
    face_count = len(mesh.GetFaceVertexCountsAttr().Get())
    assert UsdGeom.Subset.ValidateSubsets(subsets, face_count, "partition")[0]

    # screwdriver_1.mtl Kd, decoded the way AUTOLab's RTX renderer decodes display colours.
    expected = {
        "autolab_handle_blue": (0.0, 0.0, 0.752941),
        "autolab_handle_green": (0.0, 0.752941, 0.0),
        "autolab_shaft_dark": (0.250980,) * 3,
        "autolab_shaft_steel": (0.792157, 0.819608, 0.933333),
    }
    assert {subset.GetPrim().GetName() for subset in subsets} == set(expected)
    for subset in subsets:
        kd = expected[subset.GetPrim().GetName()]
        assert _diffuse(subset.GetPrim()) == pytest.approx(
            tuple(_srgb_to_linear(c) for c in kd), abs=1e-5
        )


def test_slim_screwdriver_tip_marker_is_a_visual_only_red_sphere() -> None:
    stage = Usd.Stage.Open(str(SLIM_USD))
    marker = stage.GetPrimAtPath("/Asset/Geometry/Visual_obj_00/autolab_tip_marker_00")
    assert marker.IsA(UsdGeom.Sphere)
    assert UsdGeom.Sphere(marker).GetRadiusAttr().Get() == pytest.approx(0.003)
    translation = UsdGeom.Xformable(marker).ComputeLocalToWorldTransform(0).ExtractTranslation()
    assert tuple(translation) == pytest.approx((0.175, 0.0, 0.0))
    assert not marker.HasAPI(UsdPhysics.CollisionAPI)
    assert not marker.HasAPI(UsdPhysics.MassAPI)
    assert _diffuse(marker) == pytest.approx((1.0, 0.0, 0.0))
