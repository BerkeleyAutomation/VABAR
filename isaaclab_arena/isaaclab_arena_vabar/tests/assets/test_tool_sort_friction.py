"""Task-specific friction must not leak into the shared syringe hand."""
from pathlib import Path

import pytest

Usd = pytest.importorskip("pxr.Usd")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
ROOT = Path(__file__).resolve().parents[2] / "src/isaaclab_arena_vabar/assets"


@pytest.mark.parametrize("suffix", ["", "_instanceable"])
def test_sorting_pad_materials(suffix):
    folder = ROOT / "vabar__fr3_robotiq_2f85_on_cart"
    for filename, expected in [(f"tool_sorting{suffix}.usda", (0.7, 0.6)),
                               (f"vabar__fr3_robotiq_2f85_on_cart{suffix}.usda", (4., 4.))]:
        stage = Usd.Stage.Open(str(folder / filename))
        found = []
        for prim in stage.Traverse():
            if prim.GetName() not in ("PhysicsMaterial_1", "PhysicsMaterial_2"):
                continue
            material = UsdPhysics.MaterialAPI(prim)
            index = int(prim.GetName()[-1]) - 1
            assert material.GetDynamicFrictionAttr().Get() == pytest.approx(expected[index])
            assert material.GetStaticFrictionAttr().Get() == pytest.approx(expected[index])
            if filename.startswith("tool_sorting"):
                assert prim.GetAttribute("newton:rollingFriction").Get() == pytest.approx(0.0001)
                assert prim.GetAttribute("newton:torsionalFriction").Get() == pytest.approx(0.005)
            found.append(index)
        assert sorted(found) == [0, 1]


def test_multimeter_matches_autolab_level1_scale():
    """AUTOLab's v1.0 level-1 multimeter (152 x 73 x 44 mm) is this mesh at 0.83."""
    from pxr import UsdGeom
    stage = Usd.Stage.Open(str(ROOT / "vabar__multimeter/vabar__multimeter.usda"))
    assert tuple(stage.GetDefaultPrim().GetAttribute("xformOp:scale").Get()) == (0.83,) * 3
    cache = UsdGeom.XformCache()
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    for mesh in meshes:
        transform = cache.GetLocalToWorldTransform(mesh)
        assert transform.ExtractRotationMatrix().GetRow(0).GetLength() == pytest.approx(0.83)
