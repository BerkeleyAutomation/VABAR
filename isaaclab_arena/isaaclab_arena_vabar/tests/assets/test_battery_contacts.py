"""The D cell collides as AUTOLab's sort_electrical does, at its uniform-density inertia."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")
pytest.importorskip("pxr")

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from newton import GeoType, ModelBuilder, ShapeFlags
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Usd, UsdGeom, UsdPhysics

from ib_support import ASSETS_ROOT, AUTOLAB_ROOT

_BATTERY_USD = ASSETS_ROOT / "vabar__battery" / "vabar__battery.usda"
_BODY = "/Asset/Geometry/industrial_d_battery_0_g0_visual_obj_00"
_RSS_SCENE = AUTOLAB_ROOT / "assets/vab_tool_sort/sort_electrical/scene.xml"


def _enabled_colliders(stage):
    return [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]


def test_battery_collides_as_its_mesh_bounding_box() -> None:
    stage = Usd.Stage.Open(str(_BATTERY_USD))
    (collider,) = _enabled_colliders(stage)
    assert collider.IsA(UsdGeom.Cube)
    assert UsdGeom.Imageable(collider).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
    points = np.asarray(
        UsdGeom.Mesh(
            stage.GetPrimAtPath(f"{_BODY}/industrial_d_battery_0_g0_visual_mesh_00")
        ).GetPointsAttr().Get()
    )
    bound = UsdGeom.Cube(collider).ComputeLocalBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_, UsdGeom.Tokens.guide
    )
    box = bound.ComputeAlignedRange()
    assert tuple(box.GetMin()) == pytest.approx(tuple(points.min(axis=0)), abs=1e-6)
    assert tuple(box.GetMax()) == pytest.approx(tuple(points.max(axis=0)), abs=1e-6)


@pytest.mark.skipif(not _RSS_SCENE.is_file(), reason="AUTOLab v1.0 sources are not materialized; run ./setup.sh")
def test_rss_sort_electrical_collides_the_battery_as_a_box() -> None:
    body = ET.parse(_RSS_SCENE).find('.//body[@name="industrial_d_battery_0_main"]')
    colliding = [
        geom
        for geom in body.findall("geom")
        if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0"
    ]
    assert [geom.get("type") for geom in colliding] == ["box"]


def test_newton_imports_the_battery_box_at_uniform_density_inertia() -> None:
    builder = ModelBuilder()
    NewtonMJWarpManager._register_builder_attributes(builder)
    builder.add_usd(
        Usd.Stage.Open(str(_BATTERY_USD)),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )
    colliding = [
        shape
        for shape in range(builder.shape_count)
        if builder.shape_flags[shape] & int(ShapeFlags.COLLIDE_SHAPES)
    ]
    assert [builder.shape_type[shape] for shape in colliding] == [GeoType.BOX]

    (body,) = [index for index in range(builder.body_count) if builder.body_mass[index] > 0]
    assert builder.body_mass[body] == pytest.approx(0.135, rel=1e-6)
    principal = np.linalg.eigvalsh(np.asarray(builder.body_inertia[body]).reshape(3, 3))
    # Uniform density over the visual mesh; the delivered 6.9e-5/1.2e-4 was 2.6-3.9x it.
    assert principal == pytest.approx((1.7666e-5, 4.6158e-5, 4.6167e-5), rel=1e-3)
