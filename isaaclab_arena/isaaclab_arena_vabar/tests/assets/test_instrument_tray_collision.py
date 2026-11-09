"""The instrument tray collides as the open tray it is drawn as, not as its filled hull."""

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdPhysics

from ib_support import ASSETS_ROOT

TRAY = ASSETS_ROOT / "vabar__instrument_tray/vabar__instrument_tray.usda"
BODY = "/Asset/Geometry/instrument_tray_01_obj_00"


def _colliders(stage):
    """Every enabled collider on the tray body, as (prim, local min, local max)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    transforms = UsdGeom.XformCache()
    body = transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY)).GetInverse()
    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(BODY)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        box = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        corners = np.array([(transforms.GetLocalToWorldTransform(prim) * body).Transform(box.GetCorner(i)) for i in range(8)])
        out.append((prim, corners.min(axis=0), corners.max(axis=0)))
    return out


def test_the_artist_mesh_does_not_collide():
    stage = Usd.Stage.Open(str(TRAY))
    colliders = _colliders(stage)
    assert colliders
    assert all(prim.IsA(UsdGeom.Cube) for prim, _, _ in colliders)


def test_the_cavity_is_open_down_to_a_3_mm_floor():
    stage = Usd.Stage.Open(str(TRAY))
    colliders = _colliders(stage)
    lows = np.array([lo for _, lo, _ in colliders])
    highs = np.array([hi for _, _, hi in colliders])
    np.testing.assert_allclose(lows.min(axis=0), (-0.1, -0.2, -0.006), atol=1e-6)
    np.testing.assert_allclose(highs.max(axis=0), (0.1, 0.2, 0.006), atol=1e-6)
    # A point resting on the cavity floor, anywhere inside the walls, is in no collider.
    for x, y in ((0.0, 0.0), (0.09, 0.19), (-0.09, -0.19)):
        point = np.array((x, y, -0.0029))
        assert not any(np.all(point > lo) and np.all(point < hi) for _, lo, hi in colliders)
    # ... and just below it, it is in the floor.
    assert any(np.all(np.array((0.0, 0.0, -0.0031)) > lo) and np.all(np.array((0.0, 0.0, -0.0031)) < hi) for _, lo, hi in colliders)


def test_the_proxy_keeps_the_tray_material():
    stage = Usd.Stage.Open(str(TRAY))
    proxy = stage.GetPrimAtPath(f"{BODY}/CollisionProxy")
    targets = proxy.GetRelationship("material:binding:physics").GetTargets()
    assert [str(t) for t in targets] == ["/Asset/PhysicsMaterials/physics_mat_m_opaque__metal__aluminiumBrushed_c01"]
