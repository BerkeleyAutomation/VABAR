"""MuJoCo contact meshes must preserve the J-hook's concave cavity."""

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("scipy")

from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

from ib_support import ASSETS_ROOT


def test_wrench_hook_cavity_is_not_filled_by_mujoco_convex_hull():
    asset = ASSETS_ROOT / "vabar__wrench_hooks/wrench_hook_2.usda"
    stage = Usd.Stage.Open(str(asset))
    visual = stage.GetPrimAtPath("/Asset/fixture/Geometry")
    assert not visual.HasAPI(UsdPhysics.CollisionAPI)
    transforms = UsdGeom.XformCache()
    cavity = np.array([0.04, -0.00249, 0.02])
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        count += 1
        transform = transforms.GetLocalToWorldTransform(prim)
        vertices = np.array([transform.Transform(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
        hull = ConvexHull(vertices)
        assert np.any(hull.equations[:, :3] @ cavity + hull.equations[:, 3] > 0)
    assert count > 1
