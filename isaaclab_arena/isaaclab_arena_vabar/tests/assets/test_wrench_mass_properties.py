"""RSS physical-property contract for the easy-task 19 mm wrench."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Gf, Usd, UsdPhysics

from ib_support import PACKAGE_ROOT

WRENCH_USD = (
    PACKAGE_ROOT
    / "assets/vabar__wrench_ring_19"
    / "vabar__wrench_ring_19.usda"
)
RSS_MASS_KG = 0.10
ARTIST_MASS_KG = 0.21671282949102935
ARTIST_CENTER_OF_MASS_M = np.array(
    [0.14409081637859344, 0.2217905968427658, 0.003674149978905916]
)
ARTIST_INERTIA_KG_M2 = np.array(
    [
        [1.625943332328461e-05, -6.257227141759358e-06, -2.506500095478259e-06],
        [-6.257227141759358e-06, 0.001027776044793427, -9.136887690885942e-09],
        [-2.506500095478259e-06, -9.136887690885942e-09, 0.0010400805622339249],
    ]
)


def test_wrench_uses_rss_mass_with_density_scaled_artist_inertia() -> None:
    stage = Usd.Stage.Open(str(WRENCH_USD))
    assert stage is not None
    prim = stage.GetPrimAtPath("/Asset/Geometry/base_01_obj_00")
    mass_api = UsdPhysics.MassAPI(prim)

    assert mass_api.GetMassAttr().Get() == pytest.approx(RSS_MASS_KG)
    assert np.asarray(mass_api.GetCenterOfMassAttr().Get()) == pytest.approx(
        ARTIST_CENTER_OF_MASS_M
    )

    diagonal = np.asarray(mass_api.GetDiagonalInertiaAttr().Get())
    axes = mass_api.GetPrincipalAxesAttr().Get()
    rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(axes)))
    # Gf matrices use row-vector composition; transpose that convention to
    # reconstruct the body-frame tensor consumed by Newton.
    inertia = rotation.T @ np.diag(diagonal) @ rotation
    assert inertia == pytest.approx(
        ARTIST_INERTIA_KG_M2 * RSS_MASS_KG / ARTIST_MASS_KG, abs=1e-10
    )


def test_collision_hulls_preserve_the_visual_ring_opening():
    from pxr import UsdGeom
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull

    stage = Usd.Stage.Open(str(WRENCH_USD))
    transforms = UsdGeom.XformCache()
    ring = np.array([0.024331, 0.2214774, 0.00425])
    distances = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "StoredConvexCollision" not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = transforms.GetLocalToWorldTransform(prim)
        vertices = np.array([transform.Transform(p) for p in mesh.GetPointsAttr().Get()])
        equations = ConvexHull(vertices).equations
        normals, offsets = equations[:, :3], equations[:, 3]
        closest = minimize(
            lambda p: np.sum((p - ring) ** 2), vertices.mean(axis=0),
            jac=lambda p: 2 * (p - ring), method="SLSQP",
            constraints={"type": "ineq", "fun": lambda p, normals=normals, offsets=offsets: -normals @ p - offsets,
                         "jac": lambda p, normals=normals: -normals},
            options={"ftol": 1e-14},
        )
        assert closest.success
        distances.append(np.linalg.norm(closest.x - ring))
    # The visible opening has 10.23 mm inradius. A coarser 13-piece hull set reduces
    # this to 7.28 mm, so the hook would strike collision geometry inside the hole.
    assert distances
    assert 0.009 < min(distances) < 0.0105
