"""RSS socket geometry and the imported Medium contact configuration."""

import hashlib

import numpy as np
import pytest

newton = pytest.importorskip("newton")
Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")

from ib_support import ASSETS_ROOT as ASSETS

SOCKET = ASSETS / "vabar__bulkhead/vabar__bulkhead.usda"
PLUG = ASSETS / "vabar__usbc_easy_plug/medium_plug.usda"


@pytest.mark.parametrize("path", [SOCKET, PLUG])
def test_mating_part_imports_one_hydroelastic_collider(path):
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(path), load_visual_shapes=True)
    colliders = [
        i
        for i, flags in enumerate(builder.shape_flags)
        if flags & newton.ShapeFlags.COLLIDE_SHAPES
    ]
    assert len(colliders) == 1
    shape = colliders[0]
    assert builder.shape_flags[shape] & newton.ShapeFlags.HYDROELASTIC
    assert builder.shape_sdf_target_voxel_size[shape] == pytest.approx(0.00025)
    assert builder.shape_sdf_narrow_band_range[shape] == pytest.approx((-0.005, 0.005))
    assert builder.shape_sdf_padding[shape] == pytest.approx(0.006)
    assert builder.shape_material_restitution[shape] == 0
    assert builder.custom_attributes["mujoco:condim"].values[shape] == 3
    if path == SOCKET:
        assert builder.shape_material_mu[shape] == pytest.approx(2.5)
        assert sum(builder.body_mass) == pytest.approx(0.09)


def test_socket_keeps_artist_visual_and_source_mouth_frame():
    stage = Usd.Stage.Open(str(SOCKET))
    body = "/Asset/Geometry/bulkhead_01_obj_00"
    visual = stage.GetPrimAtPath(body + "/bulkhead_01_mesh_00")
    assert visual.GetAttribute("physics:collisionEnabled").Get() is False
    assert UsdGeom.Imageable(visual).ComputeVisibility() == "inherited"
    visual_transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(visual))
    # Artist slot width is local Z; the source collider slot width is local Y.
    np.testing.assert_allclose(
        np.abs(visual_transform[2, :3]), [0, 1, 0], atol=1e-12
    )
    collider = UsdGeom.Mesh(stage.GetPrimAtPath(body + "/SourceCollisionMesh"))
    assert collider.GetPurposeAttr().Get() == "guide"
    points = np.asarray(collider.GetPointsAttr().Get())
    np.testing.assert_allclose(points.min(axis=0), [-0.0045, -0.03, -0.03], atol=1e-8)
    np.testing.assert_allclose(points.max(axis=0), [0.0615, 0.03, 0.03], atol=1e-8)
    # Golden triangle coordinates from the pinned STL with scale 2 baked in.
    indices = np.asarray(collider.GetFaceVertexIndicesAttr().Get())
    assert hashlib.sha256(points[indices].astype("<f4").tobytes()).hexdigest() == (
        "4843469169ed1687378555a4365ee03b53021b3fbae76c4a03b2fb0f37d0307b"
    )
    provenance = collider.GetPrim().GetCustomDataByKey("autolab")
    assert provenance["sourceRevision"] == "bf985d08c0e2deafdbdca3f34c3a5a6d8d19deeb"
    assert provenance["sourcePath"].endswith("Bulkhead_Fitted.stl")
