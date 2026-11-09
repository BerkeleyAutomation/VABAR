"""RSS BDDL physical-property parity for tool-hanging assets."""

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Gf, Usd, UsdPhysics

from ib_support import ASSETS_ROOT as ASSET_ROOT


@pytest.mark.parametrize(
    ("name", "mass_kg", "center_of_mass", "inertia", "friction"),
    (
        (
            "vabar__plier_combination",
            0.04508982783590909,
            (0.0015829341, 0.0009236451, 0.0039990777),
            (
                (1.6196785509e-5, 4.9321280393e-9, 2.5426725170e-9),
                (4.9321280393e-9, 7.6205287769e-5, 2.6940730462e-12),
                (2.5426725170e-9, 2.6940730462e-12, 9.1767033155e-5),
            ),
            0.8,
        ),
        (
            "vabar__plier_cutter",
            0.06018748536007479,
            (1.8062397e-7, 0.0034940792, -0.053974073),
            (
                (4.7620811529e-5, 1.5390481778e-8, -9.3937535528e-10),
                (1.5390481778e-8, 8.5307910922e-5, 2.3396932392e-8),
                (-9.3937535528e-10, 2.3396932392e-8, 3.9513513912e-5),
            ),
            0.8,
        ),
        (
            "vabar__scissor_pair_1",
            0.08351116412086412,
            (-5.9712067e-7, 2.2029785e-7, -0.034583572),
            (
                (7.4175637565e-5, -2.1799921868e-9, 2.6805879738e-9),
                (-2.1799921868e-9, 1.0505172395e-4, -1.0721448174e-9),
                (2.6805879738e-9, -1.0721448174e-9, 3.2066880522e-5),
            ),
            0.7,
        ),
        (
            "vabar__scissor_pair_2",
            0.08154969400493428,
            (0.0025335031, 0.036702383, -0.00025140584),
            (
                (6.3277075242e-5, -2.5904257654e-6, 2.7814237580e-9),
                (-2.5904257654e-6, 1.3343787941e-5, -2.3204682620e-9),
                (2.7814237580e-9, -2.3204682620e-9, 7.5877542258e-5),
            ),
            0.7,
        ),
        (
            "vabar__wrench_ring_17",
            0.1487786126060627,
            (0.15414155, 0.17876531, 0.003169366),
            (
                (8.5961737568e-6, -3.4699589833e-6, -1.4614832935e-6),
                (-3.4699589833e-6, 5.8367982274e-4, -3.6339071752e-9),
                (-1.4614832935e-6, -3.6339071752e-9, 5.8963999618e-4),
            ),
            1.0,
        ),
    ),
)
def test_tool_hanging_asset_matches_rss_physics(
    name, mass_kg, center_of_mass, inertia, friction
):
    stage = Usd.Stage.Open(str(ASSET_ROOT / name / f"{name}.usda"))
    assert stage
    mass_prims = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.MassAPI)
        and (UsdPhysics.MassAPI(prim).GetMassAttr().Get() or 0.0) > 1.0e-5
    ]
    assert len(mass_prims) == 1
    mass_api = UsdPhysics.MassAPI(mass_prims[0])
    assert float(mass_api.GetMassAttr().Get()) == pytest.approx(mass_kg, abs=1.0e-7)
    assert tuple(mass_api.GetCenterOfMassAttr().Get()) == pytest.approx(
        center_of_mass, abs=1.0e-7
    )

    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
    rotation = np.asarray(Gf.Matrix3d(principal_axes))
    diagonal = np.asarray(mass_api.GetDiagonalInertiaAttr().Get())
    composed_inertia = rotation.T @ np.diag(diagonal) @ rotation
    np.testing.assert_allclose(composed_inertia, inertia, rtol=1.0e-5, atol=1.0e-9)

    materials = [
        UsdPhysics.MaterialAPI(prim)
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.MaterialAPI)
    ]
    assert any(
        float(material.GetStaticFrictionAttr().Get()) == pytest.approx(friction)
        and float(material.GetDynamicFrictionAttr().Get()) == pytest.approx(friction)
        for material in materials
    )


def test_wrench_contact_material_matches_rss_level1():
    from pxr import UsdShade

    path = ASSET_ROOT / "vabar__wrench_ring_19/vabar__wrench_ring_19.usda"
    stage = Usd.Stage.Open(str(path))
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert collisions
    for prim in collisions:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        assert api.GetStaticFrictionAttr().Get() == pytest.approx(1.5)
        assert api.GetDynamicFrictionAttr().Get() == pytest.approx(1.5)
        assert api.GetRestitutionAttr().Get() == pytest.approx(0)
