"""The YAM USD's ``gripper`` variant set: crank_4310 default, linear_4310 swap."""

import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdPhysics

from ib_support import ASSETS_ROOT

_ASSET_DIR = ASSETS_ROOT / "vabar__i2rt_yam"
_LINK_6 = "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


def _stage(usd_name: str, gripper: str | None) -> Usd.Stage:
    stage = Usd.Stage.Open(str(_ASSET_DIR / usd_name))
    if gripper is not None:
        variant_set = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("gripper")
        variant_set.SetVariantSelection(gripper)
    return stage


def _prismatic_joints(stage: Usd.Stage) -> set[str]:
    return {
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.PrismaticJoint)
    }


@pytest.mark.parametrize("usd_name", ["i2rt_yam_default.usda", "i2rt_yam_instanceable.usda"])
def test_crank_is_the_default_gripper(usd_name):
    stage = _stage(usd_name, None)
    variant_set = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("gripper")
    assert variant_set.GetVariantNames() == ["crank_4310", "crank_4310_wide", "linear_4310"]
    assert variant_set.GetVariantSelection() == "crank_4310"
    assert _prismatic_joints(stage) == {"left_finger", "right_finger"}
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(_LINK_6)).GetMassAttr().Get()
    assert mass == pytest.approx(0.367476)


@pytest.mark.parametrize("usd_name", ["i2rt_yam_default.usda", "i2rt_yam_instanceable.usda"])
def test_linear_variant_replaces_the_crank_hand(usd_name):
    stage = _stage(usd_name, "linear_4310")
    assert _prismatic_joints(stage) == {"joint7", "joint8"}
    link_6 = stage.GetPrimAtPath(_LINK_6)
    assert sorted(child.GetName() for child in link_6.GetChildren()) == [
        "joint6",
        "linear_4310_attach",
        "linear_4310_gripper_attach_joint",
    ]
    # The hand body carries its own mass; link_6 keeps only a placeholder.
    assert UsdPhysics.MassAPI(link_6).GetMassAttr().Get() == pytest.approx(1e-4)
    colliders = {"tip_left": 0, "tip_right": 0}
    for prim in stage.Traverse():
        collision = UsdPhysics.CollisionAPI(prim)
        if "linear_4310_attach" not in str(prim.GetPath()):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not collision.GetCollisionEnabledAttr().Get():
            continue
        assert prim.GetName().startswith("CableConvex_"), prim.GetPath()
        colliders[prim.GetParent().GetName()] += 1
    # AUTOLab's CoACD fingertip hulls replace the hand's single convex hull.
    assert colliders == {"tip_left": 51, "tip_right": 52}
    joint7 = stage.GetPrimAtPath(
        f"{_LINK_6}/linear_4310_attach/Geometry/gripper/tip_left/joint7"
    )
    assert joint7.GetAttribute("newton:armature").Get() == pytest.approx(0.05)
    assert joint7.GetAttribute("newton:damping").Get() == pytest.approx(2.0)
    assert joint7.GetAttribute("newton:friction").Get() == pytest.approx(0.5)
    # AUTOLab drives each finger on its own; nothing couples joint7 to joint8.
    assert not joint7.GetRelationship("newton:mimicJoint").GetTargets()
    # Newton's MuJoCo solver reads mjc:gravcomp; the generated hand only had
    # mjc:body:gravcomp, which left its weight on the wrist.
    for body in ("gripper", "gripper/tip_left", "gripper/tip_right"):
        prim = stage.GetPrimAtPath(f"{_LINK_6}/linear_4310_attach/Geometry/{body}")
        assert prim.GetAttribute("mjc:gravcomp").Get() == 1.0


@pytest.mark.parametrize("usd_name", ["i2rt_yam_default.usda", "i2rt_yam_instanceable.usda"])
def test_wide_crank_variant_opens_each_finger_47_5_mm(usd_name):
    """AUTOLab's yam_tool_hanging MJCF: the crank hand, each finger on [-0.00205, 0.0475]."""
    stage = _stage(usd_name, "crank_4310_wide")
    joints = {
        prim.GetName(): UsdPhysics.PrismaticJoint(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.PrismaticJoint)
    }
    assert set(joints) == {"left_finger", "right_finger"}
    assert joints["left_finger"].GetLowerLimitAttr().Get() == pytest.approx(-0.00205)
    assert joints["left_finger"].GetUpperLimitAttr().Get() == pytest.approx(0.0475)
    assert joints["right_finger"].GetLowerLimitAttr().Get() == pytest.approx(-0.0475)
    assert joints["right_finger"].GetUpperLimitAttr().Get() == pytest.approx(0.00205)
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("gripper").SetVariantSelection("crank_4310")
