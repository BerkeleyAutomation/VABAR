"""Physical-asset contracts for the canonical I2RT YAM USD."""

import pytest

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")
pytest.importorskip("pxr")

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Usd, UsdGeom, UsdPhysics

from ib_support import ASSETS_ROOT

_ASSET_DIR = ASSETS_ROOT / "vabar__i2rt_yam"
_YAM_USD = _ASSET_DIR / "i2rt_yam_default.usda"
_INSTANCEABLE_YAM_USD = _ASSET_DIR / "i2rt_yam_instanceable.usda"
_FINGERTIP = (
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/"
    "link_left_finger/lf_rot/lf_down/Sphere"
)
_JOINTS = (
    "/i2rt_yam/Geometry/arm/link_1/joint1",
    "/i2rt_yam/Geometry/arm/link_1/link_2/joint2",
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/joint3",
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/joint4",
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/joint5",
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/joint6",
)
_GRIPPER_JOINT = (
    "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/"
    "link_left_finger/left_finger"
)


def test_yam_usd_is_the_single_physics_source() -> None:
    assert not (_ASSET_DIR / "bddl_contact_overrides.usda").exists()
    stage = Usd.Stage.Open(str(_YAM_USD))
    assert stage is not None
    assert not [
        prim for prim in stage.Traverse() if prim.GetTypeName() == "MjcActuator"
    ]

    fingertip = stage.GetPrimAtPath(_FINGERTIP)
    assert fingertip.GetAttribute("mjc:condim").Get() == 6
    assert fingertip.GetAttribute("mjc:solref").Get() == pytest.approx((0.004, 1.0))
    assert fingertip.GetAttribute("mjc:solimp").Get() == pytest.approx(
        (0.998, 0.998, 0.001, 0.5, 2.0)
    )

    gravcomp = [
        prim.GetAttribute("mjc:gravcomp").Get()
        for prim in stage.Traverse()
        if prim.GetAttribute("mjc:gravcomp").HasAuthoredValue()
    ]
    assert gravcomp == [1.0] * 12
    assert not any(
        prim.GetAttribute("mjc:body:gravcomp").HasAuthoredValue()
        for prim in stage.Traverse()
    )

    for path in _JOINTS:
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(path), "angular")
        assert drive
        assert drive.GetStiffnessAttr().Get() == pytest.approx(
            1.0e-3 * 3.141592653589793 / 180.0
        )
    gripper_drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(_GRIPPER_JOINT), "linear")
    assert gripper_drive
    assert gripper_drive.GetStiffnessAttr().Get() == pytest.approx(1.0e-3)


def test_arena_manager_imports_authored_yam_mujoco_attributes() -> None:
    stage = Usd.Stage.Open(str(_YAM_USD))
    builder = ModelBuilder()
    NewtonMJWarpManager._register_builder_attributes(builder)
    builder.add_usd(
        stage,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )

    assert builder.custom_attributes["mujoco:condim"].values
    assert builder.custom_attributes["mujoco:gravcomp"].values
    assert builder.custom_attributes["mujoco:solref"].values
    assert all(
        int(value) == 6
        for value in builder.custom_attributes["mujoco:condim"].values.values()
    )
    assert all(
        float(value) == pytest.approx(1.0)
        for value in builder.custom_attributes["mujoco:gravcomp"].values.values()
    )
    assert all(
        tuple(float(component) for component in value) == pytest.approx((0.004, 1.0))
        for value in builder.custom_attributes["mujoco:solref"].values.values()
    )


def test_instanceable_yam_is_an_authored_render_variant() -> None:
    stage = Usd.Stage.Open(str(_INSTANCEABLE_YAM_USD))
    assert stage is not None
    render_instances = [
        prim
        for prim in stage.Traverse()
        if prim.GetName().endswith("__render_instance")
    ]
    assert render_instances
    assert all(prim.IsInstance() for prim in render_instances)

    original_meshes = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.HasAuthoredReferences()
    ]
    assert original_meshes
    assert all(
        UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        for prim in original_meshes
    )
