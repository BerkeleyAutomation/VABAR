"""Physical-asset contracts for the canonical FR3/Robotiq USD."""

import pytest

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")
pytest.importorskip("pxr")

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Usd, UsdGeom, UsdPhysics

from ib_support import ASSETS_ROOT

_ASSET_DIR = ASSETS_ROOT / "vabar__fr3_robotiq_2f85"
_FR3_USD = _ASSET_DIR / "franka_fr3_robotiq_2f85.usda"
_INSTANCEABLE_FR3_USD = _ASSET_DIR / "franka_fr3_robotiq_2f85_instanceable.usda"
_ARM_JOINT_NAMES = {f"fr3_joint{index}" for index in range(1, 8)}
_DRIVEN_JOINT_NAMES = _ARM_JOINT_NAMES | {"left_driver_joint"}


def test_fr3_usd_is_the_single_physics_source() -> None:
    stage = Usd.Stage.Open(str(_FR3_USD))
    assert stage is not None
    assert not [
        prim for prim in stage.Traverse() if prim.GetTypeName() == "MjcActuator"
    ]

    rigid_bodies = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    gravity_compensated = {
        prim.GetName()
        for prim in rigid_bodies
        if prim.GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
    }
    assert gravity_compensated == {
        prim.GetName()
        for prim in rigid_bodies
        if prim.GetName() not in {"base", "fr3_link0"}
    }

    driven_joints = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.GetName() in _DRIVEN_JOINT_NAMES and prim.GetTypeName() != "MjcActuator"
    }
    assert set(driven_joints) == _DRIVEN_JOINT_NAMES
    for name, prim in driven_joints.items():
        drive = UsdPhysics.DriveAPI(prim, "angular")
        assert drive
        assert drive.GetStiffnessAttr().Get() == pytest.approx(1.0e-3)
        actuator_gravcomp = prim.GetAttribute("mjc:actuatorgravcomp")
        if name in _ARM_JOINT_NAMES:
            assert actuator_gravcomp.Get() is True
        else:
            assert not actuator_gravcomp.HasAuthoredValue()


def test_arena_manager_imports_authored_fr3_mujoco_attributes() -> None:
    stage = Usd.Stage.Open(str(_FR3_USD))
    builder = ModelBuilder()
    NewtonMJWarpManager._register_builder_attributes(builder)
    builder.add_usd(
        stage,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )

    condim = builder.custom_attributes["mujoco:condim"].values
    solref = builder.custom_attributes["mujoco:solref"].values
    solimp = builder.custom_attributes["mujoco:geom_solimp"].values
    gravcomp = builder.custom_attributes["mujoco:gravcomp"].values
    actuator_gravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"].values
    assert len(condim) == 4
    assert {int(value) for value in condim.values()} == {6}
    assert all(
        tuple(float(component) for component in value) == pytest.approx((0.004, 1.0))
        for value in solref.values()
    )
    assert all(
        tuple(float(component) for component in value)
        == pytest.approx((0.95, 0.99, 0.001, 0.5, 2.0))
        for value in solimp.values()
    )
    assert len(gravcomp) == 18
    assert all(float(value) == pytest.approx(1.0) for value in gravcomp.values())
    assert len(actuator_gravcomp) == 7
    assert all(bool(value) for value in actuator_gravcomp.values())

    armatures = {}
    for joint_index, label in enumerate(builder.joint_label):
        name = str(label).rsplit("/", 1)[-1]
        if name in _DRIVEN_JOINT_NAMES | {"right_driver_joint"}:
            dof_index = int(builder.joint_qd_start[joint_index])
            armatures[name] = float(builder.joint_armature[dof_index])
    assert armatures["fr3_joint1"] == pytest.approx(0.195)
    assert armatures["fr3_joint7"] == pytest.approx(0.074)
    assert armatures["left_driver_joint"] == pytest.approx(0.005)
    assert armatures["right_driver_joint"] == pytest.approx(0.005)


def test_instanceable_fr3_is_an_authored_render_variant() -> None:
    stage = Usd.Stage.Open(str(_INSTANCEABLE_FR3_USD))
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
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAuthoredReferences()
        and "__render_instance" not in str(prim.GetPath())
        and UsdGeom.Imageable(prim).ComputePurpose() in ("default", "render")
    ]
    assert original_meshes
    assert all(
        UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        for prim in original_meshes
    )


def test_right_robotiq_driver_mimics_the_driven_left() -> None:
    """One drive closes both fingers, which is why it states twice AUTOLab's per-finger gains."""
    from isaaclab_arena_vabar.embodiments.fr3_robotiq_2f85.config import (
        DRIVEN_FINGERS,
        GRIPPER_JOINT_NAME,
    )

    stage = Usd.Stage.Open(str(_FR3_USD))
    joints = {prim.GetName(): prim for prim in stage.Traverse() if prim.GetName().endswith("_driver_joint")}
    assert set(joints) == {"left_driver_joint", "right_driver_joint"}
    assert GRIPPER_JOINT_NAME == "left_driver_joint"
    leader = joints["right_driver_joint"].GetRelationship("newton:mimicJoint").GetTargets()
    assert [path.name for path in leader] == [GRIPPER_JOINT_NAME]
    assert not joints[GRIPPER_JOINT_NAME].GetRelationship("newton:mimicJoint").GetTargets()
    assert DRIVEN_FINGERS == len(joints)
