"""The 2F-85's spring-link spring on the robot every FR3 task spawns.

MuJoCo Menagerie's ``robotiq_2f85_v4`` holds each finger's spring link with 0.05 N.m/rad
toward ``springref`` 2.62 rad. AUTOLab's hand (RoboSimStudio's ``2f85.xml``, which
GaP's shadow closes on every FR3 task) raises that spring tenfold, to 0.5 N.m/rad
with 0.02 damping. The on-cart layers restate AUTOLab's values, so the fingers
close with the same spring preload GaP's shadow assumes.
"""

import pytest

pytest.importorskip("pxr")

from pxr import Usd

from ib_support import ASSETS_ROOT

_ON_CART = ASSETS_ROOT / "vabar__fr3_robotiq_2f85_on_cart"
_ROBOTS = (
    _ON_CART / "vabar__fr3_robotiq_2f85_on_cart.usda",
    _ON_CART / "vabar__fr3_robotiq_2f85_on_cart_instanceable.usda",
    _ON_CART / "tool_sorting.usda",
)
_SPRING_JOINTS = {"left_spring_link_joint", "right_spring_link_joint"}


def _spring_joints(path) -> dict[str, dict[str, float]]:
    """``{joint: {attribute: value}}`` for both spring-link joints, as composed."""
    stage = Usd.Stage.Open(str(path))
    assert stage is not None, path
    return {
        prim.GetName(): {
            name: prim.GetAttribute(f"mjc:{name}").Get() for name in ("stiffness", "damping", "springref")
        }
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.GetName() in _SPRING_JOINTS
    }


@pytest.mark.parametrize("path", _ROBOTS, ids=lambda p: p.name)
def test_spring_link_carries_autolabs_spring(path) -> None:
    joints = _spring_joints(path)
    assert set(joints) == _SPRING_JOINTS
    for spring in joints.values():
        assert spring["stiffness"] == pytest.approx(0.5)
        assert spring["damping"] == pytest.approx(0.02)
        # The rest angle is MuJoCo Menagerie's and AUTOLab's alike; only the gains differ.
        assert spring["springref"] == pytest.approx(2.62)


def test_canonical_hand_keeps_menageries_spring() -> None:
    """The vendored hand stays as generated; the FR3 profile restates the spring on top."""
    joints = _spring_joints(ASSETS_ROOT / "vabar__fr3_robotiq_2f85/franka_fr3_robotiq_2f85.usda")
    assert set(joints) == _SPRING_JOINTS
    for spring in joints.values():
        assert spring["stiffness"] == pytest.approx(0.05)


def test_newton_imports_the_spring_as_a_passive_joint_spring() -> None:
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
    from newton import ModelBuilder
    from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

    stage = Usd.Stage.Open(str(_ROBOTS[0]))
    # The cart is a remote, render-only reference; the robot's physics is all local.
    stage.GetPrimAtPath("/franka_fr3_v1/Geometry/base/fr3_link0/cart_stand").SetActive(False)
    builder = ModelBuilder()
    NewtonMJWarpManager._register_builder_attributes(builder)
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    stiffness = builder.custom_attributes["mujoco:dof_passive_stiffness"].values
    springref = builder.custom_attributes["mujoco:dof_springref"].values
    # Finalized into Model.joint_damping; on the builder it is still the MuJoCo attribute.
    damping = builder.custom_attributes["mujoco:dof_passive_damping"].values
    seen = set()
    for joint_index, label in enumerate(builder.joint_label):
        name = str(label).rsplit("/", 1)[-1]
        if name in _SPRING_JOINTS:
            dof = int(builder.joint_qd_start[joint_index])
            assert float(stiffness[dof]) == pytest.approx(0.5)
            assert float(springref[dof]) == pytest.approx(2.62)
            assert float(damping[dof]) == pytest.approx(0.02)
            seen.add(name)
    assert seen == _SPRING_JOINTS
