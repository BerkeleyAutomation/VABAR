"""Small public control contracts for the two industrial embodiments."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest


class _Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.values = (x, y, z)


class _TranslateAttribute:
    def __init__(self) -> None:
        self.value = _Vec3(0.0, 0.0, 0.0)

    def Get(self):
        return self.value

    def Set(self, value) -> bool:
        self.value = value
        return True


class _CameraPrim:
    def __init__(self, path: str) -> None:
        self.path = path
        self.translation = _TranslateAttribute()

    def GetAttribute(self, name: str):
        assert name == "xformOp:translate"
        return self.translation

    def GetPath(self) -> str:
        return self.path


class _UsdCamera:
    def __init__(self, prim: _CameraPrim) -> None:
        self.prim = prim

    def GetPrim(self) -> _CameraPrim:
        return self.prim


def test_fr3_action_and_observation_contract() -> None:
    from isaaclab_arena_vabar.embodiments.fr3_robotiq_2f85 import (
        IndustrialFr3Robotiq2f85Embodiment,
    )
    from isaaclab_arena_vabar.embodiments.fr3_robotiq_2f85.config import (
        ARM_JOINT_NAMES,
        END_EFFECTOR_BODY_NAME,
        GRIPPER_JOINT_NAME,
    )

    embodiment = IndustrialFr3Robotiq2f85Embodiment()
    assert embodiment.name == "vabar__fr3_robotiq_2f85"
    assert embodiment.action_config.arm_action.joint_names == ARM_JOINT_NAMES
    assert embodiment.action_config.arm_action.preserve_order is True
    assert embodiment.action_config.gripper_action.joint_names == [GRIPPER_JOINT_NAME]
    assert embodiment.action_config.gripper_action.scale == 0.8
    assert embodiment.action_config.gripper_action.offset == 0.0
    policy = embodiment.observation_config.policy
    assert all(
        hasattr(policy, name)
        for name in ("actions", "joint_pos", "gripper_pos", "eef_pos", "eef_quat")
    )
    assert END_EFFECTOR_BODY_NAME == "robotiq_base"
    robot = embodiment.scene_config.robot
    assert (
        robot.spawn.func == "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd"
    )
    assert robot.actuators["fr3_joint_1_2"].stiffness == 650.0
    assert robot.actuators["fr3_joint_1_2"].damping == 100.0
    # The left driver's drive carries both fingers: twice AUTOLab's per-finger 20/1.
    assert robot.actuators["robotiq_driver"].joint_names_expr == [GRIPPER_JOINT_NAME]
    assert robot.actuators["robotiq_driver"].stiffness == 40.0
    assert robot.actuators["robotiq_driver"].damping == 2.0
    assert robot.actuators["robotiq_driver"].effort_limit_sim == 1.0e6


def test_bimanual_yam_action_and_observation_contract() -> None:
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam.actions import (
        FiniteJointPositionAction,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import (
        ARM_JOINT_NAMES,
        GRIPPER_JOINT_NAME,
    )

    embodiment = IndustrialBimanualYamEmbodiment()
    actions = embodiment.action_config
    assert [field.name for field in fields(type(actions))] == [
        "left_arm_action",
        "left_gripper_action",
        "right_arm_action",
        "right_gripper_action",
    ]
    for side in ("left", "right"):
        arm = getattr(actions, f"{side}_arm_action")
        gripper = getattr(actions, f"{side}_gripper_action")
        assert arm.asset_name == f"{side}_robot"
        assert arm.joint_names == ARM_JOINT_NAMES
        assert arm.preserve_order is True
        assert arm.class_type is FiniteJointPositionAction
        assert gripper.asset_name == f"{side}_robot"
        assert gripper.joint_names == [GRIPPER_JOINT_NAME]
        for suffix in ("joint_pos", "gripper_pos", "eef_pos", "eef_quat"):
            assert hasattr(embodiment.observation_config.policy, f"{side}_{suffix}")
    assert embodiment.observation_config.policy.concatenate_terms is False


def test_rendered_camera_refresh_writes_selected_environment_prims() -> None:
    import torch

    from isaaclab_arena_vabar.embodiments.fr3_robotiq_2f85.camera_variations import (
        _write_usd_camera_translations,
    )

    prims = [_CameraPrim(f"/World/envs/env_{index}/camera") for index in range(3)]
    camera = SimpleNamespace(_sensor_prims=[_UsdCamera(prim) for prim in prims])

    _write_usd_camera_translations(
        camera,
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([[2.0, 2.1, 2.2], [0.0, 0.1, 0.2]]),
    )

    assert prims[0].translation.value.values == pytest.approx((0.0, 0.1, 0.2))
    assert prims[1].translation.value.values == pytest.approx((0.0, 0.0, 0.0))
    assert prims[2].translation.value.values == pytest.approx((2.0, 2.1, 2.2))


def test_yam_gripper_asset_commands_reset_and_feedback_agree() -> None:
    import numpy as np
    import warp as wp
    from pxr import Work

    Work.SetConcurrencyLimit(1)
    from pxr import Usd, UsdPhysics
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam.observations import gripper_pos

    opening = 0.037524
    embodiment = IndustrialBimanualYamEmbodiment()
    for instanceable in (False, True):
        embodiment.set_use_instanceable_meshes(instanceable)
        for side in ("left", "right"):
            robot = getattr(embodiment.scene_config, f"{side}_robot")
            action = getattr(embodiment.action_config, f"{side}_gripper_action")
            observation = getattr(embodiment.observation_config.policy, f"{side}_gripper_pos")
            assert robot.init_state.joint_pos["left_finger"] == opening
            assert robot.init_state.joint_pos["right_finger"] == -opening
            assert action.open_command_expr == {"left_finger": opening}
            stage = Usd.Stage.Open(robot.spawn.usd_path)
            joints = {prim.GetName(): UsdPhysics.PrismaticJoint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.PrismaticJoint)}
            assert joints["left_finger"].GetUpperLimitAttr().Get() == pytest.approx(opening)
            assert joints["right_finger"].GetLowerLimitAttr().Get() == pytest.approx(-opening)
            positions = np.array([[opening], [opening / 2], [0]], dtype=np.float32)
            data = SimpleNamespace(joint_names=["left_finger"], joint_pos=wp.array(positions, device="cpu"))
            env = SimpleNamespace(scene={f"{side}_robot": SimpleNamespace(data=data)})
            result = gripper_pos(env, **observation.params)
            np.testing.assert_allclose(result.numpy().ravel(), [0, 0.5, 1])


def test_bimanual_yam_linear_gripper_contract() -> None:
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam.config import LINEAR_4310

    embodiment = IndustrialBimanualYamEmbodiment(
        gripper="linear_4310", gripper_stiffness=600.0
    )
    for side in ("left", "right"):
        robot = getattr(embodiment.scene_config, f"{side}_robot")
        assert robot.spawn.variants == {"gripper": "linear_4310"}
        assert "left_finger" not in robot.init_state.joint_pos
        assert robot.init_state.joint_pos["joint7"] == LINEAR_4310.open_position
        assert robot.init_state.joint_pos["joint8"] == LINEAR_4310.open_position
        assert robot.actuators["gripper"].joint_names_expr == ["joint7", "joint8"]
        assert robot.actuators["gripper"].stiffness == 600.0
        assert "gripper_passive" not in robot.actuators
        action = getattr(embodiment.action_config, f"{side}_gripper_action")
        assert action.joint_names == ["joint7", "joint8"]
        assert action.open_command_expr == {"joint7": 0.0475, "joint8": 0.0475}
        params = getattr(embodiment.observation_config.policy, f"{side}_gripper_pos").params
        assert params["joint_name"] == "joint7"
        assert params["open_position"] == 0.0475


def test_bimanual_yam_crank_default_is_untouched() -> None:
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )

    robot = IndustrialBimanualYamEmbodiment().scene_config.left_robot
    assert robot.spawn.variants is None
    assert robot.actuators["gripper_passive"].joint_names_expr == ["right_finger"]


def test_bimanual_yam_rejects_unknown_gripper() -> None:
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )

    with pytest.raises(ValueError, match="gripper must be one of"):
        IndustrialBimanualYamEmbodiment(gripper="robotiq")
