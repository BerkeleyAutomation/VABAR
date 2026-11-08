"""Arena observation/action adapters satisfy the GaP robot interface."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("isaaclab_arena")
torch = pytest.importorskip("torch")

from _experiment_configs import policy_config
from isaaclab_arena_gap.msgpack_transport import GapProtocolError
from isaaclab_arena_gap.robot_adapter import (
    Fr3Adapter,
    YamAdapter,
)


class _Scene(dict):
    pass


def _action(dof: int, *, gripper: object = 1.0, joints: object | None = None):
    return {
        "left": {
            "joint_pos": [0.0] * dof if joints is None else joints,
            "gripper": gripper,
        }
    }


def _camera(num_envs: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            pos_w=torch.tensor([[0.95, 0.0, 0.90]], dtype=torch.float32).repeat(
                num_envs, 1
            ),
            quat_w_ros=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).repeat(
                num_envs, 1
            ),
            intrinsic_matrices=torch.eye(3, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_envs, 1, 1),
            output={
                "rgb": torch.zeros((num_envs, 8, 12, 4), dtype=torch.uint8),
                "distance_to_image_plane": torch.ones(
                    (num_envs, 8, 12, 1), dtype=torch.float32
                ),
            },
        )
    )


def _fr3_env(num_envs: int = 1) -> SimpleNamespace:
    body_names = ["robotiq_base", "robotiq_left_pad", "robotiq_right_pad"]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_names=[*(f"fr3_joint{i}" for i in range(1, 8)), "left_driver_joint"],
            joint_pos=torch.zeros((num_envs, 8), dtype=torch.float32),
            joint_vel=torch.zeros((num_envs, 8), dtype=torch.float32),
            root_pos_w=torch.zeros((num_envs, 3), dtype=torch.float32),
            root_quat_w=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
            ).repeat(num_envs, 1),
            body_names=body_names,
            body_pos_w=torch.zeros((num_envs, len(body_names), 3), dtype=torch.float32),
            body_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0]] * len(body_names)], dtype=torch.float32
            ).repeat(num_envs, 1, 1),
        )
    )
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=_Scene(robot=robot, exterior_cam=_camera(num_envs))
        )
    )


def _yam_robot(position: float = 0.0, finger: str = "left_finger") -> SimpleNamespace:
    names = [*(f"joint{index}" for index in range(1, 7)), finger]
    return SimpleNamespace(
        data=SimpleNamespace(
            joint_names=names,
            joint_pos=torch.full((1, len(names)), position, dtype=torch.float32),
            joint_vel=torch.zeros((1, len(names)), dtype=torch.float32),
            root_pos_w=torch.zeros((1, 3), dtype=torch.float32),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        )
    )


def _yam_env(action_dim: int = 7) -> SimpleNamespace:
    robot = _yam_robot()
    scene = _Scene(
        left_robot=robot,
        top_camera=_camera(),
        left_wrist_camera=_camera(),
    )
    if action_dim == 14:
        scene["right_robot"] = _yam_robot(0.01)
        scene["right_wrist_camera"] = _camera()
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            action_manager=SimpleNamespace(total_action_dim=action_dim),
        )
    )


def test_fr3_observation_and_action_contract() -> None:
    adapter = Fr3Adapter()
    env = _fr3_env()

    frame = adapter.build_obs_frame(env, {}, 0)
    camera = frame["robot0_robotview"]
    assert len(frame["left"]["joint_pos"]) == 8
    assert len(frame["left"]["joint_vel"]) == 8
    assert camera["images"]["rgb"].shape == (8, 12, 3)
    assert camera["depth_data"].shape == (8, 12)
    assert camera["intrinsics"]["left"]["intrinsics_matrix"].shape == (3, 3)
    assert camera["pose_mat"].shape == (4, 4)
    assert np.isfinite(camera["pose_mat"]).all()
    assert frame["passive_joints"]["names"] == []
    assert "_isaac_cap" not in frame

    assert adapter.decode_action(_action(7, gripper=1.0), env)[7] == 0.0
    assert adapter.decode_action(_action(7, gripper=0.0), env)[7] == 1.0
    assert adapter.decode_action(_action(7, gripper=0.882353), env)[7] == pytest.approx(
        0.117647
    )


def test_yam_observation_action_and_hold_contract() -> None:
    adapter = YamAdapter()
    env = _yam_env()

    frame = adapter.build_obs_frame(env, {}, 0)
    assert len(frame["left"]["joint_pos"]) == 7
    assert {"overhead", "eye_in_hand_left"} <= frame.keys()
    action = adapter.decode_action(_action(6, gripper=0.25), env)
    assert action.shape == (7,)
    assert action[6] == pytest.approx(0.75)
    held = adapter.hold_action(env)
    assert np.array_equal(held[:6], np.zeros(6, dtype=np.float32))
    assert held[6] == 1.0


def test_yam_bimanual_action_holds_the_uncommanded_right_arm() -> None:
    adapter = YamAdapter()
    env = _yam_env(action_dim=14)

    action = adapter.decode_action(_action(6, gripper=0.25), env)
    held = adapter.hold_action(env)

    assert action.shape == held.shape == (14,)
    assert action[6] == pytest.approx(0.75)
    assert np.allclose(action[7:13], 0.01)
    assert action[13] == pytest.approx(1.0 - 0.01 / 0.037524)
    assert np.array_equal(action[7:], held[7:])


def test_yam_can_map_the_gap_channel_to_the_physical_right_arm() -> None:
    adapter = YamAdapter(arms=("right",))
    env = _yam_env(action_dim=14)

    action = adapter.decode_action(_action(6, joints=[0.2] * 6, gripper=0.25), env)
    frame = adapter.build_obs_frame(env, {}, 0)

    assert np.array_equal(action[:7], adapter.hold_action(env)[:7])
    assert np.allclose(action[7:13], 0.2)
    assert action[13] == pytest.approx(0.75)
    assert frame["left"]["joint_pos"][0] == pytest.approx(0.01)
    assert "eye_in_hand_left" in frame


def test_yam_maps_two_wire_arms_to_configured_physical_arms() -> None:
    adapter = YamAdapter(arms=("right", "left"))
    env = _yam_env(action_dim=14)
    action = {
        "left": {"joint_pos": [0.2] * 6, "gripper": 0.25},
        "right": {"joint_pos": [0.3] * 6, "gripper": 0.75},
    }

    decoded = adapter.decode_action(action, env)
    frame = adapter.build_obs_frame(env, {}, 0)

    assert np.allclose(decoded[:6], 0.3)
    assert decoded[6] == pytest.approx(0.25)
    assert np.allclose(decoded[7:13], 0.2)
    assert decoded[13] == pytest.approx(0.75)
    assert frame["left"]["joint_pos"][0] == pytest.approx(0.01)
    assert frame["right"]["joint_pos"][0] == pytest.approx(0.0)
    assert adapter.command_action_slices_for(env) == {
        "left": slice(7, 14),
        "right": slice(0, 7),
    }


@pytest.mark.parametrize(
    "action",
    [
        _action(7, joints=[0.0] * 6),
        _action(7, joints=[float("nan")] * 7),
        _action(7, gripper=-0.01),
        _action(7, gripper=1.01),
        _action(7, gripper="open"),
    ],
)
def test_invalid_gap_actions_fail_closed(action) -> None:
    with pytest.raises(GapProtocolError):
        Fr3Adapter().decode_action(action, None)


def test_hold_action_preserves_the_observed_fr3_state() -> None:
    adapter = Fr3Adapter()
    env = _fr3_env()
    env.unwrapped.scene["robot"].data.joint_pos[0, :7] = 0.1
    env.unwrapped.scene["robot"].data.joint_pos[0, 7] = 0.2

    held = adapter.hold_action(env)
    assert np.allclose(held[:7], 0.1)
    assert held[7] == pytest.approx(0.25)


def test_named_camera_must_supply_matching_rgb_and_depth() -> None:
    env = _fr3_env()
    camera = env.unwrapped.scene["exterior_cam"]
    camera.data.output["distance_to_image_plane"] = torch.ones(
        (1, 7, 12, 1), dtype=torch.float32
    )
    with pytest.raises(GapProtocolError, match="RGB and depth resolutions differ"):
        Fr3Adapter().build_obs_frame(env, {}, 0)


def test_fr3_adapter_reads_only_the_selected_environment() -> None:
    adapter = Fr3Adapter()
    env = _fr3_env(num_envs=2)
    robot = env.unwrapped.scene["robot"]
    robot.data.joint_pos[0] = 0.1
    robot.data.joint_pos[1] = 0.2
    camera = env.unwrapped.scene["exterior_cam"]
    camera.data.output["rgb"][1, ..., 0] = 7

    frame = adapter.build_obs_frame(env, {}, 1)
    held = adapter.hold_action(env, env_id=1)

    assert frame["left"]["joint_pos"][0] == pytest.approx(0.2)
    assert np.all(frame["robot0_robotview"]["images"]["rgb"][..., 0] == 7)
    assert np.allclose(held[:7], 0.2)


@pytest.mark.parametrize(
    ("adapter", "env", "robot_name", "camera_name", "bridge_name"),
    (
        (Fr3Adapter(), _fr3_env(), "robot", "exterior_cam", "robot0_robotview"),
        (YamAdapter(), _yam_env(), "left_robot", "top_camera", "overhead"),
    ),
)
def test_camera_and_robot_base_use_environment_local_world_frame(
    adapter, env, robot_name: str, camera_name: str, bridge_name: str
) -> None:
    env.unwrapped.scene.env_origins = torch.tensor([[10.0, 20.0, 30.0]])
    env.unwrapped.scene[robot_name].data.root_pos_w[0] = torch.tensor(
        [10.2, 20.3, 30.4]
    )
    env.unwrapped.scene[camera_name].data.pos_w[0] = torch.tensor([10.3, 20.5, 30.7])

    pose = adapter.build_obs_frame(env, {}, 0)[bridge_name]["pose_mat"]

    assert pose[:3, 3] == pytest.approx([0.3, 0.5, 0.7], abs=2.0e-6)
    assert adapter.arm_base_position(env, 0) == pytest.approx(
        [0.2, 0.3, 0.4], abs=2.0e-6
    )


def test_wrench_easy_observation_exposes_all_rss_cameras() -> None:
    from isaaclab_arena_gap.gap_remote_policy import (
        GapRemotePolicyArgs,
        _apply_adapter_tool_env,
        _resolve_robot_adapter,
    )

    args = GapRemotePolicyArgs(
        **policy_config("tool_hanging/vabar_tool_hanging_wrench_easy_gap_baseline.yaml")
    )
    adapter = _resolve_robot_adapter(args)
    child_env = {}
    _apply_adapter_tool_env(args, child_env)
    assert set(child_env["CAP_GAP_CAMERA_NAME"].split(",")) == {
        "overhead",
        "side",
        "eye_in_hand_left",
        "eye_in_hand_right",
    }
    env = _yam_env(14)
    env.unwrapped.scene["side_camera"] = _camera()
    env.unwrapped.scene["right_wrist_camera"] = _camera()
    env.unwrapped.scene["right_wrist_camera"].data.output["rgb"].fill_(73)

    frame = adapter.build_obs_frame(env, {}, 0)

    for name in ("overhead", "side", "eye_in_hand_left", "eye_in_hand_right"):
        assert frame[name]["depth_data"].shape == (8, 12)
    assert np.all(frame["eye_in_hand_right"]["images"]["rgb"] == 73)
    assert np.all(frame["eye_in_hand_left"]["images"]["rgb"] == 0)


@pytest.mark.parametrize("difficulty", ("easy", "medium"))
def test_cable_baseline_camera_contract(difficulty):
    from isaaclab_arena_gap.gap_remote_policy import GapRemotePolicyArgs, _resolve_robot_adapter

    policy = policy_config(
        f"cable_routing/vabar_cable_routing_{difficulty}_gap_baseline.yaml"
    )
    adapter = _resolve_robot_adapter(GapRemotePolicyArgs(**policy))
    scene = _Scene(
        # Cable runs on the linear_4310 hand, whose driven finger is joint7.
        left_robot=_yam_robot(0.1, finger="joint7"),
        right_robot=_yam_robot(0.2, finger="joint7"),
        top_camera=_camera(),
        cable_camera=_camera(),
        left_wrist_camera=_camera(),
        right_wrist_camera=_camera(),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    frame = adapter.build_obs_frame(env, {}, 0)
    camera_names = {
        name
        for name, value in frame.items()
        if isinstance(value, dict) and "depth_data" in value
    }
    # Both levels stream the rig AUTOLab's cable graph reads: overhead, work-arm wrist, cable.
    assert camera_names == {"overhead", "wrist", "cable"}


def test_bimanual_yam_command_validity_is_independent_per_channel():
    from isaaclab_arena_gap.robot_adapter import BimanualYamAdapter

    env = _yam_env(14)
    adapter = BimanualYamAdapter()
    held = adapter.hold_action(env)
    row = adapter.decode_validated_action(
        {
            "right": {
                "joint_pos": [0.2] * 6,
                "arm_valid": True,
                "gripper": float("nan"),
                "gripper_valid": False,
            },
            "left": {
                "joint_pos": [float("nan")] * 6,
                "arm_valid": False,
                "gripper": 0.4,
                "gripper_valid": True,
            },
        },
        env,
    )
    assert row[:6] == pytest.approx(held[:6])
    assert row[6] == pytest.approx(0.6)
    assert row[7:13] == pytest.approx([0.2] * 6)
    assert row[13] == pytest.approx(held[13])
    with pytest.raises(GapProtocolError):
        adapter.decode_validated_action(
            {"right": {"joint_pos": [float("nan")] * 6, "arm_valid": True}}, env
        )


def test_fr3_observation_tolerates_an_arm_without_finger_joint() -> None:
    env = _fr3_env()
    data = env.unwrapped.scene["robot"].data
    data.joint_names = data.joint_names[:7]
    data.joint_pos = torch.full((1, 7), 0.1)
    data.joint_vel = torch.full((1, 7), 0.2)

    frame = Fr3Adapter().build_obs_frame(env, {}, 0)

    assert frame["left"]["joint_pos"][-1] == pytest.approx(1.0)
    assert frame["left"]["joint_vel"] == pytest.approx([0.2] * 7 + [0.0])


def test_fr3_exports_passive_joints_without_simulator_geometry():
    env = _fr3_env()
    data = env.unwrapped.scene["robot"].data
    data.joint_names += ["left_spring_link_joint", "left_follower"]
    data.joint_pos = torch.tensor([[0.0] * 8 + [0.4, -0.3]])
    data.joint_vel = torch.tensor([[0.0] * 8 + [0.2, -0.1]])
    frame = Fr3Adapter().build_obs_frame(env, {}, 0)
    assert frame["passive_joints"]["names"] == data.joint_names[8:]
    np.testing.assert_allclose(frame["passive_joints"]["positions"], [0.4, -0.3])
    np.testing.assert_allclose(frame["passive_joints"]["velocities"], [0.2, -0.1])


def test_linear_yam_adapter_reads_the_linear_hand():
    from isaaclab_arena_gap.robot_adapter import make_robot_adapter
    from isaaclab_arena_gap.yam_description import YAM_LINEAR_4310

    adapter = make_robot_adapter("yam_bimanual_linear")
    robot = _yam_robot(0.5 * YAM_LINEAR_4310.open_joint_m, finger="joint7")
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=_Scene(left_robot=robot)))
    hold = adapter._hold(env, "left_robot", 0)
    assert hold[-1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="arm contract"):
        adapter._hold(
            SimpleNamespace(
                unwrapped=SimpleNamespace(scene=_Scene(left_robot=_yam_robot()))
            ),
            "left_robot",
            0,
        )
