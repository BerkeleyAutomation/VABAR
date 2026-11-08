"""The Arena wire exposes only camera and measured robot I/O to GaP."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

JointPositionCommand = pytest.importorskip("gap.robot_io").JointPositionCommand

from isaaclab_arena_gap.bridge_robot_io import ArenaBridgeRobotIO
from isaaclab_arena_gap.cap_bimanual_yam_env import CapBimanualYamEnv
from isaaclab_arena_gap.cap_franka_real_env import CapFrankaRealEnv


def _wire(sequence: int, positions: np.ndarray) -> dict:
    return {
        "timestamp": 10.0 + sequence * 0.02,
        "left": {"joint_pos": positions.copy(), "joint_vel": np.zeros(8)},
        "front": {
            "images": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
            "depth_data": np.ones((2, 3), dtype=np.float32),
            "intrinsics": {"left": {"intrinsics_matrix": np.eye(3)}},
            "pose_mat": np.eye(4),
        },
    }


class FakeBridge:
    def __init__(self) -> None:
        self.server = SimpleNamespace(
            latest_observation=_wire(0, np.zeros(8)), exchange_lock=threading.Lock()
        )
        self._sim_step_count = 0
        self.commands: list[np.ndarray] = []

    def get_integration_failure(self):
        return None

    def set_robot_io_target(self, joints: np.ndarray, open_fraction: float) -> None:
        target = np.r_[joints, open_fraction]
        self.commands.append(target.copy())
        self.server.latest_observation = _wire(len(self.commands), target)

    def close(self) -> None:
        pass


def test_arena_bridge_reads_measured_state_and_exchanges_one_complete_target() -> None:
    bridge = FakeBridge()
    io = ArenaBridgeRobotIO(bridge, control_freq=50.0, camera_names=("front",))
    initial = io.observe()
    assert initial.sequence == 0
    assert initial.cameras["front"].rgb.shape == (2, 3, 3)
    assert initial.cameras["front"].depth_m.shape == (2, 3)

    target = np.r_[np.full(7, 0.2), 0.6]
    result = io.exchange(JointPositionCommand(target))
    assert result.sequence == 1
    np.testing.assert_allclose(bridge.commands[0], target)
    np.testing.assert_allclose(result.joint_positions, target)
    assert bridge._sim_step_count == 1


def test_exchange_returns_the_request_whose_reply_carries_the_command() -> None:
    from isaaclab_arena_gap.cap_franka_real_env import CapIdentityMsgpackServer

    server = CapIdentityMsgpackServer()
    server.latest_observation = _wire(0, np.zeros(8))
    server.latest_action = {"left": {"joint_pos": "previous"}}
    replies = {}

    def arena_request() -> None:
        request = _wire(1, np.zeros(8))
        replies[id(request)] = (request, server.accept(request))

    class Bridge(FakeBridge):
        def set_robot_io_target(self, joints, open_fraction):
            request = threading.Thread(target=arena_request)
            request.start()
            request.join(timeout=0.2)
            server.latest_action = {"left": {"joint_pos": "commanded"}}

    bridge = Bridge()
    bridge.server = server
    io = ArenaBridgeRobotIO(bridge, control_freq=50.0, camera_names=())
    sample = io.exchange(JointPositionCommand(np.r_[np.full(7, 0.2), 0.6]))
    (request, reply), = replies.values()
    assert sample.timestamp_s == request["timestamp"]
    assert reply["left"]["joint_pos"] == "commanded"


def test_arena_bridge_waits_for_first_observation() -> None:
    bridge = FakeBridge()
    bridge.server.latest_observation = None
    io = ArenaBridgeRobotIO(
        bridge, control_freq=50.0, camera_names=(), wait_timeout_s=0.5,
    )

    def publish() -> None:
        time.sleep(0.01)
        bridge.server.latest_observation = _wire(0, np.zeros(8))

    worker = threading.Thread(target=publish)
    worker.start()
    assert io.observe().sequence == 0
    worker.join()


def test_arena_bridge_rejects_unavailable_command_gains() -> None:
    io = ArenaBridgeRobotIO(FakeBridge(), control_freq=50.0, camera_names=())
    with pytest.raises(NotImplementedError, match="joint gains"):
        io.exchange(JointPositionCommand(np.zeros(8), kp=np.ones(8)))


def _unstarted(cls, **attributes):
    env = object.__new__(cls)
    env._target_lock = threading.Lock()
    env.server = SimpleNamespace(latest_action={}, latest_observation=None)
    for name, value in attributes.items():
        setattr(env, name, value)
    return env


def test_fr3_wire_target_updates_arm_and_gripper_in_one_snapshot() -> None:
    env = _unstarted(
        CapFrankaRealEnv, _target_update_count=0, _cap_gripper_commanded=False
    )
    CapFrankaRealEnv.set_robot_io_target(env, np.full(7, 0.25), 0.4)
    np.testing.assert_allclose(env._target_joints, np.full(7, 0.25))
    assert env._target_gripper == 0.4
    assert env._target_update_count == 1
    assert env._cap_gripper_commanded
    action = env.server.latest_action
    np.testing.assert_allclose(action["left"]["joint_pos"], np.full(7, 0.25))
    assert action["left"]["gripper"] == pytest.approx(0.4)
    assert action["command_valid"] is True


def test_republisher_never_overwrites_a_newer_published_target() -> None:
    env = _unstarted(
        CapFrankaRealEnv,
        _target_joints=np.zeros(7),
        _target_gripper=1.0,
        _target_update_count=0,
        _cap_gripper_commanded=False,
        _republish_stop=threading.Event(),
        _republish_period=0.0,
        _republish_count=0,
    )
    original = CapFrankaRealEnv._wire_action
    published_old = threading.Event()
    setter_done = threading.Event()

    def slow_wire_action(self, target, *args):
        if not np.any(target) and not published_old.is_set():
            published_old.set()
            setter_done.wait(0.2)
            env._republish_stop.set()
        return original(self, target, *args)

    env._wire_action = lambda *args: slow_wire_action(env, *args)
    republisher = threading.Thread(target=env._republish_loop)
    republisher.start()
    assert published_old.wait(1.0)
    setter = threading.Thread(
        target=CapFrankaRealEnv.set_robot_io_target, args=(env, np.full(7, 0.5), 0.3)
    )
    setter.start()
    setter_done.set()
    setter.join(1.0)
    republisher.join(1.0)
    np.testing.assert_allclose(
        env.server.latest_action["left"]["joint_pos"], np.full(7, 0.5)
    )


def test_republisher_crash_is_a_sticky_integration_failure() -> None:
    from isaaclab_arena_gap.stale_observation import BridgeIntegrationError

    env = _unstarted(
        CapFrankaRealEnv,
        _target_joints=np.zeros(7),
        _target_gripper=1.0,
        _target_update_count=0,
        _cap_gripper_commanded=False,
        _cap_republisher_failure=None,
        _cap_observation_guard=SimpleNamespace(failure=lambda: None),
        _republish_stop=threading.Event(),
        _republish_period=0.0,
        _republish_count=0,
    )

    def broken(*_args):
        raise TypeError("cannot encode target")

    env._wire_action = broken
    env._republish_loop()

    failure = env.get_integration_failure()
    assert isinstance(failure, BridgeIntegrationError)
    assert failure.failure_class == "runtime_error"
    assert "cannot encode target" in str(failure)
    assert env.get_integration_failure() is failure
    io = ArenaBridgeRobotIO(env, control_freq=50.0, camera_names=())
    with pytest.raises(BridgeIntegrationError):
        io.observe()


def test_two_arm_command_is_one_robot_io_exchange() -> None:
    class Bridge(FakeBridge):
        def __init__(self):
            super().__init__()
            self.server.latest_observation = {
                "timestamp": 10.0,
                **{name: {"joint_pos": np.zeros(7), "joint_vel": np.zeros(7)}
                   for name in ("left", "right")},
            }

        def set_robot_io_targets(self, arms, grippers):
            self.commands.append(np.c_[arms, grippers].reshape(-1).copy())
            self.server.latest_observation = {
                "timestamp": 10.02,
                **{name: {"joint_pos": np.r_[arms[index], grippers[index]],
                          "joint_vel": np.zeros(7)}
                   for index, name in enumerate(("left", "right"))},
            }

    bridge = Bridge()
    io = ArenaBridgeRobotIO(
        bridge, control_freq=50.0, camera_names=(), arm_dof=6,
        wire_arms=("left", "right"), joint_prefix="yam",
    )
    io.observe()
    command = np.r_[np.full(6, 0.1), 0.4, np.full(6, -0.2), 0.6]
    sample = io.exchange(JointPositionCommand(command))
    np.testing.assert_allclose(sample.joint_positions, command)
    assert sample.sequence == 1
    assert len(bridge.commands) == 1


def test_bimanual_wire_targets_are_published_atomically() -> None:
    env = _unstarted(
        CapBimanualYamEnv,
        arm_names=("left", "right"),
        _arm_targets=[np.zeros(6), np.zeros(6)],
        _gripper_targets=[1.0, 1.0],
        _arm_valid=[False, False],
        _gripper_valid=[False, False],
    )
    CapBimanualYamEnv.set_robot_io_targets(
        env, np.array([[0.1] * 6, [-0.2] * 6]), np.array([0.4, 0.6])
    )
    assert env._arm_valid == [True, True]
    assert env._gripper_valid == [True, True]
    assert env._gripper_targets == [0.4, 0.6]
    action = env.server.latest_action
    np.testing.assert_allclose(action["right"]["joint_pos"], [-0.2] * 6)
    assert action["left"]["gripper"] == pytest.approx(0.4)
    assert action["right"]["arm_valid"] and action["right"]["gripper_valid"]


@pytest.mark.parametrize("encoded_names", [False, True])
def test_passive_joint_samples_cross_robot_io_without_becoming_actuator_commands(encoded_names):
    bridge = FakeBridge()
    extra = {"names": ["left_spring_link_joint", "left_follower"],
             "positions": [0.4, -0.3], "velocities": [0.2, -0.1]}
    if encoded_names:
        extra["names"] = [name.encode() for name in extra["names"]]
    bridge.server.latest_observation["passive_joints"] = extra
    original = bridge.set_robot_io_target

    def send(joints, opened):
        original(joints, opened)
        bridge.server.latest_observation["passive_joints"] = {
            **extra, "positions": [0.5, -0.4],
        }

    bridge.set_robot_io_target = send
    io = ArenaBridgeRobotIO(bridge, control_freq=50, camera_names=(), passive_joints=True)
    assert io.joint_names[-2:] == ("left_spring_link_joint", "left_follower")
    sample = io.observe()
    np.testing.assert_allclose(sample.joint_positions[-2:], [0.4, -0.3])
    np.testing.assert_allclose(sample.joint_velocities[-2:], [0.2, -0.1])
    target = sample.joint_positions.copy()
    target[:7] = 0.1
    after = io.exchange(JointPositionCommand(target))
    assert bridge.commands[0].shape == (8,)
    np.testing.assert_allclose(after.joint_positions[-2:], [0.5, -0.4])
    bridge.server.latest_observation["passive_joints"]["names"] = ["different"]
    with pytest.raises(ValueError, match="layout changed"):
        io.observe()


@pytest.mark.parametrize(
    ("module_name", "env_class"),
    [
        ("cap_franka_real_env", "CapFrankaRealEnv"),
        ("cap_bimanual_yam_env", "CapBimanualYamEnv"),
    ],
)
def test_simulated_bridges_let_go_home_move(monkeypatch, module_name, env_class) -> None:
    import importlib

    runtime = importlib.import_module(f"isaaclab_arena_gap.{module_name}")
    monkeypatch.setenv("CAP_GAP_ARM_BASE_POSES", json.dumps([[[0, y, 0], [0, 0, 0, 1]] for y in (0.17, -0.17)]))
    monkeypatch.setattr(runtime, env_class, lambda **kwargs: SimpleNamespace(**kwargs))
    _env, config = runtime.make_env(camera_names=["front"])
    assert config.is_real is False
