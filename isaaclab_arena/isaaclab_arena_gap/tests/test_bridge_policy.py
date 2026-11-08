"""Episode lifecycle and result semantics at the GaP/Arena policy seam."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("isaaclab_arena")
torch = pytest.importorskip("torch")

import isaaclab_arena_gap.gap_remote_policy as policy_module
import isaaclab_arena_gap.gap_session as session_module
from isaaclab_arena_gap.bridge_contract import (
    GRAPH_RESULT_SCHEMA,
    ContractError,
    graph_result_reference,
    validate_graph_result,
)
from isaaclab_arena_gap.external_policy_termination import (
    requested_external_policy_terminations,
)
from isaaclab_arena_gap.gap_remote_policy import (
    GapObsAdapter,
    GapRemotePolicy,
    GapRemotePolicyArgs,
)


class _Adapter(GapObsAdapter):
    action_dim = 3

    def __init__(self) -> None:
        self.frames = 0

    def build_obs_frame(self, env, observation, env_id):
        self.frames += 1
        return {"timestamp": self.frames}

    def decode_action(self, action, env, env_id=0):
        left = action["left"]
        return np.asarray([*left["joint_pos"], left["gripper"]], dtype=np.float32)

    def hold_action(self, env, env_id=0):
        return np.asarray([9.0, 9.0, -1.0], dtype=np.float32)


class _WideAdapter(_Adapter):
    def action_dim_for(self, env, env_id=0):
        return 6

    def command_action_dim_for(self, env, env_id=0):
        return 3

    def decode_action(self, action, env, env_id=0):
        return np.concatenate(
            (super().decode_action(action, env, env_id), [8.0, 8.0, -2.0])
        ).astype(np.float32)

    def hold_action(self, env, env_id=0):
        return np.asarray([9.0, 9.0, -1.0, 8.0, 8.0, -2.0], dtype=np.float32)


class _DualAdapter(_WideAdapter):
    def command_action_dim_for(self, env, env_id=0):
        return 6

    def command_action_slices_for(self, env, env_id=0):
        return {"left": slice(3, 6), "right": slice(0, 3)}

    def decode_action(self, action, env, env_id=0):
        left = action["left"]
        right = action["right"]
        return np.asarray(
            [
                *right["joint_pos"],
                right["gripper"],
                *left["joint_pos"],
                left["gripper"],
            ],
            dtype=np.float32,
        )


class _WorldAdapter(_Adapter):
    def arm_base_position(self, env, env_id=0):
        del env
        return (0.25, 0.30 + env_id, 0.75)


class _DualWorldAdapter(_DualAdapter):
    def arm_base_positions(self, env, env_id=0):
        del env, env_id
        return ((0.25, 0.30, 0.75), (0.25, -0.30, 0.75))


class _Client:
    def __init__(self, replies) -> None:
        self.replies = iter(replies)
        self.connected = 0
        self.closed = 0
        self.frames = []

    def connect(self) -> None:
        self.connected += 1

    def round_trip(self, frame):
        self.frames.append(frame)
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.closed += 1


class _FleetClient(_Client):
    def __init__(self, env_id: int, barrier: threading.Barrier) -> None:
        super().__init__([])
        self.env_id = env_id
        self.barrier = barrier

    def round_trip(self, frame):
        self.frames.append(frame)
        self.barrier.wait(timeout=5.0)
        identity = dict(frame["_isaac_cap"])
        return {
            "left": {
                "joint_pos": [float(self.env_id), float(self.env_id) + 0.5],
                "gripper": 0.25,
            },
            "arm_valid": True,
            "gripper_valid": True,
            "_isaac_cap": identity,
        }


class _SequencedFleetClient(_Client):
    def __init__(self, env_id: int, replies) -> None:
        super().__init__(replies)
        self.env_id = env_id

    def round_trip(self, frame):
        self.frames.append(frame)
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        reply = dict(value)
        if "_isaac_cap" not in reply:
            reply["_isaac_cap"] = dict(frame["_isaac_cap"])
        return reply


def _reply(*, arm_valid=True, gripper_valid=True):
    return {
        "left": {"joint_pos": [1.0, 2.0], "gripper": 0.25},
        "arm_valid": arm_valid,
        "gripper_valid": gripper_valid,
    }


def _env(num_envs: int = 1):
    return SimpleNamespace(
        unwrapped=SimpleNamespace(num_envs=num_envs, step_dt=0.02),
    )


def _policy(
    client: _Client | None = None,
    *,
    adapter=None,
    spawn_fn=None,
    client_factory=None,
    **config,
) -> GapRemotePolicy:
    """Build a CPU policy; managed (auto_spawn) policies default to a tool-hanging graph."""
    config.setdefault("policy_device", "cpu")
    if config.get("auto_spawn"):
        config.setdefault("gap_graph", "tool_hanging/gap_perception")
    if client is not None:
        client_factory = lambda _host, _port: client  # noqa: E731
    return GapRemotePolicy(
        GapRemotePolicyArgs(**config),
        adapter=adapter or _Adapter(),
        spawn_fn=spawn_fn,
        client_factory=client_factory,
    )


def test_policy_is_lazy_and_performs_one_exchange_per_arena_step() -> None:
    client = _Client([_reply(), _reply()])
    policy = _policy(client)
    assert client.connected == 0

    first = policy.get_action(_env(), {})
    second = policy.get_action(_env(), {})

    assert client.connected == 1
    assert len(client.frames) == 2
    assert torch.equal(first, second)
    assert first.tolist() == [[1.0, 2.0, 0.25]]
    policy.close()


def test_terminal_hold_retains_each_wire_arms_last_gripper_command():
    policy = _policy(_Client([]), adapter=_DualAdapter())
    policy._sessions[0]._last_action = torch.tensor([[1.0, 2.0, 0.2, 3.0, 4.0, 0.8]])
    held = policy._sessions[0].hold_action(_env())
    assert held.tolist()[0] == pytest.approx([9.0, 9.0, 0.2, 8.0, 8.0, 0.8])
    policy.close()


@pytest.mark.parametrize(
    "close_kind, expected_steps",
    [
        ("graph_clean_close", 60),
        ("graph_declared_failure", 60),
        ("graph_exit_nonzero", 60),
        ("protocol_error", 0),
        ("io_timeout", 0),
    ],
)
def test_task_can_settle_after_graph_failure(close_kind, expected_steps) -> None:
    policy = _policy(graph_success_settle_s=1.0, graph_failure_settle_s=1.0)
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=1 / 60))
    policy._sessions[0].schedule_episode_termination(close_kind, env)
    assert policy._sessions[0]._episode_termination_countdown == expected_steps
    policy.close()


def test_default_graph_failure_still_terminates_immediately() -> None:
    policy = _policy(_Client([]))
    policy._sessions[0].schedule_episode_termination("graph_declared_failure", _env())
    assert policy._sessions[0]._episode_termination_countdown == 0
    policy.close()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_failure_settle_duration_must_be_finite_and_nonnegative(value) -> None:
    with pytest.raises(ValueError, match="graph_failure_settle_s"):
        _policy(graph_failure_settle_s=value)


def test_scalar_lifecycle_connects_and_closes_one_client() -> None:
    client = _Client([_reply()])
    policy = _policy(client_factory=lambda _host, _port: client)

    action = policy.get_action(_env(), {})
    policy.close()

    assert action.tolist() == [[1.0, 2.0, 0.25]]
    assert client.connected == 1
    assert client.closed == 1


def test_invalid_channels_hold_the_observed_robot_state() -> None:
    client = _Client([_reply(arm_valid=False, gripper_valid=True)])
    policy = _policy(client)

    action = policy.get_action(_env(), {})

    assert action.tolist() == [[9.0, 9.0, 0.25]]
    policy.close()


def test_wide_adapter_holds_uncommanded_terms_and_preserves_primary_gripper() -> None:
    client = _Client(
        [
            _reply(arm_valid=False, gripper_valid=True),
            ConnectionError("peer closed"),
        ]
    )
    policy = _policy(client, adapter=_WideAdapter())

    commanded = policy.get_action(_env(), {})
    disconnected = policy.get_action(_env(), {})

    expected = [[9.0, 9.0, 0.25, 8.0, 8.0, -2.0]]
    assert commanded.tolist() == disconnected.tolist() == expected
    policy.close()


def test_dual_adapter_applies_per_arm_validity_and_latches_both_grippers() -> None:
    reply = {
        "left": {"joint_pos": [1.0, 2.0], "gripper": 0.25},
        "right": {"joint_pos": [3.0, 4.0], "gripper": 0.75},
        "arm_valid": {"left": False, "right": True},
        "gripper_valid": {"left": True, "right": False},
    }
    client = _Client([reply, ConnectionError("peer closed")])
    policy = _policy(client, adapter=_DualAdapter())

    commanded = policy.get_action(_env(), {})
    disconnected = policy.get_action(_env(), {})

    assert commanded.tolist() == [[3.0, 4.0, -1.0, 8.0, 8.0, 0.25]]
    assert disconnected.tolist() == [[9.0, 9.0, -1.0, 8.0, 8.0, 0.25]]
    policy.close()


def test_disconnect_latches_hold_and_requests_termination_without_respawn() -> None:
    client = _Client([ConnectionError("peer closed")])
    policy = _policy(client)

    first = policy.get_action(_env(), {})
    second = policy.get_action(_env(), {})

    assert first.tolist() == second.tolist() == [[9.0, 9.0, -1.0]]
    assert client.connected == 1
    assert requested_external_policy_terminations() == (True,)
    policy.close()


def test_reset_reaps_one_managed_process_and_next_episode_spawns_fresh(
    monkeypatch,
) -> None:
    clients = [_Client([_reply()]), _Client([_reply()])]
    spawned = []
    arm_bases = []
    reaped = []

    def spawn(config, _port):
        process = SimpleNamespace(pid=100 + len(spawned), poll=lambda: None)
        arm_bases.append(config.gap_arm_base_position)
        spawned.append(process)
        return process

    monkeypatch.setattr(session_module, "_free_port", lambda: 12345 + len(spawned))
    monkeypatch.setattr(
        session_module,
        "_terminate_proc",
        lambda process, pgid: reaped.append((process.pid, pgid)),
    )
    policy = _policy(
        adapter=_WorldAdapter(),
        spawn_fn=spawn,
        client_factory=lambda _host, _port: clients[len(spawned) - 1],
        auto_spawn=True,
    )

    policy.get_action(_env(), {})
    policy.reset()
    policy.get_action(_env(), {})
    policy.close()

    assert [process.pid for process in spawned] == [100, 101]
    assert arm_bases == [(0.25, 0.30, 0.75), (0.25, 0.30, 0.75)]
    assert reaped == [(100, 100), (101, 101)]


def test_policy_rejects_unsafe_managed_configuration() -> None:
    with pytest.raises(ValueError, match="not loopback"):
        _policy(auto_spawn=True, gap_host="example.com")


def test_twenty_sessions_exchange_concurrently_and_preserve_env_order(
    monkeypatch,
) -> None:
    num_envs = 20
    barrier = threading.Barrier(num_envs)
    clients_by_port = {}
    spawned = []
    ports = iter(range(12000, 12100))

    def spawn(config, port):
        env_id = config.gap_env_id
        client = _FleetClient(env_id, barrier)
        clients_by_port[port] = client
        process = SimpleNamespace(pid=1000 + env_id, poll=lambda: None)
        spawned.append((config, port, process))
        return process

    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(session_module, "_terminate_proc", lambda *_args: None)
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=num_envs,
        gap_fleet_id="test-fleet",
        gap_base_seed=123,
    )

    action = policy.get_action(_env(num_envs=num_envs), {})

    assert action.shape == (num_envs, 3)
    assert action[:, 0].tolist() == [float(env_id) for env_id in range(num_envs)]
    assert len(spawned) == num_envs
    assert len({port for _config, port, _process in spawned}) == num_envs
    identities = [client.frames[0]["_isaac_cap"] for client in clients_by_port.values()]
    assert {item["session_id"] for item in identities} == {
        f"test-fleet/env-{env_id:03d}" for env_id in range(num_envs)
    }
    assert {item["episode_id"] for item in identities} == {0}
    assert {item["request_id"] for item in identities} == {1}
    assert len({item["seed"] for item in identities}) == num_envs
    policy.close()


def test_partial_reset_restarts_only_selected_session_and_rejects_stale_reply(
    monkeypatch,
) -> None:
    clients_by_port = {}
    spawned = []
    reaped = []
    ports = iter(range(13000, 13100))

    def spawn(config, port):
        episode = config.gap_episode_id
        env_id = config.gap_env_id
        replies = [_reply(), _reply()]
        if env_id == 1 and episode == 1:
            stale = _reply()
            stale["_isaac_cap"] = {
                "session_id": config.gap_session_id,
                "episode_id": 0,
                "request_id": 1,
            }
            replies = [stale]
        clients_by_port[port] = _SequencedFleetClient(env_id, replies)
        process = SimpleNamespace(pid=2000 + len(spawned), poll=lambda: None)
        spawned.append((env_id, episode, process.pid))
        return process

    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(
        session_module,
        "_terminate_proc",
        lambda process, pgid: reaped.append((process.pid, pgid)),
    )
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=3,
        gap_fleet_id="partial",
    )

    policy.get_action(_env(num_envs=3), {})
    policy.reset(env_ids=[1])
    action = policy.get_action(_env(num_envs=3), {})

    assert spawned[:3] == [(0, 0, 2000), (1, 0, 2001), (2, 0, 2002)]
    assert spawned[3][:2] == (1, 1)
    # The selected old episode is reaped at reset; its replacement is reaped
    # when the stale reply fails that session closed.
    assert reaped == [(2001, 2001), (2003, 2003)]
    assert action.tolist() == [
        [1.0, 2.0, 0.25],
        [9.0, 9.0, -1.0],
        [1.0, 2.0, 0.25],
    ]
    assert requested_external_policy_terminations() == (False, True, False)
    policy.close()


def test_fleet_port_is_reusable_after_session_teardown(monkeypatch) -> None:
    ports = iter((13500, 13501, 13500))
    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    policy = _policy(auto_spawn=True, num_envs=2)
    assert len(policy._sessions) == 2

    policy._assign_fleet_ports()
    assert policy._fleet_allocated_ports == {13500, 13501}
    policy._sessions[0].teardown()
    assert policy._fleet_allocated_ports == {13501}

    policy._assign_fleet_ports()
    assert policy._sessions[0].assigned_port == 13500
    assert policy._fleet_allocated_ports == {13500, 13501}
    policy.close()


def test_eager_fleet_survives_initial_reset(monkeypatch) -> None:
    clients_by_port = {}
    spawned = []
    reaped = []
    ports = iter((13600, 13601, 13602))

    def spawn(config, port):
        spawned.append((config.gap_env_id, port))
        clients_by_port[port] = _SequencedFleetClient(config.gap_env_id, [_reply()])
        return SimpleNamespace(pid=6000 + config.gap_env_id, poll=lambda: None)

    monkeypatch.setenv("CAP_EAGER_GAP_SPAWN", "1")
    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(
        session_module,
        "_terminate_proc",
        lambda process, pgid: reaped.append((process.pid, pgid)),
    )
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=3,
        gap_arm_base_position=[0.0, 0.0, 0.0],
    )

    assert sorted(env_id for env_id, _port in spawned) == [0, 1, 2]
    policy.reset()
    assert reaped == []
    policy.get_action(_env(num_envs=3), {})
    assert len(spawned) == 3
    policy.close()


def test_eager_spawn_waits_for_runtime_arm_bases(monkeypatch) -> None:
    spawned_with_bases = []
    client = _Client(
        [
            {
                "left": {"joint_pos": [1.0, 2.0], "gripper": 0.25},
                "right": {"joint_pos": [3.0, 4.0], "gripper": 0.75},
                "arm_valid": {"left": True, "right": True},
                "gripper_valid": {"left": True, "right": True},
            }
        ]
    )

    def spawn(config, _port):
        spawned_with_bases.append(config.gap_arm_base_positions)
        return SimpleNamespace(pid=6100, poll=lambda: None)

    monkeypatch.setenv("CAP_EAGER_GAP_SPAWN", "1")
    monkeypatch.setattr(session_module, "_terminate_proc", lambda _proc, _pgid: None)
    policy = _policy(
        adapter=_DualWorldAdapter(),
        spawn_fn=spawn,
        client_factory=lambda _host, _port: client,
        robot_profile="fr3",
        auto_spawn=True,
        gap_graph="gear_mesh/gap_perception",
    )

    assert spawned_with_bases == []
    policy.get_action(_env(), {})
    assert spawned_with_bases == [((0.25, 0.30, 0.75), (0.25, -0.30, 0.75))]
    policy.close()


class _BimanualWorldAdapter(_DualWorldAdapter):
    def base_poses(self, env, env_id=0):
        del env, env_id
        return [[[0.25, 0.30, 0.75], [0, 0, 0, 1]], [[0.25, -0.30, 0.75], [0, 0, 0, 1]]]


_DUAL_REPLY = {
    "left": {"joint_pos": [1.0, 2.0], "gripper": 0.25},
    "right": {"joint_pos": [3.0, 4.0], "gripper": 0.75},
    "arm_valid": {"left": True, "right": True},
    "gripper_valid": {"left": True, "right": True},
}
_EXPLICIT_POSES = [[[0, 1, 2], [0, 0, 0, 1]], [[0, -1, 2], [0, 0, 0, 1]]]


@pytest.mark.parametrize(
    "profile, adapter, explicit, expected",
    [
        ("fr3", _WorldAdapter(), {}, "0.25,0.3,0.75"),
        (
            "fr3",
            _WorldAdapter(),
            {"gap_arm_base_position": [0.0, 0.0, 1.35]},
            "0.0,0.0,1.35",
        ),
        (
            "fr3",
            _WorldAdapter(),
            {"gap_arm_base_positions": [[0.0, 0.0, 1.35]]},
            "0.0,0.0,1.35",
        ),
        (
            "yam_bimanual",
            _BimanualWorldAdapter(),
            {},
            _BimanualWorldAdapter().base_poses(None),
        ),
        (
            "yam_bimanual",
            _BimanualWorldAdapter(),
            {"gap_arm_base_poses": _EXPLICIT_POSES},
            _EXPLICIT_POSES,
        ),
    ],
)
def test_eager_spawn_never_starts_a_child_without_its_arm_bases(
    monkeypatch, tmp_path, profile, adapter, explicit, expected
) -> None:
    import json

    spawned = []

    def spawn(config, _port):
        env = {}
        policy_module._apply_adapter_tool_env(config, env)
        spawned.append(
            env.get("CAP_GAP_ARM_BASE_POSITION")
            if profile == "fr3"
            else json.loads(env["CAP_GAP_ARM_BASE_POSES"])
        )
        return SimpleNamespace(pid=6200, poll=lambda: None)

    status_path = tmp_path / "bridge-status.json"
    monkeypatch.setenv("CAP_EAGER_GAP_SPAWN", "1")
    monkeypatch.setattr(session_module, "_terminate_proc", lambda _proc, _pgid: None)
    policy = _policy(
        adapter=adapter,
        spawn_fn=spawn,
        client_factory=lambda _host, _port: _Client(
            [_reply() if profile == "fr3" else _DUAL_REPLY]
        ),
        robot_profile=profile,
        auto_spawn=True,
        bridge_status_path=str(status_path),
        **explicit,
    )

    assert len(spawned) == (1 if explicit else 0)
    policy.get_action(_env(), {})
    assert spawned == [json.loads(json.dumps(expected))]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["perf_knobs"]["CAP_EAGER_GAP_SPAWN"] == (
        "1" if explicit else "deferred"
    )
    policy.close()


def _fixture(position, *, kinematic=True):
    rigid_props = SimpleNamespace(kinematic_enabled=kinematic)
    return SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(rigid_props=rigid_props)),
        data=SimpleNamespace(
            root_pos_w=torch.tensor([position], dtype=torch.float64),
            root_quat_w=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64),
        ),
    )


def test_eager_spawn_waits_for_the_scene_fixtures(monkeypatch) -> None:
    import json

    spawned = []

    def spawn(config, _port):
        env = {}
        policy_module._apply_adapter_tool_env(config, env)
        spawned.append(json.loads(env["CAP_GAP_SCENE_STATICS"]))
        return SimpleNamespace(pid=6201, poll=lambda: None)

    monkeypatch.setenv("CAP_EAGER_GAP_SPAWN", "1")
    monkeypatch.setattr(session_module, "_terminate_proc", lambda _proc, _pgid: None)
    policy = _policy(
        adapter=_BimanualWorldAdapter(),
        spawn_fn=spawn,
        client_factory=lambda _host, _port: _Client([_DUAL_REPLY]),
        robot_profile="yam_bimanual",
        auto_spawn=True,
        gap_arm_base_poses=_EXPLICIT_POSES,
        gap_scene_statics={"peg0": "peg_0"},
    )
    assert spawned == []
    env = _env()
    env.unwrapped.scene = {"peg_0": _fixture([0.1, 0.2, 0.8])}
    policy.get_action(env, {})
    assert spawned == [{"peg0": [0.1, 0.2, 0.8, 0.0, 0.0, 1.0, 0.0]}]
    policy.close()


def test_one_episode_per_env_retires_fast_slots_without_respawn(monkeypatch) -> None:
    clients_by_port = {}
    spawned = []
    ports = iter(range(14000, 14100))

    def spawn(config, port):
        env_id = config.gap_env_id
        replies = (
            [ConnectionError("fast slot complete")]
            if env_id == 0
            else [_reply(), ConnectionError("slow slot complete")]
        )
        clients_by_port[port] = _SequencedFleetClient(env_id, replies)
        process = SimpleNamespace(pid=3000 + len(spawned), poll=lambda: None)
        spawned.append((env_id, config.gap_episode_id))
        return process

    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(session_module, "_terminate_proc", lambda *_args: None)
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=2,
        gap_fleet_id="one-wave",
        one_episode_per_env=True,
    )

    first = policy.get_action(_env(num_envs=2), {})
    assert first.tolist()[0] == [9.0, 9.0, -1.0]
    assert requested_external_policy_terminations() == (True, False)
    policy.reset(env_ids=[0])

    second = policy.get_action(_env(num_envs=2), {})
    assert second.tolist()[0] == [9.0, 9.0, -1.0]
    assert requested_external_policy_terminations() == (False, True)
    policy.reset(env_ids=[1])

    assert requested_external_policy_terminations() == (False, False)
    assert policy.get_action(_env(num_envs=2), {}).tolist() == [
        [9.0, 9.0, -1.0],
        [9.0, 9.0, -1.0],
    ]
    assert spawned == [(0, 0), (1, 0)]
    policy.close()


def test_unexpected_child_failure_holds_and_terminates_only_that_slot(
    monkeypatch,
) -> None:
    clients_by_port = {}
    spawned = []
    ports = iter(range(15000, 15100))

    def spawn(config, port):
        failure = RuntimeError("child implementation crashed")
        replies = [failure] if config.gap_env_id == 1 else [_reply()]
        clients_by_port[port] = _SequencedFleetClient(config.gap_env_id, replies)
        process = SimpleNamespace(pid=4000 + len(spawned), poll=lambda: None)
        spawned.append(process)
        return process

    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(session_module, "_terminate_proc", lambda *_args: None)
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=3,
    )

    action = policy.get_action(_env(num_envs=3), {})

    assert action.tolist() == [
        [1.0, 2.0, 0.25],
        [9.0, 9.0, -1.0],
        [1.0, 2.0, 0.25],
    ]
    assert requested_external_policy_terminations() == (False, True, False)
    policy.close()


def test_adapter_settings_are_propagated_to_children() -> None:
    config = GapRemotePolicyArgs(
        gap_control_frequency_hz=60.0,
        gap_arm_base_position=(0.25, 0.30, 0.75),
        gap_work_arm="right",
        gap_adaptation="rss_usbc_insert_easy",
    )
    env = {"CAP_GAP_WORK_ARM": "left", "GAP_ADAPTATION": "stale"}

    policy_module._apply_adapter_tool_env(config, env)

    assert env["CAP_GAP_ARM_BASE_POSITION"] == "0.25,0.3,0.75"
    assert env["CAP_GAP_CONTROL_FREQUENCY_HZ"] == "60.0"
    assert env["CAP_GAP_WORK_ARM"] == "right"
    assert env["GAP_ADAPTATION"] == "rss_usbc_insert_easy"
    policy_module._apply_adapter_tool_env(GapRemotePolicyArgs(), env)
    assert "CAP_GAP_WORK_ARM" not in env
    assert "GAP_ADAPTATION" not in env


def test_unbound_robot_profiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="robot_profile"):
        _policy(robot_profile="yam")


def test_default_launcher_is_the_rss_adapter(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **kwargs: captured.update(command=command, env=kwargs["env"]),
    )
    policy_module._default_spawn_gap(GapRemotePolicyArgs(gap_graph="gear_mesh/gap_perception"), 12345)
    assert captured["command"][0].endswith("isaaclab_arena_gap/scripts/run_rss_graph.sh")
    assert captured["env"]["GAP_PORT"] == "12345"


@pytest.mark.parametrize("frequency", (0.0, -1.0, float("nan")))
def test_control_frequency_must_be_positive_and_finite(frequency: float) -> None:
    with pytest.raises(ValueError, match="gap_control_frequency_hz"):
        _policy(gap_control_frequency_hz=frequency)


def test_each_fleet_session_spawns_its_own_child(monkeypatch) -> None:
    spawned = []
    clients_by_port = {}
    ports = iter((16000, 16001))

    def spawn(config, port):
        spawned.append((config.gap_env_id, config.gap_session_id, port))
        clients_by_port[port] = _SequencedFleetClient(config.gap_env_id, [_reply()])
        return SimpleNamespace(pid=5000 + config.gap_env_id, poll=lambda: None)

    monkeypatch.setattr(session_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(session_module, "_terminate_proc", lambda *_args: None)
    policy = _policy(
        spawn_fn=spawn,
        client_factory=lambda _host, port: clients_by_port[port],
        auto_spawn=True,
        num_envs=2,
    )

    policy.get_action(_env(num_envs=2), {})
    policy.close()

    assert sorted(spawned) == [
        (0, f"{policy._fleet_id}/env-000", 16000),
        (1, f"{policy._fleet_id}/env-001", 16001),
    ]


def _result(outcome: str) -> dict:
    success = outcome == "success"
    return {
        "schema_version": GRAPH_RESULT_SCHEMA,
        "timestamp": "2026-09-02T00:00:00Z",
        "campaign_id": "campaign",
        "plan_sha256": "plan",
        "image_ref": "image",
        "sample_id": "sample",
        "attempt_id": "attempt",
        "outcome": outcome,
        "success": success,
        "exit_status": "success" if success else "failure",
        "failure_class": None if success else "other",
        "error_type": None,
        "error": None,
        "duration_s": 1.0,
        "trace_path": None,
    }


def test_graph_result_is_identity_bound_but_does_not_replace_arena_scoring() -> None:
    result = _result("success")
    outcome = validate_graph_result(
        result,
        campaign_id="campaign",
        plan_sha256="plan",
        image_ref="image",
        sample_id="sample",
        attempt_id="attempt",
    )
    reference = graph_result_reference(result, relative_path="graph_result.json")

    assert outcome == "success"
    assert reference["relative_path"] == "graph_result.json"
    assert set(reference) == {"schema_version", "relative_path", "canonical_sha256"}
    with pytest.raises(ContractError, match="mismatched sample_id"):
        validate_graph_result(
            result,
            campaign_id="campaign",
            plan_sha256="plan",
            image_ref="image",
            sample_id="different",
            attempt_id="attempt",
        )


def test_spawn_passes_workflow_inputs_and_both_base_poses(monkeypatch):
    import json
    import subprocess

    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(subprocess, "Popen", popen)
    bases = [([0, 0.25, 0.8], [0, 0, 0, 1]), ([0, -0.25, 0.8], [0, 0, 1, 0])]
    config = GapRemotePolicyArgs(
        robot_profile="yam_bimanual",
        gap_graph="tool_hanging/gap_perception",
        gap_inputs={"instruction": "hang the wrench on the hook"},
        gap_arm_base_poses=bases,
    )
    policy_module._default_spawn_gap(config, 12345)
    assert captured["command"][1] == "--inputs"
    key, value = captured["command"][2].split("=", 1)
    assert key == "instruction"
    assert json.loads(value) == "hang the wrench on the hook"
    assert json.loads(captured["env"]["CAP_GAP_ARM_BASE_POSES"]) == json.loads(
        json.dumps(bases)
    )


def test_workspace_is_passed_as_task_config_outside_robot_io():
    import json

    env = {}
    config = GapRemotePolicyArgs(gap_workspace={"surface_z": -0.117})
    policy_module._apply_adapter_tool_env(config, env)
    assert json.loads(env["CAP_GAP_WORKSPACE"]) == {"surface_z": -0.117}
    policy_module._apply_adapter_tool_env(GapRemotePolicyArgs(), env)
    assert "CAP_GAP_WORKSPACE" not in env


@pytest.mark.parametrize(
    "exit_code, state, close_kind, error_type",
    [
        (0, "ended", "graph_clean_close", None),
        (3, "failed", "graph_exit_nonzero", "ConnectionError"),
    ],
)
def test_channel_close_status_records_an_error_only_for_failures(
    monkeypatch, tmp_path, exit_code, state, close_kind, error_type
) -> None:
    import json

    status_path = tmp_path / "bridge-status.json"
    monkeypatch.setattr(session_module, "_terminate_proc", lambda *_args: None)
    policy = _policy(
        spawn_fn=lambda _config, _port: SimpleNamespace(
            pid=6300, poll=lambda: exit_code, wait=lambda timeout=None: exit_code
        ),
        client_factory=lambda _host, _port: _Client([ConnectionError("peer closed")]),
        auto_spawn=True,
        bridge_status_path=str(status_path),
    )

    policy.get_action(_env(), {})

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert (status["state"], status["close_kind"]) == (state, close_kind)
    assert status["error_type"] == error_type
    assert (status["error"] is None) == (error_type is None)
    policy.close()
