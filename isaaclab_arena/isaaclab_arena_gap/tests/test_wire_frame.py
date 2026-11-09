"""Arena sends GaP exactly the adapter's observation frame, with nothing simulated added."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# The wire tests drive the real policy and session with a scripted client.
class _Client:
    def __init__(self):
        self.frames = []

    def connect(self):
        pass

    def round_trip(self, frame):
        self.frames.append(frame)
        return {"left": {"joint_pos": [1.0, 2.0], "gripper": 0.5}, "arm_valid": True, "gripper_valid": True}

    def close(self):
        pass


def _policy(client=None, **config):
    policy_module = pytest.importorskip("isaaclab_arena_gap.gap_remote_policy")
    pytest.importorskip("torch")

    class Adapter(policy_module.GapObsAdapter):
        action_dim = 3

        def build_obs_frame(self, env, observation, env_id):
            return {"timestamp": 1.0, "left": {"joint_pos": [0.0, 0.0, 1.0]}}

        def decode_action(self, action, env, env_id=0):
            return np.asarray([*action["left"]["joint_pos"], action["left"]["gripper"]], dtype=np.float32)

        def hold_action(self, env, env_id=0):
            return np.zeros(3, dtype=np.float32)

    return policy_module.GapRemotePolicy(
        policy_module.GapRemotePolicyArgs(policy_device="cpu", **config),
        adapter=Adapter(),
        client_factory=lambda _host, _port: client,
    )


def _exchange(**config):
    client = _Client()
    _policy(client, **config).get_action(
        SimpleNamespace(unwrapped=SimpleNamespace(num_envs=1, step_dt=0.02)), {}
    )
    return client.frames[0]


def test_default_wire_frame_is_exactly_the_adapter_frame():
    assert _exchange() == {"timestamp": 1.0, "left": {"joint_pos": [0.0, 0.0, 1.0]}}
