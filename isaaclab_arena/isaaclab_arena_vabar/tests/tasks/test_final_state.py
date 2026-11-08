"""Tool hanging scores its final state, exactly as AUTOLab v1.0 does."""

from types import SimpleNamespace
import re

import pytest

pytest.importorskip("torch")

import torch


class _Asset:
    def __init__(self, name: str):
        self.name = name

    def get_contact_sensor_cfg(self, contact_against_object):
        from isaaclab.sensors import ContactSensorCfg

        return ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{self.name}",
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{contact_against_object.name}"],
        )

    def get_prim_path(self) -> str:
        return f"{{ENV_REGEX_NS}}/{self.name}"


def test_fixture_contact_filter_matches_nested_newton_shape_labels() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.task import _newton_fixture_contact_sensor

    sensor = _newton_fixture_contact_sensor(_Asset("plier"), (_Asset("rack"),))
    pattern = sensor.filter_shape_prim_expr[0].replace("{ENV_REGEX_NS}", "/World/envs/env_[^/]+")
    assert re.fullmatch(pattern, "/World/envs/env_0/rack/Geometry/hook_mesh")
    assert re.fullmatch(pattern, "/World/envs/env_0/rack")


class _Scene:
    """Two envs, each with one tool resting in its fixture, in contact and at rest."""

    def __init__(self):
        from isaaclab_arena_vabar.tasks.tool_hanging.task import ToolHangingTask

        self.task = ToolHangingTask(
            tools=[_Asset("plier")],
            fixtures=[_Asset("rack")],
            goals=[
                {
                    "tool": "plier",
                    "fixture": "rack",
                    "containment": {"minimum_xyz": (-0.2, -0.2, -0.2), "maximum_xyz": (0.2, 0.2, 0.2)},
                    "contact_force_n": 0.2,
                    "speed_m_s": 0.15,
                }
            ],
            episode_length_s=10.0,
        )
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        (goal,) = self.task.goal_specs
        self.env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            scene={
                "plier": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_pos_w=torch.zeros((2, 3)),
                        root_link_quat_w=identity.clone(),
                        root_lin_vel_w=torch.zeros((2, 3)),
                    )
                ),
                "rack": SimpleNamespace(
                    data=SimpleNamespace(root_link_pos_w=torch.zeros((2, 3)), root_link_quat_w=identity.clone())
                ),
                goal.contact_sensor_name: SimpleNamespace(
                    data=SimpleNamespace(force_matrix_w=torch.full((2, 1, 1, 3), 1.0))
                ),
            },
            episode_length_buf=torch.tensor([10, 10]),
            max_episode_length=600,
            external_policy_termination_requested=lambda: (False, False),
        )
        self.cfg = self.task.termination_cfg.success[0].predicate_sequence[0]

    def score(self) -> list[bool]:
        self.env.episode_length_buf += 1
        return self.cfg.func(self.env, **self.cfg.params).tolist()


def test_a_goal_met_mid_episode_does_not_end_it() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.terminations import tool_hanging_final_state_success

    scene = _Scene()
    assert scene.cfg.func is tool_hanging_final_state_success
    assert scene.score() == [False, False]
    # Only the env whose policy stopped is scored, on that step.
    scene.env.external_policy_termination_requested = lambda: (False, True)
    assert scene.score() == [False, True]


def test_the_time_limit_scores_the_final_state_without_a_policy_request() -> None:
    scene = _Scene()
    del scene.env.external_policy_termination_requested
    assert scene.score() == [False, False]
    scene.env.episode_length_buf[0] = scene.env.max_episode_length - 1
    assert scene.score() == [True, False]


def test_the_final_state_needs_every_clause() -> None:
    scene = _Scene()
    scene.env.external_policy_termination_requested = lambda: (True, True)
    assert scene.score() == [True, True]
    scene.env.scene["plier"].data.root_lin_vel_w[1, 0] = 0.2
    assert scene.score() == [True, False]
