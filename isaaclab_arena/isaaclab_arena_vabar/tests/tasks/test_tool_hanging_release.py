"""Tool hanging scores as the RSS version does: hung, at rest and released, held for a second."""

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


class _Robot:
    """One arm whose ``link_6`` stands at ``x`` along world x, unrotated."""

    def __init__(self, x: float):
        identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
        self.data = SimpleNamespace(
            body_names=["link_6"],
            body_link_pos_w=torch.tensor([[[x, 0.0, 0.0]]]).repeat(2, 1, 1),
            body_link_quat_w=identity,
        )


class _Scene:
    """Two envs, each with one tool resting in its fixture, in contact and at rest, hands 0.5 m away."""

    def __init__(self):
        from isaaclab_arena_vabar.tasks.tool_hanging.task import ToolHangingTask
        from isaaclab_arena_vabar.tasks.tool_hanging.terminations import HandTcp

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
                    "hands_clear_m": 0.11,
                }
            ],
            episode_length_s=10.0,
            hands=(
                HandTcp("left_robot", "link_6", (0.0, 0.0, 0.0)),
                HandTcp("right_robot", "link_6", (0.0, 0.0, 0.0)),
            ),
        )
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        (goal,) = self.task.goal_specs
        self.env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.25,
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
                "left_robot": _Robot(0.5),
                "right_robot": _Robot(-0.5),
                goal.contact_sensor_name: SimpleNamespace(
                    data=SimpleNamespace(force_matrix_w=torch.full((2, 1, 1, 3), 1.0))
                ),
            },
            episode_length_buf=torch.tensor([10, 10]),
            max_episode_length=600,
        )
        self.cfg = self.task.termination_cfg.success[0].predicate_sequence[0]

    def score(self) -> list[bool]:
        self.env.episode_length_buf += 1
        return self.cfg.func(self.env, **self.cfg.params).tolist()


def test_success_needs_the_goal_held_for_a_second() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.terminations import tool_hanging_held_success

    scene = _Scene()
    assert scene.cfg.func is tool_hanging_held_success
    # 0.25 s steps: held time counts from the first step the goal held, so the fifth step is 1 s.
    assert [scene.score() for _ in range(5)] == [[False, False]] * 4 + [[True, True]]


def test_a_hand_within_the_release_radius_resets_the_hold() -> None:
    scene = _Scene()
    for _ in range(3):
        scene.score()
    scene.env.scene["left_robot"].data.body_link_pos_w[1, 0, 0] = 0.1
    assert scene.score() == [False, False]
    scene.env.scene["left_robot"].data.body_link_pos_w[1, 0, 0] = 0.5
    assert [scene.score() for _ in range(5)] == [[True, False]] * 4 + [[True, True]]


def test_every_clause_is_needed() -> None:
    scene = _Scene()
    scene.env.scene["plier"].data.root_lin_vel_w[1, 0] = 0.2
    assert [scene.score() for _ in range(5)][-1] == [True, False]


def test_a_reset_clears_the_hold() -> None:
    scene = _Scene()
    for _ in range(4):
        scene.score()
    scene.env.episode_length_buf[:] = 0
    assert [scene.score() for _ in range(5)] == [[False, False]] * 4 + [[True, True]]
