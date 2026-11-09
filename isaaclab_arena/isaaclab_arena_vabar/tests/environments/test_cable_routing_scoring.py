"""Easy scoring follows the pinned RSS evaluator, not fixture orientation."""

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from isaaclab_arena_vabar.environments.cable_routing import environment as cable
from isaaclab_arena_vabar.embodiments.bimanual_yam.config import (
    END_EFFECTOR_BODY_NAME,
    GRASP_SITE_OFFSET,
)
from isaaclab_arena_vabar.tasks.cable_routing import terminations


@pytest.mark.parametrize("difficulty", ("easy", "medium"))
def test_solver_configuration_is_accepted_by_pinned_newton(difficulty):
    from isaaclab_newton.physics.newton_manager import NewtonManager

    cfg = SimpleNamespace(
        sim=SimpleNamespace(), scene=SimpleNamespace(num_envs=1, left_robot=object())
    )
    cable._configure_cable_routing_newton_physics(
        cfg, variant=getattr(cable, f"_{difficulty}_variant")(10046)
    )
    physics = cfg.sim.physics
    mode = NewtonManager._resolve_deterministic_mode(physics.deterministic_mode)
    NewtonManager._validate_deterministic_solver_cfg(physics.solver_cfg, mode)
    assert physics.deterministic_mode == "not_guaranteed"
    assert physics.solver_cfg.entries[0].solver_cfg.disable_sensors is True
    assert cfg.sim.dt == 1 / 60
    assert cfg.decimation == 1
    assert physics.num_substeps == 16


@pytest.mark.parametrize("fixture_yaw", (0.0, math.pi / 4, math.pi / 2))
@pytest.mark.parametrize("xy,expected", (((0.04, 0.04), True), ((0.055, 0.0), False)))
def test_easy_goal_regions_are_translated_world_boxes(fixture_yaw, xy, expected):
    env, goal = _scene(fixture_yaw, xy)
    env.external_policy_termination_requested = lambda: (True,)
    score = _score(env, goal)
    assert score.tolist() == [expected]
    env.external_policy_termination_requested = lambda: (False,)
    assert _score(env, goal).tolist() == [False]
    del env.external_policy_termination_requested
    assert _score(env, goal).tolist() == [False]
    # Native scores the final state at truncation, even without graph termination.
    env.episode_length_buf[:] = env.max_episode_length
    assert _score(env, goal).tolist() == [expected]


def _score(env, goal):
    return terminations.terminated_cable_route_success(
        env,
        peg_names=("peg",),
        port_name="port",
        goal=goal,
        segment_half_lengths=(0.5 * math.dist((0, 0, 0), (0, 0, 0.01)),),
        seat_frame_z=-0.5 * cable._PEG_HEIGHT,
        work_hand=terminations.CableWorkHand(
            robot_name="right_robot",
            body_name=END_EFFECTOR_BODY_NAME,
            tcp_offset=GRASP_SITE_OFFSET,
        ),
    )


def _scene(fixture_yaw, xy):
    # The RSS goal region (``goals.bind._region``) adds the fixture origin to the
    # declared bounds; it does not rotate the box. The two cases cover both
    # a false negative and a false positive under a rotated box.
    body_q = wp.array(
        [[*xy, 0.02, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transform,
        device="cpu",
    )
    state = SimpleNamespace(
        body_q=body_q,
        body_qd=wp.zeros(1, dtype=wp.spatial_vector, device="cpu"),
    )
    fixture = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros((1, 3)),
            root_quat_w=torch.tensor(
                [[0.0, 0.0, math.sin(fixture_yaw / 2), math.cos(fixture_yaw / 2)]]
            ),
        )
    )
    robot = SimpleNamespace(
        find_bodies=lambda _: ([0], [END_EFFECTOR_BODY_NAME]),
        data=SimpleNamespace(
            body_link_pos_w=torch.ones((1, 1, 3)),
            body_link_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"peg": fixture, "port": fixture, "right_robot": robot},
        sim=SimpleNamespace(
            physics_manager=SimpleNamespace(
                get_state_0=lambda: state,
                cable_body_state=lambda: (torch.tensor([0]), None),
            )
        ),
        external_policy_termination_requested=lambda: (True,),
        episode_length_buf=torch.tensor([12999]),
        max_episode_length=13000,
    )
    goal = terminations.TerminatedCableGoal(
        seat_regions=((-0.045, -0.045, 0.045, 0.045),),
        seat_min_fractions=(1.0,),
        port_region=(-0.045, -0.045, 0.045, 0.045),
        port_min_fraction=1.0,
        min_tcp_distance=0.05,
        max_mean_speed=0.05,
    )
    return env, goal


@pytest.mark.parametrize(
    "region,inside,expected",
    [
        ("seat", 2, False),
        ("seat", 3, True),
        ("port", 3, False),
        ("port", 4, True),
    ],
)
def test_medium_fraction_comparisons_preserve_native_float64(
    monkeypatch, region, inside, expected
):
    env, goal = _scene(0.0, (0.0, 0.0))
    points = torch.zeros((1, 71, 3))
    points[..., 2] = 0.02
    points[:, :-inside, 0] = 0.1
    monkeypatch.setattr(terminations, "capsule_centerline", lambda *args: points)
    goal.seat_min_fractions = (0.028169014085 if region == "seat" else 0.0,)
    goal.port_min_fraction = 0.042253521127 if region == "port" else 0.0
    assert _score(env, goal).tolist() == [expected]
