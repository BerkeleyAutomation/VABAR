"""Fixed-start scenes replay each env's first placement layout on every reset."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab_arena")


class _Pool:
    """Arena's pool surface: each draw is a new layout, so a re-draw is visible."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.objects = []
        self.draws: list[list[int]] = []

    def sample_for_envs(self, env_ids):
        self.draws.append(list(env_ids))
        return {
            env_id: SimpleNamespace(name=f"draw{len(self.draws)}-env{env_id}", success=True, positions={})
            for env_id in env_ids
        }


def _placement_event(num_envs=2):
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.relations.placement_events import PlacementPoolHandle, solve_and_place_objects

    handle = PlacementPoolHandle(_Pool(num_envs))
    return EventTermCfg(func=solve_and_place_objects, mode="reset", params={"placement_pool": handle})


def _env(num_envs=2):
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)))


@pytest.fixture
def writes(monkeypatch):
    from isaaclab_arena.relations import placement_events

    written = []
    monkeypatch.setattr(
        placement_events,
        "write_layout_to_sim",
        lambda env, env_id, layout, anchors, base: written.append((env_id, layout.name)),
    )
    return written


def test_every_reset_writes_the_layout_each_env_drew_first(writes):
    from isaaclab_arena_vabar.fixed_layouts import replay_first_layouts

    event = _placement_event()
    replay_first_layouts(SimpleNamespace(events=SimpleNamespace(placement_reset=event)))
    handle = event.params["placement_pool"]
    env = _env()
    term = event.func(event, env)

    term(env, torch.tensor([0, 1]), handle)
    first = list(writes)
    assert first == [(0, "draw1-env0"), (1, "draw1-env1")]
    for _ in range(3):
        writes.clear()
        term(env, torch.tensor([0, 1]), handle)
        assert writes == first
    # A partial reset replays only the resetting env's own first layout.
    writes.clear()
    term(env, torch.tensor([1]), handle)
    assert writes == [(1, "draw1-env1")]
    assert handle.pool.draws == [[0, 1]]
    # Arena's reset-results reader, which random_around_solution uses, sees the replayed layout.
    assert {k: v.name for k, v in handle.last_results.items()} == {1: "draw1-env1"}


def test_an_env_first_reset_later_draws_once_for_itself(writes):
    from isaaclab_arena_vabar.fixed_layouts import ReplayFirstLayouts

    event = _placement_event()
    handle = event.params["placement_pool"]
    env = _env()
    term = ReplayFirstLayouts(event, env)
    term(env, torch.tensor([1]), handle)
    term(env, torch.tensor([0, 1]), handle)
    term(env, torch.tensor([0, 1]), handle)
    assert handle.pool.draws == [[1], [0]]
    assert writes[-2:] == [(0, "draw2-env0"), (1, "draw1-env1")]


def test_replay_override_refuses_an_unknown_placement_reset():
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena_vabar.fixed_layouts import ReplayFirstLayouts, replay_first_layouts

    event = _placement_event()
    cfg = SimpleNamespace(events=SimpleNamespace(placement_reset=event))
    replay_first_layouts(cfg)
    replay_first_layouts(cfg)
    assert event.func is ReplayFirstLayouts

    other = EventTermCfg(func=lambda env, env_ids: None, mode="reset", params={})
    with pytest.raises(AssertionError, match="not Arena's pooled placement reset"):
        replay_first_layouts(SimpleNamespace(events=SimpleNamespace(placement_reset=other)))


def test_spread_is_redrawn_around_the_replayed_center(writes):
    """USB-C's random_around_solution composes: fixed center, new spread each reset."""
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.relations.placement_events import PlacementPoolHandle
    from isaaclab_arena.relations.relations import RandomAroundSolution
    from isaaclab_arena_vabar.environments.random_around_solution import (
        PlacementSeed,
        randomize_around_solution,
    )
    from isaaclab_arena_vabar.fixed_layouts import ReplayFirstLayouts

    marker = RandomAroundSolution(x_half_m=0.004, y_half_m=0.004, yaw_half_rad=0.05)

    class _Port:
        name = "port"

        def __init__(self):
            self.poses = []

        def get_relations(self):
            return [marker]

        def get_bounding_box(self):
            return SimpleNamespace(
                min_point=torch.tensor([[-0.02, -0.02, 0.0]]), max_point=torch.tensor([[0.02, 0.02, 0.05]])
            )

        def write_layout_pose_to_sim(self, env, env_id, pose):
            self.poses.append(pose.position_xyz)

    port = _Port()
    center = (0.44, 0.0148, 0.8233)

    class _CenterPool(_Pool):
        def sample_for_envs(self, env_ids):
            self.draws.append(list(env_ids))
            # Each draw would land somewhere else; only the first may ever be used.
            offset = 0.1 * (len(self.draws) - 1)
            return {
                e: SimpleNamespace(
                    name="layout", success=True, positions={port: (center[0] + offset, *center[1:])}, orientations={}
                )
                for e in env_ids
            }

    handle = PlacementPoolHandle(_CenterPool(1))
    handle.pool.objects = [port]
    event = EventTermCfg(func=ReplayFirstLayouts, mode="reset", params={"placement_pool": handle})
    env = _env(1)
    env.unwrapped = env
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: event)
    term = ReplayFirstLayouts(event, env)
    seed = PlacementSeed(SimpleNamespace(placement_seed=45))
    for _ in range(3):
        term(env, torch.tensor([0]), handle)
        randomize_around_solution(env, torch.tensor([0]), seed)

    assert handle.pool.draws == [[0]]
    spread = np.array(port.poses) - center
    assert np.abs(spread[:, :2]).max() <= 0.004 + 1e-9
    assert len({tuple(p) for p in port.poses}) == 3
    assert not math.isclose(port.poses[0][0], port.poses[1][0])


def test_syringe_tray_trials_replay_the_first_layout():
    from isaaclab_arena_vabar.environments.syringe_disposal import tray_layouts
    from isaaclab_arena_vabar.fixed_layouts import ReplayFirstLayouts

    event = _placement_event()
    event.params["placement_pool"].pool.layouts_per_env = lambda: [[], []]
    cfg = SimpleNamespace(events=SimpleNamespace(placement_reset=event))
    tray_layouts.apply_tray_layouts(
        cfg,
        file_name="pair_layouts.json",
        syringe_names=("syringe_0", "syringe_1"),
        placer_params=SimpleNamespace(placement_seed=42),
    )
    assert event.func is ReplayFirstLayouts
