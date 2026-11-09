# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared spawn spreads, applied on top of Arena's pooled placement."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("isaaclab_arena")

BULKHEAD = SimpleNamespace(
    x_half_m=0.004, y_half_m=0.004, z_half_m=0.0, roll_half_rad=0.0, pitch_half_rad=0.0, yaw_half_rad=np.radians(3.0)
)
BULKHEAD_BASE = tuple(Rotation.from_euler("xz", [90.0, 90.0], degrees=True).as_quat())


def _draws(seed: int, count: int = 200):
    from isaaclab_arena_vabar.environments.random_around_solution import perturbed_pose

    rng = np.random.default_rng((seed, 0, 0))
    return [perturbed_pose((0.44, 0.0148, 0.8233), BULKHEAD_BASE, BULKHEAD, rng) for _ in range(count)]


def test_spread_stays_within_the_declared_half_ranges_and_turns_about_vertical():
    draws = _draws(45)
    xy = np.array([p[:2] for p, _ in draws]) - (0.44, 0.0148)
    assert np.abs(xy).max() <= 0.004 + 1e-9
    assert np.abs(xy).max() > 0.003
    assert {p[2] for p, _ in draws} == {0.8233}
    # The bore axis (local x) stays horizontal and turns by at most the yaw half-range.
    axes = np.array([Rotation.from_quat(q).apply([1.0, 0.0, 0.0]) for _, q in draws])
    assert np.abs(axes[:, 2]).max() < 1e-9
    yaw = np.degrees(np.arctan2(axes[:, 1], axes[:, 0]))
    assert np.abs(yaw - 90.0).max() <= 3.0 + 1e-6
    assert np.abs(yaw - 90.0).max() > 2.0


def test_placement_seed_alone_decides_the_draw():
    assert _draws(45, 3) == _draws(45, 3)
    assert _draws(45, 3) != _draws(46, 3)


def _marker(**half):
    from isaaclab_arena.relations.relations import RandomAroundSolution

    return RandomAroundSolution(**half)


def test_every_marked_graph_gets_the_term_before_its_family_callbacks(monkeypatch):
    import copy

    from isaaclab_arena_vabar.environments import random_around_solution
    from isaaclab_arena_vabar.environments.usbc_insertion import environment as module

    import isaaclab_arena_vabar.physics as physics

    monkeypatch.setattr(physics, "install_benchmark_physics", lambda *a, **k: None)
    placer = SimpleNamespace(placement_seed=None)
    plain = SimpleNamespace(get_relations=lambda: [])
    marked = SimpleNamespace(get_relations=lambda: [_marker(yaw_half_rad=0.1)])
    light = object()  # no relations at all

    unmarked_env = SimpleNamespace(
        scene=SimpleNamespace(assets={"a": plain, "light": light}), placer_params=placer, env_cfg_callback=None
    )
    random_around_solution.install(unmarked_env)
    assert unmarked_env.env_cfg_callback is None

    arena_env = SimpleNamespace(
        scene=SimpleNamespace(assets={"a": plain, "b": marked, "light": light}),
        placer_params=placer,
        env_cfg_callback=None,
    )
    random_around_solution.install(arena_env)
    physics.append_env_cfg_callback(arena_env, partial(module._configure_env_cfg, cable_links=24))
    events = SimpleNamespace(placement_reset=object())
    arena_env.env_cfg_callback(SimpleNamespace(events=events, sim=SimpleNamespace(physics=None)))

    assert list(vars(events)) == ["placement_reset", random_around_solution.EVENT_NAME, "reset_usbc_cables"]
    handle = getattr(events, random_around_solution.EVENT_NAME).params["placement_seed"]
    # Shared through config deep copies, and read at reset, after the builder sets the seed.
    assert copy.deepcopy(handle) is handle
    assert handle.seed == 0
    placer.placement_seed = 47
    assert handle.seed == 47


IDENTITY = (0.0, 0.0, 0.0, 1.0)
BAR = ((-0.1, -0.02, 0.0), (0.1, 0.02, 0.02))
"""A 20 x 4 cm bar, 2 cm tall."""


def test_draws_never_cut_into_a_neighbour_and_stay_in_range():
    from isaaclab_arena_vabar.environments.random_around_solution import Solid, spread_layout

    poses = {"a": ((0.0, 0.0, 0.0), IDENTITY), "b": ((0.0, 0.06, 0.0), IDENTITY)}
    boxes = {"a": BAR, "b": BAR}
    markers = {"a": _marker(x_half_m=0.05, y_half_m=0.05, yaw_half_rad=np.radians(5.0))}
    blocked = 0
    for seed in range(300):
        drawn = spread_layout(poses, boxes, markers, np.random.default_rng((seed, 0, 0)))
        (x, y, z), q = drawn["a"]
        assert abs(x) <= 0.05 + 1e-12 and abs(y) <= 0.05 + 1e-12 and abs(z) < 1e-12
        assert not Solid.of(BAR, *drawn["a"]).collides(Solid.of(BAR, *poses["b"]))
        blocked += y > 0.02
    # Without the check about a third of these draws would reach into b.
    assert blocked == 0


def test_stacked_bodies_do_not_veto_and_an_existing_overlap_may_not_grow():
    from isaaclab_arena_vabar.environments.random_around_solution import Solid, spread_layout

    table = ((-0.5, -0.5, -0.02), (0.5, 0.5, 0.0))
    # "fixture" overlaps "a" in the solved layout, as an authored contact would.
    poses = {
        "a": ((0.0, 0.0, 0.0), IDENTITY),
        "table": ((0.0, 0.0, 0.0), IDENTITY),
        "fixture": ((0.09, 0.0, 0.0), IDENTITY),
    }
    boxes = {"a": BAR, "table": table, "fixture": BAR}
    markers = {"a": _marker(x_half_m=0.01, y_half_m=0.01)}
    fixture = Solid.of(BAR, *poses["fixture"])
    solved = Solid.of(BAR, *poses["a"]).overlap(fixture)
    for seed in range(50):
        drawn = spread_layout(poses, boxes, markers, np.random.default_rng(seed))
        # The table under it never refuses a draw; the fixture refuses only a deeper overlap.
        assert drawn["a"] != poses["a"]
        assert Solid.of(BAR, *drawn["a"]).overlap(fixture) <= solved


def test_a_layout_with_no_clear_draw_keeps_its_solved_pose():
    from isaaclab_arena_vabar.environments.random_around_solution import spread_layout

    # Four bars 0.2 mm off each side of a: almost every draw cuts into one,
    # so none of the 64 draws clears.
    poses = {"a": ((0.0, 0.0, 0.0), IDENTITY)}
    boxes = {"a": BAR}
    for name, x, y in (("n", 0.0, 0.0402), ("s", 0.0, -0.0402), ("e", 0.2002, 0.0), ("w", -0.2002, 0.0)):
        poses[name] = ((x, y, 0.0), IDENTITY)
        boxes[name] = BAR
    markers = {"a": _marker(x_half_m=0.05, y_half_m=0.05)}
    drawn = spread_layout(poses, boxes, markers, np.random.default_rng(0))
    assert drawn["a"] == poses["a"]


def test_one_seed_draws_one_layout():
    from isaaclab_arena_vabar.environments.random_around_solution import spread_layout

    poses = {"a": ((0.0, 0.0, 0.0), IDENTITY), "b": ((0.0, 0.06, 0.0), IDENTITY)}
    boxes = {"a": BAR, "b": BAR}
    markers = {name: _marker(x_half_m=0.05, y_half_m=0.05, yaw_half_rad=0.1) for name in poses}

    def draw(seed):
        return spread_layout(poses, boxes, markers, np.random.default_rng((seed, 0, 0)))

    assert draw(42) == draw(42)
    assert draw(42) != draw(43)


def test_a_yaw_draw_turns_the_body_in_place_not_about_a_distant_origin():
    from isaaclab_arena_vabar.environments.random_around_solution import Solid, spread_layout

    # The ring wrench's body sits 22 cm off its origin.
    offset_bar = ((0.01, 0.2, 0.0), (0.25, 0.24, 0.01))
    poses = {"w": ((0.7, -0.1, 0.75), IDENTITY)}
    boxes = {"w": offset_bar}
    markers = {"w": _marker(yaw_half_rad=np.pi)}
    centre = Solid.of(offset_bar, *poses["w"]).footprint.centroid
    for seed in range(50):
        drawn = spread_layout(poses, boxes, markers, np.random.default_rng(seed))
        assert Solid.of(offset_bar, *drawn["w"]).footprint.centroid.distance(centre) < 1e-6


def test_later_draws_never_inherit_an_overlap_an_earlier_draw_made():
    from isaaclab_arena_vabar.environments.random_around_solution import Solid, spread_layout

    names = ("a", "b", "c")
    poses = {name: ((0.0, 0.05 * i, 0.0), IDENTITY) for i, name in enumerate(names)}
    boxes = {name: BAR for name in names}
    markers = {name: _marker(x_half_m=0.05, y_half_m=0.05, yaw_half_rad=0.3) for name in names}
    for seed in range(200):
        drawn = spread_layout(poses, boxes, markers, np.random.default_rng(seed))
        solids = [Solid.of(BAR, *drawn[name]) for name in names]
        assert not any(solids[i].collides(solids[j]) for i in range(3) for j in range(i + 1, 3))
