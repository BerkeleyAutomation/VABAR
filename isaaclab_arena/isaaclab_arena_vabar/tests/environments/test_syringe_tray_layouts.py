"""Cluttered and both-syringe scenes use AUTOLab's own tray trials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from isaaclab_arena_vabar.environments.syringe_disposal import tray_layouts


def test_clutter_layouts_are_six_syringes_lying_across_the_tray():
    layouts = tray_layouts.load_tray_layouts("clutter_layouts.json")
    assert len(layouts) == 7
    for layout in layouts:
        assert len(layout.syringes) == 6
        # The YAML's rotate_around_solution keeps the tray at AUTOLab's heading.
        assert layout.tray_yaw is None
        for x, y, yaw in layout.syringes:
            # Inside the 0.2 x 0.4 m tray, each barrel across it within AUTOLab's
            # YAW_JITTER_DEG (35 deg).
            assert abs(x) < 0.1 and abs(y) < 0.2
            assert abs(abs(yaw) - math.pi / 2) <= math.radians(35)


def test_pair_layouts_are_two_barrels_across_the_tray_90_mm_apart():
    layouts = tray_layouts.load_tray_layouts("pair_layouts.json")
    assert len(layouts) == 30
    for layout in layouts:
        (x0, y0, yaw0), (x1, y1, yaw1) = layout.syringes
        # AUTOLab's spacing: an open hand's fingers sweep ~58 mm, so 90 mm leaves
        # the neighbour clear on the descent.
        assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(0.09, abs=1e-3)
        assert yaw0 == pytest.approx(math.pi / 2, abs=1e-3)
        assert yaw1 == pytest.approx(math.pi / 2, abs=1e-3)
        # make_tray_pair_trials.py: the tray at 140.4 deg, yawed uniform(+-25 deg).
        assert abs(layout.tray_yaw - 2.4504) <= math.radians(25) + 1e-3


@dataclass(frozen=True)
class _Asset:
    name: str


def _solved(names, tray_yaw):
    assets = {n: _Asset(n) for n in (tray_layouts.TRAY_NAME, *names)}
    tray = assets[tray_layouts.TRAY_NAME]
    layout = SimpleNamespace(
        positions={tray: (0.5, -0.2, 0.79), **{assets[n]: (0.0, 0.0, 0.80) for n in names}},
        orientations={tray: tray_yaw},
    )
    return assets, layout


def test_apply_layout_composes_with_the_solved_tray_pose_and_keeps_z():
    names = tuple(f"syringe_{i}" for i in range(6))
    assets, layout = _solved(names, math.pi / 2)
    trial = tray_layouts.TrayLayout(syringes=tuple((0.01 * i, 0.05 * i, 0.3) for i in range(6)))
    tray_layouts.apply_layout(layout, names, trial)
    for i, n in enumerate(names):
        x, y, z = layout.positions[assets[n]]
        # Tray yaw +90 deg maps tray (x, y) to world (-y, x).
        assert (x, y) == pytest.approx((0.5 - 0.05 * i, -0.2 + 0.01 * i))
        assert z == pytest.approx(0.80)
        assert layout.orientations[assets[n]] == pytest.approx(math.pi / 2 + 0.3)


def test_a_trial_with_its_own_tray_heading_turns_the_tray_and_its_syringes():
    names = ("syringe_0", "syringe_1")
    assets, layout = _solved(names, 2.4504)
    trial = tray_layouts.TrayLayout(syringes=((0.0, 0.045, math.pi / 2), (0.0, -0.045, math.pi / 2)), tray_yaw=math.pi)
    tray_layouts.apply_layout(layout, names, trial)
    tray = assets[tray_layouts.TRAY_NAME]
    assert layout.orientations[tray] == pytest.approx(math.pi)
    assert layout.positions[tray] == (0.5, -0.2, 0.79)
    # Tray yaw 180 deg maps tray (x, y) to world (-x, -y).
    assert layout.positions[assets["syringe_0"]][:2] == pytest.approx((0.5, -0.245))
    assert layout.positions[assets["syringe_1"]][:2] == pytest.approx((0.5, -0.155))
    for n in names:
        assert abs(layout.orientations[assets[n]]) == pytest.approx(math.pi / 2)


def test_a_layout_must_match_the_scene_syringe_count():
    names = ("syringe_0",)
    _, layout = _solved(names, 0.0)
    with pytest.raises(ValueError, match="syringe count"):
        tray_layouts.apply_layout(layout, names, tray_layouts.load_tray_layouts("pair_layouts.json")[0])


def test_each_placement_seed_starts_from_its_own_trial():
    # Seeds 42-51 are the benchmark's seed sweep: one trial each, none shared.
    trials = [tray_layouts.trial_index(seed, 0, 30) for seed in range(42, 52)]
    assert trials == [tray_layouts.trial_index(seed, 0, 30) for seed in range(42, 52)]
    assert len(set(trials)) == 10
    # Clutter has seven trials, so a sweep covers all of them.
    assert {tray_layouts.trial_index(seed, 0, 7) for seed in range(42, 52)} == set(range(7))


def test_vector_envs_start_from_the_trials_of_consecutive_scalar_seeds():
    assert [tray_layouts.trial_index(42, env_id, 30) for env_id in range(3)] == [
        tray_layouts.trial_index(seed, 0, 30) for seed in (42, 43, 44)
    ]


class _Pool:
    def __init__(self, solved):
        self._solved = solved

    def layouts_per_env(self):
        return self._solved


def _apply_with_seed(file_name, names, num_envs, seed):
    """Run ``apply_tray_layouts`` over ``num_envs`` envs of two solved candidates each."""
    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.relations.placement_events import PlacementPoolHandle, solve_and_place_objects

    solved = [[_solved(names, 2.4504) for _ in range(2)] for _ in range(num_envs)]
    pool = _Pool([[layout for _, layout in per_env] for per_env in solved])
    event = EventTermCfg(func=solve_and_place_objects, mode="reset", params={"placement_pool": PlacementPoolHandle(pool)})
    placer_params = SimpleNamespace(placement_seed=None)
    # The builder writes the seed after the environment is built, before this callback runs.
    placer_params.placement_seed = seed
    tray_layouts.apply_tray_layouts(
        SimpleNamespace(events=SimpleNamespace(placement_reset=event)),
        file_name=file_name,
        syringe_names=names,
        placer_params=placer_params,
    )
    return solved


def _syringe_xy(assets, layout, names):
    return tuple(tuple(round(v, 9) for v in layout.positions[assets[n]][:2]) for n in names)


def test_apply_tray_layouts_lays_each_env_out_from_its_seeded_trial():
    names = ("syringe_0", "syringe_1")
    trials = tray_layouts.load_tray_layouts("pair_layouts.json")
    solved = _apply_with_seed("pair_layouts.json", names, num_envs=2, seed=43)
    for env_id, per_env in enumerate(solved):
        trial = trials[tray_layouts.trial_index(43, env_id, len(trials))]
        expected_names, expected = _solved(names, 2.4504)
        tray_layouts.apply_layout(expected, names, trial)
        for assets, layout in per_env:
            assert layout.orientations[assets[tray_layouts.TRAY_NAME]] == pytest.approx(trial.tray_yaw)
            assert _syringe_xy(assets, layout, names) == _syringe_xy(expected_names, expected, names)


def test_the_same_tray_pose_gives_different_trials_for_different_seeds():
    # The trial used to be hashed from the solved tray X/Y, so seeds that solved one
    # tray pose shared a trial. Here every seed solves the same tray pose.
    names = ("syringe_0", "syringe_1")
    starts = {}
    for seed in range(42, 52):
        ((assets, layout), _), = _apply_with_seed("pair_layouts.json", names, num_envs=1, seed=seed)
        starts[seed] = _syringe_xy(assets, layout, names)
    assert len(set(starts.values())) == 10
    ((assets, layout), _), = _apply_with_seed("pair_layouts.json", names, num_envs=1, seed=42)
    assert _syringe_xy(assets, layout, names) == starts[42]


def test_unknown_layout_files_are_refused():
    with pytest.raises(ValueError, match="unknown tray layout file"):
        tray_layouts.load_tray_layouts("../both.yaml")


def test_both_and_cluttered_opt_in():
    from isaaclab_arena_vabar.environments.syringe_disposal.registration import ENVIRONMENTS

    assert {d.environment_name: d.autolab_tray_layouts for d in ENVIRONMENTS} == {
        "vabar_syringe_disposal__single": None,
        "vabar_syringe_disposal__both": "pair_layouts.json",
        "vabar_syringe_disposal__designated": None,
        "vabar_syringe_disposal__cluttered": "clutter_layouts.json",
    }
