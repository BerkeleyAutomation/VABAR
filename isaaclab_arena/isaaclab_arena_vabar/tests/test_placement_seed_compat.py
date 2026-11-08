"""Placement seeds are spread before Arena samples candidates, until Arena decorrelates them."""

from __future__ import annotations

import sys

import pytest

from isaaclab_arena_vabar import placement_seed_compat as compat


def test_consecutive_seeds_spread_beyond_arena_candidate_windows() -> None:
    spread = [compat.spread_placement_seed(seed) for seed in range(40, 50)]
    assert spread == [compat.spread_placement_seed(seed) for seed in range(40, 50)]
    gaps = [abs(a - b) for i, a in enumerate(spread) for b in spread[i + 1 :]]
    assert min(gaps) > 100_000


def test_consecutive_placement_seeds_give_different_layouts(monkeypatch) -> None:
    if sys.version_info < (3, 12):
        pytest.skip("Arena requires Python 3.12")
    generator_module = pytest.importorskip("isaaclab_arena.relations.placement_candidate_generator")
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    generator_class = generator_module.PlacementCandidateGenerator
    monkeypatch.setattr(generator_class, "generate_candidates", generator_class.generate_candidates)
    monkeypatch.setattr(compat, "_installed", False)
    compat.install_placement_seed_compat()

    def place(seed: int) -> tuple[float, float, float]:
        desk = DummyObject(
            name="desk",
            bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
        )
        desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        desk.add_relation(IsAnchor())
        box = DummyObject(
            name="box",
            bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
        )
        neighbour = DummyObject(
            name="neighbour",
            bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.15, 0.15, 0.15)),
        )
        box.add_relation(On(desk, clearance_m=0.01))
        neighbour.add_relation(On(desk, clearance_m=0.01))
        neighbour.add_relation(NextTo(box, side=Side.POSITIVE_X, distance_m=0.05))
        params = ObjectPlacerParams(placement_seed=seed, solver_params=RelationSolverParams(max_iters=10))
        (result,) = ObjectPlacer(params=params).place(objects=[desk, box, neighbour])
        assert params.placement_seed == seed
        return result.positions[box]

    assert place(42) != place(43)
    assert place(42) == place(42)
