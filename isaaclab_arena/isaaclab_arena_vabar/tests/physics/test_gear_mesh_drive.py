"""The gear board's fixed root must follow Arena's reset placement into MuJoCo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

LIVE = os.environ.get("ISAAC_CAP_RUN_LIVE_TESTS") == "1"
LIVE_CHECK = Path(__file__).with_name("gear_mesh_drive_check.py")


@pytest.mark.parametrize("method", ["write_root_link_pose_to_sim_index", "write_root_link_pose_to_sim_mask",
                                    "write_root_com_pose_to_sim_index", "write_root_com_pose_to_sim_mask"])
@pytest.mark.parametrize("fixed_base", [True, False])
def test_root_pose_write_notifies_joint_properties(monkeypatch, method, fixed_base):
    """A fixed root's pose lives in joint_X_p, which MuJoCo re-reads only on JOINT_PROPERTIES."""
    from isaaclab_newton.assets import Articulation
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelFlags

    from isaaclab_arena_vabar.assets.fixed_base_articulation import PlacedFixedBaseArticulation

    writes = []
    monkeypatch.setattr(Articulation, method, lambda self, **kwargs: writes.append(kwargs))
    monkeypatch.setattr(PlacedFixedBaseArticulation, "is_fixed_base", property(lambda self: fixed_base))
    monkeypatch.setattr(NewtonManager, "_model_changes", set())

    board = object.__new__(PlacedFixedBaseArticulation)
    getattr(board, method)(root_pose="pose", env_ids=None)

    assert writes == [{"root_pose": "pose", "env_ids": None}]
    assert (ModelFlags.JOINT_PROPERTIES in NewtonManager._model_changes) is fixed_base


def test_gear_board_spawns_the_synced_articulation():
    pytest.importorskip("isaaclab_newton")
    from isaaclab_arena_vabar.assets.fixed_base_articulation import PlacedFixedBaseArticulation
    from isaaclab_arena_vabar.assets.gear_insertion import make_gear_mesh_board_20

    cfg = make_gear_mesh_board_20()._generate_articulation_cfg()

    assert cfg.class_type is PlacedFixedBaseArticulation


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="set ISAAC_CAP_RUN_LIVE_TESTS=1 to run the fresh-process gear drive rig")
# Seeds whose reset placement moves the board away from its construction pose,
# so the seated gear turns at its ratio only if the board's root follows the placement.
@pytest.mark.parametrize("seed", [42, 43, 48])
def test_seated_gear_turns_at_its_ratio(seed: int) -> None:
    subprocess_utils = pytest.importorskip("isaaclab_arena.tests.utils.subprocess")
    result = subprocess_utils.run_subprocess(
        [os.sys.executable, str(LIVE_CHECK), "--seed", str(seed)], timeout_sec=1800, capture_output=True
    )
    assert result is not None
    line = next((l for l in result.stdout.splitlines() if l.startswith("GEAR_MESH_DRIVE ")), None)
    assert line is not None, result.stderr
    report = json.loads(line.removeprefix("GEAR_MESH_DRIVE "))
    assert report["anchor_offset_mm"] < 0.01, report
    assert report["ratio"] > 0.9, report
