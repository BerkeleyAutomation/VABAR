"""Declarative medium placement and two-distractor contracts."""

import pytest

from ib_support import AUTOLAB_ROOT, load_scene

FAMILIES = ("wrench", "scissors", "plier", "screwdriver")


def scene(family):
    return load_scene(f"tool_hanging/{family}_medium.yaml")


@pytest.mark.parametrize("family", FAMILIES)
def test_medium_uses_arena_tool_randomization_and_fixed_fixtures(family):
    config = scene(family)
    task = config["task"]["subtasks"][0]["params"]
    relations = {(r["subject"], r["kind"]): r for r in config["relations"]}
    assert config["background"]["registry_name"] == "vabar__yam_workcell_table"
    for tool in task["tools"]:
        assert relations[tool, "on"]["reference"] == "table"
        assert "z" not in relations[tool, "at_position"]["params"]
        randomization = relations[tool, "random_around_solution"]["params"]
        assert randomization == pytest.approx(
            dict(x_half_m=0.05, y_half_m=0.05, yaw_half_rad=0.08726646259971647)
        )
    for fixture in task["fixtures"]:
        assert (fixture, "is_anchor") in relations
        assert (fixture, "random_around_solution") not in relations
    assert task["episode_length_s"] == pytest.approx(22000 / 60)
    assert len(task["goals"]) == 2
    assert all(
        g["contact_force_n"] == 0.2 and g["speed_m_s"] == 0.15 for g in task["goals"]
    )
    if family == "screwdriver":
        objects = {o["id"]: o for o in config["objects"]}
        assert objects["rack"]["registry_name"] == "vabar__screwdriver_box"


@pytest.mark.parametrize("family", FAMILIES)
def test_medium_has_two_ordinary_distractors(family):
    config = scene(family)
    assert not config.get("object_sets")
    distractors = [o for o in config["objects"] if o["id"].startswith("distractor_")]
    assert len(distractors) == 2
    for obj in distractors:
        assert obj["registry_name"].startswith("vabar__")
        assert obj["params"]["instance_name"] == obj["id"]
        assert any(
            r["kind"] == "on" and r["subject"] == obj["id"] for r in config["relations"]
        )


def test_medium_scenes_validate_with_arena():
    pytest.importorskip("isaaclab_arena")
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    register_components()
    for family in FAMILIES:
        ArenaEnvGraphSpec(**scene(family))


@pytest.mark.parametrize("level", ("easy", "medium"))
def test_screwdriver_box_matches_stored_rss_mesh(level):
    """The stored STL, not its generator script's comments, is the RSS asset.

    AUTOLab's screwdrivers_level1 and screwdrivers_level2 both stand this one
    SCREWDRIVER_BOX record, so easy and medium share its geometry.
    """
    import numpy as np
    import trimesh
    from isaaclab_arena_vabar.assets.registrations.industrial_workcell_fixtures import (
        _ASSET_ENTRY_POINTS,
    )
    from pxr import Usd, UsdGeom, UsdPhysics

    source = (
        AUTOLAB_ROOT
        / "tasks/tool_hanging/assets/screwdriver_holders/screwdriver_box.stl"
    )
    if not source.exists():
        pytest.skip("pinned RSS task assets are not materialized")
    mesh = trimesh.load(source, force="mesh")
    # The mesh also repeats two coplanar triangles on the floor's top face.
    parts = [p for p in mesh.split(only_watertight=False) if np.all(p.extents > 1e-8)]
    assert len(parts) == 11
    config = load_scene(f"tool_hanging/screwdriver_{level}.yaml")
    obj = next(o for o in config["objects"] if o["id"] == "rack")
    asset = _ASSET_ENTRY_POINTS[obj["registry_name"]](**obj["params"])
    stage = Usd.Stage.Open(asset.usd_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == len(parts)
    for prim in colliders:
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        bounds = np.array([extent.GetMin(), extent.GetMax()])
        assert any(np.allclose(bounds, p.bounds, atol=2e-8, rtol=0) for p in parts)


@pytest.mark.parametrize("family", ("scissors", "wrench"))
def test_medium_hooks_match_rss_level2_hooks(family):
    # AUTOLab's level-2 hooks hang flat at +/-100 mm from the board centre; the
    # hook mesh's bounds centre shifts each pin site by 2.49 mm along y.
    hooks = [
        obj["params"]["initial_pose"]
        for obj in scene(family)["objects"]
        if obj["registry_name"] == "vabar__wrench_hook_2"
    ]
    assert len(hooks) == 2
    for pose, y in zip(hooks, (0.1, -0.1), strict=True):
        assert pose["position_xyz"] == pytest.approx([0.88, y - 0.002491525, 1.128515762])
    assert [pose["rotation_xyzw"] for pose in hooks] == [[0.0, 0.0, 1.0, 0.0]] * 2
