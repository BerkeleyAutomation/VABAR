"""Replay the observed artist wrench through AUTOLab v1.0's CAD registration."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from isaaclab_arena_gap.task_sources import autolab_root
from isaaclab_arena_gap.tool_hanging_asset_profile import artist_wrench_profiles

GRAPH = autolab_root() / "tasks/tool_hanging/graphs/gap_perception"


def _script(name):
    pytest.importorskip("gap")
    if not GRAPH.is_dir():
        pytest.skip("AUTOLab v1.0 sources are not materialized; run ./setup.sh")
    spec = importlib.util.spec_from_file_location(name, GRAPH / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_artist_wrench_ring_is_registered_on_closed_end():
    pytest.importorskip("pxr")
    pytest.importorskip("trimesh")
    # Seed42 overhead cloud, mass100 eval before grasp. Known ring from the
    # authored mesh and measured object pose: (0.7665, 0.001), not (0.568, 0.002).
    cloud = np.load(Path(__file__).parent / "fixtures/artist_wrench_cloud.npz")["positions"]
    workflow = json.loads((GRAPH / "workflow.json").read_text())
    profiles = workflow["subgraphs"]["target_sg"]["nodes"]["perceive"]["inputs"]["feature_profiles"]
    original = json.dumps(profiles)
    profile = next(p for p in artist_wrench_profiles(profiles) if p["kind"] == "wrench")
    rotation, origin, rmse = _script("perceive_target")._register_planar_model(
        cloud, profile["model_path"]
    )
    ring = rotation @ np.asarray(profile["feature_landmarks_xy"][0]) + origin
    assert np.linalg.norm(ring - [0.7665, 0.001]) < 0.006
    assert rmse < 0.001
    assert json.dumps(profiles) == original
    feature = np.eye(4)
    feature[:2, :2] = rotation
    feature[:2, 3] = ring
    feature[2, 3] = np.mean(cloud[:, 2])
    held = _script("register_held")
    registration = profile["registration_profile"]
    initial = held._initial_wrench_pose({"points": cloud}, feature, registration)
    fitted, error, inliers = held._register_wrench_cad(cloud, initial, registration)
    assert fitted is not None
    actual = held._feature_from_wrench_pose(fitted, registration)
    assert np.linalg.norm(actual[:2, 3] - [0.7665, 0.001]) < 0.01
    assert error < 0.003
    assert inliers > 100


def _patched_workflow():
    from isaaclab_arena_gap.rss_adapter.runner import adapted_graph

    if not GRAPH.is_dir():
        pytest.skip("AUTOLab v1.0 sources are not materialized; run ./setup.sh")
    staged = Path(adapted_graph(GRAPH, "rss_tool_hanging_artist_wrench"))
    return json.loads((staged / "workflow.json").read_text())


def _wrench_fixture_profile(workflow):
    profiles = workflow["subgraphs"]["fixture_sg"]["nodes"]["perceive"]["inputs"]["fixture_profiles"]
    return next(p for p in profiles if p["source_kind"] == "wrench")


def test_artist_patch_keeps_autolabs_mating_offsets():
    adapted = _wrench_fixture_profile(_patched_workflow())
    rss_profile = _wrench_fixture_profile(json.loads((GRAPH / "workflow.json").read_text()))
    assert adapted["mating_profile"] == rss_profile["mating_profile"]


@pytest.mark.parametrize("sag", [0.003, 0.008])
def test_artist_mating_profile_crosses_and_seats_inside_ring_under_load_sag(sag):
    # The loaded YAM arm ends 3-8 mm below its Cartesian target, as it does in the RSS run; AUTOLab's
    # vertical offsets compensate. Commanded at the axis, the ring crossed below the shank and
    # contact_seat stalled 4-9 mm short.
    profile = _wrench_fixture_profile(_patched_workflow())
    tip = {"x": 0.82, "y": 0.06, "z": 1.16}
    fixture = {
        "pose": {"position": tip}, "usable_length": 0.06,
        "mating_profile": profile["mating_profile"],
    }
    poses = {
        name: {"position": {"x": x, "y": 0.07, "z": 1.18},
               "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        for name, x in (("approach_pose", 0.79), ("engaged_pose", 0.84), ("mate_pose", 0.84))
    }
    _script("compute_mate")._move_loop_poses_to_shaft_interior(
        poses, {"position": {"x": 0, "y": 0, "z": 0}}, fixture,
        {"x": -1, "y": 0, "z": 0}, 0.0102, 0.00249,
    )
    for name in ("approach_pose", "mate_pose"):
        reached = dict(poses[name]["position"], z=poses[name]["position"]["z"] - sag)
        radial_error = np.linalg.norm([reached[axis] - tip[axis] for axis in "yz"])
        assert radial_error < 0.0102 - 0.00249, name


def test_hook_cad_recovers_centerline_from_recorded_oblique_depth():
    # The hook CAD is Arena's USD: this runs in the root and AUTOLab environments, which carry
    # usd-core, and skips in GaP's, whose upstream lock does not.
    pytest.importorskip("pxr")
    pytest.importorskip("trimesh")
    from isaaclab_arena_gap.tool_hanging_asset_profile import refine_artist_hook

    cloud = np.load(Path(__file__).parent / "fixtures/artist_hook_cloud.npz")["positions"]
    # The RSS heuristic returned 1.161609; authored/measured shank center
    # was 1.154716. Fit the recorded camera cloud without its simulator pose.
    result = {
        "fixture_cloud": {"points": cloud},
        "fixture_axis": {"x": -1.0, "y": 0.0, "z": 0.0},
        "hook_tip": {"x": 0.820106566, "y": 0.060516836, "z": 1.161609029},
        "fixture_feature": {"pose": {"position": {}}, "radius_outer": 0.002861517},
    }
    refined = refine_artist_hook(result)
    tip = np.array([refined["hook_tip"][key] for key in "xyz"])
    assert np.linalg.norm(tip - [0.8189076, 0.06, 1.154716]) < 0.0015
    assert result["hook_tip"]["z"] == 1.161609029
    assert refined["fixture_feature"]["pose"]["position"] == refined["hook_tip"]
