"""Checks for the seeded Easy cable-routing layout distribution."""

import json

import pytest

from ib_support import AUTOLAB_ROOT, load_scene
from isaaclab_arena_vabar.environments.cable_routing.distribution import (
    sample_cable_routing_layout,
)

SCENE = "cable_routing/easy.yaml"


def test_regression_seed_reproduces_berkeley_sample_10() -> None:
    spec = load_scene(SCENE)
    layout = sample_cable_routing_layout(
        spec["distribution"], spec["distribution"]["default_seed"]
    )

    assert (
        layout.bow_fraction,
        layout.shift_fraction,
        layout.yaw_fraction,
        layout.wave_fraction,
    ) == pytest.approx(
        (
            0.04467945582986155,
            0.12236874520167551,
            0.20747168831416163,
            -0.1252732721765486,
        )
    )
    assert tuple(
        value for xy in layout.peg_positions_xy for value in xy
    ) == pytest.approx(
        (
            -0.06014228460135129,
            -0.2710538148325847,
            -0.057914071329495886,
            0.0027501271795930538,
        )
    )
    assert layout.port_pose_xyyaw == pytest.approx(
        (-0.010872356158660557, 0.07932383933181138, 3.161636600701201)
    )


def test_all_layouts_match_latest_rss_manifest() -> None:
    manifest_path = (
        AUTOLAB_ROOT / "benchmark/cable_tasks/dataset/manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("RSS cable dataset is not materialized; run ./setup.sh")
    manifest = json.loads(manifest_path.read_text())
    case = next(
        case
        for case in manifest["cases"]
        if case["name"] == "cablerouting-easy-routecable"
    )
    distribution = load_scene(SCENE)["distribution"]
    assert len(case["scenes"]) == 100
    for scene in case["scenes"]:
        episode = scene["episode"]
        expected = dict(episode["overrides"])
        layout = sample_cable_routing_layout(distribution, episode["seed"])
        assert (
            layout.bow_fraction,
            layout.shift_fraction,
            layout.yaw_fraction,
            layout.wave_fraction,
        ) == pytest.approx(
            tuple(expected[f"cable_{key}"] for key in ("bow", "shift", "yaw", "wave"))
        ), scene["name"]
        poses = expected["static_poses"]
        for index, position in enumerate(layout.peg_positions_xy):
            assert position == pytest.approx(poses[f"peg{index}"][:2], abs=1e-12), (
                scene["name"]
            )
        assert layout.port_pose_xyyaw == pytest.approx(
            (poses["port"][0], poses["port"][1], poses["port"][3]), abs=1e-12
        ), scene["name"]
