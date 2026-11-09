"""Native cable tier contracts without constructing a simulator."""

import json
import math
import re
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab")

from ib_support import AUTOLAB_ROOT
from isaaclab_arena_vabar.environments.cable_routing import environment as cable
from isaaclab_arena_vabar.embodiments.bimanual_yam import (
    IndustrialBimanualYamEmbodiment,
)


def _manifest(difficulty):
    # The AUTOLab v1.0 clone carries the declared source revisions in its history.
    source = AUTOLAB_ROOT
    declaration = (
        cable._EASY_SCENE_LAYOUT if difficulty == "easy" else cable._SCENE_LAYOUT
    )
    revision = declaration["source"]["revision"]
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "show",
            f"{revision}:benchmark_specs/cable_tasks/dataset/manifest.json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip(f"fetch the declared {difficulty} source {revision}")
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "difficulty,segments,guides,steps",
    [
        ("Easy", 50, 2, 13000),
        ("Medium", 70, 4, 24000),
    ],
)
def test_native_tier_scene_contract(
    stub_scene_module, difficulty, segments, guides, steps
):
    cfg_type = getattr(cable, f"CableRouting{difficulty}EnvironmentCfg")
    factory = getattr(cable, f"CableRouting{difficulty}Environment")()
    cfg = cfg_type(enable_cameras=True, layout_seed=10000)
    variant = factory._variant_for_cfg(cfg)
    assert variant.cable_num_segments == segments
    assert len(variant.peg_positions) == guides
    assert variant.physics.radius == pytest.approx(0.0035)
    assert variant.anchor_position is not None and variant.port_position is not None
    goal = variant.terminated_goal
    assert goal.min_tcp_distance == 0.05 and goal.max_mean_speed == 0.05
    assert goal.seat_min_fractions == (
        (0.039999999999,) * guides
        if difficulty == "Easy"
        else (0.028169014085,) * guides
    )
    assert goal.port_min_fraction == (
        0.059999999999 if difficulty == "Easy" else 0.042253521127
    )
    env = factory.build(cfg)
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    assert isinstance(env.embodiment, IndustrialBimanualYamEmbodiment)
    assert env.task.episode_length_s == steps / 60
    scene = env.task.get_scene_cfg()
    for index in range(guides):
        assert getattr(scene, f"peg_{index}") is not None
    for index in range(guides, 4):
        assert getattr(scene, f"peg_{index}") is None
    termination = env.task.get_termination_cfg()
    assert isinstance(termination, TaskTerminationCfg)
    assert termination.timeout_s == steps / 60
    assert termination.success[0].predicate_sequence[0].params["peg_names"] == tuple(
        f"peg_{index}" for index in range(guides)
    )
    cameras = env.embodiment.camera_config
    assert {"top_camera", "cable_camera", "right_wrist_camera"} <= set(
        cameras.camera_names()
    )
    mounts = [
        getattr(env.embodiment.scene_config, f"{side}_robot").init_state.pos
        for side in ("left", "right")
    ]
    assert mounts == [(-0.335, 0.31, 0.767), (-0.335, -0.31, 0.767)]


def test_two_variants_in_one_process_keep_their_own_scoring(stub_scene_module):
    """Building a second cable environment must not rebind the first one."""
    built = {}
    for difficulty in ("Easy", "Medium"):
        cfg_type = getattr(cable, f"CableRouting{difficulty}EnvironmentCfg")
        factory = getattr(cable, f"CableRouting{difficulty}Environment")()
        cfg = cfg_type(layout_seed=10000)
        built[difficulty] = (factory._variant_for_cfg(cfg), factory.build(cfg))
    for difficulty, segments in (("Easy", 50), ("Medium", 70)):
        variant, env = built[difficulty]
        (success,) = env.task.get_termination_cfg().success[0].predicate_sequence
        assert success.params["goal"] == variant.terminated_goal
        assert len(success.params["segment_half_lengths"]) == segments
        cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=1, left_robot=True), sim=SimpleNamespace()
        )
        env.env_cfg_callback(cfg)
        manager = cfg.sim.physics.solver_cfg.class_type
        assert manager.cable_variant.cable_num_segments == segments
        assert cfg.sim.physics.default_shape_cfg.mu == variant.physics.fixture_friction


@pytest.mark.parametrize("difficulty", ("easy", "medium"))
def test_all_guides_are_owned_and_exposed_to_cable_solver(difficulty):
    factory = getattr(cable, f"CableRouting{difficulty.title()}Environment")()
    cfg_type = getattr(cable, f"CableRouting{difficulty.title()}EnvironmentCfg")
    variant = factory._variant_for_cfg(cfg_type(layout_seed=10009))
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1, left_robot=True), sim=SimpleNamespace()
    )
    cable._configure_cable_routing_newton_physics(cfg, variant=variant)
    coupler = cfg.sim.physics.solver_cfg
    rigid = next(entry for entry in coupler.entries if entry.name == "rigid")
    (proxy,) = coupler.proxies
    assert (proxy.source, proxy.destination) == ("rigid", "cable")
    for index in range(len(variant.peg_positions)):
        label = f"/World/envs/env_0/Peg{index}"
        assert any(re.fullmatch(pattern, label) for pattern in rigid.bodies), label
        assert any(re.fullmatch(pattern, label) for pattern in proxy.bodies), label


@pytest.mark.parametrize("difficulty", ("easy", "medium"))
def test_published_peg_and_port_rows_match_current_sampler(difficulty):
    manifest = _manifest(difficulty)
    case = next(
        c
        for c in manifest["cases"]
        if c["name"] == f"cablerouting-{difficulty}-routecable"
    )
    assert len(case["scenes"]) == 100
    factory = getattr(cable, f"CableRouting{difficulty.title()}Environment")()
    cfg_type = getattr(cable, f"CableRouting{difficulty.title()}EnvironmentCfg")
    for scene in case["scenes"]:
        episode = scene["episode"]
        variant = factory._variant_for_cfg(cfg_type(layout_seed=episode["seed"]))
        poses = dict(episode["overrides"])["static_poses"]
        for index, actual in enumerate(variant.peg_positions):
            expected = poses[f"peg{index}"]
            assert actual == pytest.approx(
                (expected[0] + 0.0125, expected[1], expected[2] + 0.767 + 0.0235 / 2),
                abs=1e-12,
            ), scene["name"]
        # AUTOLab's current anchors are checked independently in test_cable_native_declarations;
        # the fixed Medium corpus keeps the anchor poses it was generated with.
        for terminal in ("port",):
            expected = poses[terminal]
            assert getattr(variant, terminal + "_position") == pytest.approx(
                (expected[0] + 0.0125, expected[1], expected[2] + 0.767), abs=1e-12
            ), scene["name"]
            assert getattr(variant, terminal + "_rotation_xyzw") == pytest.approx(
                (0, 0, math.sin(expected[3] / 2), math.cos(expected[3] / 2)), abs=1e-12
            ), scene["name"]
