"""Opt-in CPU parity check of the Cable Easy/Medium variants against AUTOLab's own cable runtime.

The test skips unless ``CABLE_NATIVE_PYTHON`` names a Python interpreter whose environment can
import AUTOLab's pre-v1.0 cable runtime (``run_cable``, ``robosimstudio_bench``, Newton, Warp) and
``CABLE_NATIVE_ROOT`` names a checkout of that runtime (AUTOLab-GaP-NVIDIA-Collaboration at
``82e414c``, e.g. ``git -C <AUTOLab checkout> worktree add <dir> 82e414c``; v1.0 no longer
ships ``run_cable``). For all 200 published scenes it compares cable points, physics
coefficients, pegs, anchor and port poses, goal regions and thresholds, episode length, and the
AUTOLab camera declaration. No simulator, renderer, or GPU is used.
"""

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaaclab")

from isaaclab_arena_vabar.environments.cable_routing import environment as cable


def test_current_native_tasks_match_seeded_arena_declarations():
    native_python = os.environ.get("CABLE_NATIVE_PYTHON")
    native_root = os.environ.get("CABLE_NATIVE_ROOT")
    if not (native_python and native_root):
        pytest.skip("set CABLE_NATIVE_PYTHON and CABLE_NATIVE_ROOT to AUTOLab's pre-v1.0 cable runtime")
    source = Path(native_root)
    # Use the official loader's discovery names; direct task-module imports can
    # register the same assets twice. No Env, renderer or solver is constructed.
    script = r"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import run_cable
run_cable.configure()
from tools.fixed_scenes import base_document, plain
from robosimstudio_bench.taskbuild import get_task
from robosimstudio_bench.taskbuild.episodes import make_episode
from newton.utils import create_straight_cable_points_and_quaternions
from robosimstudio.domains.rod.entity import _bow_cable
import warp as wp
dataset = Path("benchmark_specs/cable_tasks/dataset")
manifest = json.loads((dataset / "manifest.json").read_text())
rows = []
for case in manifest["cases"]:
    if case["name"] not in ("cablerouting-easy-routecable", "cablerouting-medium-routecable"):
        continue
    tier = case["name"].split("-")[1]
    task = get_task("cable_route/port_" + tier)
    document = base_document(task.full_name)[1]
    for scene in case["scenes"]:
        seeded = replace(task, episodes=replace(task.episodes, seed=scene["episode"]["seed"], count=1))
        episode = make_episode(seeded, 0)
        spec = seeded.deformables[0]
        points, quats = create_straight_cable_points_and_quaternions(
            start=wp.vec3(*spec.start), direction=wp.vec3(*spec.direction),
            length=spec.length, num_segments=spec.segments)
        points, _ = _bow_cable(points, quats, spec,
                              episode)
        rows.append(dict(tier=tier, name=scene["name"], seed=scene["episode"]["seed"],
                         spec=plain(spec), points=[[float(v) for v in p] for p in points],
                         episode=plain(episode), goal=plain(document.goal),
                         regions=plain(task.regions), cameras=plain(task.cameras),
                         control_hz=task.physics.control_hz, max_steps=task.max_steps))
print(json.dumps(rows))
"""
    result = subprocess.run(
        [native_python, "-c", script],
        cwd=source,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PXR_WORK_THREAD_LIMIT": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    rows = json.loads(result.stdout.splitlines()[-1])
    assert len(rows) == 200
    fields = {
        "radius": "radius",
        "density": "density",
        "stretch_stiffness": "stretch_ke",
        "stretch_damping": "stretch_kd",
        "bend_stiffness": "bend_ke",
        "bend_damping": "bend_kd",
        "cable_friction": "mu_cable",
        "fixture_friction": "mu_base",
        "table_friction": "mu_surface",
        "contact_stiffness": "contact_ke",
        "contact_damping": "contact_kd",
        "contact_gap": "contact_gap",
    }
    for row in rows:
        factory = getattr(cable, f"CableRouting{row['tier'].title()}Environment")()
        cfg_type = getattr(cable, f"CableRouting{row['tier'].title()}EnvironmentCfg")
        variant = factory._variant_for_cfg(cfg_type(layout_seed=row["seed"]))
        # Newton's vec3 spawn is float32; compare in the declared surface frame,
        # before either environment applies its world translation or settles.
        np.testing.assert_allclose(
            variant.cable_local_positions,
            row["points"],
            rtol=0,
            atol=2e-7,
            err_msg=row["name"],
        )
        for cap_name, native_name in fields.items():
            assert getattr(variant.physics, cap_name) == pytest.approx(
                row["spec"][native_name], rel=0, abs=1e-12
            ), (row["name"], cap_name)
        assert variant.cable_num_segments == row["spec"]["segments"]
        assert row["spec"]["anchors"] == ["start"]
        assert row["spec"]["allow_grasp_latch"] is False
        assert row["spec"]["filter_fingertip_scenery"] is False
        assert variant.episode_length_s == row["max_steps"] / row["control_hz"]
        poses = dict(row["episode"]["overrides"])["static_poses"]
        for index, actual in enumerate(variant.peg_positions):
            expected = poses[f"peg{index}"]
            assert actual == pytest.approx(
                (expected[0] + 0.0125, expected[1], expected[2] + 0.767 + 0.0235 / 2),
                abs=1e-12,
            ), row["name"]
        for terminal in ("anchor", "port"):
            expected = poses[terminal]
            assert getattr(variant, terminal + "_position") == pytest.approx(
                (expected[0] + 0.0125, expected[1], expected[2] + 0.767),
                abs=1e-12,
            ), row["name"]
            assert getattr(variant, terminal + "_rotation_xyzw") == pytest.approx(
                (0, 0, math.sin(expected[3] / 2), math.cos(expected[3] / 2)),
                abs=1e-12,
            ), row["name"]
        goal = variant.terminated_goal
        containment, released = row["goal"]["children"]
        assert row["goal"]["head"] == "worst_of"
        assert containment["head"] == released["head"] == "and"
        terms = containment["children"]
        assert tuple(term["value"] for term in terms[:-2]) == goal.seat_min_fractions
        assert terms[-2]["value"] == goal.port_min_fraction
        assert terms[-1]["args"][0]["path"] == ["end_b"] and terms[-1]["value"] == 1
        assert [(t["head"], t["op"], t["value"]) for t in released["children"]] == [
            ("tcp-distance", ">", goal.min_tcp_distance),
            ("speed", "<=", goal.max_mean_speed),
        ]
        assert (
            tuple(tuple(r["xy_ranges"]) for r in row["regions"][:-1])
            == goal.seat_regions
        )
        assert tuple(row["regions"][-1]["xy_ranges"]) == goal.port_region
        camera = next(c for c in row["cameras"] if c["name"] == "cable")
        assert (
            camera["name"],
            camera["width"],
            camera["height"],
            camera["fov_y"],
            camera["channels"],
        ) == (
            "cable",
            1280,
            720,
            50,
            ["rgb", "depth"],
        )
