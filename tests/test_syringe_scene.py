# SPDX-License-Identifier: Apache-2.0
"""The syringe tray scene: colliders where they are drawn, and the same hand and renderer in every config.

No GPU: this reads the lowered specs, the layouts and the documents. What the scene scored is in
``docs/validation/syringe_working_scene.md``.
"""

from __future__ import annotations

import json

import pytest

import vabar
from vabar import benchmark
from vabar import config as cfg
from vabar.run import config_flags, runner_flags

TASK = vabar.TASKS / "syringe_packing"
POSES = TASK / "poses"
FROZEN = vabar.REPO / "benchmark" / "syringe_packing"
HULLS = TASK / "assets" / "syringe" / "collision_coacd"
CELLS = ["syringe_packing/L1", "syringe_packing/L2"]
CONFIGS = ["default", "fast", "paper"]

CAVITY_HALF = (0.097, 0.197)
SYRINGE_HALF_LENGTH = 0.075
SYRINGE_HALF_WIDTH = 0.011783
FOOTPRINT_TOLERANCE_M = 0.0002


def _rotate(q, v):
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _margin(entry, tray_xyz, tray_quat) -> float:
    """The least distance [m] from a syringe's footprint to the drawn walls, in the tray's frame."""
    inverse = (-tray_quat[0], -tray_quat[1], -tray_quat[2], tray_quat[3])
    centre = _rotate(inverse, [a - b for a, b in zip(entry["pose"], tray_xyz, strict=True)])
    along = _rotate(inverse, _rotate(entry["quat"], (0.0, 1.0, 0.0)))
    across = _rotate(inverse, _rotate(entry["quat"], (1.0, 0.0, 0.0)))
    reach = [abs(along[i]) * SYRINGE_HALF_LENGTH + abs(across[i]) * SYRINGE_HALF_WIDTH for i in range(2)]
    return min(CAVITY_HALF[i] - (abs(centre[i]) + reach[i]) for i in range(2))


def _syringes(layout: dict) -> dict:
    return {name: entry for name, entry in layout["objects"].items() if name.startswith("syringe") and "pose" in entry}


@pytest.fixture(scope="module")
def suite():
    vabar.install()
    from robosimstudio_bench.taskbuild.registry import get_suite

    return get_suite("syringe_packing")


def test_hull_set_is_the_one_measured():
    assert len(json.loads((HULLS / "manifest.json").read_text())["hulls"]) == 61
    assert len(list(HULLS.glob("hull_*.obj"))) == 61


@pytest.mark.parametrize("path", sorted(POSES.glob("tray_*.json")), ids=lambda p: p.stem)
def test_layout_lies_in_the_pan(path):
    layout = json.loads(path.read_text())
    xform = layout["objects"]["syringe_tray"]["xform"]
    syringes = _syringes(layout)
    assert syringes
    for name, entry in syringes.items():
        assert _margin(entry, xform[:3], xform[3:]) >= -FOOTPRINT_TOLERANCE_M, name


def test_level2_holds_176_syringes():
    counts = [len(_syringes(json.loads((POSES / f"tray_clutter_t{i:02d}.json").read_text()))) for i in range(30)]
    assert sum(counts) == 176
    assert min(counts) >= 4 and max(counts) <= 8


@pytest.mark.parametrize(("level", "count"), [(1, 31), (2, 30)])
def test_frozen_copies_are_the_documents(level, count):
    frozen = sorted((FROZEN / f"level{level}" / "tasks").glob("*.task"))
    assert len(frozen) == count
    for path in frozen:
        assert path.read_text() == (TASK / path.name).read_text(), path.name


@pytest.mark.parametrize("variant", ["tray_pair_both_t00", "tray_clutter_t01", "tray_pair_both", "tray_pair"])
def test_documents_state_buffers_sized_for_the_hulls(suite, variant):
    """On 1024 a full L2 cell overflowed its broadphase, which drops collision pairs and carries on."""
    params = dict(suite.variant(variant).physics.backend_for("*").params)
    assert params["njmax"] >= 8192 and params["nconmax"] >= 4096, variant
    assert params["enable_multiccd"]


def test_arrangements_differ_only_in_buffers_and_rate(suite):
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    base, hulls, fast = (get_arrangement(n) for n in ("vab_500hz@v1", "vab_sharps_union@v1", "vab_syringe_200hz@v1"))
    assert hulls.substeps == base.substeps == 10 and fast.substeps == 4
    assert hulls.collision == base.collision == fast.collision
    assert hulls.control_hz == base.control_hz == fast.control_hz
    strip = lambda a: {k: v for k, v in a.backend_for("*").params if k not in {"njmax", "nconmax"}}
    assert strip(hulls) == strip(base) == strip(fast)
    assert dict(base.backend_for("*").params)["njmax"] == 1024, "vab_500hz@v1 is tool_sort's too"


def test_fast_maps_the_documents_physics(config_sandbox):
    table = cfg.load("fast").table("syringe_packing/tray_pair_both_t00")
    assert table["physics"] == {"vab_sharps_union@v1": "vab_syringe_200hz@v1"}
    assert table["workers"] == 16


def test_the_reference_hand_declares_the_graph(suite):
    binding = suite.variant("tray_pair_both_t00").binding_for("robosuite_bench", "panda_robotiq")
    assert "gap_perception" in binding.graphs


@pytest.mark.parametrize("variant", ["tray_pair_both_t00", "tray_clutter_t01"])
def test_the_look_is_bare(suite, variant):
    from robosimstudio.specs import RenderSpec

    assert suite.variant(variant).render == RenderSpec()


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("cell", CELLS)
def test_every_config_benches_on_tiled_and_the_robotiq(cell, config, config_sandbox):
    flags = runner_flags()
    plan = benchmark.plan_cell(
        cell,
        benchmark.load()["cells"][cell],
        cfg.resolve(config, []),
        flags=flags,
        budget=lambda task: 1000,
        workers=lambda config, units: (1, "test"),
    )
    for job in plan.jobs:
        assert job.argv.count("--renderer") == 1
        assert flags.value(job.argv, "--renderer") == "tiled"
        assert flags.value(job.argv, "--embodiment") == "panda_robotiq"
        assert flags.value(job.argv, "--scene") == "robosuite_bench"
    assert {spec.renderer for spec in benchmark.job_specs(plan, flags)} == {"tiled"}


@pytest.mark.parametrize("config", CONFIGS)
def test_a_plain_run_gets_the_same_hand_and_renderer(config, config_sandbox):
    """`vabar run` has no cell to read, so the suite's table has to say it."""
    flags = runner_flags()
    add, _, _ = config_flags(
        cfg.resolve(config, []),
        "syringe_packing/tray_pair_both_t00",
        solve="graph",
        vision=True,
        flags=flags,
        bench=False,
    )
    assert flags.value(add, "--renderer") == "tiled"
    assert flags.value(add, "--embodiment") == "panda_robotiq"


@pytest.mark.parametrize(
    ("override", "expected"),
    [("renderer=rtx", "rtx"), ("runtime.renderer=rtx", "rtx"), ("suites.syringe_packing.renderer=gl", "gl")],
)
def test_the_command_line_overrides_the_suites_renderer(override, expected, config_sandbox):
    assert cfg.resolve("fast", [override]).renderer("syringe_packing/tray_pair_both_t00") == expected


def test_another_suite_keeps_the_runtimes_renderer(config_sandbox):
    assert cfg.resolve("fast", []).renderer("tool_hanging/pliers_level1") == "rtx"
