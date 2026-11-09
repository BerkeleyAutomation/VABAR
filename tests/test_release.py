# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: the suites register on the pinned RoboSimStudio, and the benchmark names real tasks.

uv run pytest tests/
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

import vabar
from vabar import benchmark

SUITES = {"cable_route", "gear_mesh", "syringe_packing", "tool_hanging", "tool_sort", "usbc_insert"}


def test_install_is_idempotent():
    from robosimstudio.roots import task_roots

    first = vabar.install()
    second = vabar.install()
    assert first == second == vabar.TASKS
    assert sum(1 for root in task_roots() if root.resolve() == vabar.TASKS.resolve()) == 1


def test_routines_are_installed_where_upstream_looks():
    from robosimstudio_bench.solutions.declared import ROUTINES_MODULE, routines_installed

    vabar.install_routines()
    assert routines_installed()
    assert importlib.util.find_spec(ROUTINES_MODULE) is not None
    assert sys.modules[ROUTINES_MODULE].SolutionRoutine.__module__ == "vabar.routines"


@pytest.fixture(scope="module")
def tasks() -> set[str]:
    vabar.install()
    from robosimstudio_bench.taskbuild.registry import list_tasks

    return set(list_tasks())


def test_every_suite_registers(tasks):
    assert {name.split("/", 1)[0] for name in tasks} == SUITES


def test_manifest_is_current():
    assert benchmark.load() == json.loads(json.dumps(benchmark.build()))


def test_manifest_names_registered_tasks(tasks):
    cells = benchmark.load()["cells"]
    assert len(cells) == 12
    missing = sorted({i["task"] for c in cells.values() for i in c["instances"]} - tasks)
    assert not missing


def test_manifest_cells_hold_thirty_or_say_why():
    for key, cell in benchmark.load()["cells"].items():
        assert len(cell["instances"]) == 30 or (not cell["instances"] and cell["notes"]), key
        assert len({(i["task"], i["episode"]) for i in cell["instances"]}) == len(cell["instances"]), key


def test_reference_graphs_are_declared(tasks):
    from robosimstudio_bench.taskbuild.registry import get_task

    for key, cell in benchmark.load()["cells"].items():
        for task in {i["task"] for i in cell["instances"]}:
            assert cell["graph"] in get_task(task).binding_for().graphs, (key, task)


def test_tool_sort_level2_is_frozen_and_its_assets_ship():
    """The frozen L2 copies are the documents that run, and every file their exports name is in the tree."""
    for i in range(1, 31):
        doc = vabar.TASKS / "tool_sort" / f"level2_{i}.task"
        assert (vabar.REPO / "benchmark" / "tool_sort" / "level2" / "tasks" / doc.name).read_text() == doc.read_text()
        # The Robotiq profile welds its own hand on: scene.xml here would stand two arms in the cell.
        assert "(:file scene_norobot.xml)" in doc.read_text()
        assert "(:entity tool_sort_workcell)" in doc.read_text()
        scene = vabar.REPO / "assets" / "vab_tool_sort" / "level2" / f"level2_{i}" / "scene_norobot.xml"
        missing = [f for f in re.findall(r'file="([^"]+)"', scene.read_text()) if not (scene.parent / f).is_file()]
        assert not missing, (i, missing)


def test_tool_sort_attempt_log_comes_round_the_loop():
    """A call-site input outranks the loop's own output, so binding this one fed `select_pair` an empty log every visit."""
    graph = vabar.TASKS / "tool_sort" / "graphs" / "gap_perception_loop"
    workflow = json.loads((graph / "workflow.json").read_text())
    assert "attempted_json" not in workflow["nodes"]["pair"]["inputs"]
    assert "attempted_json" in workflow["subgraphs"]["pair_sg"]["outputs"]
    # select_pair's every exit keeps the log: a retry that dropped it would start the count again.
    script = (graph / "scripts" / "select_pair.py").read_text()
    retry = script[script.index('_empty_result("retry")') :]
    assert 'result["attempted_json"]' in retry[: retry.index("return result")]


def test_tool_sort_blocked_descent_stops_on_a_stall():
    """A blocked descent used to spend 1,000 steps, over 40% of an episode's, on a move that rarely ended holding."""
    workflow = json.loads((vabar.TASKS / "tool_sort" / "graphs" / "gap_perception_loop" / "workflow.json").read_text())
    for subgraph in ("grasp_sg", "concave_grasp_sg"):
        descend = workflow["subgraphs"][subgraph]["nodes"]["descend"]
        assert descend["tool"] == "robot.move_cartesian_until_contact"
        assert descend["inputs"]["stall_steps"] == 60


def test_tool_hanging_landmark_beads_stay_out_of_the_cameras(tasks):
    """The debugging beads render in every camera a perception graph or a policy reads; VABAR_LANDMARK_MARKERS=1 opts in."""
    from robosimstudio_bench.taskbuild.registry import get_task

    for doc in (vabar.TASKS / "tool_hanging").glob("*.task"):
        assert all(not spec.visual_markers for spec in get_task(f"tool_hanging/{doc.stem}").objects), doc.stem
