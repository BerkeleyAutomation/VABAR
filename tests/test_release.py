# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: the suites register on the pinned RoboSimStudio, and the benchmark names real tasks.

uv run pytest tests/
"""

from __future__ import annotations

import importlib.util
import json
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
