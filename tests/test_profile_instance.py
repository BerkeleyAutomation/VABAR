# SPDX-License-Identifier: Apache-2.0
"""tools/profile_instance.py on a synthetic run directory (and on real ones, when this machine has them)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("profile_instance", REPO / "tools" / "profile_instance.py")
profile_instance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_instance)


def node(name, kind, seconds, *, start, script=None):
    return {
        "name": name,
        "node_type": kind,
        "script": script,
        "started_at": start,
        "finished_at": start + seconds,
        "duration_ms": seconds * 1000.0,
        "visits": 1,
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "th_graph"
    (run / "traces" / "0003").mkdir(parents=True)
    meta = {
        "task": "tool_hanging/wrenches_level1",
        "solution": "graph:gap_perception",
        "renderer": "rtx",
        "device": "cuda:0",
        "episodes": 1,
        "successes": 1,
        "provenance": {
            "workers": 1,
            "wall_s": 100.0,
            "run_wall_s": 130.0,
            "control_steps": 2000,
            "steps_per_s": 20.0,
            "steps_per_s_stepping": 20.0,
            "builds": [{"task": "tool_hanging/wrenches_level1", "substeps": 8, "control_hz": 60.0, "build_s": 12.5}],
        },
    }
    (run / "run_meta.json").write_text(json.dumps(meta))
    row = {"episode_id": 3, "success": True, "reward": 1.0, "episode_length": 2000, "close_kind": "policy_request"}
    (run / "episode_results.jsonl").write_text(json.dumps({**row, "wall_s": 100.0}) + "\n")
    t0 = 1000.0
    trace = {
        "nodes": [
            node("target", "subgraph", 0.0, start=0.0),
            node("target_sg.observe", "tool", 2.0, start=t0),
            node("target_sg.perceive", "script", 10.0, start=t0 + 2, script="scripts/perceive_target.py"),
            node("register_sg.register", "script", 5.0, start=t0 + 12, script="scripts/register_held.py"),
            node("disengage_sg.plan", "tool", 4.0, start=t0 + 17),
            node("motion_plan_sg.plan", "script", 1.0, start=t0 + 21, script="scripts/plan_clearance_motion.py"),
            node("insert_sg.execute", "script", 60.0, start=t0 + 22, script="scripts/execute_placement_plan.py"),
            node("grasp_sg.close", "tool", 8.0, start=t0 + 82),
        ],
        "events": [
            {"event_type": "subcall_recorded", "detail": {"tool": "sam3.segment_text"}},
            {"event_type": "subcall_recorded", "detail": {"tool": "sam3.segment_text"}},
        ],
    }
    (run / "traces" / "0003" / "dag_trace.json").write_text(json.dumps(trace))
    workflow = {
        "subgraphs": {
            "target_sg": {"nodes": {"observe": {"type": "tool", "tool": "robot.get_observation"}}},
            "disengage_sg": {"nodes": {"plan": {"type": "tool", "tool": "motion.plan_support_disengagement"}}},
            "grasp_sg": {"nodes": {"close": {"type": "tool", "tool": "robot.close_gripper"}}},
        }
    }
    (run / "traces" / "0003" / "workflow.json").write_text(json.dumps(workflow))
    return run


def test_groups(run_dir):
    p = profile_instance.trace_profile(run_dir / "traces" / "0003")
    assert p["wall_s"] == pytest.approx(90.0)
    assert p["groups"] == pytest.approx(
        {"perception scripts": 15.0, "motion scripts": 60.0, "robot tools": 10.0, "planning": 5.0}
    )
    tools = {n["name"]: n["what"] for n in p["nodes"]}
    assert tools["disengage_sg.plan"] == "motion.plan_support_disengagement"
    assert p["subcalls"]["sam3.segment_text"] == 2


def test_report(run_dir, capsys):
    assert profile_instance.main([str(run_dir.parent)]) == 0
    out = capsys.readouterr().out
    assert "build_s    12.5" in out and "steps 2000" in out and "steps/s 20.0" in out and "wall_s 100.0" in out
    assert "      3  yes" in out and "policy_request" in out
    assert "graph wall 90.0 s, 90% of the episode's 100 s" in out
    assert "motion scripts            60.0 s  66.7%" in out
    assert "robot.get_observation 2.0 s" in out


REAL = os.environ.get("VABAR_PROFILE_RUNS", "")
"""A directory of real graph runs to profile too: ``th_L1_full`` (serial) and ``th_L1_fleet4_rtx`` (a 4-world wave)."""


@pytest.mark.parametrize("name", ["th_L1_full", "th_L1_fleet4_rtx"])
def test_real_runs(name, capsys):
    run = Path(REAL) / name
    if not REAL or not (run / "run_meta.json").is_file():
        pytest.skip("set VABAR_PROFILE_RUNS to a directory holding real graph runs")
    assert profile_instance.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "perception scripts" in out and "planning" in out and "insert_sg.execute" in out
