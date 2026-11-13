"""Contracts of the RoboSimStudio-runtime adapter that hold without AUTOLab's environment installed."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena_gap import robot_io_types
from isaaclab_arena_gap.task_sources import autolab_python, autolab_root

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOLAB_VABAR = autolab_root()
# Every shipped experiment runs through the adapter: the baselines and the cable campaigns.
RSS_CONFIGS = sorted((REPO_ROOT / "isaaclab_arena_gap/experiment_configs").glob("*/*.yaml"))
BASELINES = [p for p in RSS_CONFIGS if p.name.endswith("_gap_baseline.yaml")]
# Documents outside v1.0's cells, each with why no v1.0 instance fits: the document's file
# and the graph its bound graph name resolves to, which must be the one v1.0 runs.
NON_V1_DOCUMENTS = {
    # v1.0's syringe cells are both-syringes (L1) and clutter (L2); Arena's single and
    # designated variants post one syringe from the tray, AUTOLab's tray_pair task.
    "syringe_packing/tray_pair": ("tray_pair.task", "graphs/gap_perception"),
}
# v1.0 recorded no gear selection and takes the first 30 documents by index. The gear
# campaigns pin each seed to the document the RSS comparison runs used; these
# are past index 30, in the same generated families, read by the suite's one graph.
GEAR_DOCUMENTS_PAST_V1 = {
    *(f"gear_mesh/gearmesh-easy-pair-{n}" for n in ("35", "47", "48", "59")),
    *(f"gear_mesh/gearmesh-medium-train-{n}" for n in ("040", "041", "046", "050", "054", "083", "092")),
}


def _v1_cells() -> dict:
    benchmark = AUTOLAB_VABAR / "benchmark/vabar_v1.json"
    if not benchmark.exists():
        pytest.skip("AUTOLab v1.0 is not materialized")
    return json.loads(benchmark.read_text())["cells"]


def _rss_policy(path: Path) -> dict:
    """A baseline's one policy, or a campaign's shared one."""
    import yaml

    document = yaml.safe_load(path.read_text())
    if "policy" in document.get("shared", {}):
        return document["shared"]["policy"]
    (run,) = document["runs"].values()
    return run["policy"]


def _run_policies(path: Path) -> list[dict]:
    """Every run's policy, with a campaign's per-run inputs over its shared ones."""
    import yaml

    document = yaml.safe_load(path.read_text())
    shared = document.get("shared", {}).get("policy", {})
    policies = []
    for run in document["runs"].values():
        policy = {**shared, **run.get("policy", {})}
        policy["gap_inputs"] = {**shared.get("gap_inputs", {}), **run.get("policy", {}).get("gap_inputs", {})}
        policies.append(policy)
    return policies


@pytest.mark.parametrize("name", ["CameraSample", "RobotSample", "JointPositionCommand"])
def test_fallback_wire_types_mirror_gap_robot_io(name):
    gap_robot_io = pytest.importorskip("gap.robot_io")
    fields = lambda cls: [(f.name, f.default) for f in dataclasses.fields(cls)]  # noqa: E731
    assert fields(getattr(robot_io_types, name)) == fields(getattr(gap_robot_io, name))


def test_relative_launcher_resolves_against_the_repository(monkeypatch):
    from isaaclab_arena_gap import gap_process

    spawned = {}
    monkeypatch.setattr(gap_process.subprocess, "Popen", lambda command, **kw: spawned.setdefault("cmd", command))
    monkeypatch.setattr(gap_process, "apply_adapter_tool_env", lambda config, env: None)
    config = SimpleNamespace(
        gap_launcher="isaaclab_arena_gap/scripts/run_rss_graph.sh", gap_graph="gear_mesh/gap_perception",
        graph_result_path="", gap_fleet_session=False, gap_inputs={},
    )
    gap_process.default_spawn_gap(config, 9000)
    assert spawned["cmd"][0] == str(REPO_ROOT / "isaaclab_arena_gap/scripts/run_rss_graph.sh")


def test_bindings_run_graphs_as_autolabs_benchmark_does():
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    cells = _v1_cells()
    for graph, binding in BINDINGS.items():
        suite, name = graph.split("/", 1)
        matching = [c for c in cells.values() if c["suite"] == suite and c["graph"] == name]
        assert matching, f"no v1.0 cell runs {graph}"
        for cell in matching:
            flags = cell["flags"]
            value = lambda flag: flags[flags.index(flag) + 1] if flag in flags else ""  # noqa: E731
            assert binding.flags["privileged"] is ("--no-privileged" not in flags)
            assert binding.flags["perception"] == value("--perception")
            assert binding.flags.get("motion", "") == value("--motion")
            assert binding.flags["bus_timeout"] == float(value("--bus-timeout"))
            assert binding.flags.get("settle_steps", 60) == int(value("--settle-steps") or 60)
            # Only the cell's own environment, where Arena keeps it: USB-C drops the palette (see BINDINGS).
            assert set(cell["env"]) <= {"ROBOSIMSTUDIO_SEGMENTATION_BACKEND"}
            assert set(binding.environment) <= set(cell["env"].items())
        assert (AUTOLAB_VABAR / "tasks" / suite / "graphs" / name / "workflow.json").exists()


def test_a_chosen_document_is_a_v1_instance_of_its_graph():
    """Graph names resolve through the document (an L2 tool-hanging document's
    ``gap_perception`` is ``graphs/gap_perception_level2``), so the one an
    experiment names must be one v1.0 runs that graph on."""
    from isaaclab_arena_gap.rss_adapter.runner import document_for

    cells = _v1_cells()
    chosen = [(path, policy) for path in RSS_CONFIGS for policy in _run_policies(path)]
    chosen = [(path, policy) for path, policy in chosen if "autolab_task" in policy.get("gap_inputs", {})]
    assert chosen
    for path, policy in chosen:
        suite, name = policy["gap_graph"].split("/", 1)
        instances = {i["task"] for c in cells.values() if (c["suite"], c["graph"]) == (suite, name) for i in c["instances"]}
        document = document_for(policy["gap_graph"], policy["gap_inputs"])
        assert document in instances or document in NON_V1_DOCUMENTS or document in GEAR_DOCUMENTS_PAST_V1, (
            path.name,
            document,
        )


def test_gear_documents_past_v1_are_documents_of_v1s_gear_families():
    cells = _v1_cells()
    gear = [c for c in cells.values() if c["suite"] == "gear_mesh"]
    families = {i["task"].rsplit("-", 1)[0] for c in gear for i in c["instances"]}
    assert {c["graph"] for c in gear} == {"gap_perception"}
    for document in GEAR_DOCUMENTS_PAST_V1:
        assert document.rsplit("-", 1)[0] in families
        suite, stem = document.split("/", 1)
        assert (AUTOLAB_VABAR / "tasks" / suite / f"{stem}.task").is_file()


@pytest.mark.parametrize("document", sorted(NON_V1_DOCUMENTS))
def test_a_document_outside_v1_resolves_the_graph_v1_runs(document):
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    cells = _v1_cells()
    suite = document.split("/", 1)[0]
    (graph,) = {name.split("/", 1)[1] for name in BINDINGS if name.startswith(f"{suite}/")}
    assert any((c["suite"], c["graph"]) == (suite, graph) for c in cells.values())
    filename, resolved = NON_V1_DOCUMENTS[document]
    text = (AUTOLAB_VABAR / "tasks" / suite / filename).read_text()
    assert text.count(f"(:graph {graph} ") == 1
    assert f'(:graph {graph} "{resolved}")' in text
    assert (AUTOLAB_VABAR / "tasks" / suite / resolved / "workflow.json").is_file()


def test_every_binding_default_document_is_a_v1_instance_or_named_exception():
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    cells = _v1_cells()
    tasks = {i["task"] for c in cells.values() for i in c["instances"]}
    # gear_mesh's default is AUTOLab's single-gear document, Arena's easy scene; v1.0's
    # L1 starts at the pair. Experiments for v1.0's own levels name their document.
    defaults = {b.task for b in BINDINGS.values()} - tasks
    assert defaults == {"gear_mesh/gearmesh-easy-single-01"}


def test_the_experiment_chooses_the_document():
    from isaaclab_arena_gap.rss_adapter.runner import document_for, experiment_inputs

    argv = ["--inputs", 'autolab_task="usbc_insert/insert_medium"', 'instruction="insert the plug"']
    assert document_for("usbc_insert/mate", experiment_inputs(argv)) == "usbc_insert/insert_medium"
    assert document_for("usbc_insert/mate", {}) == "usbc_insert/insert_easy"
    with pytest.raises(SystemExit):
        document_for("usbc_insert/mate", {"autolab_task": "tool_hanging/pliers_level2"})


def test_every_baseline_runs_through_the_adapter():
    assert BASELINES and len(BASELINES) == 22


@pytest.mark.parametrize("path", RSS_CONFIGS, ids=lambda p: p.stem)
def test_rss_experiments_name_the_launcher_and_a_bound_graph(path):
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS, ROBOTS, document_for

    policy = _rss_policy(path)
    launcher = REPO_ROOT / policy["gap_launcher"]
    assert launcher.exists() and os.access(launcher, os.X_OK)
    # Only the adapter's own scene adaptations, never the vendored path's patches.
    assert policy.get("gap_adaptation", "rss_").startswith("rss_")
    binding = BINDINGS[policy["gap_graph"]]
    assert policy["robot_profile"] == binding.robot_profile
    assert binding.robot_profile in ROBOTS
    # A named document must be in the graph's suite (document_for stops the run otherwise).
    inputs = policy.get("gap_inputs", {})
    if "autolab_task" in inputs:
        assert document_for(policy["gap_graph"], inputs) == inputs["autolab_task"]


def test_the_experiment_instruction_reaches_the_graph():
    from isaaclab_arena_gap.rss_adapter.runner import experiment_inputs

    # GapRemotePolicy's launch command for the pliers config (gap_process.spawn).
    argv = ["--inputs", 'instruction="hang the pliers on the triangular support"']
    assert experiment_inputs(argv) == {"instruction": "hang the pliers on the triangular support"}
    assert experiment_inputs([]) == {}


def _staged_adaptation(path: Path) -> tuple[Path, Path]:
    from isaaclab_arena_gap.rss_adapter.runner import adapted_graph

    policy = _rss_policy(path)
    suite, graph = policy["gap_graph"].split("/", 1)
    if str((policy.get("gap_inputs") or {}).get("autolab_task", "")).endswith("_level2"):
        graph += "_level2"  # AUTOLab's L2 documents resolve gap_perception to its level-2 workflow
    source = AUTOLAB_VABAR / "tasks" / suite / "graphs" / graph
    if not source.is_dir():
        pytest.skip("AUTOLab v1.0 sources are not materialized; run ./setup.sh")
    return source, Path(adapted_graph(source, policy["gap_adaptation"]))


ADAPTED_CONFIGS = [p for p in RSS_CONFIGS if "gap_adaptation" in _rss_policy(p)]


@pytest.mark.parametrize("path", ADAPTED_CONFIGS, ids=lambda p: p.stem)
def test_rss_adaptations_apply_to_the_pinned_v1_graph(path):
    source, staged = _staged_adaptation(path)
    assert (staged / "workflow.json").is_file()
    assert staged != source
    assert (staged / "workflow.json").read_text() != (source / "workflow.json").read_text()


def test_rss_wrench_medium_registers_arenas_artist_wrench_on_the_level2_graph():
    path = REPO_ROOT / "isaaclab_arena_gap/experiment_configs/tool_hanging/vabar_tool_hanging_wrench_medium_gap_baseline.yaml"
    policy = _rss_policy(path)
    assert policy["gap_adaptation"] == "rss_tool_hanging_artist_wrench_level2"

    source, staged = _staged_adaptation(path)
    assert source.name == "gap_perception_level2"
    # The level-2 graph runs the easy graph's scripts from the sibling directory; both are patched.
    scripts = staged.parent / "gap_perception/scripts"
    assert "artist_wrench_profiles" in (scripts / "perceive_target.py").read_text()
    assert "refine_artist_hook" in (scripts / "perceive_fixture.py").read_text()
    assert "artist_wrench_profiles" not in (source.parent / "gap_perception/scripts/perceive_target.py").read_text()

    adapted = json.loads((staged / "workflow.json").read_text())
    pinned = json.loads((source / "workflow.json").read_text())
    for number in ("1", "2"):
        select = ("subgraphs", f"pair_sg_{number}", "nodes", "select", "inputs", "source_categories")
        plan = ("subgraphs", f"motion_plan_sg_{number}", "nodes", "plan", "inputs", "motion_profiles")

        def at(workflow, keys):
            for key in keys:
                workflow = workflow[key]
            return workflow

        assert at(adapted, select)[0]["query"] == "wrench"
        assert at(pinned, select)[0]["query"] == "adjustable wrench"
        assert at(adapted, plan)[0]["transit_speed_scale"] == 0.5
        assert at(pinned, plan)[0]["transit_speed_scale"] == 1.0


def test_rss_level2_wrench_adaptation_refuses_the_easy_graph():
    from isaaclab_arena_gap.rss_adapter.runner import adapted_graph

    easy = AUTOLAB_VABAR / "tasks/tool_hanging/graphs/gap_perception"
    if not easy.is_dir():
        pytest.skip("AUTOLab v1.0 sources are not materialized; run ./setup.sh")
    # The level-2 patch names the level-2 workflow; on the easy graph it must stop the run
    # rather than return the easy graph with only its scripts patched.
    with pytest.raises(SystemExit):
        adapted_graph(easy, "rss_tool_hanging_artist_wrench_level2")


@pytest.mark.parametrize("path", [p for p in ADAPTED_CONFIGS if "usbc" in p.stem], ids=lambda p: p.stem)
def test_rss_usbc_adaptations_segment_the_artist_plug(path):
    source, staged = _staged_adaptation(path)
    assert "sam3.segment_text" in (staged / "workflow.json").read_text()
    assert 'PLUG_QUERY = "orange and silver USB-C plug"' in (staged / "scripts/ground_vision.py").read_text()
    assert "orange and silver" not in (source / "scripts/ground_vision.py").read_text()


def test_rss_wrench_easy_registers_arenas_artist_wrench_with_autolabs_mating():
    from isaaclab_arena_gap.rss_adapter.runner import document_for

    path = REPO_ROOT / "isaaclab_arena_gap/experiment_configs/tool_hanging/vabar_tool_hanging_wrench_easy_gap_baseline.yaml"
    policy = _rss_policy(path)
    assert policy["gap_launcher"] == "isaaclab_arena_gap/scripts/run_rss_graph.sh"
    assert policy["gap_adaptation"] == "rss_tool_hanging_artist_wrench"
    assert policy["robot_profile"] == "yam_bimanual_wide"
    assert policy["gap_inputs"] == {"instruction": "hang the wrench on the hook"}
    assert document_for(policy["gap_graph"], policy["gap_inputs"]) == "tool_hanging/wrenches_level1"

    source, staged = _staged_adaptation(path)
    # Arena's artist CAD and hook fit, from isaaclab_arena_gap (on the adapter's PYTHONPATH).
    target = (staged / "scripts/perceive_target.py").read_text()
    assert "from isaaclab_arena_gap.tool_hanging_asset_profile import artist_wrench_profiles" in target
    assert "refine_artist_hook" in (staged / "scripts/perceive_fixture.py").read_text()
    assert "isaaclab_arena_gap" not in (source / "scripts/perceive_target.py").read_text()
    for script in staged.glob("scripts/*.py"):
        assert "autolab_robot" not in script.read_text(), script.name

    def wrench(workflow, subgraph, node, key, kind_key):
        items = workflow["subgraphs"][subgraph]["nodes"][node]["inputs"][key]
        return next(item for item in items if item[kind_key] == "wrench")

    adapted = json.loads((staged / "workflow.json").read_text())
    pinned = json.loads((source / "workflow.json").read_text())
    # AUTOLab's mating offsets compensate the loaded arm's sag and are kept.
    fixture = ("fixture_sg", "perceive", "fixture_profiles", "source_kind")
    assert wrench(adapted, *fixture)["mating_profile"] == wrench(pinned, *fixture)["mating_profile"]
    assert wrench(adapted, *fixture)["mating_profile"]["mate_settling_offset_m"] == pytest.approx(0.010)
    categories = adapted["subgraphs"]["pair_sg"]["nodes"]["select"]["inputs"]["source_categories"]
    assert next(c for c in categories if c["kind"] == "wrench")["query"] == "wrench"


CABLE_CONFIGS = [p for p in RSS_CONFIGS if _rss_policy(p)["gap_graph"] == "cable_route/gt"]


def _document_fixtures(task: str) -> list[str]:
    """The ``(:fixtures ...)`` names of an AUTOLab task document, read as text."""
    import re

    suite, name = task.split("/", 1)
    path = AUTOLAB_VABAR / "tasks" / suite / f"{name}.task"
    if not path.is_file():
        pytest.skip("AUTOLab v1.0 is not materialized")
    block = re.search(r"\(:fixtures(.*?)\n  \)", path.read_text(), re.S).group(1)
    return re.findall(r"^\s*(\w+)\s+-\s", block, re.M)


def test_cable_runs_both_levels_through_the_v1_graph():
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    binding = BINDINGS["cable_route/gt"]
    # Two arms, as the suite's (:arms 2) and the weave's support hand need; the linear_4310 hand.
    assert binding.robot_profile == "yam_bimanual_linear"
    assert dict(binding.environment) == {"ROBOSIMSTUDIO_SEGMENTATION_BACKEND": "palette"}
    levels = {_rss_policy(p)["gap_inputs"]["autolab_task"]: p for p in CABLE_CONFIGS}
    assert set(levels) == {"cable_route/port_easy", "cable_route/port_medium"}


@pytest.mark.parametrize("path", CABLE_CONFIGS, ids=lambda p: p.stem)
def test_cable_experiments_pass_exactly_the_documents_fixtures(path):
    policy = _rss_policy(path)
    fixtures = _document_fixtures(policy["gap_inputs"]["autolab_task"])
    statics = policy["gap_scene_statics"]
    assert sorted(statics) == sorted(fixtures)
    # Arena's pegs, anchor and port -- never the cable, whose state the graph must perceive.
    pegs = sorted(n for n in fixtures if n.startswith("peg"))
    assert {k: statics[k] for k in pegs} == {n: f"peg_{n[3:]}" for n in pegs}
    assert (statics["anchor"], statics["port"]) == ("anchor", "port")
    assert policy["gap_work_arm"] == "right"
    # The graph reads its cameras by the rig's names.
    assert {policy["gap_bridge_camera"], policy["gap_wrist_bridge_camera"], policy["gap_aux_bridge_camera"]} == {
        "overhead", "wrist", "cable"}


def test_scene_statics_reach_the_graph_process():
    from isaaclab_arena_gap import gap_process

    try:
        from isaaclab_arena_gap.gap_remote_policy import GapRemotePolicyArgs
    except ImportError:
        pytest.skip("the policy config needs Arena's environment")

    env = {"CAP_GAP_SCENE_STATICS": "stale"}
    gap_process.apply_adapter_tool_env(GapRemotePolicyArgs(), env)
    assert "CAP_GAP_SCENE_STATICS" not in env
    pose = [0.1, 0.2, 0.78, 0.0, 0.0, 0.0, 1.0]
    gap_process.apply_adapter_tool_env(GapRemotePolicyArgs(gap_scene_static_poses={"peg0": pose}), env)
    assert json.loads(env["CAP_GAP_SCENE_STATICS"]) == {"peg0": pose}


def test_runner_reads_the_scene_statics(monkeypatch):
    from isaaclab_arena_gap.rss_adapter.runner import _scene_statics

    monkeypatch.delenv("CAP_GAP_SCENE_STATICS", raising=False)
    assert _scene_statics() == {}
    monkeypatch.setenv("CAP_GAP_SCENE_STATICS", json.dumps({"port": [0, 0, 0.77, 0, 0, 1, 0]}))
    assert _scene_statics() == {"port": [0, 0, 0.77, 0, 0, 1, 0]}


def test_only_kinematic_fixtures_are_measured():
    torch = pytest.importorskip("torch")
    from isaaclab_arena_gap.gap_session import scene_static_poses

    def asset(position, kinematic):
        return SimpleNamespace(
            cfg=SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(kinematic_enabled=kinematic))),
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[9.0, 9.0, 9.0], position], dtype=torch.float64),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071, 0.7071]], dtype=torch.float64),
            ),
        )

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float64)

    scene = Scene(anchor=asset([5.1, -0.35, 0.767], True), cable=asset([5.0, 0.0, 0.77], False))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    poses = scene_static_poses(env, 1, {"anchor": "anchor"})
    assert poses["anchor"] == pytest.approx([0.1, -0.35, 0.767, 0.0, 0.0, 0.7071, 0.7071])
    with pytest.raises(ValueError, match="not a kinematic fixture"):
        scene_static_poses(env, 1, {"cable": "cable"})


def _arena_cable_mounts() -> list[tuple[float, ...]]:
    """The cable cell's two YAM bases as Arena stands them (x, y, z, qx, qy, qz, qw), both levels.

    ``cable_routing/environment.py`` reads the embodiment frame off ``medium.yaml`` for every level.
    """
    import yaml

    layout = REPO_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/cable_routing/medium.yaml"
    x, y, z = yaml.safe_load(layout.read_text())["embodiment_frame"]["midpoint_position_xyz"]
    return [(x, y + side * 0.31, z, 0.0, 0.0, 0.0, 1.0) for side in (1.0, -1.0)]


def test_cable_shadow_runs_in_autolabs_frame():
    """The weave plans against AUTOLab world constants, so its shadow stands where AUTOLab's cell does."""
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS, frame_offset

    mounts = _arena_cable_mounts()
    offset = frame_offset(BINDINGS["cable_route/gt"], mounts)
    assert offset == pytest.approx((0.5875, 0.0, -0.017))
    # plan_station_stages' final retreat, x = WEAVE_CABLE_X - 0.12, is 47.5 mm in front of the
    # bases in AUTOLab's cell. Read in Arena's world it was 635 mm in front, under the cable camera.
    retreat_x, base_x = 0.42 - 0.12, mounts[0][0]
    assert retreat_x - (base_x + offset[0]) == pytest.approx(0.0475)
    assert retreat_x - base_x == pytest.approx(0.635)


def test_other_shadows_run_in_arenas_world():
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS, frame_offset

    mounts = [(0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0)]
    for name, binding in BINDINGS.items():
        if name != "cable_route/gt":
            assert frame_offset(binding, mounts) == (0.0, 0.0, 0.0), name
    assert frame_offset(None, mounts) == (0.0, 0.0, 0.0)


def test_task_frame_is_a_translation_only():
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS, frame_offset

    yawed = [(-0.335, 0.31, 0.767, 0.0, 0.0, 0.3826834, 0.9238795)]
    with pytest.raises(ValueError, match="rotated"):
        frame_offset(BINDINGS["cable_route/gt"], yawed)


def test_autolab_cable_cell_stands_its_bases_at_the_bound_midpoint():
    python = autolab_python()
    if not python.is_file():
        pytest.skip("AUTOLab v1.0 environment is missing; run ./setup.sh")
    check = """
import vabar
vabar.install()
from robosimstudio.robots.embodiments.yam_i2rt import ACTIVE_BASE, IDLE_BASE
from robosimstudio_bench.taskbuild.registry import get_task
from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

midpoint = BINDINGS["cable_route/gt"].task_base_midpoint
assert get_task("cable_route/port_easy").embodiment == "yam_i2rt"
assert ((ACTIVE_BASE[0] + IDLE_BASE[0]) / 2, (ACTIVE_BASE[1] + IDLE_BASE[1]) / 2) == midpoint[:2], midpoint
import sys
(layout,) = [m for name, m in sys.modules.items() if name.endswith("cable_route.assets.layout")]
assert layout.WEAVE_CABLE_X == 0.42, layout.WEAVE_CABLE_X
"""
    result = subprocess.run(
        [str(python), "-c", check],
        cwd=AUTOLAB_VABAR,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "isaaclab_arena_gap/src")},
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_static_handles_are_the_documents_fixtures_at_arenas_poses():
    # RoboSimStudio is AUTOLab v1.0's, so the check runs in the environment the adapter runs in.
    python = autolab_python()
    if not python.is_file():
        pytest.skip("AUTOLab v1.0 environment is missing; run ./setup.sh")
    check = """
import vabar
vabar.install()
from robosimstudio_bench.taskbuild.registry import get_task
from isaaclab_arena_gap.rss_adapter.arena_env import static_handles

task = get_task("cable_route/port_easy")
(handle,) = static_handles(task, {"peg0": [0.0, -0.26, 0.78, 0, 0, 0, 1]}).values()
assert (handle.name, handle.static, handle.body, handle.xform[:2]) == ("peg0", True, -1, (0.0, -0.26))
(moved,) = static_handles(task, {"port": [0.0, 0.08, 0.767, 0, 0, 1, 0]}, (0.5875, 0.0, -0.017)).values()
assert all(abs(a - b) < 1e-12 for a, b in zip(moved.xform, (0.5875, 0.08, 0.75, 0, 0, 1, 0))), moved.xform

import numpy as np
from types import SimpleNamespace
from isaaclab_arena_gap.rss_adapter.arena_env import ArenaCamera

arena_pose = np.eye(4)
arena_pose[:3, 3] = (0.3321, -0.06, 1.1527)
env = SimpleNamespace(frame_offset=np.array([0.5875, 0.0, -0.017]),
                      sample=SimpleNamespace(cameras={"cable": SimpleNamespace(pose=arena_pose)}))
camera = ArenaCamera("cable", env)
# AUTOLab's own cable camera eye; Arena's measured pose is left as it was.
assert np.allclose(camera.world_matrix[:3, 3], (0.9196, -0.06, 1.1357)), camera.world_matrix
assert np.allclose(arena_pose[:3, 3], (0.3321, -0.06, 1.1527))

# A wrist camera rides a shadow body that already stands in the task frame: its pose on that
# body is what Arena measures, translation or not, however often it is read.
from scipy.spatial.transform import Rotation
body_rot = Rotation.from_euler("xyz", (0.3, -0.7, 1.1))
body = np.eye(4)
body[:3, :3], body[:3, 3] = body_rot.as_matrix(), (0.1, -0.2, 0.9)
on_body = np.eye(4)
on_body[:3, :3], on_body[:3, 3] = Rotation.from_euler("z", 0.4).as_matrix(), (0.01, 0.02, 0.03)
offset = np.array([0.5875, 0.0, -0.017])
shadow_body = body.copy()
shadow_body[:3, 3] += offset
env = SimpleNamespace(
    frame_offset=offset,
    sample=SimpleNamespace(cameras={"wrist": SimpleNamespace(pose=body @ on_body)}),
    state=SimpleNamespace(body_q=SimpleNamespace(
        numpy=lambda: np.array([[*shadow_body[:3, 3], *body_rot.as_quat()]]))),
)
wrist = ArenaCamera("wrist", env, parent_body=0)
for _ in range(2):
    assert np.allclose(wrist.world_matrix, on_body), wrist.world_matrix
try:
    static_handles(task, {"cable": [0, 0, 0, 0, 0, 0, 1]})
except KeyError as error:
    assert "no static fixture" in str(error)
else:
    raise AssertionError("a moving entity was offered as a static fixture")
"""
    result = subprocess.run(
        [str(python), "-c", check],
        cwd=AUTOLAB_VABAR,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "isaaclab_arena_gap/src")},
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
