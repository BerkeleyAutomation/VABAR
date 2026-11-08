"""All shipped baseline Arena Experiments satisfy one catalog contract."""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path

import pytest
import yaml
from _experiment_configs import CONFIG_ROOT, load_run
from isaaclab_arena_gap.task_sources import autolab_root

ROOT = CONFIG_ROOT.parent
WORKSPACE_ROOT = ROOT.parent
LIVE_CHECK = Path(__file__).with_name("arena_experiment_smoke_check.py")
EXPERIMENT_CONFIGS = tuple(sorted(CONFIG_ROOT.glob("*/*.yaml")))
CONFIGS = tuple(sorted(CONFIG_ROOT.glob("*/*_baseline.yaml")))
GRAPH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
LIVE = os.environ.get("ISAAC_CAP_RUN_LIVE_TESTS") == "1"
# Environment -> (robot profile, graph adaptation, AUTOLab document) each GaP baseline
# must declare. Every baseline runs AUTOLab's graph through the RoboSimStudio adapter
# (isaaclab_arena_gap.rss_adapter); ``None`` is the binding's default document. Kept
# explicit so a dropped adaptation, swapped profile or changed document fails here.
EXPECTED_GAP_BASELINES = {
    "vabar_tool_sorting__easy_1": ("fr3", None, None),
    "vabar_tool_sorting__easy_2": ("fr3", None, None),
    "vabar_tool_sorting__easy_3": ("fr3", None, None),
    "vabar_cable_routing__easy": ("yam_bimanual_linear", None, "cable_route/port_easy"),
    "vabar_cable_routing__medium": ("yam_bimanual_linear", None, "cable_route/port_medium"),
    "vabar_gear_insertion__easy": ("fr3", None, None),
    "vabar_gear_insertion__easy_pair": ("fr3", None, "gear_mesh/gearmesh-easy-pair-01"),
    "vabar_gear_insertion__medium_train": ("fr3", None, "gear_mesh/gearmesh-medium-train-001"),
    "vabar_usbc_insertion__easy": ("yam_bimanual_linear", "rss_usbc_insert_easy", "usbc_insert/insert_easy"),
    "vabar_usbc_insertion__medium": ("yam_bimanual_linear", "rss_usbc_insert_medium", "usbc_insert/insert_medium"),
    "vabar_tool_hanging__plier_easy": ("yam_bimanual_wide", None, "tool_hanging/pliers_level1"),
    "vabar_tool_hanging__plier_medium": ("yam_bimanual_wide", None, "tool_hanging/pliers_level2"),
    "vabar_tool_hanging__scissors_easy": ("yam_bimanual_wide", None, "tool_hanging/scissors_level1"),
    "vabar_tool_hanging__scissors_medium": ("yam_bimanual_wide", None, "tool_hanging/scissors_level2"),
    "vabar_tool_hanging__screwdriver_easy": ("yam_bimanual_wide", None, "tool_hanging/screwdrivers_level1"),
    "vabar_tool_hanging__screwdriver_medium": ("yam_bimanual_wide", None, "tool_hanging/screwdrivers_level2"),
    "vabar_tool_hanging__wrench_easy": ("yam_bimanual_wide", "rss_tool_hanging_artist_wrench", None),
    "vabar_tool_hanging__wrench_medium": ("yam_bimanual_wide", "rss_tool_hanging_artist_wrench_level2", "tool_hanging/wrenches_level2"),
    "vabar_syringe_disposal__single": ("fr3", None, "syringe_packing/tray_pair"),
    "vabar_syringe_disposal__both": ("fr3", None, "syringe_packing/tray_pair_both_t00"),
    "vabar_syringe_disposal__designated": ("fr3", None, "syringe_packing/tray_pair"),
    "vabar_syringe_disposal__cluttered": ("fr3", None, "syringe_packing/tray_clutter_t00"),
}
ADAPTER_LAUNCHER = "isaaclab_arena_gap/scripts/run_rss_graph.sh"
EXPECTED_ENVIRONMENTS = set(EXPECTED_GAP_BASELINES)


def _run(path: Path) -> dict:
    """Expose one typed Run in the legacy-shaped view used by these assertions."""
    name, run = load_run(path)
    environment = dict(run["environment"])
    environment_type = environment.pop("type")
    builder = dict(run.get("environment_builder") or {})
    policy = dict(run["policy"])
    policy_type = policy.pop("type")
    rollout_limit = run.get("rollout_limit") or {}
    return {
        "name": name,
        "arena_env_args": {
            "environment": environment_type,
            **environment,
            **builder,
        },
        "policy_type": policy_type,
        "policy_config_dict": policy,
        "num_steps": rollout_limit.get("num_steps"),
        "num_episodes": rollout_limit.get("num_episodes"),
        "num_rebuilds": run.get("num_rebuilds", 1),
        "variations": run.get("variations", {}),
    }


def test_every_baseline_experiment_has_reproducible_placement() -> None:
    for config in CONFIGS:
        assert _run(config)["arena_env_args"]["placement_seed"] == 42, config


FR3_TABLE_TOP_Z = 0.78
"""The FR3 workcell table top in Arena's world frame, 0.132 m below the arm base."""
AUTOLAB_PANDA_HOME_JOINTS = [0.0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483]
"""AUTOLab's FR3 home, which its connector reads the resting tool height from."""
SCENE_ROOT = WORKSPACE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"


def test_fr3_experiments_declare_the_workcell_table_top() -> None:
    fr3 = [
        (config, name, run.get("policy") or {})
        for config in EXPERIMENT_CONFIGS
        for name, run in yaml.safe_load(config.read_text(encoding="utf-8"))["runs"].items()
        if (run.get("policy") or {}).get("robot_profile") == "fr3"
    ]
    assert fr3
    for config, name, policy in fr3:
        assert policy.get("gap_workspace", {}).get("surface_z") == FR3_TABLE_TOP_Z, (config, name)


def test_fr3_scenes_start_at_autolab_home() -> None:
    scenes = [
        path for path in sorted(SCENE_ROOT.glob("*/*.yaml"))
        if (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("embodiment", {}).get("registry_name")
        == "vabar__fr3_robotiq_2f85"
    ]
    assert len(scenes) == 10
    for scene in scenes:
        params = yaml.safe_load(scene.read_text(encoding="utf-8"))["embodiment"]["params"]
        assert params.get("initial_joint_pose") == pytest.approx(AUTOLAB_PANDA_HOME_JOINTS), scene


def test_experiments_leave_reset_rerendering_to_the_environment() -> None:
    """Environment specs own num_rerenders_on_reset; runs must not override it."""
    for config in EXPERIMENT_CONFIGS:
        experiment = yaml.safe_load(config.read_text(encoding="utf-8"))
        for run in experiment["runs"].values():
            assert "num_rerenders_on_reset" not in run.get("environment_builder", {})


def test_experiment_filenames_match_run_names() -> None:
    for config in CONFIGS:
        expected_run_name = config.stem.removesuffix("_baseline")
        experiment = yaml.safe_load(config.read_text(encoding="utf-8"))
        assert set(experiment["runs"]) == {expected_run_name}, config


def _policy_type(job: dict):
    module_name, _, class_name = job["policy_type"].rpartition(".")
    assert module_name and class_name
    module = importlib.import_module(module_name)
    return getattr(module, class_name), getattr(module, f"{class_name}Args")


def test_experiment_catalog_has_unique_run_names_and_supported_policy_families() -> (
    None
):
    runs = [_run(path) for path in CONFIGS]
    names = [run["name"] for run in runs]
    assert names and len(names) == len(set(names))
    assert {run["policy_type"] for run in runs} == {
        "isaaclab_arena_gap.gap_remote_policy.GapRemotePolicy"
    }


def test_cable_routing_campaign_keeps_qualification_settings_and_selected_order() -> (
    None
):
    folder = CONFIG_ROOT / "cable_routing"
    baseline = next(
        iter(
            yaml.safe_load(
                (folder / "vabar_cable_routing_easy_gap_baseline.yaml").read_text()
            )["runs"].values()
        )
    )
    campaign = yaml.safe_load(
        (folder / "vabar_cable_routing_easy_gap_published_layouts.yaml").read_text()
    )
    shared = campaign["shared"]
    for key in ("environment_builder", "policy", "rollout_limit", "num_rebuilds"):
        assert shared[key] == baseline[key]
    assert shared["environment"] == {
        key: value
        for key, value in baseline["environment"].items()
        if key != "layout_seed"
    }
    manifest = autolab_root() / "benchmark/cable_tasks/dataset/manifest.json"
    if not manifest.is_file():
        pytest.skip("RSS cable dataset is not materialized; run ./setup.sh")
    case = next(
        c
        for c in json.loads(manifest.read_text())["cases"]
        if c["name"] == "cablerouting-easy-routecable"
    )
    seeds = {s["name"]: s["episode"]["seed"] for s in case["scenes"]}
    assert list(campaign["runs"].values()) == [
        {"environment": {"layout_seed": seeds[name]}} for name in case["selected_tests"]
    ]


# Benchmark seed -> AUTOLab document for the gear families: the documents the RSS
# comparison runs used. Easy has no document to pin; it pins the tooth counts of
# AUTOLab's gearmesh-easy-single-01..10, in order (our choice, not an AUTOLab mapping).
GEAR_SEED_DOCUMENTS = {
    "easy_pair": dict(zip(range(42, 52), "15 59 18 06 48 23 03 19 47 35".split())),
    "medium_train": dict(zip(range(42, 52), "046 054 050 009 026 040 092 010 041 083".split())),
}
GEAR_EASY_SEED_TEETH = dict(zip(range(42, 52), (20, 20, 24, 24, 16, 16, 16, 24, 24, 20)))


def _gear_campaign(stem: str) -> tuple[dict, dict]:
    folder = CONFIG_ROOT / "gear_insertion"
    (baseline,) = yaml.safe_load((folder / f"vabar_gear_insertion_{stem}_gap_baseline.yaml").read_text())[
        "runs"
    ].values()
    campaign = yaml.safe_load((folder / f"vabar_gear_insertion_{stem}_gap_published_layouts.yaml").read_text())
    return baseline, campaign


@pytest.mark.parametrize("family", sorted(GEAR_SEED_DOCUMENTS))
def test_gear_campaigns_pin_one_document_per_seed(family: str) -> None:
    """Each seed builds and tells the graph the same AUTOLab document."""
    baseline, campaign = _gear_campaign(family)
    shared = campaign["shared"]
    # Seed sweeps clone the baseline and change only the seeds, so it stays unpinned:
    # the seeded draw picks its row.
    assert "layout_names" not in baseline["environment"]
    for key in ("rollout_limit", "num_rebuilds"):
        assert shared[key] == baseline[key]
    assert shared["environment"] == baseline["environment"]
    assert shared["policy"] == {k: v for k, v in baseline["policy"].items() if k != "gap_inputs"}
    prefix = "easy-pair" if family == "easy_pair" else "medium-train"
    expected = {
        f"vabar_gear_insertion_{family}_gap_s{seed}": {
            "environment": {"layout_names": [f"gearmesh-{prefix}-{doc}"]},
            "environment_builder": {"placement_seed": seed, "seed": seed},
            "policy": {"gap_inputs": {"autolab_task": f"gear_mesh/gearmesh-{prefix}-{doc}"}},
        }
        for seed, doc in GEAR_SEED_DOCUMENTS[family].items()
    }
    assert campaign["runs"] == expected


def test_gear_easy_campaign_pins_one_tooth_count_per_seed() -> None:
    baseline, campaign = _gear_campaign("easy")
    shared = campaign["shared"]
    # The baseline draws its tooth count from the seed; only the campaign pins it.
    assert "variations" not in baseline
    assert shared == {k: v for k, v in baseline.items() if k != "environment_builder"}
    assert campaign["runs"] == {
        f"vabar_gear_insertion_easy_gap_s{seed}": {
            "environment_builder": {"placement_seed": seed, "seed": seed},
            "variations": {"board": {"gear_family": {"tooth_counts": [teeth]}}},
        }
        for seed, teeth in GEAR_EASY_SEED_TEETH.items()
    }
    singles = autolab_root() / "tasks/gear_mesh"
    if not singles.is_dir():
        pytest.skip("AUTOLab tasks are not materialized")
    for index, teeth in enumerate(GEAR_EASY_SEED_TEETH.values(), start=1):
        text = (singles / f"gearmesh-easy-single-{index:02d}.task").read_text()
        assert f"gear_a - gear_wheel_{teeth}t" in text


@pytest.mark.parametrize(
    "config", sorted((CONFIG_ROOT / "tool_hanging").glob("*.yaml")), ids=lambda path: path.stem
)
def test_tool_hanging_scores_where_the_graph_leaves_the_tool(config: Path) -> None:
    """v1.0 scores the tool-hanging state 60 steps after the graph ends.

    The RSS solution holds those settle steps itself before it closes, so Arena adds none.
    """
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    assert "settle_steps" not in BINDINGS["tool_hanging/gap_perception"].flags
    (policy,) = _policies(config)
    assert policy["graph_success_settle_s"] == 0.0
    assert policy["graph_failure_settle_s"] == 0.0


@pytest.mark.parametrize(
    "config", sorted((CONFIG_ROOT / "cable_routing").glob("*.yaml")), ids=lambda path: path.stem
)
def test_cable_routing_scores_where_autolab_does(config: Path) -> None:
    """v1.0 scores cable 60 steps after the graph ends (--settle-steps 60).

    The RSS solution steps those settle steps itself before it closes, so Arena adds none.
    """
    from isaaclab_arena_gap.rss_adapter.runner import BINDINGS

    assert BINDINGS["cable_route/gt"].flags["settle_steps"] == 60
    policies = _policies(config)
    assert policies, config
    for policy in policies:
        assert policy["graph_success_settle_s"] == 0.0
        assert policy["graph_failure_settle_s"] == 0.0


def _policies(config: Path) -> list[dict]:
    """Every policy an experiment config declares: each run's over a campaign's shared one, and the shared one."""
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    shared = document.get("shared", {}).get("policy")
    policies = [
        {**(shared or {}), **run["policy"]} for run in document["runs"].values() if "policy" in run
    ]
    if shared is not None:
        policies.append(shared)
    return policies


@pytest.mark.parametrize("config", EXPERIMENT_CONFIGS, ids=lambda path: path.stem)
def test_no_experiment_runs_the_vendored_gap_path(config: Path) -> None:
    """The adapter is the only path: its launcher, an ``rss_*`` patch at most, a bound robot profile."""
    policies = _policies(config)
    assert policies, config
    for policy in policies:
        assert policy.get("gap_launcher") == ADAPTER_LAUNCHER, config
        assert str(policy.get("gap_adaptation", "rss_")).startswith("rss_"), config
        assert policy["robot_profile"] in {"fr3", "yam_bimanual_wide", "yam_bimanual_linear"}, config


def test_every_environment_has_one_managed_gap_baseline() -> None:
    runs = [_run(config) for config in CONFIGS]
    gap_runs = [run for run in runs if run["policy_type"].endswith("GapRemotePolicy")]
    environments = [run["arena_env_args"]["environment"] for run in gap_runs]

    assert set(environments) == EXPECTED_ENVIRONMENTS
    assert len(environments) == len(set(environments))
    assert all(
        re.fullmatch(r"vabar_(?:[a-z_]+__[a-z0-9_]+|tool_sorting_easy_[123])", name)
        for name in environments
    )
    assert all("newton" not in name and "yam" not in name for name in environments)
    for run in gap_runs:
        policy = run["policy_config_dict"]
        assert policy["auto_spawn"] is True
        assert policy["gap_graph"]
        assert "gap_port" not in policy
        profile, adaptation, document = EXPECTED_GAP_BASELINES[
            run["arena_env_args"]["environment"]
        ]
        assert policy["gap_launcher"] == ADAPTER_LAUNCHER, run["name"]
        assert policy["robot_profile"] == profile, run["name"]
        assert policy.get("gap_adaptation") == adaptation, run["name"]
        assert policy.get("gap_inputs", {}).get("autolab_task") == document, run["name"]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_run_references_resolve_and_are_single_environment(config: Path) -> None:
    job = _run(config)
    assert job["num_episodes"] == 1
    assert job["num_rebuilds"] == 1
    env_args = job["arena_env_args"]
    assert isinstance(env_args, dict) and env_args.get("environment")
    assert env_args.get("num_envs", 1) == 1
    policy = job["policy_config_dict"]
    assert isinstance(policy, dict)

    plan_path = policy.get("plan_path")
    if plan_path:
        resolved = ROOT / plan_path
        assert resolved.is_file(), resolved
        plan = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        assert isinstance(plan, dict) and plan

    graph_path = policy.get("graph_path")
    if graph_path:
        resolved = (
            WORKSPACE_ROOT
            / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"
            / graph_path
        )
        assert resolved.is_file(), resolved

    gap_graph = policy.get("gap_graph")
    if gap_graph:
        assert GRAPH_NAME.fullmatch(gap_graph)
        adaptation = policy.get("gap_adaptation")
        if adaptation:
            assert (ROOT / "graph_adaptations" / f"{adaptation}.patch").is_file()


def test_every_run_constructs_its_registered_policy_configuration() -> None:
    pytest.importorskip("isaaclab_arena")
    if os.sys.version_info < (3, 12):
        pytest.skip("Arena policy configuration types require Python 3.12")
    for config in CONFIGS:
        job = _run(config)
        policy_type, config_type = _policy_type(job)
        assert policy_type.name
        config_type(**job["policy_config_dict"])


def _smoke_command(config: Path) -> list[str]:
    return [os.sys.executable, str(LIVE_CHECK), str(config)]


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE,
    reason="set ISAAC_CAP_RUN_LIVE_TESTS=1 to run fresh-process Arena Experiment smokes",
)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_experiment_environment_builds_resets_and_steps(config: Path) -> None:
    subprocess_utils = pytest.importorskip("isaaclab_arena.tests.utils.subprocess")
    result = subprocess_utils.run_subprocess(
        _smoke_command(config), timeout_sec=1800, capture_output=True
    )
    assert result is not None
    assert "ARENA_EXPERIMENT_SMOKE" in result.stdout, result.stderr
