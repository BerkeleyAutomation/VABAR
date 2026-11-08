"""Each Run of a multi-Run Experiment runs in its own evaluator process."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from isaaclab_arena_gap.isolated_runs import (
    arguments_for_run,
    plan_isolation,
    run_isolated,
    single_run_experiment,
)

pytest.importorskip("isaaclab_arena.evaluation.arena_experiment_result")


def _run(seed: int) -> dict:
    return {
        "environment": {"type": "vabar_tool_sorting__easy_1", "enable_cameras": True},
        "environment_builder": {"placement_seed": seed, "seed": seed},
        "policy": {"type": "isaaclab_arena_gap.gap_remote_policy.GapRemotePolicy"},
        "rollout_limit": {"num_episodes": 1},
        "num_rebuilds": 1,
    }


def _experiment(tmp_path: Path, names=("a__s44", "b__s45", "c__s46")) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump({"runs": {name: _run(44 + i) for i, name in enumerate(names)}}),
        encoding="utf-8",
    )
    return path


def _arguments(config: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--kit_args",
        "--/app/tokens/cache=/cache/kit",
        "--experiment_config",
        str(config),
        "--experiment_output_directory",
        str(output),
        "--viz",
        "none",
        "--continue_on_error",
        *extra,
    ]


class FakeChild:
    """Writes what one Arena child writes; ``crash`` names Runs whose process dies mid-episode."""

    def __init__(self, crash=()):
        self.crash = set(crash)
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> int:
        self.calls.append(arguments)
        config = Path(arguments[arguments.index("--experiment_config") + 1])
        output = Path(arguments[arguments.index("--experiment_output_directory") + 1])
        (run_name,) = yaml.safe_load(config.read_text(encoding="utf-8"))["runs"]
        run_dir = output / run_name
        run_dir.mkdir(parents=True)
        if run_name in self.crash:
            return -11  # SIGSEGV before any episode or result
        episode = {
            "job_name": run_name,
            "env_id": 0,
            "episode_in_env": 0,
            "seed": 44,
            "success": True,
            "episode_length": 7000,
            "language_instruction": "sort",
            "timestamp": "2026-11-07T00:00:00",
        }
        (run_dir / "episode_results_rebuild0.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
        (run_dir / "arena_experiment_timings.json").write_text("[]\n", encoding="utf-8")
        result = {
            "runs": {
                run_name: {
                    "environment": {"name": "vabar_tool_sorting__easy_1", "definition": "vabar_tool_sorting__easy_1"},
                    "policy_variant": "gap_remote",
                    "status": "completed",
                    "rebuilds": [{"index": 0, "episodes": [episode]}],
                }
            }
        }
        (output / "arena_experiment_result.json").write_text(json.dumps(result), encoding="utf-8")
        return 0


def test_only_multi_run_yaml_with_an_exact_output_directory_is_split(tmp_path, monkeypatch):
    monkeypatch.delenv("CAP_ISOLATED_RUN_CHILD", raising=False)
    config = _experiment(tmp_path)
    assert plan_isolation(_arguments(config, tmp_path / "out")) is not None
    (tmp_path / "one").mkdir()
    single = _experiment(tmp_path / "one", names=("only",))
    assert plan_isolation(_arguments(single, tmp_path / "out")) is None
    assert plan_isolation(["--experiment_config", str(config), "--output_base_dir", str(tmp_path)]) is None
    monkeypatch.setenv("CAP_ISOLATED_RUN_CHILD", "1")
    assert plan_isolation(_arguments(config, tmp_path / "out")) is None


def test_every_run_gets_its_own_process_and_one_experiment_output(tmp_path, monkeypatch):
    monkeypatch.delenv("CAP_ISOLATED_RUN_CHILD", raising=False)
    output = tmp_path / "out"
    plan = plan_isolation(_arguments(_experiment(tmp_path), output))
    child = FakeChild()

    assert run_isolated(plan, child_runner=child) == 0

    assert len(child.calls) == 3
    for call in child.calls:
        assert call[:2] == ["--kit_args", "--/app/tokens/cache=/cache/kit"]
        assert "--continue_on_error" in call
        assert call.count("--experiment_config") == 1
    result = json.loads((output / "arena_experiment_result.json").read_text(encoding="utf-8"))
    assert set(result["runs"]) == {"a__s44", "b__s45", "c__s46"}
    assert all(run["status"] == "completed" for run in result["runs"].values())
    assert all(len(run["rebuilds"][0]["episodes"]) == 1 for run in result["runs"].values())
    assert (output / "arena_experiment_timings.json").is_file()
    assert sorted(p.name for p in output.iterdir()) == [
        "a__s44",
        "arena_experiment_result.json",
        "arena_experiment_timings.json",
        "b__s45",
        "c__s46",
    ]


def test_a_crashed_run_is_failed_and_later_runs_still_run(tmp_path, monkeypatch):
    """A mid-batch crash: the second Run's process dies; the third must still be scored."""
    monkeypatch.delenv("CAP_ISOLATED_RUN_CHILD", raising=False)
    output = tmp_path / "out"
    plan = plan_isolation(_arguments(_experiment(tmp_path), output))

    assert run_isolated(plan, child_runner=FakeChild(crash={"b__s45"})) == 0

    runs = json.loads((output / "arena_experiment_result.json").read_text(encoding="utf-8"))["runs"]
    assert runs["a__s44"]["status"] == "completed"
    assert runs["c__s46"]["status"] == "completed"
    assert runs["b__s45"]["status"] == "failed"
    assert runs["b__s45"]["environment"]["name"] == "vabar_tool_sorting__easy_1"
    assert runs["b__s45"]["policy_variant"] == "gap_remote"


def test_without_continue_on_error_a_crash_stops_and_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("CAP_ISOLATED_RUN_CHILD", raising=False)
    output = tmp_path / "out"
    arguments = [a for a in _arguments(_experiment(tmp_path), output) if a != "--continue_on_error"]
    child = FakeChild(crash={"a__s44"})

    assert run_isolated(plan_isolation(arguments), child_runner=child) == 1

    assert len(child.calls) == 1
    runs = json.loads((output / "arena_experiment_result.json").read_text(encoding="utf-8"))["runs"]
    assert {name: run["status"] for name, run in runs.items()} == {
        "a__s44": "failed",
        "b__s45": "failed",
        "c__s46": "failed",
    }


def test_run_overrides_reach_only_their_run_and_option_values_pass_untouched():
    arguments = [
        "--kit_args",
        "--/app/tokens/cache=/cache/kit x=y",
        "--viz",
        "none",
        "runs.a.rollout_limit.num_episodes=2",
        "runs.b.policy.gap_io_timeout_s=60",
        "shared.x=1",
    ]
    assert arguments_for_run(arguments, "a") == [
        "--kit_args",
        "--/app/tokens/cache=/cache/kit x=y",
        "--viz",
        "none",
        "runs.a.rollout_limit.num_episodes=2",
        "shared.x=1",
    ]


def test_run_names_that_look_like_config_files_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.delenv("CAP_ISOLATED_RUN_CHILD", raising=False)
    output = tmp_path / "out"
    plan = plan_isolation(_arguments(_experiment(tmp_path, names=("a", "a.yaml")), output))
    assert run_isolated(plan, child_runner=FakeChild()) == 0
    runs = json.loads((output / "arena_experiment_result.json").read_text(encoding="utf-8"))["runs"]
    assert {name: run["status"] for name, run in runs.items()} == {"a": "completed", "a.yaml": "completed"}


def test_single_run_experiment_keeps_shared_defaults():
    experiment = {"shared": {"policy": {"policy_device": "cuda"}}, "runs": {"a": {}, "b": {}}}
    assert single_run_experiment(experiment, "b") == {"shared": {"policy": {"policy_device": "cuda"}}, "runs": {"b": {}}}
