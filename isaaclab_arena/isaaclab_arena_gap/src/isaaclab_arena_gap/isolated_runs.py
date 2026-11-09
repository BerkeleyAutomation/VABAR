"""Run each Run of a multi-Run Experiment in its own evaluator process.

Arena's typed Experiment runner executes every Run in one Isaac process. A second Run in
the same process can crash at its start (a CUDA fault in ``warp.so!wp_free_device_async``,
then a ``StaleObservationError``) and take every later Run of the batch with it. One
fresh process per Run removes that state from the benchmark, the way Arena's
``--chunk_size`` already does for legacy JSON Experiments.

The parent never starts Isaac. It writes one single-Run YAML per Run, runs this same
entry point on it with an exact output directory, then assembles the output an in-process
run writes: one directory per Run, ``arena_experiment_result.json`` and
``arena_experiment_timings.json``. A Run whose process dies is recorded as failed, with
whatever episodes it finished, and the remaining Runs still run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ISOLATED_DIRECTORY = ".isolated_runs"
RESULT_FILENAME = "arena_experiment_result.json"
TIMINGS_FILENAME = "arena_experiment_timings.json"

ChildRunner = Callable[[list[str]], int]


@dataclass(frozen=True)
class IsolationPlan:
    """A multi-Run typed Experiment to split, and the arguments to rebuild per Run."""

    experiment: dict
    output_directory: Path
    arguments: tuple[str, ...]
    """Every argument except the Experiment path and the output directory, in order."""
    continue_on_error: bool


def _option_value(arguments: Sequence[str], name: str) -> tuple[str | None, list[str]]:
    """Return ``name``'s value and the arguments without it (``--name v`` or ``--name=v``)."""
    remaining: list[str] = []
    value = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == name and index + 1 < len(arguments):
            value = arguments[index + 1]
            index += 2
            continue
        if argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
        else:
            remaining.append(argument)
        index += 1
    return value, remaining


_RUN_OVERRIDE = re.compile(r"^[+~]*runs\.([^.=]+)\.")


def plan_isolation(arguments: Sequence[str]) -> IsolationPlan | None:
    """Plan per-Run processes, or ``None`` when the arguments run one Run in-process.

    Only a typed YAML Experiment with more than one Run and an exact
    ``--experiment_output_directory`` (how evaluator tasks run) is split.
    """
    if os.environ.get("CAP_ISOLATED_RUN_CHILD") == "1":
        return None
    config, remaining = _option_value(arguments, "--experiment_config")
    if config is None:
        config, remaining = _option_value(arguments, "--eval_jobs_config")
    output, remaining = _option_value(remaining, "--experiment_output_directory")
    if config is None or output is None or not config.endswith((".yaml", ".yml")):
        return None
    if "--list_variations" in remaining:
        return None
    experiment = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    runs = experiment.get("runs") if isinstance(experiment, dict) else None
    if not isinstance(runs, dict) or len(runs) < 2:
        return None
    return IsolationPlan(
        experiment=experiment,
        output_directory=Path(output),
        arguments=tuple(remaining),
        continue_on_error="--continue_on_error" in remaining,
    )


def arguments_for_run(arguments: Sequence[str], run_name: str) -> list[str]:
    """Every argument, minus Hydra overrides addressed to another Run (``runs.<other>.…=``).

    Options and their values pass through untouched; Arena's own parser (``allow_abbrev``
    off) separates them from overrides in the child.
    """
    kept = []
    for argument in arguments:
        match = _RUN_OVERRIDE.match(argument)
        if match is not None and "=" in argument and match.group(1) != run_name:
            continue
        kept.append(argument)
    return kept


def single_run_experiment(experiment: dict, run_name: str) -> dict:
    """The Experiment with only ``run_name``; shared defaults and other keys stay."""
    return {**experiment, "runs": {run_name: experiment["runs"][run_name]}}


def _python_child_runner(arguments: list[str]) -> int:
    environment = {**os.environ, "CAP_ISOLATED_RUN_CHILD": "1"}
    command = [sys.executable, "-m", "isaaclab_arena_gap.container_runner", *arguments]
    return subprocess.run(command, env=environment, check=False).returncode


def _fallback_metadata(run: dict, known: dict[str, dict]) -> dict:
    """Metadata for a Run whose process died before writing its result.

    A sibling Run with the same environment and policy type already carries the registered
    names; otherwise the YAML's environment and policy types stand in.
    """
    environment = str((run.get("environment") or {}).get("type") or "unknown")
    policy = str((run.get("policy") or {}).get("type") or "unknown")
    for metadata in known.values():
        if metadata["environment"]["name"] == environment and metadata.get("_policy_type") == policy:
            return {"environment": metadata["environment"], "policy_variant": metadata["policy_variant"]}
    return {
        "environment": {"name": environment, "definition": environment},
        "policy_variant": policy.rsplit(".", 1)[-1],
    }


def run_isolated(plan: IsolationPlan, child_runner: ChildRunner = _python_child_runner) -> int:
    """Run every Run in its own process and assemble one Experiment output."""
    from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult

    output = plan.output_directory
    output.mkdir(parents=True, exist_ok=True)
    assert not [p for p in output.iterdir()], f"Experiment output directory '{output}' is not empty"
    staging = output / ISOLATED_DIRECTORY
    metadata_by_name: dict[str, dict] = {}
    failed: list[str] = []
    runs = plan.experiment["runs"]
    for run_name, run in runs.items():
        ArenaExperimentResult.assert_run_name_is_safe_path_component(run_name)
        child_output = staging / "outputs" / run_name
        child_output.mkdir(parents=True)
        child_config = staging / "configs" / f"{run_name}.yaml"
        child_config.parent.mkdir(parents=True, exist_ok=True)
        child_config.write_text(
            yaml.safe_dump(single_run_experiment(plan.experiment, run_name), sort_keys=False),
            encoding="utf-8",
        )
        print(f"[vabar] Run '{run_name}' in its own process", flush=True)
        exit_code = child_runner(
            [
                *arguments_for_run(plan.arguments, run_name),
                "--experiment_config",
                str(child_config),
                "--experiment_output_directory",
                str(child_output),
            ]
        )
        if (child_output / run_name).is_dir():
            shutil.move(str(child_output / run_name), str(output / run_name))
        result_path = child_output / RESULT_FILENAME
        child_runs = (
            json.loads(result_path.read_text(encoding="utf-8")).get("runs", {})
            if result_path.is_file()
            else {}
        )
        if run_name in child_runs:
            entry = child_runs[run_name]
            status = entry["status"] if exit_code == 0 else "failed"
            metadata_by_name[run_name] = {
                "environment": entry["environment"],
                "policy_variant": entry["policy_variant"],
                "status": status,
                "_policy_type": str((run.get("policy") or {}).get("type")),
            }
        else:
            print(f"[vabar] Run '{run_name}' process exited {exit_code} without a result", flush=True)
            metadata_by_name[run_name] = {"status": "failed", "_pending": run}
        if metadata_by_name[run_name]["status"] != "completed":
            failed.append(run_name)
            if not plan.continue_on_error:
                break

    known = {n: m for n, m in metadata_by_name.items() if "_pending" not in m}
    for run_name, metadata in metadata_by_name.items():
        if "_pending" in metadata:
            metadata_by_name[run_name] = {**_fallback_metadata(metadata["_pending"], known), "status": "failed"}
    for run_name in runs:
        if run_name not in metadata_by_name:  # never started: an earlier Run failed without --continue_on_error
            metadata_by_name[run_name] = {**_fallback_metadata(runs[run_name], known), "status": "failed"}
    run_metadata = {
        name: {key: value for key, value in metadata.items() if not key.startswith("_")}
        for name, metadata in metadata_by_name.items()
    }
    ArenaExperimentResult(output, run_metadata).write()
    _write_timings(output, [n for n, m in run_metadata.items() if m["status"] == "completed"])
    shutil.rmtree(staging, ignore_errors=True)
    return 1 if failed and not plan.continue_on_error else 0


def _write_timings(output: Path, completed: list[str]) -> None:
    from isaaclab_arena.evaluation.experiment_timings import aggregate_experiment_timings

    with_timings = [n for n in completed if (output / n / TIMINGS_FILENAME).is_file()]
    aggregate_experiment_timings(output, with_timings)
