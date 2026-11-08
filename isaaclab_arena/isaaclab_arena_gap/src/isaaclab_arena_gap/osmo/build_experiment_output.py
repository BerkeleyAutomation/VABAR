#!/usr/bin/env python3
"""Collect parallel or batched Arena Run outputs without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena.visualization.report import RunExecutionReport, build_report

EXPERIMENT_RUNNER_RESULT_FILE_NAME = "experiment_runner_result.json"
ARENA_EXPERIMENT_RESULT_FILE_NAME = "arena_experiment_result.json"
PROVENANCE_FILENAMES = ("image-manifest.json", "run-provenance.json")
VALID_STATUSES = frozenset({"completed", "failed"})


def _assert_safe_run_name(run_name: str) -> None:
    assert isinstance(run_name, str) and run_name.strip(), "Run name must be non-empty"
    assert run_name not in {".", ".."} and "/" not in run_name and "\\" not in run_name
    assert not PureWindowsPath(run_name).drive


def load_runner_result(
    runner_output_directory: Path, run_name: str
) -> tuple[RunExecutionReport, dict[str, object]]:
    """Select one Run from a runner result that may describe a serial batch."""
    result_path = runner_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = result["execution_status"]
    process_exit_code = result["process_exit_code"]
    assert status in VALID_STATUSES, f"Invalid runner status in {result_path}: {status!r}"
    assert (status == "completed") == (process_exit_code == 0), (
        f"Runner result for Run {run_name!r} is inconsistent: {result_path}"
    )
    runs = result["runs"]
    assert isinstance(runs, dict) and run_name in runs, (
        f"Runner result {result_path} has no metadata for Run {run_name!r}"
    )
    metadata = runs[run_name]
    assert isinstance(metadata, dict), f"Metadata for Run {run_name!r} must be an object"
    arena_result_path = runner_output_directory / ARENA_EXPERIMENT_RESULT_FILE_NAME
    # Per-Run status can refine only a wrapper execution that completed cleanly.
    if status == "completed" and arena_result_path.is_file():
        arena_result = json.loads(arena_result_path.read_text(encoding="utf-8"))
        arena_runs = arena_result["runs"]
        assert isinstance(arena_runs, dict) and run_name in arena_runs, (
            f"Arena result {arena_result_path} has no Run {run_name!r}"
        )
        arena_run = arena_runs[run_name]
        assert isinstance(arena_run, dict), (
            f"Arena result metadata for Run {run_name!r} must be an object"
        )
        status = arena_run["status"]
        assert status in VALID_STATUSES, (
            f"Invalid Run status in {arena_result_path}: {status!r}"
        )
        # A serial runner can complete its process after recording a failed Run.
        # The report needs a nonzero code for that Run even though the wrapper's
        # process-level exit code is zero.
        process_exit_code = 0 if status == "completed" else max(process_exit_code, 1)
    return RunExecutionReport(run_name, status, process_exit_code), metadata


def load_runner_directories(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, "Runner directories must be a non-empty mapping"
    result: dict[str, Path] = {}
    for run_name, directory in data.items():
        _assert_safe_run_name(run_name)
        assert isinstance(directory, str) and directory
        result[run_name] = Path(directory)
    return result


def _copy_provenance(source: Path, destination: Path) -> None:
    """Preserve the evaluator digest and manifest even for failed Runs."""
    source_paths = [source / filename for filename in PROVENANCE_FILENAMES]
    for source_path in source_paths:
        assert source_path.is_file(), (
            f"Runner output {source} is missing required provenance file "
            f"{source_path.name!r}"
        )
    for source_path in source_paths:
        if not (destination / source_path.name).exists():
            shutil.copy2(source_path, destination / source_path.name)


def collect_outputs(
    runner_directories: Mapping[str, Path], experiment_output_directory: Path
) -> tuple[list[RunExecutionReport], dict[str, dict[str, object]]]:
    reports: list[RunExecutionReport] = []
    metadata_by_name: dict[str, dict[str, object]] = {}
    for run_name, runner_directory in runner_directories.items():
        report, metadata = load_runner_result(runner_directory, run_name)
        reports.append(report)
        metadata_by_name[run_name] = {**metadata, "status": str(report.status)}
        destination = experiment_output_directory / run_name
        source = runner_directory / run_name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            assert report.status != "completed", (
                f"Completed Run {run_name!r} is missing {source}"
            )
            destination.mkdir(parents=True)
        shutil.copy2(
            runner_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME,
            destination / EXPERIMENT_RUNNER_RESULT_FILE_NAME,
        )
        _copy_provenance(runner_directory, destination)
    return sorted(reports, key=lambda report: report.run_name), metadata_by_name


def build_experiment_output(
    runner_directories: Mapping[str, Path], experiment_output_directory: Path
) -> Path:
    reports, metadata = collect_outputs(runner_directories, experiment_output_directory)
    ArenaExperimentResult(experiment_output_directory, metadata).write()
    return build_report(experiment_output_directory, run_executions=reports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-runner-output-directories-file", required=True, type=Path
    )
    parser.add_argument("--experiment-output-directory", required=True, type=Path)
    args = parser.parse_args()
    build_experiment_output(
        load_runner_directories(args.experiment_runner_output_directories_file),
        args.experiment_output_directory,
    )


if __name__ == "__main__":
    main()
