"""OSMO workflow adaptation, Isaac cache wiring, and result collection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/IsaacLab-Arena"))

from isaaclab_arena_gap.osmo.submit_gap_experiment import (  # noqa: E402
    CAP_CACHE_RUNNER,
    CAP_COLLECTOR,
    CAP_PROVENANCE_POSTFLIGHT,
    CAP_RUNNER,
    DEFAULT_EXPERIMENT,
    DEFAULT_WORKSTREAMS,
    GAP_RUNNER_ENVIRONMENT,
    _with_evaluator_log,
    adapt_gap_workflow,
    build_gap_workflow,
    require_immutable_image,
)

IMAGE = "nvcr.io/example/cap@sha256:" + "a" * 64
COLLECTOR_IMAGE = "nvcr.io/example/cap-collector@sha256:" + "b" * 64


def _minimal_osmo_workflow(
    image: str,
    runner_count: int,
    *,
    platform: str = "ovx-l40s",
) -> dict:
    runner_names = [f"experiment-runner-{index}" for index in range(runner_count)]
    runner_groups = [
        {
            "name": f"arena-run-{index}",
            "tasks": [
                {
                    "name": runner_name,
                    "image": image,
                    "credentials": {},
                    "inputs": [],
                    "files": [
                        {
                            "path": "/tmp/entry.sh",
                            "contents": (
                                "/isaac-sim/python.sh "
                                "isaaclab_arena/evaluation/experiment_runner.py "
                                "--experiment_config /tmp/experiment.yaml "
                                "--enable_cameras --record_camera_video\n"
                                "# Publish the result for the collector"
                            ),
                        }
                    ],
                }
            ],
        }
        for index, runner_name in enumerate(runner_names)
    ]
    collector_group = {
        "name": "arena-experiment-output",
        "tasks": [
            {
                "name": "collect-experiment-outputs",
                "image": image,
                "credentials": {},
                "inputs": [{"task": name} for name in runner_names],
                "files": [
                    {
                        "path": "/tmp/entry.sh",
                        "contents": (
                            "/isaac-sim/python.sh /tmp/arena_build_experiment_output.py"
                        ),
                    }
                ],
            }
        ],
    }
    return {
        "workflow": {
            "resources": {"default": {"platform": platform}},
            "groups": [*runner_groups, collector_group],
        }
    }


def test_osmo_submission_requires_an_image_digest() -> None:
    require_immutable_image(IMAGE)
    with pytest.raises(ValueError, match="immutable"):
        require_immutable_image("nvcr.io/example/cap:latest")
    workflow = _minimal_osmo_workflow("nvcr.io/example/cap:latest", runner_count=1)
    with pytest.raises(ValueError, match="immutable"):
        adapt_gap_workflow(workflow, collector_image=COLLECTOR_IMAGE)


def test_default_osmo_experiment_is_shipped() -> None:
    assert DEFAULT_EXPERIMENT.is_file()


def test_osmo_adaptation_rewrites_runner_and_collector_tasks() -> None:
    workflow = _minimal_osmo_workflow(IMAGE, runner_count=1, platform="test-gpu")
    for group in workflow["workflow"]["groups"]:
        group["tasks"][0]["credentials"] = {
            "omni_svc": {"OMNI_PASS": "omni_pass", "OMNI_USER": "omni_user"}
        }
        group["tasks"][0]["environment"] = {"UPSTREAM_ONLY": "1"}
    adapted = adapt_gap_workflow(workflow, collector_image=COLLECTOR_IMAGE)

    runner = adapted["workflow"]["groups"][0]["tasks"][0]
    script = runner["files"][0]["contents"]
    assert script.index(CAP_RUNNER) < script.index(CAP_PROVENANCE_POSTFLIGHT)
    # With caching off the runner is launched directly, never through the wrapper.
    assert CAP_CACHE_RUNNER not in script
    assert [item["path"] for item in runner["files"]] == ["/tmp/entry.sh"]
    assert "--enable_cameras" not in script
    assert "--record_camera_video" not in script
    environment = runner["environment"]
    assert environment["CAP_IMAGE_DIGEST"] == IMAGE
    assert environment["CAP_OSMO_PLATFORM"] == "test-gpu"
    assert environment["CUDA_CACHE_PATH"] == "/cache/nvidia/ComputeCache"
    assert environment["__GL_SHADER_DISK_CACHE_PATH"] == "/cache/nvidia/GLCache"
    # GaP traces go to OSMO's writable output mount, and robot_descriptions may
    # use its root-owned Git cache as the non-root OSMO user.
    assert environment.items() >= GAP_RUNNER_ENVIRONMENT.items()
    assert environment["ARENA_OUTPUT_DIR"] == "/osmo/data/output"
    assert environment["GIT_CONFIG_COUNT"] == "1"
    assert environment["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert environment["GIT_CONFIG_VALUE_0"].endswith("/example-robot-data")
    assert environment["UPSTREAM_ONLY"] == "1"
    # The launcher supplies the VLM route; OSMO injects only the API key.
    assert not any(name.startswith("GAP_VLM_") for name in environment)
    assert runner["credentials"]["cap-vlm"] == {"GAP_VLM_API_KEY": "gap_vlm_api_key"}
    assert "omni_svc" not in runner["credentials"]
    assert runner["credentials"]["omni_cred"] == {
        "OMNI_PASS": "omni_pass",
        "OMNI_USER": "omni_user",
    }
    collector = adapted["workflow"]["groups"][1]["tasks"][0]
    assert collector["image"] == COLLECTOR_IMAGE
    assert collector["credentials"] == {}
    # The collector does not inherit the upstream (Isaac Sim) task environment.
    assert collector["environment"] == {"PYTHONUNBUFFERED": "1"}
    assert collector["files"][0]["contents"] == CAP_COLLECTOR


def test_osmo_workflow_is_labelled_with_its_workstream() -> None:
    workflow = _minimal_osmo_workflow(IMAGE, runner_count=1)

    adapted = adapt_gap_workflow(workflow, collector_image=COLLECTOR_IMAGE)
    assert adapted["workflow"]["labels"] == {"workstreams": DEFAULT_WORKSTREAMS}

    shared = adapt_gap_workflow(
        workflow, collector_image=COLLECTOR_IMAGE, workstreams="vabar,isaac_lab_arena"
    )
    # OSMO labels are single strings; several workstreams share one, comma-separated.
    assert shared["workflow"]["labels"] == {"workstreams": "vabar,isaac_lab_arena"}


def test_osmo_workflow_requires_a_workstream() -> None:
    workflow = _minimal_osmo_workflow(IMAGE, runner_count=1)
    with pytest.raises(ValueError, match="workstreams"):
        adapt_gap_workflow(workflow, collector_image=COLLECTOR_IMAGE, workstreams=" ")


def _assert_runner_uses_cache_wrapper(runner: dict) -> None:
    script = next(
        item["contents"] for item in runner["files"] if item["path"] == "/tmp/entry.sh"
    )
    assert f"{CAP_CACHE_RUNNER} {CAP_RUNNER}" in script
    assert any(
        item["path"] == CAP_CACHE_RUNNER.removeprefix("bash ") for item in runner["files"]
    )


@pytest.mark.parametrize("runner_count", [1, 2])
def test_osmo_cache_miss_builds_and_publishes_once(
    runner_count: int,
) -> None:
    evaluator_image, collector_image = IMAGE, COLLECTOR_IMAGE
    runner_names = [f"experiment-runner-{index}" for index in range(runner_count)]

    adapted = adapt_gap_workflow(
        _minimal_osmo_workflow(evaluator_image, runner_count),
        collector_image=collector_image,
        cache_mode="auto",
        cache_artifact_root_url="swift://cache-bucket/cap/isaac",
        cache_driver_branch="580",
        cache_recipe_epoch="hold-v7",
        cache_warmup_experiment="runs:\n  cache_warmup: {}\n",
        cache_artifact_exists=lambda _: False,
    )

    groups = adapted["workflow"]["groups"]
    builder = groups[0]["tasks"][0]
    assert groups[0]["name"] == "isaac-cache-builder"
    assert builder["name"] == "build-isaac-cache"
    assert builder["image"] == evaluator_image
    assert builder["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert builder["environment"]["CAP_IMAGE_DIGEST"] == evaluator_image
    first_runner = groups[1]["tasks"][0]
    assert builder["environment"]["CAP_ISAAC_CACHE_RECIPE_EPOCH"] == "hold-v7"
    assert first_runner["environment"]["CAP_ISAAC_CACHE_RECIPE_EPOCH"] == "hold-v7"
    assert builder["environment"]["CAP_EXPECTED_DRIVER_BRANCH"] == "580"
    assert builder["environment"]["CAP_OSMO_PLATFORM"] == "ovx-l40s"
    assert builder["inputs"] == []
    artifact_url = (
        "swift://cache-bucket/cap/isaac/schema-1/recipe-hold-v7/"
        "kit-f9bf0dda/platform-ovx-l40s/driver-580"
    )
    assert builder["outputs"] == [{"url": artifact_url}]
    assert builder["environment"]["CAP_ISAAC_CACHE_ARTIFACT_DIR"] == "{{output}}"
    builder_script = next(
        item["contents"] for item in builder["files"] if item["path"] == "/tmp/entry.sh"
    )
    assert builder["environment"]["CAP_ISAAC_CACHE_MODE"] == "builder"
    assert "--record_camera_video" not in builder_script
    assert "/tmp/cap_cache_warmup.yaml" in builder_script
    assert any(
        item["path"] == "/tmp/cap_cache_warmup.yaml"
        and item["contents"] == "runs:\n  cache_warmup: {}\n"
        for item in builder["files"]
    )
    for group in groups[1 : runner_count + 1]:
        runner = group["tasks"][0]
        _assert_runner_uses_cache_wrapper(runner)
        assert runner["inputs"] == [{"task": "build-isaac-cache"}]
        assert runner["environment"]["CAP_ISAAC_CACHE_MODE"] == "reuse"
        assert runner["environment"]["CAP_ISAAC_CACHE_ARTIFACT_DIR"] == (
            "{{input:build-isaac-cache}}"
        )
    collector = groups[-1]["tasks"][0]
    assert collector["inputs"] == [{"task": name} for name in runner_names]


def test_osmo_cache_hit_omits_gpu_builder_and_reuses_url() -> None:
    evaluator_image, collector_image = IMAGE, COLLECTOR_IMAGE

    adapted = adapt_gap_workflow(
        _minimal_osmo_workflow(evaluator_image, 2, platform="ovx-l40"),
        collector_image=collector_image,
        cache_mode="auto",
        cache_artifact_root_url="swift://cache-bucket/cap/isaac",
        cache_driver_branch="580",
        cache_warmup_experiment="runs:\n  cache_warmup: {}\n",
        cache_artifact_exists=lambda _: True,
    )

    assert [group["name"] for group in adapted["workflow"]["groups"]] == [
        "arena-run-0",
        "arena-run-1",
        "arena-experiment-output",
    ]
    assert all(
        group["tasks"][0]["inputs"]
        == [
            {
                "url": (
                    "swift://cache-bucket/cap/isaac/schema-1/"
                    "recipe-isaac-sim-6.1.0-arena-main-v1/kit-f9bf0dda/"
                    "platform-ovx-l40/driver-580"
                )
            }
        ]
        for group in adapted["workflow"]["groups"][:2]
    )
    for group in adapted["workflow"]["groups"][:2]:
        runner = group["tasks"][0]
        _assert_runner_uses_cache_wrapper(runner)
        assert runner["environment"]["CAP_ISAAC_CACHE_ARTIFACT_DIR"] == "{{input:0}}"
        assert runner["environment"]["CAP_ISAAC_CACHE_MODE"] == "reuse"


def test_enabled_osmo_cache_requires_explicit_driver_and_artifact_root() -> None:
    image, collector = IMAGE, COLLECTOR_IMAGE
    with pytest.raises(ValueError, match="explicit cache_driver_branch"):
        adapt_gap_workflow(
            _minimal_osmo_workflow(image, 1),
            collector_image=collector,
            cache_mode="auto",
            cache_artifact_root_url="swift://cache/root",
            cache_warmup_experiment="runs: {}",
        )


def test_collector_selects_runs_from_one_serial_runner_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    from isaaclab_arena_gap.osmo.build_experiment_output import collect_outputs

    runner = tmp_path / "runner"
    runner.mkdir()
    for run_name in ("camera_left", "camera_right"):
        run = runner / run_name
        run.mkdir()
        (run / "episode_results.jsonl").write_text(
            json.dumps(
                {
                    "job_name": run_name,
                    "env_id": 0,
                    "episode_in_env": 0,
                    "seed": 1,
                    "success": False,
                    "episode_length": 2,
                    "language_instruction": None,
                    "timestamp": "2026-09-04T00:00:00Z",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (runner / "experiment_runner_result.json").write_text(
        json.dumps(
            {
                "execution_status": "completed",
                "process_exit_code": 0,
                "runs": {
                    run_name: {
                        "environment": {"name": "hammer", "definition": "hammer"},
                        "policy_variant": "gap",
                    }
                    for run_name in ("camera_left", "camera_right")
                },
            }
        ),
        encoding="utf-8",
    )
    (runner / "arena_experiment_result.json").write_text(
        json.dumps(
            {
                "runs": {
                    "camera_left": {"status": "completed"},
                    "camera_right": {"status": "failed"},
                }
            }
        ),
        encoding="utf-8",
    )
    (runner / "image-manifest.json").write_text("{}\n", encoding="utf-8")
    (runner / "run-provenance.json").write_text("{}\n", encoding="utf-8")

    output = tmp_path / "output"
    reports, metadata = collect_outputs(
        {"camera_left": runner, "camera_right": runner}, output
    )
    assert [
        (report.run_name, report.status, report.process_exit_code) for report in reports
    ] == [
        ("camera_left", "completed", 0),
        ("camera_right", "failed", 1),
    ]
    assert set(metadata) == {"camera_left", "camera_right"}
    for run_name in metadata:
        assert (output / run_name / "image-manifest.json").is_file()
        assert (output / run_name / "run-provenance.json").is_file()
        assert (output / run_name / "episode_results.jsonl").is_file()
    assert metadata["camera_left"]["status"] == "completed"
    assert metadata["camera_right"]["status"] == "failed"

    (runner / "run-provenance.json").unlink()
    with pytest.raises(AssertionError, match="missing required provenance file"):
        collect_outputs({"camera_left": runner}, tmp_path / "missing-provenance")


def test_collector_preserves_wrapper_failure_after_arena_result_is_written(
    tmp_path: Path,
) -> None:
    from isaaclab_arena_gap.osmo.build_experiment_output import load_runner_result

    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "experiment_runner_result.json").write_text(
        json.dumps(
            {
                "execution_status": "failed",
                "process_exit_code": 137,
                "runs": {"syringe": {}},
            }
        ),
        encoding="utf-8",
    )
    (runner / "arena_experiment_result.json").write_text(
        json.dumps({"runs": {"syringe": {"status": "completed"}}}),
        encoding="utf-8",
    )

    report, _ = load_runner_result(runner, "syringe")

    assert report.status == "failed"
    assert report.process_exit_code == 137


@pytest.mark.parametrize("serial_runs", [False, True])
@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Arena main uses Python 3.12 types.get_original_bases",
)
def test_linked_workflow_publishes_runners_and_builds_reports(
    serial_runs: bool,
) -> None:
    _, workflow = build_gap_workflow(
        ROOT
        / "isaaclab_arena_gap/experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml",
        ["experiment_runner.image=nvcr.io/example/eval@sha256:" + "a" * 64],
        collector_image="nvcr.io/example/collector@sha256:" + "b" * 64,
        serial_runs=serial_runs,
    )
    tasks = [
        task
        for group in workflow.generate_workflow()["workflow"]["groups"]
        for task in group["tasks"]
    ]
    for task in tasks:
        if task["name"].startswith("experiment-runner-"):
            assert task["outputs"][0]["url"].endswith("/runner-outputs/" + task["name"])
        elif task["name"] == "collect-experiment-outputs":
            assert all(set(item) == {"task"} for item in task["inputs"])
            script = next(
                item["contents"]
                for item in task["files"]
                if item["path"] == "/tmp/entry.sh"
            )
            assert "python /opt/cap/collector/build_experiment_output.py" in script


@pytest.mark.parametrize("exit_code", [0, 7])
def test_evaluator_log_preserves_empty_output_and_exit_status(tmp_path, exit_code):
    output = tmp_path / "output with spaces"
    output.mkdir()
    script = (
        'test -z "$(ls -A "{{output}}")" || exit 91\n'
        'printf "evaluator stdout\\n"\n'
        'printf "evaluator stderr\\n" >&2\n'
        'touch "{{output}}/result.json"\n'
        f"exit {exit_code}\n"
    )
    wrapped = _with_evaluator_log(script).replace("{{output}}", str(output))
    result = subprocess.run(
        ["bash", "-c", wrapped], capture_output=True, text=True, check=False
    )
    assert result.returncode == exit_code
    assert (output / "result.json").exists()
    assert (output / "evaluator-full.log").stat().st_mode & 0o444 == 0o444
    assert (output / "evaluator-full.log").read_text() == (
        "evaluator stdout\nevaluator stderr\n"
    )
