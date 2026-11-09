#!/usr/bin/env python3
"""Submit an Arena-owned typed Experiment with the exact-pin GaP image."""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENT = (
    Path(__file__).resolve().parents[3]
    / "experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml"
)
DEFAULT_ARENA_SOURCE_ROOT = (
    Path(__file__).resolve().parents[4] / "external/IsaacLab-Arena"
)
IMAGE_PATTERN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")

CAP_PYTHON = "/workspace/isaaclab_arena/.venv/bin/python"
UPSTREAM_RUNNER = "/isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py"
CAP_KIT_ARGS = (
    "--kit_args "
    "'--/app/tokens/exe-cache=/cache/kit "
    "--/app/tokens/cache=/cache/kit "
    "--/app/tokens/data=/cache/data "
    "--/app/tokens/logs=/cache/logs "
    "--/app/tokens/documents=/cache/documents'"
)
CAP_PROVENANCE_POSTFLIGHT = (
    f'{CAP_PYTHON} -c "from pathlib import Path; '
    "from isaaclab_arena_gap.container_runner import publish_image_provenance; "
    "publish_image_provenance(Path(r'{{output}}'), "
    "Path('/opt/cap/image-manifest.json'))\""
)
CAP_RUNNER = f"{CAP_PYTHON} -m isaaclab_arena_gap.container_runner {CAP_KIT_ARGS}"
UPSTREAM_CAMERA_FLAG = " --enable_cameras"
UPSTREAM_RECORD_CAMERA_VIDEO_FLAG = " --record_camera_video"
UPSTREAM_CONTINUE_ON_ERROR_FLAG = " --continue_on_error"
UPSTREAM_RESULT_MARKER = "# Publish the result for the collector"
UPSTREAM_COLLECTOR = "/isaac-sim/python.sh /tmp/arena_build_experiment_output.py"
CAP_COLLECTOR = "python /opt/cap/collector/build_experiment_output.py"
CAP_CACHE_SCRIPT_PATH = "/tmp/cap_run_with_isaac_cache.sh"
CAP_CACHE_RUNNER = f"bash {CAP_CACHE_SCRIPT_PATH}"
CAP_CACHE_SCRIPT = Path(__file__).with_name("run_with_isaac_cache.sh")
CAP_CACHE_BUILDER_TASK = "build-isaac-cache"
CAP_CACHE_WARMUP_EXPERIMENT_PATH = "/tmp/cap_cache_warmup.yaml"
CAP_CACHE_WARMUP_OUTPUT_PATH = "/tmp/cap_cache_warmup_output"
DEFAULT_CACHE_RECIPE_EPOCH = "isaac-sim-6.1.0-arena-main-v1"
CACHE_MODES = ("off", "auto", "rebuild")
# OSMO attributes GPU usage by the workflow's `workstreams` label: one string, with several
# workstreams separated by commas. Tags are listed on the Isaac OSMO allocation sheet.
DEFAULT_WORKSTREAMS = "vabar"


def _add_arena_osmo_import_path() -> None:
    """Expose Arena's repository-owned submission helpers without packaging them."""
    arena_source_root = Path(
        os.environ.get("ARENA_SOURCE_ROOT", DEFAULT_ARENA_SOURCE_ROOT)
    ).resolve()
    if not (arena_source_root / "osmo").is_dir():
        raise RuntimeError(f"Arena OSMO helpers are missing from {arena_source_root}")
    source = str(arena_source_root)
    if source not in sys.path:
        sys.path.insert(0, source)


# The evaluator image already sets OMNI_KIT_ACCEPT_EULA, ARENA_SOURCE_ROOT, and
# ARENA_ENV_ROOT. Upstream task environments set their own ISAACLAB_PATH.
CAP_TASK_ENVIRONMENT = {
    "ISAACLAB_PATH": "/workspace/isaaclab_arena/external/IsaacLab-Arena/submodules/IsaacLab",
    "PYTHONPATH": (
        "/workspace/isaaclab_arena/isaaclab_arena_gap/src:"
        "/workspace/isaaclab_arena/isaaclab_arena_vabar/src:"
        "/workspace/isaaclab_arena/external/IsaacLab-Arena"
    ),
    # Newton defers its CUDA graph capture to the first step() under RTX, and on
    # L40 GPUs that capture has taken about 225 s.  The observation
    # bridge's startup watchdog defaults to 180 s, so GaP aborted the graph
    # before the simulator ever published its second frame.  Allow the one-off
    # capture to finish; the 60 s steady-state watchdog still guards the stream
    # once it is advancing.
    "CAP_STARTUP_OBSERVATION_TIMEOUT_S": "600",
}
GAP_RUNNER_ENVIRONMENT = {
    # OSMO's runner output mount is writable by the non-root image user.  GaP
    # derives a unique per-process trace directory below this root.
    "ARENA_OUTPUT_DIR": "/osmo/data/output",
    "GAP_SKILLS_PATH": "/workspace/isaaclab_arena/external/gap/open-robot-skills",
    # robot_descriptions shells out to Git even when its exact cached revision
    # is already baked into the image.  The cache is intentionally root-owned,
    # while OSMO runs as UID 1000, so authorize this one immutable repository
    # without writing a per-user Git configuration at runtime.
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "safe.directory",
    "GIT_CONFIG_VALUE_0": "/opt/cap/robot-descriptions/example-robot-data",
}
UPSTREAM_OMNI_CREDENTIAL = "omni_svc"
CAP_OMNI_CREDENTIAL = "omni_cred"


def require_immutable_image(image: str) -> None:
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError(
            "experiment_runner.image must be an immutable "
            "image@sha256:<64 hex> reference"
        )


def _entry_script(task: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item for item in task.get("files", []) if item.get("path") == "/tmp/entry.sh"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"OSMO task {task.get('name')!r} must contain one /tmp/entry.sh"
        )
    return matches[0]


def _replace_once(script: str, old: str, new: str, task_name: str) -> str:
    if script.count(old) != 1:
        raise ValueError(
            f"OSMO task {task_name!r} does not contain the expected upstream command"
        )
    return script.replace(old, new, 1)


def _with_evaluator_log(script: str) -> str:
    """Publish full logs after Arena has checked that its output is empty."""
    return (
        "cap_evaluator_log=$(mktemp) || exit 1\n"
        "cap_run_evaluator() {\n" + script + "\n}\n"
        'cap_run_evaluator 2>&1 | tee "$cap_evaluator_log"\n'
        'cap_evaluator_status=("${PIPESTATUS[@]}")\n'
        'cp "$cap_evaluator_log" "{{output}}/evaluator-full.log" || exit 1\n'
        'chmod a+r "{{output}}/evaluator-full.log" || exit 1\n'
        'rm -f "$cap_evaluator_log"\n'
        "if (( cap_evaluator_status[0] != 0 )); then\n"
        '  exit "${cap_evaluator_status[0]}"\n'
        "fi\n"
        'exit "${cap_evaluator_status[1]}"\n'
    )


def adapt_gap_workflow(
    workflow: dict[str, Any],
    *,
    collector_image: str,
    cache_mode: str = "off",
    cache_artifact_root_url: str | None = None,
    cache_driver_branch: str | None = None,
    cache_recipe_epoch: str = DEFAULT_CACHE_RECIPE_EPOCH,
    cache_warmup_experiment: str | None = None,
    cache_artifact_exists: Callable[[str], bool] | None = None,
    workstreams: str = DEFAULT_WORKSTREAMS,
) -> dict[str, Any]:
    """Apply only the image/runtime deltas required by VABAR on Isaac Lab Arena and GaP."""
    from isaaclab_arena_gap.osmo_cache import CacheCompatibility, persistent_artifact_exists

    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}; got {cache_mode!r}")
    if cache_mode != "off" and (not cache_artifact_root_url or not cache_driver_branch):
        raise ValueError(
            "cache_artifact_root_url and explicit cache_driver_branch are required "
            "when cache_mode is enabled"
        )
    if cache_mode != "off" and cache_warmup_experiment is None:
        raise ValueError(
            "enabled cache mode requires a representative warmup experiment"
        )
    if cache_artifact_exists is None:
        cache_artifact_exists = persistent_artifact_exists
    if not workstreams.strip():
        raise ValueError("workstreams must name at least one OSMO workstream tag")

    adapted = copy.deepcopy(workflow)
    adapted["workflow"].setdefault("labels", {})["workstreams"] = workstreams
    runner_count = 0
    collector_count = 0
    runner_tasks: list[dict[str, Any]] = []
    for group in adapted["workflow"]["groups"]:
        for task in group["tasks"]:
            require_immutable_image(task["image"])
            credentials = task.setdefault("credentials", {})
            task_name = task["name"]
            entry = _entry_script(task)
            script = str(entry["contents"])
            if task_name.startswith("experiment-runner-"):
                runner_count += 1
                resource_name = task.get("resource", "default")
                resources = adapted["workflow"].get("resources")
                if not isinstance(resources, dict):
                    raise ValueError("OSMO workflow has no valid resources")
                if not isinstance(resource_name, str) or not resource_name:
                    raise ValueError(f"OSMO task {task_name!r} has no valid resource")
                resource = resources.get(resource_name)
                if not isinstance(resource, dict):
                    raise ValueError(
                        f"OSMO resource {resource_name!r} is missing for task {task_name!r}"
                    )
                platform = resource.get("platform")
                if not isinstance(platform, str) or not platform.strip():
                    raise ValueError(
                        f"OSMO resource {resource_name!r} has no valid platform"
                    )
                task.setdefault("environment", {}).update(CAP_TASK_ENVIRONMENT)
                task["environment"].update(
                    {
                        "CAP_IMAGE_DIGEST": task["image"],
                        "CAP_KIT_BUILD_ID": "f9bf0dda",
                        "CAP_OSMO_PLATFORM": platform,
                        "CUDA_CACHE_PATH": "/cache/nvidia/ComputeCache",
                        "__GL_SHADER_DISK_CACHE_PATH": "/cache/nvidia/GLCache",
                    }
                )
                task["environment"].update(GAP_RUNNER_ENVIRONMENT)
                if UPSTREAM_OMNI_CREDENTIAL in credentials:
                    if CAP_OMNI_CREDENTIAL in credentials:
                        raise ValueError(
                            f"OSMO task {task_name!r} contains both Omniverse credential names"
                        )
                    credentials[CAP_OMNI_CREDENTIAL] = credentials.pop(
                        UPSTREAM_OMNI_CREDENTIAL
                    )
                credentials["cap-vlm"] = {"GAP_VLM_API_KEY": "gap_vlm_api_key"}
                replacement_runner = CAP_RUNNER
                if cache_mode != "off":
                    replacement_runner = f"{CAP_CACHE_RUNNER} {CAP_RUNNER}"
                script = _replace_once(
                    script, UPSTREAM_RUNNER, replacement_runner, task_name
                )
                # Arena's typed Experiment runner derives camera startup from the
                # Experiment itself.  Its OSMO task still emits the removed
                # top-level CLI flag, which the pinned runner rejects before Kit
                # starts, so consume that legacy workflow-only argument here.
                script = _replace_once(script, UPSTREAM_CAMERA_FLAG, "", task_name)
                # Arena 3.0 EA tries to encode RGB-D depth observations as RGB.
                # Keep cameras available to policies without enabling its recorder.
                script = _replace_once(
                    script, UPSTREAM_RECORD_CAMERA_VIDEO_FLAG, "", task_name
                )
                script = _replace_once(
                    script,
                    UPSTREAM_RESULT_MARKER,
                    f"{CAP_PROVENANCE_POSTFLIGHT}\n\n{UPSTREAM_RESULT_MARKER}",
                    task_name,
                )
                entry["contents"] = type(entry["contents"])(_with_evaluator_log(script))
                if cache_mode != "off":
                    task.setdefault("files", []).append(
                        {
                            "path": CAP_CACHE_SCRIPT_PATH,
                            "contents": type(entry["contents"])(
                                CAP_CACHE_SCRIPT.read_text(encoding="utf-8")
                            ),
                        }
                    )
                runner_tasks.append(task)
            elif task_name == "collect-experiment-outputs":
                collector_count += 1
                require_immutable_image(collector_image)
                if collector_image == task["image"]:
                    raise ValueError(
                        "collector image must be distinct from evaluator image"
                    )
                task["image"] = collector_image
                task["environment"] = {"PYTHONUNBUFFERED": "1"}
                task["credentials"] = {}
                # Arena main mounts each complete runner output directory. This repository's
                # collector validates provenance from that directory directly.
                task["inputs"] = [
                    {"task": name}
                    for name in dict.fromkeys(
                        item["task"] for item in task.get("inputs", [])
                    )
                ]
                entry["contents"] = type(entry["contents"])(
                    _replace_once(script, UPSTREAM_COLLECTOR, CAP_COLLECTOR, task_name)
                )

    if runner_count < 1 or collector_count != 1:
        raise ValueError(
            "Arena experiment workflow must contain at least one runner and one collector"
        )
    if cache_mode != "off":
        platforms = {task["environment"]["CAP_OSMO_PLATFORM"] for task in runner_tasks}
        if len(platforms) != 1:
            raise ValueError(
                "one task-neutral cache artifact requires exactly one evaluator platform; "
                f"got {sorted(platforms)}"
            )
        compatibility = CacheCompatibility(
            recipe_epoch=cache_recipe_epoch,
            kit_build="f9bf0dda",
            platform=next(iter(platforms)),
            driver_branch=str(cache_driver_branch),
        )
        artifact_url = compatibility.artifact_url(str(cache_artifact_root_url))
        cache_hit = cache_mode == "auto" and cache_artifact_exists(artifact_url)
        common_cache_environment = {
            "CAP_ISAAC_CACHE_RECIPE_EPOCH": compatibility.recipe_epoch,
            "CAP_EXPECTED_DRIVER_BRANCH": compatibility.driver_branch,
        }
        for runner_task in runner_tasks:
            runner_task["environment"].update(common_cache_environment)
            runner_task["environment"]["CAP_ISAAC_CACHE_MODE"] = "reuse"

        if cache_hit:
            for runner_task in runner_tasks:
                input_index = len(runner_task.setdefault("inputs", []))
                runner_task["inputs"].append({"url": artifact_url})
                runner_task["environment"]["CAP_ISAAC_CACHE_ARTIFACT_DIR"] = (
                    "{{input:" + str(input_index) + "}}"
                )
            return adapted

        if not artifact_url.startswith("swift://"):
            raise ValueError(
                "cache miss/rebuild publication requires a writable swift:// artifact root; "
                f"got {cache_artifact_root_url!r}"
            )

        first_runner = runner_tasks[0]
        builder_environment = {
            **CAP_TASK_ENVIRONMENT,
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            **{
                name: first_runner["environment"][name]
                for name in (
                    "CAP_IMAGE_DIGEST",
                    "CAP_KIT_BUILD_ID",
                    "CAP_OSMO_PLATFORM",
                    "CUDA_CACHE_PATH",
                    "__GL_SHADER_DISK_CACHE_PATH",
                )
            },
            **common_cache_environment,
            "CAP_ISAAC_CACHE_MODE": "builder",
            "CAP_ISAAC_CACHE_ARTIFACT_DIR": "{{output}}",
        }
        entry_contents = "\n".join(
            (
                "set -euo pipefail",
                f"rm -rf {CAP_CACHE_WARMUP_OUTPUT_PATH}",
                (
                    f"{CAP_CACHE_RUNNER} {CAP_RUNNER} "
                    f"--experiment_config {CAP_CACHE_WARMUP_EXPERIMENT_PATH} "
                    f"--experiment_output_directory {CAP_CACHE_WARMUP_OUTPUT_PATH} "
                    "--viz none"
                ),
                "",
            )
        )
        builder = {
            "name": CAP_CACHE_BUILDER_TASK,
            "args": ["/tmp/entry.sh"],
            "command": ["bash"],
            "credentials": {
                name: copy.deepcopy(mapping)
                for name, mapping in first_runner.get("credentials", {}).items()
                if name == CAP_OMNI_CREDENTIAL
            },
            "downloadType": "download",
            "environment": builder_environment,
            "files": [
                {"path": "/tmp/entry.sh", "contents": entry_contents},
                {
                    "path": CAP_CACHE_WARMUP_EXPERIMENT_PATH,
                    "contents": cache_warmup_experiment,
                },
                {
                    "path": CAP_CACHE_SCRIPT_PATH,
                    "contents": CAP_CACHE_SCRIPT.read_text(encoding="utf-8"),
                },
            ],
            "image": first_runner["image"],
            "inputs": [],
            "outputs": [{"url": artifact_url}],
            "lead": True,
            "resource": first_runner.get("resource", "default"),
        }
        for runner_task in runner_tasks:
            runner_task.setdefault("inputs", []).append(
                {"task": CAP_CACHE_BUILDER_TASK}
            )
            runner_task["environment"]["CAP_ISAAC_CACHE_ARTIFACT_DIR"] = (
                "{{input:" + CAP_CACHE_BUILDER_TASK + "}}"
            )
        adapted["workflow"]["groups"].insert(
            0,
            {"name": "isaac-cache-builder", "tasks": [builder]},
        )
    return adapted


def build_gap_workflow(
    experiment_cfg_path: str | Path,
    overrides: list[str] | None = None,
    *,
    collector_image: str,
    serial_runs: bool = False,
    cache_mode: str = "off",
    cache_artifact_root_url: str | None = None,
    cache_driver_branch: str | None = None,
    cache_recipe_epoch: str = DEFAULT_CACHE_RECIPE_EPOCH,
    workstreams: str = DEFAULT_WORKSTREAMS,
) -> tuple[Any, Any]:
    """Compose an upstream Arena workflow and add the narrow GaP adapter."""
    _add_arena_osmo_import_path()
    from isaaclab_arena_gap.arena.registration import register_arena_components
    from isaaclab_arena_gap.gap_remote_policy import GapRemotePolicyArgs
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
    from osmo.submit_arena_experiment import build_arena_experiment_submission_cfg
    from osmo.tasks.experiment_runner_task import ExperimentRunnerTask
    from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow

    register_arena_components()
    submission_cfg = build_arena_experiment_submission_cfg(
        experiment_cfg_path=experiment_cfg_path,
        overrides=overrides,
    )
    require_immutable_image(submission_cfg.experiment_runner.image)
    require_immutable_image(collector_image)
    non_gap_runs = [
        run_name
        for run_name, run_cfg in submission_cfg.experiment_cfg.runs.items()
        if not isinstance(run_cfg.policy, GapRemotePolicyArgs)
    ]
    if non_gap_runs:
        raise ValueError(
            "submit_gap_experiment accepts only GapRemotePolicy runs; got "
            f"{non_gap_runs}"
        )

    cache_warmup_experiment = None
    if cache_mode != "off":
        from dataclasses import replace

        from isaaclab_arena_gap.hold_policy import CapHoldPolicyArgs
        from isaaclab_arena.evaluation.arena_run import RolloutLimitCfg
        from isaaclab_arena.hydra.typed_experiment_serializer import (
            serialize_arena_experiment_to_yaml,
        )

        representative_run = copy.deepcopy(
            next(iter(submission_cfg.experiment_cfg.runs.values()))
        )
        representative_run = replace(
            representative_run,
            name="isaac_cache_warmup",
            policy=CapHoldPolicyArgs(),
            rollout_limit=RolloutLimitCfg(num_steps=8),
            num_rebuilds=1,
        )
        cache_warmup_experiment = serialize_arena_experiment_to_yaml(
            ArenaExperimentCfg(runs={representative_run.name: representative_run})
        )

    class GapArenaExperimentWorkflow(ArenaExperimentWorkflow):
        def _get_group_dicts(self) -> list[dict[str, Any]]:
            if not serial_runs:
                return super()._get_group_dicts()
            task_name = "experiment-runner-0"
            task = ExperimentRunnerTask(
                task_cfg=self.task_cfg,
                experiment_cfg=copy.deepcopy(self.experiment_cfg),
                lead=True,
                task_name=task_name,
                published_output_url=(
                    f"{self.workflow_cfg.output_url.rstrip('/')}/runner-outputs/{task_name}"
                ),
            )
            task_dict = task.create_task_dict()
            entry = _entry_script(task_dict)
            entry["contents"] = type(entry["contents"])(
                _replace_once(
                    str(entry["contents"]),
                    UPSTREAM_CAMERA_FLAG,
                    f"{UPSTREAM_CONTINUE_ON_ERROR_FLAG}{UPSTREAM_CAMERA_FLAG}",
                    task_name,
                )
            )
            runner_names = {
                run_name: task_name for run_name in self.experiment_cfg.runs
            }
            return [
                {"name": "arena-runs-serial", "tasks": [task_dict]},
                self._create_experiment_output_group_dict(runner_names),
            ]

        def _create_run_group_dict(
            self,
            run_index: int,
            run_name: str,
            run_config: Any,
        ) -> tuple[dict[str, Any], str]:
            assert isinstance(run_config.policy, GapRemotePolicyArgs)
            task_name = f"experiment-runner-{run_index}"
            task = ExperimentRunnerTask(
                task_cfg=self.task_cfg,
                experiment_cfg=ArenaExperimentCfg(
                    runs={run_name: copy.deepcopy(run_config)}
                ),
                lead=True,
                task_name=task_name,
                published_output_url=(
                    f"{self.workflow_cfg.output_url.rstrip('/')}/runner-outputs/{task_name}"
                ),
            )
            return {
                "name": f"arena-run-{run_index}",
                "tasks": [task.create_task_dict()],
            }, task_name

        def generate_workflow(self) -> dict[str, Any]:
            return adapt_gap_workflow(
                super().generate_workflow(),
                collector_image=collector_image,
                cache_mode=cache_mode,
                cache_artifact_root_url=cache_artifact_root_url,
                cache_driver_branch=cache_driver_branch,
                cache_recipe_epoch=cache_recipe_epoch,
                cache_warmup_experiment=cache_warmup_experiment,
                workstreams=workstreams,
            )

    workflow = GapArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo,
        experiment_cfg=submission_cfg.experiment_cfg,
        task_cfg=submission_cfg.experiment_runner,
    )
    return submission_cfg, workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s --experiment_cfg PATH [--dry_run] [OVERRIDE ...]",
    )
    parser.add_argument("--experiment_cfg", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--list_overrides", action="store_true")
    parser.add_argument("--collector_image", required=True)
    parser.add_argument("--serial_runs", action="store_true")
    parser.add_argument("--cache_mode", choices=CACHE_MODES, default="off")
    parser.add_argument("--cache_artifact_root_url")
    parser.add_argument("--cache_driver_branch")
    parser.add_argument(
        "--cache_recipe_epoch",
        default=DEFAULT_CACHE_RECIPE_EPOCH,
    )
    parser.add_argument(
        "--workstreams",
        default=DEFAULT_WORKSTREAMS,
        help="OSMO workstream tag(s) for GPU attribution; separate several with commas",
    )
    parser.allow_abbrev = False
    return parser


def main(cli_args: list[str] | None = None) -> int:
    parser = _parser()
    args, overrides = parser.parse_known_args(cli_args)

    _add_arena_osmo_import_path()
    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
    from osmo.submit_arena_experiment import submission_cfg_to_str

    assert_hydra_overrides(overrides, parser)
    if args.dry_run:
        overrides = [*overrides, "osmo.dry_run=true"]
    submission_cfg, workflow = build_gap_workflow(
        args.experiment_cfg,
        overrides,
        collector_image=args.collector_image,
        serial_runs=args.serial_runs,
        cache_mode=args.cache_mode,
        cache_artifact_root_url=args.cache_artifact_root_url,
        cache_driver_branch=args.cache_driver_branch,
        cache_recipe_epoch=args.cache_recipe_epoch,
        workstreams=args.workstreams,
    )
    if args.list_overrides:
        print(submission_cfg_to_str(submission_cfg), end="")
        return 0
    return workflow.submit_workflow().returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
