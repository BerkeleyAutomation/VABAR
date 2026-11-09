"""Render five-second benchmark camera GIFs and generate the environment overview."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_ROOT = (
    WORKSPACE_ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"
)
EXPECTED_ENVIRONMENT_COUNT = 22
MAX_RENDER_ATTEMPTS = 5
RENDER_CONTRACT = "rtx-hdr-only-v1"
GIF_SIZE = (320, 180)
GIF_DURATION_SECONDS = 5.0
GIF_FPS = 10
GIF_FRAME_COUNT = int(GIF_DURATION_SECONDS * GIF_FPS)
GIF_FRAME_DURATION_MS = int(1000 / GIF_FPS)
FAMILY_ORDER = (
    "tool_sorting",
    "tool_hanging",
    "syringe_disposal",
    "usbc_insertion",
    "gear_insertion",
    "cable_routing",
)
FAMILY_TITLES = {
    "tool_sorting": "Tool sorting",
    "tool_hanging": "Tool hanging",
    "syringe_disposal": "Syringe disposal",
    "usbc_insertion": "USB-C insertion",
    "gear_insertion": "Gear insertion",
    "cable_routing": "Cable routing",
}
RESET_RANDOMIZATION = {
    "tool_sorting/easy_1.yaml": "Four tools placed on the inset bin floor; randomized XY and full yaw; fixed destination labels.",
    "tool_sorting/easy_2.yaml": "Four tools placed on the inset bin floor; randomized XY and full yaw; fixed destination labels.",
    "tool_sorting/easy_3.yaml": "Four tools placed on the inset bin floor; randomized XY and full yaw; fixed destination labels.",
    "tool_hanging/plier_easy.yaml": "One pair of pliers in a 4 x 4 cm XY region at the RSS table center; fixed heading.",
    "tool_hanging/plier_medium.yaml": (
        "Two matching pairs of pliers with +/-5 cm XY and +/-5 deg yaw; two distractors; fixed fixtures."
    ),
    "tool_hanging/scissors_easy.yaml": "One pair of scissors in a 4 x 4 cm XY region at the RSS table center; fixed heading.",
    "tool_hanging/scissors_medium.yaml": (
        "Two matching pairs of scissors with +/-5 cm XY and +/-5 deg yaw; two distractors; fixed fixtures."
    ),
    "tool_hanging/screwdriver_easy.yaml": "One screwdriver in a 4 x 4 cm XY region at the RSS table center; fixed heading.",
    "tool_hanging/screwdriver_medium.yaml": (
        "Two matching screwdrivers with +/-5 cm XY and +/-5 deg yaw; two distractors; fixed fixtures."
    ),
    "tool_hanging/wrench_easy.yaml": "One wrench in a 4 x 4 cm XY region; fixed heading.",
    "tool_hanging/wrench_medium.yaml": (
        "Two matching wrenches with +/-5 cm XY and +/-5 deg yaw; two distractors; fixed fixtures."
    ),
    "syringe_disposal/single.yaml": "Dispose of one syringe. The tray varies in a 10 x 10 cm XY region and the box in a 7 x 10 cm XY region; syringes use random on-tray positions with a 1 cm edge margin; fixed headings.",
    "syringe_disposal/both.yaml": "Dispose of both syringes, laid out as one of AUTOLab's 30 tray_pair trials: both barrels across the tray, 90 mm apart along it, the tray at 140 deg +/-25 deg. The tray varies in a 10 x 10 cm XY region and the box in a 7 x 10 cm XY region.",
    "syringe_disposal/designated.yaml": "Dispose of the red-cap syringe, leaving its blank spare. The tray varies in a 10 x 10 cm XY region and the box in a 7 x 10 cm XY region; syringes use random on-tray positions with a 1 cm edge margin; fixed headings.",
    "syringe_disposal/cluttered.yaml": "Dispose of six syringes (five in three trials, the sixth parked), one red-cap and the rest blank spares, in one of AUTOLab's clutter layouts: a row along the tray, each barrel across it with up to 35 deg of yaw jitter, neighbours allowed to cross. The tray keeps AUTOLab's 140 deg heading and varies in a 10 x 10 cm XY region; the box varies in a 7 x 10 cm XY region.",
    "cable_routing/easy.yaml": (
        "`layout_seed` jointly samples the cable shape, its anchor, and the valid "
        "two-peg board. Seed 10009 is the regression layout."
    ),
    "cable_routing/medium.yaml": (
        "`layout_seed` jointly samples the 840 mm cable, anchor, four-guide board and port."
    ),
    "gear_insertion/easy.yaml": (
        "One gear (16, 20, or 24 teeth, with the matching board) on the mat with full yaw; "
        "the board varies in a 6 x 6 cm XY region with +/-10 deg yaw."
    ),
    "gear_insertion/easy_pair.yaml": "One jointly sampled RSS two-station layout per rebuild.",
    "gear_insertion/medium_train.yaml": "One jointly sampled RSS train layout per rebuild.",
    "usbc_insertion/easy.yaml": (
        "The cabled plug moves +/-1.2 cm in X/Y with +/-8 deg yaw; the bolted port moves "
        "+/-0.5 cm across and +/-0.15 cm along the bore with +/-5 deg yaw; fixed Z."
    ),
    "usbc_insertion/medium.yaml": (
        "The plug moves +/-1.8 cm in X/Y with +/-50 deg yaw; the bulkhead moves "
        "+/-0.4 cm in X/Y with +/-3 deg yaw."
    ),
}
DOCUMENTATION_CAMERAS = (
    "chain_camera",
    "front_camera",
    "iso_camera",
    "left_wrist_camera",
    "oblique_camera",
    "right_wrist_camera",
    "top_camera",
    "wrist_camera",
)
FR3_DOCUMENTATION_CAMERAS = {
    "gear_mesh": "iso_camera",
    "syringe": "oblique_camera",
    "tool_sorting": "oblique_camera",
}
"""The documented view of each FR3 ``camera_rig``; YAM scenes use ``top_camera``."""


@dataclass(frozen=True)
class EnvironmentEntry:
    family: str
    name: str
    yaml_path: Path
    relative_yaml: Path
    camera_name: str = "top_camera"

    @property
    def animation_relative_path(self) -> Path:
        return Path("media") / self.family / f"{self.name}.gif"

    @property
    def reset_randomization(self) -> str:
        return RESET_RANDOMIZATION[self.relative_yaml.as_posix()]


def _entries_from_documents(
    environment_root: Path, documents: dict[Path, dict]
) -> tuple[EnvironmentEntry, ...]:
    discovered_paths = {path.as_posix() for path in documents}
    documented_paths = set(RESET_RANDOMIZATION)
    if discovered_paths != documented_paths:
        raise ValueError(
            "reset-randomization documentation mismatch: "
            f"missing={sorted(discovered_paths - documented_paths)} "
            f"extra={sorted(documented_paths - discovered_paths)}"
        )
    entries: list[EnvironmentEntry] = []
    family_names: set[tuple[str, str]] = set()
    for relative_yaml, document in documents.items():
        if len(relative_yaml.parts) != 2 or relative_yaml.parts[0] not in FAMILY_ORDER:
            raise ValueError(
                f"unsupported environment family for {relative_yaml.as_posix()}"
            )
        family = relative_yaml.parts[0]
        name = document.get("env_name") if isinstance(document, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{relative_yaml.as_posix()} is missing string env_name")
        if name != relative_yaml.stem:
            raise ValueError(
                f"{relative_yaml.as_posix()} env_name {name!r} must match YAML stem"
            )
        family_name = (family, name)
        if family_name in family_names:
            raise ValueError(f"duplicate environment name in {family}: {name}")
        family_names.add(family_name)
        embodiment = document.get("embodiment")
        embodiment_name = (
            embodiment.get("registry_name") if isinstance(embodiment, dict) else None
        )
        camera_name = "top_camera"
        if isinstance(embodiment_name, str) and "_fr3_" in embodiment_name:
            params = embodiment.get("params") or {}
            camera_rig = params.get("camera_rig")
            if camera_rig not in FR3_DOCUMENTATION_CAMERAS:
                raise ValueError(
                    f"{relative_yaml.as_posix()} has no documented FR3 camera_rig: "
                    f"{camera_rig!r}"
                )
            camera_name = FR3_DOCUMENTATION_CAMERAS[camera_rig]
        entries.append(
            EnvironmentEntry(
                family=family,
                name=name,
                yaml_path=environment_root / relative_yaml,
                relative_yaml=relative_yaml,
                camera_name=camera_name,
            )
        )
    family_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return tuple(
        sorted(entries, key=lambda entry: (family_index[entry.family], entry.name))
    )


def _discover_environments(
    environment_root: Path = ENVIRONMENT_ROOT,
    *,
    expected_count: int | None = EXPECTED_ENVIRONMENT_COUNT,
) -> tuple[EnvironmentEntry, ...]:
    documents: dict[Path, dict] = {}
    for path in sorted(environment_root.glob("*/*.yaml")):
        if path.name.endswith("_env_config.yaml"):
            continue
        relative = path.relative_to(environment_root)
        document = yaml.safe_load(path.read_text())
        documents[relative] = document
    entries = _entries_from_documents(environment_root, documents)
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"expected {expected_count} environment YAMLs, found {len(entries)}"
        )
    return entries


def _overview_readme(entries: tuple[EnvironmentEntry, ...]) -> str:
    sections = []
    for family in FAMILY_ORDER:
        rows = []
        for entry in entries:
            if entry.family != family:
                continue
            yaml_link = entry.relative_yaml.as_posix()
            animation_link = entry.animation_relative_path.as_posix()
            preview = (
                f'<a href="{animation_link}"><img src="{animation_link}" '
                f'alt="{entry.name} {entry.camera_name}" width="320" height="180"></a>'
            )
            rows.append(
                f"| [`{entry.name}`]({yaml_link}) | [`{yaml_link}`]({yaml_link}) | "
                f"{entry.reset_randomization} | {preview} |"
            )
        if not rows:
            raise ValueError(f"environment family has no YAMLs: {family}")
        sections.append(
            f"## {FAMILY_TITLES[family]}\n\n"
            "| Environment | Definition | Reset randomization | First five seconds |\n"
            "|---|---|---|---|\n" + "\n".join(rows)
        )
    return (
        "# Industrial benchmark environments\n\n"
        "Generated from the package-owned environment YAMLs with:\n\n"
        "```bash\n"
        "uv run tools/generate_environment_docs.py\n"
        "```\n\n"
        "Each GIF shows the first five simulated seconds after reset at 10 FPS. "
        "FR3 gear environments use "
        "`iso_camera`, other FR3 environments use `oblique_camera`, and YAM "
        "environments use `top_camera`. No floor plane "
        "is rendered; the HDR panorama supplies the visible floor background. "
        "Reset randomization covers task-relevant scene state; unlisted robots, cameras, "
        "and fixtures remain fixed.\n\n" + "\n\n".join(sections) + "\n"
    )


def _arena_roots() -> tuple[Path, Path, Path]:
    source_root = (
        Path(
            os.environ.get(
                "ARENA_SOURCE_ROOT", WORKSPACE_ROOT / "external/IsaacLab-Arena"
            )
        )
        .expanduser()
        .resolve()
    )
    env_root = (
        Path(os.environ.get("ARENA_ENV_ROOT", WORKSPACE_ROOT)).expanduser().resolve()
    )
    arena_python = (
        Path(os.environ.get("ARENA_PYTHON", env_root / ".venv/bin/python"))
        .expanduser()
        # resolve() would follow the venv's python symlink out of the venv.
        .absolute()
    )
    return source_root, env_root, arena_python


def _child_environment(source_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("ACCEPT_EULA", "Y")
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(source_root),
            str(WORKSPACE_ROOT / "isaaclab_arena_gap/src"),
            str(WORKSPACE_ROOT / "isaaclab_arena_vabar/src"),
        )
    )
    return environment


def _render_command(
    *,
    arena_python: Path,
    entry: EnvironmentEntry,
    animation_path: Path,
    report_path: Path,
) -> list[str]:
    return [
        str(arena_python),
        str(Path(__file__).resolve()),
        "--render-one",
        entry.relative_yaml.as_posix(),
        "--render-output",
        str(animation_path),
        "--render-report",
        str(report_path),
    ]


def _gif_metadata(path: Path) -> tuple[tuple[int, int], int, int]:
    from PIL import Image

    with Image.open(path) as animation:
        if animation.format != "GIF":
            raise ValueError(f"not a GIF: {path}")
        frame_count = int(getattr(animation, "n_frames", 1))
        durations_ms = []
        for frame_index in range(frame_count):
            animation.seek(frame_index)
            duration_ms = animation.info.get("duration")
            if not isinstance(duration_ms, int) or duration_ms <= 0:
                raise ValueError(f"invalid GIF frame duration: {path}")
            durations_ms.append(duration_ms)
        return animation.size, frame_count, sum(durations_ms)


def _validate_render_artifact(
    entry: EnvironmentEntry, animation_path: Path, report_path: Path
) -> dict:
    if not animation_path.is_file() or not report_path.is_file():
        raise ValueError(f"missing camera evidence for {entry.name}")
    gif_size, gif_frame_count, gif_duration_ms = _gif_metadata(animation_path)
    if gif_size != GIF_SIZE:
        raise ValueError(f"unexpected GIF dimensions for {entry.name}")
    if gif_frame_count != GIF_FRAME_COUNT:
        raise ValueError(f"unexpected GIF frame count for {entry.name}")
    if gif_duration_ms != int(GIF_DURATION_SECONDS * 1000):
        raise ValueError(f"unexpected GIF duration for {entry.name}")
    report = json.loads(report_path.read_text())
    capture_shape = report.get("capture_shape")
    if (
        not isinstance(capture_shape, list)
        or len(capture_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in capture_shape
        )
        or capture_shape[2] != 3
    ):
        raise ValueError(f"invalid capture_shape in camera report for {entry.name}")
    expected = {
        "camera": entry.camera_name,
        "environment": entry.name,
        "relative_yaml": entry.relative_yaml.as_posix(),
        "render_contract": RENDER_CONTRACT,
        "encoded_size": list(GIF_SIZE),
        "duration_s": GIF_DURATION_SECONDS,
        "fps": GIF_FPS,
        "frame_count": GIF_FRAME_COUNT,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"invalid {key} in camera report for {entry.name}")
    for key in ("min", "max", "mean", "std"):
        value = report.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"invalid {key} in camera report for {entry.name}")
    step_dt_s = report.get("step_dt_s")
    simulated_duration_s = report.get("simulated_duration_s")
    if (
        not isinstance(step_dt_s, (int, float))
        or not math.isfinite(float(step_dt_s))
        or not 0 < float(step_dt_s) <= 1.0
    ):
        raise ValueError(f"invalid step_dt_s in camera report for {entry.name}")
    if (
        not isinstance(simulated_duration_s, (int, float))
        or not math.isfinite(float(simulated_duration_s))
        or not GIF_DURATION_SECONDS - 1e-6
        <= float(simulated_duration_s)
        < GIF_DURATION_SECONDS + float(step_dt_s) + 1e-6
    ):
        raise ValueError(
            f"invalid simulated_duration_s in camera report for {entry.name}"
        )
    if report["max"] <= 0:
        raise ValueError(f"black camera frame for {entry.name}")
    if report["std"] <= 1.0:
        raise ValueError(f"uniform camera frame for {entry.name}")
    return report


def _accepted_teardown_codes() -> set[int]:
    return {
        0,
        128 + signal.SIGSEGV,
        -signal.SIGSEGV,
        128 + signal.SIGABRT,
        -signal.SIGABRT,
    }


def _render_with_retries(
    *,
    arena_python: Path,
    entry: EnvironmentEntry,
    animation_path: Path,
    report_path: Path,
    environment: dict[str, str],
) -> list[dict]:
    command = _render_command(
        arena_python=arena_python,
        entry=entry,
        animation_path=animation_path,
        report_path=report_path,
    )
    blockers: list[dict] = []
    for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
        animation_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                env=environment,
                check=False,
                timeout=600,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = None
        validation_error = None
        try:
            _validate_render_artifact(entry, animation_path, report_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            validation_error = str(exc)
        if validation_error is None and returncode in _accepted_teardown_codes():
            return blockers
        blockers.append(
            {
                "environment": entry.name,
                "attempt": attempt,
                "process_returncode": returncode,
                "validation_error": validation_error,
            }
        )
        if attempt < MAX_RENDER_ATTEMPTS:
            print(
                f"Environment {entry.name} did not produce complete evidence; "
                f"retrying in a fresh process ({attempt}/{MAX_RENDER_ATTEMPTS}).",
                flush=True,
            )
    raise RuntimeError(
        f"environment render failed after {MAX_RENDER_ATTEMPTS} attempts: "
        f"{entry.name}; "
        f"blockers={blockers}"
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_docs(
    environment_root: Path,
    entries: tuple[EnvironmentEntry, ...],
    rendered_media: Path,
) -> None:
    media = environment_root / "media"
    incoming = environment_root / f".environment-media-{uuid.uuid4().hex}.new"
    backup = environment_root / f".environment-media-{uuid.uuid4().hex}.old"
    shutil.copytree(rendered_media, incoming)
    moved_existing = False
    try:
        if media.exists():
            os.replace(media, backup)
            moved_existing = True
        os.replace(incoming, media)
        _atomic_write(environment_root / "README.md", _overview_readme(entries))
    except BaseException:
        if media.exists():
            shutil.rmtree(media)
        if moved_existing and backup.exists():
            os.replace(backup, media)
        raise
    finally:
        if incoming.exists():
            shutil.rmtree(incoming)
        if backup.exists():
            shutil.rmtree(backup)


def generate_environment_docs(
    *,
    environment_root: Path = ENVIRONMENT_ROOT,
    work_dir: Path = WORKSPACE_ROOT / "outputs/environment-docs",
    arena_python: Path | None = None,
    expected_count: int | None = EXPECTED_ENVIRONMENT_COUNT,
    resume_dir: Path | None = None,
) -> Path:
    entries = _discover_environments(environment_root, expected_count=expected_count)
    source_root, _env_root, default_python = _arena_roots()
    arena_python = arena_python or default_python
    if resume_dir is None:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_dir))
    else:
        run_dir = resume_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"resume directory does not exist: {run_dir}")
    media_root = run_dir / "media"
    report_root = run_dir / "reports"
    environment = _child_environment(source_root)
    failures: list[str] = []
    blockers: list[dict] = []
    for entry in entries:
        animation_path = media_root / entry.family / f"{entry.name}.gif"
        report_path = report_root / entry.family / f"{entry.name}.json"
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _validate_render_artifact(entry, animation_path, report_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            print(f"Reusing {entry.relative_yaml.as_posix()}", flush=True)
            continue
        print(f"Rendering {entry.relative_yaml.as_posix()}", flush=True)
        try:
            blockers.extend(
                _render_with_retries(
                    arena_python=arena_python,
                    entry=entry,
                    animation_path=animation_path,
                    report_path=report_path,
                    environment=environment,
                )
            )
        except RuntimeError as exc:
            failures.append(f"{entry.name}: {exc}")
    (run_dir / "render-campaign.json").write_text(
        json.dumps(
            {
                "environment_count": len(entries),
                "failures": failures,
                "blockers": blockers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if failures:
        raise RuntimeError(
            "environment documentation incomplete: " + "; ".join(failures)
        )
    _publish_docs(environment_root, entries, media_root)
    return run_dir


def check_environment_docs(
    environment_root: Path = ENVIRONMENT_ROOT,
    *,
    expected_count: int | None = EXPECTED_ENVIRONMENT_COUNT,
    include_media: bool = True,
) -> None:
    entries = _discover_environments(environment_root, expected_count=expected_count)
    readme = environment_root / "README.md"
    if not readme.is_file() or readme.read_text() != _overview_readme(entries):
        raise ValueError("generated environment README is missing or stale")
    if not include_media:
        return
    expected_animations = {entry.animation_relative_path for entry in entries}
    media_root = environment_root / "media"
    actual_media = (
        {
            path.relative_to(environment_root)
            for path in media_root.rglob("*")
            if path.is_file()
        }
        if media_root.is_dir()
        else set()
    )
    if actual_media != expected_animations:
        raise ValueError(
            "environment animation closure mismatch: "
            f"missing={sorted(expected_animations - actual_media)} "
            f"extra={sorted(actual_media - expected_animations)}"
        )
    for entry in entries:
        dimensions, frame_count, duration_ms = _gif_metadata(
            environment_root / entry.animation_relative_path
        )
        if (
            dimensions != GIF_SIZE
            or frame_count != GIF_FRAME_COUNT
            or duration_ms != int(GIF_DURATION_SECONDS * 1000)
        ):
            raise ValueError(
                f"unexpected camera animation metadata for {entry.name}: "
                f"size={dimensions} frames={frame_count} duration_ms={duration_ms}"
            )


def _build_arena_environment(relative_yaml: Path):
    from isaaclab_arena_vabar.environments.factories import factory_name_for
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    register_components()
    registry = EnvironmentRegistry()
    factory_name = factory_name_for(relative_yaml.as_posix(), registry)
    if factory_name is None:
        spec = ArenaEnvGraphSpec.from_yaml(str(ENVIRONMENT_ROOT / relative_yaml))
        return spec.to_arena_env(enable_cameras=True)
    factory_type = registry.get_component_by_name(factory_name)
    cfg_type = registry.get_environment_cfg_type(factory_type)
    return factory_type().build(cfg_type(enable_cameras=True))


def _retain_documentation_camera(env_cfg, camera_name: str) -> None:
    """Disable camera sensors and observations not published in the task catalog."""
    camera_observations = getattr(env_cfg.observations, "camera_obs", None)
    for candidate in DOCUMENTATION_CAMERAS:
        if candidate == camera_name:
            continue
        if hasattr(env_cfg.scene, candidate):
            setattr(env_cfg.scene, candidate, None)
        if camera_observations is None:
            continue
        for suffix in ("rgb", "distance_to_image_plane"):
            observation_name = f"{candidate}_{suffix}"
            if hasattr(camera_observations, observation_name):
                setattr(camera_observations, observation_name, None)


def _render_one(relative_yaml: Path, output: Path, report_path: Path) -> None:
    import numpy as np
    import torch
    from isaaclab_arena_gap.arena_show import hold_action
    from isaaclab_arena_gap.newton_camera_rendering import configure_newton_camera_renderers
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from PIL import Image

    entry = next(
        (
            candidate
            for candidate in _discover_environments(ENVIRONMENT_ROOT)
            if candidate.relative_yaml == relative_yaml
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"unknown environment YAML: {relative_yaml.as_posix()}")
    arena_env = _build_arena_environment(relative_yaml)
    builder_kwargs = {"num_envs": 1, "placement_seed": 0}
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(**builder_kwargs))
    _name, env_cfg, env_kwargs = builder.build_registered()
    _retain_documentation_camera(env_cfg, entry.camera_name)
    configure_newton_camera_renderers(env_cfg)
    env = None
    try:
        env = builder.make_registered(env_cfg, env_kwargs, render_mode=None)
        env.reset()
        unwrapped = env.unwrapped
        with torch.inference_mode():
            for _ in range(5):
                unwrapped.sim.render()
            camera = unwrapped.scene.sensors.get(entry.camera_name)
            if camera is None:
                raise ValueError(
                    f"environment has no {entry.camera_name} sensor: {entry.name}"
                )
            action = hold_action(unwrapped)
            step_dt = float(unwrapped.step_dt)
            if not math.isfinite(step_dt) or step_dt <= 0:
                raise ValueError(f"invalid environment step_dt: {step_dt}")
            elapsed_s = 0.0
            frames = []
            minimum = 255
            maximum = 0
            value_sum = 0.0
            square_sum = 0.0
            value_count = 0
            capture_shape: tuple[int, int, int] | None = None
            for frame_index in range(GIF_FRAME_COUNT):
                target_s = (frame_index + 1) / GIF_FPS
                while elapsed_s + 1e-9 < target_s:
                    env.step(action)
                    elapsed_s += step_dt
                rgb = camera.data.output["rgb"][0, ..., :3]
                if hasattr(rgb, "torch"):
                    rgb = rgb.torch
                frame_shape = tuple(int(value) for value in rgb.shape)
                if (
                    len(frame_shape) != 3
                    or any(value <= 0 for value in frame_shape)
                    or frame_shape[2] != 3
                ):
                    raise ValueError(
                        f"unexpected {entry.camera_name} shape: {tuple(rgb.shape)}"
                    )
                if capture_shape is None:
                    capture_shape = frame_shape
                elif frame_shape != capture_shape:
                    raise ValueError(
                        f"inconsistent {entry.camera_name} shape: {frame_shape}; "
                        f"expected {capture_shape}"
                    )
                if not bool(torch.isfinite(rgb).all()):
                    raise ValueError(f"{entry.camera_name} produced non-finite RGB")
                array = rgb.detach().cpu().numpy()
                if array.dtype != np.uint8:
                    if float(array.max()) <= 1.0:
                        array = array * 255.0
                    array = array.clip(0, 255).astype(np.uint8)
                minimum = min(minimum, int(array.min()))
                maximum = max(maximum, int(array.max()))
                values = array.astype(np.float64, copy=False)
                value_sum += float(values.sum())
                square_sum += float(np.square(values).sum())
                value_count += int(values.size)
                frames.append(
                    Image.fromarray(array, mode="RGB").resize(
                        GIF_SIZE, resample=Image.Resampling.LANCZOS
                    )
                )
        mean = value_sum / value_count
        std = math.sqrt(max(0.0, square_sum / value_count - mean * mean))
        assert capture_shape is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            frames[0].save(
                temporary,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=GIF_FRAME_DURATION_MS,
                loop=0,
                optimize=False,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_write(
            report_path,
            json.dumps(
                {
                    "camera": entry.camera_name,
                    "environment": entry.name,
                    "relative_yaml": entry.relative_yaml.as_posix(),
                    "render_contract": RENDER_CONTRACT,
                    "capture_shape": list(capture_shape),
                    "encoded_size": list(GIF_SIZE),
                    "duration_s": GIF_DURATION_SECONDS,
                    "simulated_duration_s": elapsed_s,
                    "step_dt_s": step_dt,
                    "fps": GIF_FPS,
                    "frame_count": GIF_FRAME_COUNT,
                    "min": minimum,
                    "max": maximum,
                    "mean": mean,
                    "std": std,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    finally:
        if env is not None:
            env.close()


def _run_render_child(relative_yaml: Path, output: Path, report: Path) -> None:
    from isaaclab.app import AppLauncher
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    simulation_args = parser.parse_args([])
    simulation_args.enable_cameras = True
    simulation_args.limit_cpu_threads = 1
    with SimulationAppContext(simulation_args):
        _render_one(relative_yaml, output, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs/environment-docs",
        help="Intermediate render evidence directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ENVIRONMENT_ROOT,
        help="Environment directory that receives README.md and media/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate committed README/media closure without starting Isaac Sim.",
    )
    parser.add_argument(
        "--check-text",
        action="store_true",
        help="Validate only the committed README text against the environment YAMLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fresh-process render commands without starting Isaac Sim.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Reuse validated evidence in an incomplete run directory.",
    )
    parser.add_argument("--render-one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-report", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.render_one is not None:
        if args.render_output is None or args.render_report is None:
            raise SystemExit(
                "--render-one requires --render-output and --render-report"
            )
        _run_render_child(args.render_one, args.render_output, args.render_report)
        return 0
    output = args.output.expanduser().resolve()
    if args.check or args.check_text:
        check_environment_docs(output, include_media=args.check)
        print(f"Environment documentation is current: {output}")
        return 0
    entries = _discover_environments(ENVIRONMENT_ROOT)
    _source_root, _env_root, arena_python = _arena_roots()
    if not arena_python.is_file():
        raise SystemExit(
            f"Pinned Arena Python does not exist: {arena_python}\n"
            "Run `./setup.sh` or set ARENA_PYTHON."
        )
    if args.dry_run:
        work_dir = args.work_dir.expanduser().resolve() / "dry-run"
        for entry in entries:
            command = _render_command(
                arena_python=arena_python,
                entry=entry,
                animation_path=work_dir / entry.animation_relative_path,
                report_path=work_dir / "reports" / entry.family / f"{entry.name}.json",
            )
            print("$ " + " ".join(shlex.quote(part) for part in command))
        return 0
    args.work_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    run_dir = generate_environment_docs(
        environment_root=output,
        work_dir=args.work_dir,
        arena_python=arena_python,
        resume_dir=args.resume,
    )
    print(f"Generated environment documentation: {output}")
    print(f"Intermediate evidence: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
