"""One-command setup and local Arena launcher for VABAR on Isaac Lab Arena."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from isaaclab_arena_gap.paths import ARENA_GAP_ROOT, WORKSPACE_ROOT

BOOTSTRAP = ARENA_GAP_ROOT / "scripts" / "bootstrap.sh"
ARENA_LAUNCHER = ARENA_GAP_ROOT / "scripts" / "run_arena_eval.sh"
SHOW_LAUNCHER = ARENA_GAP_ROOT / "scripts" / "run_arena_show.sh"
DEFAULT_CONFIG = (
    ARENA_GAP_ROOT
    / "experiment_configs"
    / "tool_hanging"
    / "vabar_tool_hanging_wrench_easy_gap_baseline.yaml"
)
DEFAULT_SCENE = (
    WORKSPACE_ROOT
    / "isaaclab_arena_vabar"
    / "src"
    / "isaaclab_arena_vabar"
    / "environments"
    / "tool_hanging"
    / "wrench_easy.yaml"
)

def _add_bootstrap_options(
    parser: argparse.ArgumentParser, *, include_source_only: bool
) -> None:
    parser.add_argument(
        "--offline", action="store_true", help="Use only cached UV packages."
    )
    parser.add_argument(
        "--materialized-source",
        action="store_true",
        help="Use an exported source tree without Git metadata.",
    )
    if include_source_only:
        parser.add_argument(
            "--source-only",
            action="store_true",
            help="Initialize and verify source pins without creating environments.",
        )
    parser.add_argument(
        "--skip-skills",
        action="store_true",
        help="Skip the five GaP skill environments.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the final runtime, credential, and GPU preflight.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set up VABAR on Isaac Lab Arena's pinned runtimes and launch Arena "
            "through one UV command."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and environment without changing anything.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser(
        "setup", help="Initialize sources and synchronize every locked environment."
    )
    _add_bootstrap_options(setup, include_source_only=True)

    run = subparsers.add_parser(
        "run", help="Ensure the environments are ready, then run an Arena job."
    )
    run.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Arena Experiment YAML (default: the supported easy wrench-hanging task).",
    )
    run.add_argument(
        "--skip-setup",
        action="store_true",
        help="Launch immediately from environments prepared by an earlier setup.",
    )
    _add_viewer_option(run)
    _add_visualizer_option(run)
    _add_bootstrap_options(run, include_source_only=False)

    show = subparsers.add_parser(
        "show",
        help="Open a scene in the Isaac Sim viewer with no policy, and hold it there.",
        description=(
            "Build a scene and step it with zero actions until interrupted, so the "
            "environment can be inspected interactively. Accepts a scene YAML or a "
            "registered environment name; policies and episode budgets are ignored."
        ),
    )
    show.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Scene YAML (default: the easy wrench scene).",
    )
    show.add_argument(
        "--env",
        default=None,
        help="Registered Arena environment name instead of a scene YAML.",
    )
    show.add_argument(
        "--skip-setup",
        action="store_true",
        help="Launch immediately from environments prepared by an earlier setup.",
    )
    show.add_argument(
        "--presets",
        default=None,
        choices=("newton",),
        help="Newton physics preset. Direct scene YAML defaults to 'newton'.",
    )
    show.add_argument(
        "--job",
        default=None,
        help="Job name to inspect when the supplied JSON contains multiple jobs.",
    )
    show.add_argument(
        "--seed", type=int, default=None, help="Override the job's simulation seed."
    )
    show.add_argument(
        "--enable_cameras",
        action="store_true",
        help="Instantiate the scene's cameras even when the job disables them.",
    )
    _add_visualizer_option(show)
    show.add_argument(
        "--headless",
        action="store_true",
        help="Run without the viewer (useful when only the physics dump is wanted).",
    )
    _add_bootstrap_options(show, include_source_only=False)
    return parser


# Mirrors AppLauncher._parse_visualizer_csv, which also accepts "none".
VISUALIZER_BACKENDS = ("kit", "newton", "rerun", "viser", "none")


def _visualizer_csv(value: str) -> str:
    """Validate a CSV of Isaac Lab visualizer backends.

    Mirrors ``isaaclab.visualizers.visualizer._VISUALIZER_TYPES``. Checking here
    means a typo fails immediately instead of after the several minutes Isaac Sim
    takes to start.
    """
    backends = [item.strip() for item in value.split(",") if item.strip()]
    if not backends:
        raise argparse.ArgumentTypeError("no visualizer backend given")
    unknown = [item for item in backends if item not in VISUALIZER_BACKENDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown visualizer backend(s) {', '.join(unknown)}; "
            f"choose from {', '.join(VISUALIZER_BACKENDS)}"
        )
    return ",".join(backends)


def _add_visualizer_option(parser: argparse.ArgumentParser) -> None:
    """Add the Isaac Lab visualizer-backend passthrough.

    ``kit`` renders through RTX and needs a display; ``newton``, ``rerun`` and
    ``viser`` draw physics geometry and work headless, which is both what a
    remote session can use and what shows collision shapes rather than visuals.
    """
    parser.add_argument(
        "--viz",
        "--visualizer",
        dest="viz",
        default=None,
        type=_visualizer_csv,
        metavar="BACKENDS",
        help=(
            "Visualizer backends as CSV, e.g. 'rerun' or 'kit,rerun'. "
            f"One or more of: {', '.join(VISUALIZER_BACKENDS)}."
        ),
    )


def _add_viewer_option(parser: argparse.ArgumentParser) -> None:
    """Add the flag that opens the Isaac Sim viewport.

    Isaac Lab develop derives Kit headlessness from explicit visualizer intent.
    Selecting ``kit`` opens the Kit viewport; standalone visualizers keep Kit
    headless while opening their own frontend.
    """
    group = parser.add_argument_group("viewer")
    group.add_argument(
        "--no-headless",
        "--gui",
        dest="no_headless",
        action="store_true",
        help="Open the Isaac Sim viewer instead of running headless.",
    )


def _bootstrap_args(args: argparse.Namespace) -> list[str]:
    command = [str(BOOTSTRAP)]
    for option in (
        "offline",
        "materialized_source",
        "source_only",
        "skip_skills",
        "skip_preflight",
    ):
        if getattr(args, option, False):
            command.append(f"--{option.replace('_', '-')}")
    return command


def _format_command(
    command: Sequence[str], env_updates: dict[str, str] | None = None
) -> str:
    assignments = [
        f"{name}={shlex.quote(value)}" for name, value in (env_updates or {}).items()
    ]
    return "$ " + " ".join([*assignments, *(shlex.quote(part) for part in command)])


def _run_setup(args: argparse.Namespace) -> None:
    command = _bootstrap_args(args)
    if args.dry_run:
        print(_format_command(command))
        return
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def _run_arena(
    args: argparse.Namespace, parser: argparse.ArgumentParser, forwarded: Sequence[str]
) -> None:
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        workspace_config = WORKSPACE_ROOT / config
        member_config = ARENA_GAP_ROOT / config
        config = workspace_config if workspace_config.is_file() else member_config
    config = config.resolve()
    if not config.is_file():
        parser.error(f"Arena Experiment config does not exist: {config}")

    if not args.skip_setup:
        _run_setup(args)

    arena_args = list(forwarded)
    if arena_args[:1] == ["--"]:
        arena_args.pop(0)
    viz = getattr(args, "viz", None)
    if getattr(args, "no_headless", False) and viz is None:
        viz = "kit"
    if viz:
        arena_args = ["--viz", viz, *arena_args]
    command = [str(ARENA_LAUNCHER), *arena_args]
    env_updates = {"ARENA_EXPERIMENT_CONFIG": str(config)}
    if viz is not None and "newton" in viz.split(","):
        env_updates["ISAACLAB_DISABLE_LIVE_PLOTS"] = os.environ.get(
            "ISAACLAB_DISABLE_LIVE_PLOTS", "1"
        )
    if args.dry_run:
        print(_format_command(command, env_updates))
        return

    environment = os.environ.copy()
    environment.update(env_updates)
    os.execve(ARENA_LAUNCHER, command, environment)


def _run_show(
    args: argparse.Namespace, parser: argparse.ArgumentParser, forwarded: Sequence[str]
) -> None:
    if args.env is not None and args.scene is not None:
        parser.error("show accepts either a scene/job path or --env, not both")

    scene = None
    if args.env is None:
        scene = Path(args.scene or DEFAULT_SCENE).expanduser()
        if not scene.is_absolute():
            workspace_scene = WORKSPACE_ROOT / scene
            member_scene = ARENA_GAP_ROOT / scene
            scene = workspace_scene if workspace_scene.is_file() else member_scene
        scene = scene.resolve()
        if not scene.is_file():
            parser.error(f"Scene config does not exist: {scene}")

    if not args.skip_setup:
        _run_setup(args)

    show_args = list(forwarded)
    if show_args[:1] == ["--"]:
        show_args.pop(0)
    for option in ("job", "seed"):
        value = getattr(args, option, None)
        if value is not None:
            show_args = [f"--{option}", str(value), *show_args]
    if args.enable_cameras:
        show_args = ["--enable_cameras", *show_args]
    if args.presets:
        show_args = ["--presets", args.presets, *show_args]
    viz = args.viz
    if viz is None and not args.headless:
        viz = "kit"
    if viz:
        show_args = ["--viz", viz, *show_args]
    if args.env is not None:
        show_args = ["--env", args.env, *show_args]
    command = [str(SHOW_LAUNCHER), *show_args]
    visualizers = set(viz.split(",")) if viz else set()
    headless = args.headless or (bool(visualizers) and "kit" not in visualizers)
    env_updates = {"ARENA_HEADLESS": "1" if headless else "0"}
    if scene is not None:
        env_updates["ARENA_SHOW_SCENE"] = str(scene)
    if args.dry_run:
        print(_format_command(command, env_updates))
        return

    environment = os.environ.copy()
    environment.update(env_updates)
    os.execve(SHOW_LAUNCHER, command, environment)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, forwarded = parser.parse_known_args(argv)
    if args.command == "setup":
        if forwarded:
            parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
        _run_setup(args)
        return
    if args.command == "show":
        _run_show(args, parser, forwarded)
        return
    _run_arena(args, parser, forwarded)


if __name__ == "__main__":
    main()
