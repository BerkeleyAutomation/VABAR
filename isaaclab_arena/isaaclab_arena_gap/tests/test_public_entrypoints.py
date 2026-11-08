"""The supported registration and command-line entrypoints remain usable."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("injected", "succeeds"),
    (
        ({}, False),
        ({"GAP_VLM_API_KEY": "injected"}, True),
        ({"CAP_BAKED_RUNTIME": "1"}, False),
        ({"CAP_BAKED_RUNTIME": "1", "GAP_VLM_API_KEY": "injected"}, True),
    ),
)
def test_gap_launcher_accepts_process_vlm_credentials(
    tmp_path, run_launcher, injected, succeeds
) -> None:
    result, record = run_launcher(
        GAP_VLM_ENV=str(tmp_path / "missing-vlm.env"),
        **injected,
    )
    if succeeds:
        assert result.returncode == 0, result.stderr
        assert record is not None
    else:
        assert result.returncode == 1
        assert "Missing VLM env file" in result.stderr
        assert record is None


@pytest.mark.parametrize("exported_key", (False, True))
def test_preflight_accepts_the_launcher_vlm_credentials(
    tmp_path, exported_key
) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GAP_VLM_API_KEY", "OPENROUTER_API_KEY"}
    }
    if exported_key:
        environment["GAP_VLM_API_KEY"] = "injected"
    result = subprocess.run(
        [str(ROOT / "scripts/check_setup.sh")],
        env={**environment, "GAP_VLM_ENV": str(tmp_path / "missing-vlm.env")},
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    exported = "OK:   VLM API credential is exported as GAP_VLM_API_KEY"
    missing = "FAIL: VLM env file is missing and GAP_VLM_API_KEY is not exported"
    assert (exported in output) is exported_key, output
    assert (missing in output) is not exported_key, output


def test_gap_launcher_vlm_route_is_overrideable_and_skill_overrides_win(
    tmp_path, run_launcher
) -> None:
    _, default = run_launcher()
    for name in ("GAP_VLM_PROVIDER", "GAP_VLM_BASE_URL", "GAP_VLM_MODEL"):
        assert default["env"][name]
    _, custom = run_launcher(GAP_VLM_MODEL="custom/model")
    assert custom["env"]["GAP_VLM_MODEL"] == "custom/model"
    # The repository's VLM override shadows AUTOLab's bundles, which shadow the generic registry.
    assert custom["env"]["GAP_SKILLS_PATH"].split(":") == [
        str(ROOT / "skill_overrides"),
        str(tmp_path / "fake-autolab/robot_skills"),
        str(tmp_path / "skills"),
    ]


def test_gap_launcher_runs_the_rss_adapter_in_autolabs_environment(
    tmp_path, run_launcher
) -> None:
    result, record = run_launcher("--inputs", 'instruction="hang the wrench"')
    assert result.returncode == 0, result.stderr
    assert record["cwd"] == str(tmp_path / "fake-autolab")
    assert record["argv"] == [
        "-u", "-m", "isaaclab_arena_gap.rss_adapter.runner", "--inputs", 'instruction="hang the wrench"',
    ]
    assert record["env"]["PYTHONPATH"].split(":")[0] == str(ROOT / "src")
    assert record["env"]["GAP_TRACE_DIR"] == str(tmp_path / "trace")


def test_gap_launcher_requires_autolabs_environment(tmp_path, run_launcher) -> None:
    result, record = run_launcher(AUTOLAB_VABAR_ROOT=str(tmp_path / "missing"))
    assert result.returncode == 1
    assert "AUTOLab v1.0 environment missing" in result.stderr
    assert record is None


def test_arena_runner_import_does_not_boot_simulation() -> None:
    forbidden = {"isaaclab", "omni", "pxr"}
    sys.modules.pop("isaaclab_arena_gap.arena_runner", None)
    imported_before = set(sys.modules)

    importlib.import_module("isaaclab_arena_gap.arena_runner")

    newly_imported = set(sys.modules) - imported_before
    assert {name.partition(".")[0] for name in newly_imported}.isdisjoint(forbidden)


def test_newton_15_passive_tendon_compat_skips_missing_actuators(
    monkeypatch,
) -> None:
    from isaaclab_arena_gap import newton_15_tendon_compat as compat

    tendon_module = pytest.importorskip("isaaclab_newton.physics.mjwarp_tendon_control")
    MjWarpTendonControl = tendon_module.MjWarpTendonControl

    original = MjWarpTendonControl.__dict__["create"]
    monkeypatch.setattr(MjWarpTendonControl, "create", original)
    monkeypatch.setattr(compat, "_installed", False)
    monkeypatch.setattr(compat, "version", lambda _package: "1.5.2")

    compat.install_newton_15_passive_tendon_compat()

    model = SimpleNamespace(custom_frequency_counts={"mujoco:tendon": 2})
    assert MjWarpTendonControl.create(None, model) is None


@pytest.fixture
def fake_experiment_runner(monkeypatch):
    """Install a fake Arena experiment runner whose main enters SimulationAppContext."""
    events = []
    runner_module = ModuleType("isaaclab_arena.evaluation.experiment_runner")

    def parse_args():
        return object(), []

    class SimulationAppContext:
        def __init__(self, _args_cli):
            pass

        def __enter__(self):
            events.append("entered")

        def __exit__(self, exception_type, _exception, _traceback):
            events.append(("closed", exception_type))

    def runner_main():
        with runner_module.SimulationAppContext(object()):
            events.append("runner")

    runner_module.parse_experiment_runner_args = parse_args
    runner_module.SimulationAppContext = SimulationAppContext
    runner_module.main = runner_main
    for name in ("isaaclab_arena", "isaaclab_arena.evaluation"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    yield events
    # The runner patches Arena only for the duration of main().
    assert runner_module.parse_experiment_runner_args is parse_args
    assert runner_module.SimulationAppContext is SimulationAppContext


def test_arena_runner_initializes_components_inside_simulation_context(
    monkeypatch, fake_experiment_runner
) -> None:
    from isaaclab_arena_gap import arena_runner

    monkeypatch.setattr(
        arena_runner,
        "initialize_arena_runtime",
        lambda: fake_experiment_runner.append("initialize"),
    )
    arena_runner.main()
    assert fake_experiment_runner == [
        "entered",
        "initialize",
        "runner",
        ("closed", None),
    ]


def test_arena_runner_closes_simulation_context_if_initialization_fails(
    monkeypatch, fake_experiment_runner
) -> None:
    from isaaclab_arena_gap import arena_runner

    def fail_initialization():
        raise RuntimeError("registration failed")

    monkeypatch.setattr(arena_runner, "initialize_arena_runtime", fail_initialization)
    with pytest.raises(RuntimeError, match="registration failed"):
        arena_runner.main()
    assert fake_experiment_runner == ["entered", ("closed", RuntimeError)]


def test_arena_runner_limits_openusd_threads_when_unspecified() -> None:
    from argparse import Namespace

    from isaaclab_arena_gap.arena_runner import _with_safe_openusd_thread_limit

    wrapped = _with_safe_openusd_thread_limit(lambda: (Namespace(), []))

    args, overrides = wrapped()
    assert args.limit_cpu_threads == 1
    assert overrides == []


def test_arena_runner_disables_geometry_streaming_and_keeps_kit_args() -> None:
    from argparse import Namespace

    from isaaclab_arena_gap.arena_runner import _with_render_kit_args

    wrapped = _with_render_kit_args(
        lambda: (Namespace(kit_args="--ext-folder=/x"), ["a=1"])
    )

    args, overrides = wrapped()
    assert args.kit_args == (
        "--ext-folder=/x --/UJITSO/enabled=false --/UJITSO/geometry=false"
    )
    assert overrides == ["a=1"]
    assert _with_render_kit_args(lambda: (args, []))()[0].kit_args == args.kit_args


@pytest.fixture
def fake_newton_cfg(monkeypatch):
    """Install a minimal ``isaaclab_newton.physics.NewtonCfg`` and return it."""
    physics_module = ModuleType("isaaclab_newton.physics")

    class NewtonCfg:
        pass

    physics_module.NewtonCfg = NewtonCfg
    newton_package = ModuleType("isaaclab_newton")
    newton_package.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab_newton", newton_package)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics_module)
    return NewtonCfg


def _run_cfg(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        policy=SimpleNamespace(),
        environment_builder=SimpleNamespace(
            record_trajectories=False,
            recorder_dataset_export_dir_path=None,
            recorder_dataset_filename=None,
        ),
    )


def test_arena_runner_rejects_non_newton_environment(fake_newton_cfg) -> None:

    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=object()),
        recorders=None,
    )
    builder = SimpleNamespace(build_registered=lambda: ("env", env_cfg, {}))
    execution = SimpleNamespace(build_arena_builder_from_run_cfg=lambda _cfg: builder)
    cfg = _run_cfg("unsupported")

    from isaaclab_arena_gap.arena_runner import _build_environment_for_policy

    with pytest.raises(TypeError, match="supports only Newton"):
        _build_environment_for_policy(cfg, None, run_execution=execution)


def test_arena_runner_exposes_policy_completion_to_task_scoring(
    monkeypatch, fake_newton_cfg
) -> None:
    from isaaclab_arena_gap import arena_runner
    from isaaclab_arena_gap.external_policy_termination import (
        request_external_policy_termination,
    )

    managers_module = ModuleType("isaaclab.managers")
    managers_module.TerminationTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    isaaclab_package = ModuleType("isaaclab")
    isaaclab_package.__path__ = []
    camera_module = ModuleType("isaaclab_arena_gap.newton_camera_rendering")
    camera_module.configure_newton_camera_renderers = lambda _cfg: 0
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_package)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)
    monkeypatch.setitem(sys.modules, "isaaclab_arena_gap.newton_camera_rendering", camera_module)
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=fake_newton_cfg()),
        scene=SimpleNamespace(num_envs=1),
        terminations=SimpleNamespace(),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace())
    builder = SimpleNamespace(
        build_registered=lambda: ("env", env_cfg, {}),
        make_registered=lambda *_args, **_kwargs: env,
    )
    execution = SimpleNamespace(build_arena_builder_from_run_cfg=lambda _cfg: builder)
    cfg = _run_cfg("cable")

    assert (
        arena_runner._build_environment_for_policy(cfg, None, run_execution=execution)
        is env
    )
    request_external_policy_termination([True])
    assert env.unwrapped.external_policy_termination_requested() == (True,)


def test_arena_show_closes_app_when_rejecting_non_newton(fake_newton_cfg) -> None:

    app = SimpleNamespace(closed=False)
    app.close = lambda: setattr(app, "closed", True)
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=object()))

    from isaaclab_arena_gap.arena_show import _guarded_setup, _require_newton

    with pytest.raises(TypeError, match="supports only Newton"):
        _guarded_setup(app, lambda: _require_newton(env_cfg, "Test environment"))
    assert app.closed


def test_public_registration_delegates_and_is_idempotent(monkeypatch) -> None:
    registries = pytest.importorskip("isaaclab_arena.assets.registries")
    if sys.version_info < (3, 12):
        pytest.skip("Arena registries require Python 3.12")
    benchmark = importlib.import_module("isaaclab_arena_vabar.registration")
    from isaaclab_arena_gap.arena.registration import register_arena_components

    calls = []
    register_components = benchmark.register_components

    def spy() -> None:
        calls.append("registered")
        register_components()

    monkeypatch.setattr(benchmark, "register_components", spy)

    def registered() -> tuple[set[str], set[str]]:
        return (
            set(registries.AssetRegistry().get_all_keys()),
            set(registries.EnvironmentRegistry().get_all_keys()),
        )

    register_arena_components()
    first = registered()
    register_arena_components()
    assert calls == ["registered", "registered"]
    assert registered() == first
    assert "vabar__syringe_sort_bin" in first[0]
    assert "vabar_tool_hanging__scissors_easy" in first[1]


def test_console_scripts_do_not_shadow_autolab_vabar() -> None:
    # AUTOLab's own runner is `vabar`; ours must not install a command of that name.
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {
        "vabar-arena": "isaaclab_arena_gap.cli:main",
        "vabar-eval": "isaaclab_arena_gap.container_cli:main",
    }


def test_cli_dry_runs_the_default_job_and_registered_environment(capsys) -> None:
    from isaaclab_arena_gap import cli

    cli.main(["--dry-run", "run", "--skip-setup"])
    run_output = capsys.readouterr().out
    assert "run_arena_eval.sh" in run_output
    assert "vabar_tool_hanging_wrench_easy_gap_baseline.yaml" in run_output

    cli.main(
        [
            "--dry-run",
            "show",
            "--skip-setup",
            "--env",
            "vabar_cable_routing__medium",
            "--headless",
        ]
    )
    show_output = capsys.readouterr().out
    assert "run_arena_show.sh" in show_output
    assert "--env vabar_cable_routing__medium" in show_output


@pytest.mark.parametrize("preset", ["physx", "newton_dense_clutter"])
def test_cli_rejects_unsupported_preset(preset: str) -> None:
    from isaaclab_arena_gap import cli

    with pytest.raises(SystemExit):
        cli.main(["--dry-run", "show", "--skip-setup", "--presets", preset])
