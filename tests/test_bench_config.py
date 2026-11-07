# SPDX-License-Identifier: Apache-2.0
"""`vabar bench` under experiment configs: v1.0 parity, planning, the job file, the registry rewrite.

No GPU. The vendored runner is imported (for its flags and ``job_from_args``)
and the suite registry is loaded where a test needs a real ``TaskSpec``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from vabar import benchmark
from vabar import config as cfg
from vabar.run import RunnerFlags, config_flags, prepare, runner_flags

GOLDEN = Path(__file__).parent / "data" / "v1_print"
CASES = {k: v for k, v in json.loads((GOLDEN / "cases.json").read_text()).items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def flags() -> RunnerFlags:
    return runner_flags()


def cell(key: str) -> dict:
    return benchmark.load()["cells"][key]


def plan(key: str, config: cfg.Config, flags, *, workers: int = 4, extra=(), budget=None, **kw) -> benchmark.Plan:
    return benchmark.plan_cell(
        key,
        cell(key),
        config,
        extra=list(extra),
        flags=flags,
        budget=budget or (lambda task: 1000),
        workers=lambda config, units: (min(workers, units), "test"),
        **kw,
    )


# ----------------------------------------------------------------------------- v1.0 parity
@pytest.mark.parametrize("case", sorted(CASES))
def test_paper_prints_the_v1_commands(case, capsys, config_sandbox):
    """`--config paper --print` writes, on stdout, exactly what v1.0's `--print` wrote."""
    assert benchmark.main([*CASES[case], "--config", "paper", "--print"]) == 0
    out = capsys.readouterr()
    assert out.out == (GOLDEN / case).read_text()
    assert "config paper" in out.err  # the schedule goes to stderr


def test_paper_jobs_are_the_v1_runs(flags):
    """Each paper job is the JobSpec the v1.0 `vabar run` command parses to, renderer default included."""
    from vabar._upstream import bench_run
    from vabar.run import with_default_renderer

    p = plan("tool_hanging/L1", cfg.load("paper"), flags, workers=1)
    specs = benchmark.job_specs(p, flags)
    for argv, spec in zip(benchmark.commands(cell("tool_hanging/L1")), specs, strict=True):
        assert spec == bench_run.job_from_args(flags.parser.parse_args(with_default_renderer(argv)))
    assert [s.renderer for s in specs] == ["auto"] * 4 and {s.num_worlds for s in specs} == {1}
    assert p.workers == 1 and [j.task for j in p.jobs] == list(benchmark._groups(cell("tool_hanging/L1")["instances"]))


# ----------------------------------------------------------------------------- planning
def test_default_plan(flags):
    p = plan("tool_hanging/L1", cfg.load("default"), flags)
    assert [j.num_worlds for j in p.jobs] == [8, 8, 7, 7]  # min(8, the task's instances)
    assert sum(j.units for j in p.jobs) == 4 and p.workers == 4
    assert p.out == Path("outputs/default/tool_hanging/L1")
    first = p.jobs[0]
    assert flags.value(first.argv, "--renderer") == "rtx"
    assert flags.value(first.argv, "--num-worlds") == "8"
    assert flags.value(first.argv, "--max-steps") == "8500"  # the cell's own budget
    assert first.env == {cfg.LINGER_ENV: "1", cfg.PLANNER_CACHE_ENV: "1", cfg.TRACE_ASSETS_ENV: "0"}
    assert p.invocation(flags) == [
        "--jobs",
        "outputs/default/tool_hanging/L1/jobs.py",
        "--workers",
        "4",
        "--out",
        "outputs/default/tool_hanging/L1",
    ]
    line = p.line(first)
    assert line.startswith(
        "GAP_TOOL_BUNDLE_LINGER=1 ROBOSIMSTUDIO_PLANNER_CACHE=1 GAP_TRACE_ASSETS=0 python -m vabar run --task "
    )
    # The runtime flags appear once the vendored runner accepts them (they are
    # dropped with a warning on a runner that predates the interface contract).
    runtime = " --clock lockstep --obs-render on-demand" if flags.accepts("--clock") else ""
    assert line.endswith(f"--renderer rtx{runtime} --num-worlds 8 --workers 4")


def test_cable_cells_keep_their_environment_and_take_the_configs_renderer(flags):
    p = plan("cable_insertion/L1", cfg.load("default", ["renderer=gl", "bus_timeout=900"]), flags)
    (job,) = p.jobs
    assert job.env["ROBOSIMSTUDIO_SEGMENTATION_BACKEND"] == "palette"
    # Replaced in place, not appended: one --renderer, one --bus-timeout.
    assert job.argv.count("--renderer") == 1 and flags.value(job.argv, "--renderer") == "gl"
    assert job.argv.count("--bus-timeout") == 1 and flags.value(job.argv, "--bus-timeout") == "900"
    # One world: its episodes rebuild the model and its IK planners are not fleet-safe (default.toml).
    assert job.num_worlds == 1 and len(job.episodes) == 30


def _episodes(p) -> int:
    return sum(len(j.episodes) for j in p.jobs)


def test_fast_runs_the_first_15_instances_and_says_so(flags):
    p = plan("tool_hanging/L1", cfg.load("fast"), flags)
    assert sorted(e for j in p.jobs for e in j.episodes) == list(range(1, 16))
    assert any("the first 15 of 30 instances; --limit 30" in note for note in p.notes)


def test_the_other_configs_run_the_whole_cell(flags):
    for name in ("default", "paper"):
        p = plan("tool_hanging/L1", cfg.load(name), flags)
        assert _episodes(p) == 30 and not any("instances;" in note for note in p.notes)


def test_limit_and_set_beat_the_configs_instances(flags):
    fast = cfg.load("fast")
    assert _episodes(plan("tool_hanging/L1", fast, flags, limit=30)) == 30
    assert _episodes(plan("tool_hanging/L1", fast, flags, limit=3)) == 3
    assert not any("instances;" in note for note in plan("tool_hanging/L1", fast, flags, limit=3).notes)
    assert _episodes(plan("tool_hanging/L1", cfg.load("fast", ("instances=0",)), flags)) == 30
    assert _episodes(plan("tool_hanging/L1", cfg.load("default", ("instances=4",)), flags)) == 4


def test_instances_must_be_a_count():
    with pytest.raises(cfg.ConfigError, match="instances"):
        cfg.load("fast", ("instances=-1",))


def test_gear_meshing_runs_one_world_per_job(flags):
    p = plan("gear_meshing/L1", cfg.load("default"), flags)
    assert len(p.jobs) == 30 and {j.num_worlds for j in p.jobs} == {1}
    assert "--num-worlds" not in p.jobs[0].argv and p.workers == 4


def test_longest_first_only_when_pooled(flags):
    budgets = {"tool_hanging/wrenches_level1": 100, "tool_hanging/scissors_level1": 300}
    config = cfg.load("default", ["num_worlds=1"])
    for name, value in (("pliers_level1", "200"), ("screwdrivers_level1", "400")):
        config = cfg.load("default", [*config.overrides, f"suites.tool_hanging/{name}.max_steps={value}"])
    # Without a cell budget the registry's is used; here a stand-in for it.
    no_budget = dict(cell("tool_hanging/L1"), flags=["--no-privileged", "--perception", "vision"])
    pooled = benchmark.plan_cell(
        "tool_hanging/L1",
        no_budget,
        config,
        flags=flags,
        budget=budgets.__getitem__,
        workers=lambda c, u: (4, "test"),
    )
    assert [j.task.split("/")[1] for j in pooled.jobs] == [
        "screwdrivers_level1",
        "scissors_level1",
        "pliers_level1",
        "wrenches_level1",
    ]
    assert [j.budget for j in pooled.jobs] == [400, 300, 200, 100]
    serial = benchmark.plan_cell(
        "tool_hanging/L1", no_budget, config, flags=flags, budget=budgets.__getitem__, workers=lambda c, u: (1, "")
    )
    assert [j.task for j in serial.jobs] == list(benchmark._groups(no_budget["instances"]))


def test_suite_overrides_replace_cell_flags_and_mark_the_line(flags):
    config = cfg.load("default", ["suites.tool_hanging.max_steps=6000", 'suites.tool_hanging.flags=["--roll", "held"]'])
    p = plan("tool_hanging/L1", config, flags)
    job = p.jobs[0]
    assert job.argv.count("--max-steps") == 1 and flags.value(job.argv, "--max-steps") == "6000"
    assert flags.value(job.argv, "--roll") == "held"
    # A rewritten variant is only reproduced under the same config, so the line says which.
    line = p.line(job)
    assert line.startswith(
        "GAP_TOOL_BUNDLE_LINGER=1 ROBOSIMSTUDIO_PLANNER_CACHE=1 GAP_TRACE_ASSETS=0 VABAR_CONFIG=default VABAR_SET="
    )


def test_caller_flags_win(flags):
    extra = ["--renderer", "gl", "--num-worlds", "2", "--workers", "3", "--out", "elsewhere", "--stagger", "1"]
    p = plan("tool_hanging/L1", cfg.load("default"), flags, extra=extra)
    job = p.jobs[0]
    assert "--renderer" not in job.argv and job.num_worlds == 2 and "--num-worlds" not in job.argv
    assert p.workers == 3 and p.out == Path("elsewhere")
    assert p.line(job).endswith(" ".join(extra))  # the caller's flags, verbatim, last
    assert p.invocation(flags) == [
        "--jobs",
        "elsewhere/jobs.py",
        "--workers",
        "3",
        "--out",
        "elsewhere",
        "--stagger",
        "1",
    ]


def test_variant_mapping(flags, tmp_path):
    path = tmp_path / "map.toml"
    path.write_text('inherits = "default"\n[suites.tool_hanging]\nvariants = { wrenches_level1 = "wrenches" }\n')
    p = plan("tool_hanging/L1", cfg.load(str(path)), flags)
    mapped = [j for j in p.jobs if j.instance_task == "tool_hanging/wrenches_level1"]
    assert [j.task for j in mapped] == ["tool_hanging/wrenches"]
    assert mapped[0].episodes == [1, 5, 9, 13, 17, 21, 25, 29]
    assert flags.value(mapped[0].argv, "--task") == "tool_hanging/wrenches"
    assert any("runs tool_hanging/wrenches_level1 as tool_hanging/wrenches" in n for n in p.notes)


# ----------------------------------------------------------------------------- the job file
def test_job_file_round_trips(flags, tmp_path):
    from robosimstudio_bench.run import load_jobs

    p = plan("tool_hanging/L1", cfg.load("default"), flags)
    specs = benchmark.job_specs(p, flags)
    path = benchmark.write_jobs_file(tmp_path / "jobs.py", specs, header=["tool_hanging/L1, config default"])
    assert load_jobs(path) == specs
    first = specs[0]
    assert first.task == "tool_hanging/wrenches_level1" and first.num_worlds == 8
    assert first.episode_list == (1, 5, 9, 13, 17, 21, 25, 29)
    assert first.renderer == "rtx" and first.max_steps == 8500 and not first.record.enabled
    assert dict(first.solution_kwargs)["bus_timeout"] == 180.0
    assert dict(first.solution_kwargs)["perception"] == "vision"
    text = path.read_text()
    assert text.startswith("# Generated by `vabar bench`") and "JOBS = [" in text


def test_a_mapping_that_collides_is_refused(flags, tmp_path):
    path = tmp_path / "clash.toml"
    path.write_text(
        'inherits = "default"\n[suites.tool_hanging]\n'
        'variants = { wrenches_level1 = "wrenches", scissors_level1 = "wrenches" }\n'
    )
    p = plan("tool_hanging/L1", cfg.load(str(path)), flags)
    with pytest.raises(SystemExit, match="share run directories"):
        benchmark.job_specs(p, flags)


# ----------------------------------------------------------------------------- feature detection
def fake_flags(with_contract: bool, fields=()) -> RunnerFlags:
    parser = argparse.ArgumentParser()
    for flag in ("--task", "--solve", "--perception", "--renderer", "--obs-renderer", "--bus-timeout", "--max-steps"):
        parser.add_argument(flag)
    if with_contract:
        parser.add_argument("--clock", default="free")
        parser.add_argument("--obs-render", default="per-step")
        parser.add_argument("--render-every", type=int, default=1)
    return RunnerFlags(parser, set(fields))


def test_runtime_flags_are_feature_detected():
    default, paper = cfg.load("default"), cfg.load("paper")
    kw = {"solve": "graph", "vision": True, "bench": True}

    add, dropped, fields = config_flags(default, "tool_hanging/x", flags=fake_flags(True), **kw)
    assert add == ["--renderer", "rtx", "--clock", "lockstep", "--obs-render", "on-demand"] and not dropped
    # Paper asks for the runner's defaults, which need no flag: nothing added either way.
    assert config_flags(paper, "tool_hanging/x", flags=fake_flags(True), **kw) == ([], [], {})
    assert config_flags(paper, "tool_hanging/x", flags=fake_flags(False), **kw) == ([], [], {})

    add, dropped, fields = config_flags(default, "tool_hanging/x", flags=fake_flags(False), **kw)
    assert add == ["--renderer", "rtx"]
    assert dropped == ["--clock lockstep", "--obs-render on-demand"] and fields == {}
    # A JobSpec that has the field carries it even though the parser lacks the flag.
    _, _, fields = config_flags(default, "tool_hanging/x", flags=fake_flags(False, {"clock", "obs_render"}), **kw)
    assert fields == {"clock": "lockstep", "obs_render": "on-demand"}


def test_vabar_run_applies_the_config_without_overriding_the_command_line(flags):
    base = ["--task", "tool_hanging/wrenches_level1", "--solve", "graph", "--perception", "vision"]
    argv, _, _ = prepare(base, cfg.load("default", ["bus_timeout=240"]), flags)
    assert flags.value(argv, "--renderer") == "rtx" and flags.value(argv, "--bus-timeout") == "240"
    argv, _, _ = prepare([*base, "--bus-timeout", "60", "--obs-renderer", "gl"], cfg.load("default"), flags)
    assert flags.value(argv, "--bus-timeout") == "60" and "--renderer" not in argv
    argv, _, _ = prepare(base, cfg.load("paper"), flags)
    assert argv == [*base, "--renderer", "auto"]  # v1.0's default for a vision run
    argv, _, _ = prepare(["--task", "tool_hanging/wrenches_level1", "--solve", "routine"], cfg.load("default"), flags)
    assert "--renderer" not in argv  # a run without cameras gets no renderer


# ----------------------------------------------------------------------------- refused waves
def test_refused_waves_are_found_by_their_rows(tmp_path):
    from robosimstudio.specs import JobSpec

    specs = [
        JobSpec(name="a", task="gear_mesh/x", num_worlds=4),
        JobSpec(name="b", task="tool_hanging/y", num_worlds=4),
        JobSpec(name="c", task="gear_mesh/z", num_worlds=1),
    ]
    (tmp_path / "a").mkdir()
    row = {"task": "gear_mesh/x", "close_kind": "error", "error": "CannotReplicate: gear_mesh/x cannot run replicated"}
    (tmp_path / "a" / "episode_results.jsonl").write_text(json.dumps(row) + "\n")
    assert benchmark._refused(specs, tmp_path, None) == ["a"]

    class CannotReplicate(Exception):
        pass

    raised = CannotReplicate("tool_hanging/y cannot run replicated (4 worlds) yet: ...")
    assert benchmark._refused(specs, tmp_path, raised) == ["a", "b"]


# ----------------------------------------------------------------------------- the registry rewrite
def test_config_rewrites_registered_variants(config_sandbox, tmp_path):
    import vabar

    vabar.install()
    from robosimstudio_bench.taskbuild.registry import get_task

    assert get_task("tool_hanging/wrenches_level1").physics.substeps == 8
    path = tmp_path / "th960.toml"
    path.write_text(
        'inherits = "default"\n[suites."tool_hanging/wrenches_level1"]\nphysics = "tool_hanging_960hz@v1"\n'
    )
    cfg.activate(
        cfg.load(str(path), ["suites.tool_hanging.max_steps=1234", "suites.tool_hanging.cameras=tool_hanging_views@v1"])
    )
    task = get_task("tool_hanging/wrenches_level1")
    assert task.physics.substeps == 16 and task.max_steps == 1234
    assert [c.name for c in task.cameras] == [c.name for c in get_task("tool_hanging/wrenches").cameras]
    assert get_task("tool_hanging/scissors_level1").physics.substeps == 8  # the variant table is the variant's
    assert get_task("tool_hanging/scissors_level1").max_steps == 1234  # the suite table is every variant's
    cfg.apply_overrides()
    assert get_task("tool_hanging/wrenches_level1") is task  # idempotent
    cfg.activate(cfg.load("paper"))
    restored = get_task("tool_hanging/wrenches_level1")
    assert restored.physics.substeps == 8 and restored.max_steps == 6200


def test_an_unregistered_arrangement_is_named(config_sandbox):
    import vabar

    vabar.install()
    from robosimstudio_bench.taskbuild.registry import get_task

    with pytest.raises(cfg.ConfigError, match="not a registered physics arrangement"):
        cfg.activate(cfg.load("default", ["suites.tool_hanging.physics=nope@v1"]))
        get_task("tool_hanging/wrenches_level1")
    cfg._ACTIVE = cfg.load("paper")
    cfg.apply_overrides()
