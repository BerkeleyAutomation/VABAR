# SPDX-License-Identifier: Apache-2.0
"""Experiment configs: parsing, inheritance, --set, the environment they add, workers, provenance.

No GPU, and no suite registry except where a test says so.
"""

from __future__ import annotations

import json
import os

import pytest

from vabar import config as cfg


def write(path, text: str):
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------- the built-in configs
def test_builtin_configs():
    assert {"default", "fast", "paper"} <= set(cfg.names())
    default, fast, paper = cfg.load("default"), cfg.load("fast"), cfg.load("paper")

    rt = default.runtime
    assert (rt["clock"], rt["obs_render"], rt["renderer"]) == ("lockstep", "on-demand", "rtx")
    assert rt["keep_tool_servers"] and rt["cache_planner"]
    assert rt["trace_assets"] is False
    assert default.env()[cfg.TRACE_ASSETS_ENV] == "0"
    assert (rt["num_worlds"], rt["workers"]) == (8, 1)
    assert not default.registry_overrides(), "default must leave physics, cameras and budgets alone"

    # fast is default plus suite tables the sweeps fill: same runtime, default first in its chain.
    assert fast.runtime == default.runtime
    assert [p.rsplit("/", 1)[-1] for p in fast.chain] == ["default.toml", "fast.toml"]
    assert fast.num_worlds("gear_mesh/gearmesh-easy-pair-01") == 1

    rt = paper.runtime
    assert (rt["clock"], rt["obs_render"], rt["renderer"]) == ("free", "per-step", "")
    assert not rt["keep_tool_servers"] and not rt["cache_planner"]
    assert (rt["num_worlds"], rt["workers"], rt["bus_timeout"]) == (1, 1, 0.0)
    assert paper.env() == {} and paper.suites == {}
    assert paper.ref == "paper" and default.sha256 != paper.sha256


# ----------------------------------------------------------------------------- inheritance
def test_inheritance_merges_tables_and_replaces_values(tmp_path):
    base = write(
        tmp_path / "base.toml",
        """
        [runtime]
        workers = 3
        env = { A = "1", B = "2" }
        [suites.tool_hanging]
        max_steps = 7000
        flags = ["--settle-steps", "30"]
        """,
    )
    child = write(
        tmp_path / "child.toml",
        """
        inherits = "base.toml"
        [runtime]
        env = { B = "two", C = "3" }
        [suites.tool_hanging]
        flags = ["--roll", "free"]
        """,
    )
    config = cfg.load(str(child))
    assert config.name == "child" and config.ref == str(child.resolve())
    assert config.chain == (str(base.resolve()), str(child.resolve()))
    assert config.runtime["workers"] == 3
    assert config.runtime["env"] == {"A": "1", "B": "two", "C": "3"}
    assert config.suites["tool_hanging"] == {"max_steps": 7000, "flags": ["--roll", "free"]}
    # Keys a file does not state are the v1.0 behaviour, not the parent built-in's.
    assert config.runtime["clock"] == ""


def test_inherits_a_builtin_by_name(tmp_path):
    config = cfg.load(str(write(tmp_path / "mine.toml", 'inherits = "default"\n[runtime]\nworkers = 2\n')))
    assert config.runtime["clock"] == "lockstep" and config.runtime["workers"] == 2


def test_inheritance_cycle_is_refused(tmp_path):
    write(tmp_path / "a.toml", 'inherits = "b.toml"\n')
    write(tmp_path / "b.toml", 'inherits = "a.toml"\n')
    with pytest.raises(cfg.ConfigError, match="cycle"):
        cfg.load(str(tmp_path / "a.toml"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[runtime]\nclok = 'free'\n", "unknown runtime key"),
        ("[runtime]\nclock = 'fast'\n", "runtime.clock"),
        ("[runtime]\nobs_render = 'sometimes'\n", "runtime.obs_render"),
        ("[runtime]\nworkers = 0\n", "runtime.workers"),
        ("[runtime]\nworkers = 'many'\n", "runtime.workers"),
        ("[runtime]\nnum_worlds = 2.5\n", "runtime.num_worlds"),
        ("[runtime]\nkeep_tool_servers = 'yes'\n", "keep_tool_servers"),
        ("[suites.tool_hangin]\nmax_steps = 1\n", "names no suite"),
        ("[suites.tool_hanging]\nphysic = 'x@v1'\n", "unknown key"),
        ("[suites.tool_hanging]\nphysics = 'unversioned'\n", "versioned"),
        ("[suites.tool_hanging]\nmax_steps = 0\n", "max_steps"),
        ('[suites."tool_hanging/wrenches_level1"]\nvariants = { a = "b" }\n', "renames variants"),
        ("[suites.tool_hanging]\nflags = '--x'\n", "flags"),
        ("speed = 'max'\n", "unknown top-level"),
        ("[runtime\n", "toml"),
    ],
)
def test_bad_configs_are_refused(tmp_path, text, match):
    with pytest.raises(cfg.ConfigError, match=match):
        cfg.load(str(write(tmp_path / "bad.toml", text)))


def test_unknown_config_name():
    with pytest.raises(cfg.ConfigError, match="unknown config 'nope'"):
        cfg.load("nope")


# ----------------------------------------------------------------------------- --set
def test_set_overrides_win_over_the_file():
    config = cfg.load(
        "default",
        [
            "clock=free",
            "runtime.workers=3",
            "runtime.env.FOO=bar",
            "suites.tool_hanging.max_steps=9000",
            "suites.tool_hanging/wrenches_level1.physics=tool_hanging_960hz@v1",
            'suites.usbc_insert.flags=["--settle-steps", "30"]',
            "suites.usbc_insert.variants.insert_easy=insert_easy_nocable",
            "bus_timeout=300",
        ],
    )
    rt = config.runtime
    assert (rt["clock"], rt["workers"], rt["env"], rt["bus_timeout"]) == ("free", 3, {"FOO": "bar"}, 300.0)
    assert config.suites["tool_hanging"]["max_steps"] == 9000
    assert cfg.resolve_workers(config, 30, suite="tool_hanging")[0] == 3  # --set beats the suite table
    assert config.suites["tool_hanging/wrenches_level1"] == {"physics": "tool_hanging_960hz@v1"}
    assert config.suites["usbc_insert"]["flags"] == ["--settle-steps", "30"]
    assert config.map_task("usbc_insert/insert_easy") == "usbc_insert/insert_easy_nocable"
    assert config.overrides[0] == "clock=free"
    assert config.sha256 != cfg.load("default").sha256
    assert cfg.load("default", ["workers=auto"]).runtime["workers"] == "auto"


@pytest.mark.parametrize("expression", ["clock", "=free", "nonsense=1", "runtime=1", "clock=fast"])
def test_bad_set_expressions(expression):
    with pytest.raises(cfg.ConfigError):
        cfg.load("default", [expression])


def test_from_argv_strips_its_flags_and_falls_back_to_the_environment(config_sandbox, monkeypatch):
    config, rest = cfg.from_argv(["--task", "x/y", "--config", "paper", "--set", "workers=2", "--set=clock=lockstep"])
    assert rest == ["--task", "x/y"]
    assert config.name == "paper" and config.overrides == ("workers=2", "clock=lockstep")

    monkeypatch.setenv(cfg.ENV_CONFIG, "paper")
    monkeypatch.setenv(cfg.ENV_SET, json.dumps(["workers=2"]))
    config, rest = cfg.from_argv(["--set", "num_worlds=4"])
    assert config.name == "paper" and config.overrides == ("workers=2", "num_worlds=4") and rest == []
    # An explicit --config starts afresh rather than stacking on the environment's overrides.
    assert cfg.from_argv(["--config", "default"])[0].overrides == ()


def test_activate_exports_what_a_worker_needs(config_sandbox, tmp_path):
    path = write(tmp_path / "mine.toml", 'inherits = "default"\n')
    config = cfg.activate(cfg.load(str(path), ["workers=2"]))
    assert os.environ[cfg.ENV_CONFIG] == str(path.resolve())
    assert json.loads(os.environ[cfg.ENV_SET]) == ["workers=2"]
    # A fresh process resolves the same config from those two variables.
    assert cfg.from_env().sha256 == config.sha256
    cfg.activate(cfg.load("paper"))
    assert os.environ[cfg.ENV_CONFIG] == "paper" and cfg.ENV_SET not in os.environ


# ----------------------------------------------------------------------------- per task
def test_suite_and_variant_tables(tmp_path):
    config = cfg.load(
        str(
            write(
                tmp_path / "t.toml",
                """
                inherits = "default"
                [suites.tool_hanging]
                max_steps = 7000
                env = { A = "suite", B = "suite" }
                num_worlds = 4
                variants = { wrenches_level1 = "wrenches_level2" }
                [suites."tool_hanging/scissors_level1"]
                max_steps = 9000
                env = { B = "variant" }
                """,
            )
        )
    )
    assert config.table("tool_hanging/scissors_level1")["max_steps"] == 9000
    assert config.table("tool_hanging/pliers_level1")["max_steps"] == 7000
    assert config.env("tool_hanging/scissors_level1")["B"] == "variant"
    assert config.env("tool_hanging/scissors_level1")["A"] == "suite"
    assert config.map_task("tool_hanging/wrenches_level1") == "tool_hanging/wrenches_level2"
    assert config.map_task("tool_hanging/pliers_level1") == "tool_hanging/pliers_level1"
    assert config.num_worlds("tool_hanging/pliers_level1") == 4
    assert config.num_worlds("syringe_packing/tray_pair_both_t00") == 8
    assert config.rewrites("tool_hanging/pliers_level1") and not config.rewrites("gear_mesh/x")


def test_environment(config_sandbox, monkeypatch):
    default = cfg.load("default")
    assert default.env() == {cfg.LINGER_ENV: "1", cfg.PLANNER_CACHE_ENV: "1", cfg.TRACE_ASSETS_ENV: "0"}
    assert cfg.load("paper").env() == {}
    # The caller's environment wins over the config's, as the cells' own env always did.
    monkeypatch.setenv(cfg.LINGER_ENV, "0")
    for name in (cfg.PLANNER_CACHE_ENV, "X_CELL"):
        monkeypatch.setenv(name, "")  # recorded, so the undo removes what apply_env sets
        monkeypatch.delenv(name)
    effective = cfg.apply_env(default, extra={"X_CELL": "palette"})
    assert effective == {
        "X_CELL": "palette",
        cfg.LINGER_ENV: "0",
        cfg.PLANNER_CACHE_ENV: "1",
        cfg.TRACE_ASSETS_ENV: "0",
    }


def test_a_suite_env_pin_wins_over_the_shell(config_sandbox, monkeypatch, capsys):
    """The VLM a suite measures with is pinned: an export left in the shell must not change it."""
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-2.5-flash")
    fast = cfg.load("fast")
    task = "tool_sort/level1_1"
    effective = cfg.apply_env(fast, extra=fast.env(task), tasks=[task])  # as `vabar bench` calls it
    assert effective["GAP_VLM_MODEL"] == os.environ["GAP_VLM_MODEL"] == "gemini-3.1-flash-lite"
    assert "the suite pins it" in capsys.readouterr().err
    # Runtime env still yields to the shell, and a suite the run does not touch pins nothing.
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv(cfg.LINGER_ENV, "0")
    other = "gear_mesh/gearmesh-easy-pair-01"
    effective = cfg.apply_env(fast, extra=fast.env(other), tasks=[other])
    assert effective[cfg.LINGER_ENV] == "0"
    assert os.environ["GAP_VLM_MODEL"] == "gemini-2.5-flash"
    assert cfg.load("default").table("tool_sort/level1_1")["env"] == {"GAP_VLM_MODEL": "gemini-2.5-flash"}


# ----------------------------------------------------------------------------- workers
def test_workers(monkeypatch):
    assert cfg.resolve_workers(cfg.load("default"), 30)[0] == 1  # default is one process
    default = cfg.load("default", ["workers=auto"])
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (93.0, "GPU 2"))
    monkeypatch.setattr(cfg.os, "cpu_count", lambda: 240)
    assert cfg.resolve_workers(default, 30)[0] == 4  # 93 GB / 20 GB
    assert cfg.resolve_workers(default, 4)[0] == 4  # no more workers than waves
    monkeypatch.setattr(cfg.os, "cpu_count", lambda: 8)
    assert cfg.resolve_workers(default, 30)[0] == 2  # one per four cores
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (5.0, "GPU 0"))
    assert cfg.resolve_workers(default, 30)[0] == 1
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: None)
    assert cfg.resolve_workers(default, 30) == (1, "auto: nvidia-smi unavailable, so one")
    assert cfg.resolve_workers(cfg.load("default", ["workers=3"]), 30)[0] == 3
    assert cfg.resolve_workers(cfg.load("paper"), 30)[0] == 1


def test_shared_tool_servers_are_counted_once(monkeypatch):
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (47.0, "GPU 0"))
    config = cfg.load("default", ["suites.gear_mesh.worker_gb=5", "suites.gear_mesh.shared_gb=6.5"])
    n, why = cfg.resolve_workers(config, 30, suite="gear_mesh")
    assert n == 8  # (47 - 6.5) // 5; the suite's ceiling is 8
    config = cfg.load("default", ["suites.gear_mesh.worker_gb=5", "share_tool_servers=false"])
    n, why = cfg.resolve_workers(config, 30, suite="gear_mesh")
    assert n == int(47.0 // (5 + config.suites["gear_mesh"]["shared_gb"]))  # every worker starts its own
    assert "held to" in why
    monkeypatch.setattr(cfg, "free_gpu_gb", lambda: (5.0, "GPU 0"))
    assert cfg.resolve_workers(cfg.load("default"), 30, suite="gear_mesh")[0] == 1  # never fewer than one


# ----------------------------------------------------------------------------- provenance
def test_provenance(tmp_path):
    config = cfg.load("default", ["workers=2"])
    path = cfg.write_provenance(tmp_path / "run", config, cell="tool_hanging/L1")
    record = json.loads(path.read_text())
    assert path.name == "vabar_config.json"
    assert record["config"] == "default" and record["set"] == ["workers=2"] and record["cell"] == "tool_hanging/L1"
    assert record["sha256"] == config.sha256 and record["resolved"] == config.resolved()
    assert record["vabar_rev"] and record["vabar_rev"] != "unknown"


def test_replacement_map_only_swaps_the_old_name(tmp_path, monkeypatch):
    """`physics = { old = new }` rewrites a variant only when it is staged on `old`."""
    path = tmp_path / "swap.toml"
    path.write_text(
        'inherits = "paper"\n[suites.tool_hanging]\nphysics = { "tool_hanging_480hz@v1" = "tool_hanging_960hz@v1" }\n'
    )
    config = cfg.load(str(path))
    assert config.table("tool_hanging/wrenches_level1")["physics"] == {"tool_hanging_480hz@v1": "tool_hanging_960hz@v1"}
    with pytest.raises(cfg.ConfigError):
        bad = tmp_path / "bad.toml"
        bad.write_text('[suites.tool_hanging]\nphysics = { "tool_hanging_480hz@v1" = "unversioned" }\n')
        cfg.load(str(bad))


def test_trace_assets_per_config():
    """default and fast skip trace assets; paper keeps v1.0's full traces."""
    assert cfg.load("fast").env()[cfg.TRACE_ASSETS_ENV] == "0"
    assert cfg.TRACE_ASSETS_ENV not in cfg.load("paper").env()
    assert cfg.TRACE_ASSETS_ENV not in cfg.load("default", ["trace_assets=true"]).env()
