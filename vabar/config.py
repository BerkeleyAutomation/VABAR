# SPDX-License-Identifier: Apache-2.0
"""Experiment configs: HOW a VABAR run executes, kept apart from WHAT it measures.

A benchmark cell (``benchmark/vabar_v1.json``) fixes the instances, the graph and
the flags the graph was evaluated under. A config fixes everything else: the
clock the simulator keeps while a graph thinks, when observations are rendered,
how many worlds share a model, how many processes run, and -- per suite -- the
physics arrangement, camera rig, step budget and variant a faster setup swaps in.

Three ship in ``vabar/configs/``:

* ``default`` -- what a single-GPU run wants: the simulator waits for the graph
  (``clock = "lockstep"``), observations are rendered when a graph asks for one
  (``obs_render = "on-demand"``), tool servers and the planner stay up across
  episodes, a cell's episodes run as fleet waves, and the worker count follows
  free GPU memory. Physics is untouched.
* ``fast`` -- ``default`` plus per-suite physics, cameras and budgets tuned for
  throughput, on the first 15 of a cell's instances, for trying a change quickly.
  Its suite tables are filled by measurement sweeps.
* ``paper`` -- exactly the v1.0 protocol: a free-running clock, per-step
  observation rendering, and the cells' own flags, serially.

Selection, lowest precedence first: the config file (and whatever it
``inherits``), then ``--set key=value`` on the command line. ``--config NAME``
names a built-in or a path to a ``.toml``; ``$VABAR_CONFIG`` does the same when
no flag is given. The choice is exported as ``$VABAR_CONFIG`` (and the
overrides as ``$VABAR_SET``) so that pool and fleet workers -- fresh
interpreters that import :mod:`vabar` from a suite's ``staging.py`` -- resolve
the same config and apply the same per-suite overrides.

The file format::

    inherits = "default"            # a built-in name, or a path relative to this file
    description = "..."

    [runtime]
    clock = "lockstep"              # free | lockstep            ("" = the runner's own)
    obs_render = "on-demand"        # per-step | on-demand       ("" = the runner's own)
    render_every = 0                # render observations every N steps (0 = the runner's own)
    renderer = "rtx"                # rtx | gl | tiled | auto    ("" = leave it to the cell / vabar run)
    keep_tool_servers = true        # GAP_TOOL_BUNDLE_LINGER=1: SAM 3, DINO, ... survive between episodes
    cache_planner = true            # ROBOSIMSTUDIO_PLANNER_CACHE=1: cuRobo is built once per worker
    trace_assets = false            # GAP_TRACE_ASSETS=0: graph traces keep JSON, skip PNG/npy assets
    num_worlds = 8                  # bench: worlds per fleet wave, min(this, a task's episodes)
    workers = "auto"                # bench: processes; "auto" = free GPU memory / worker_gb
    worker_gb = 20.0                # GPU memory one warm graph worker holds, for workers = "auto"
    shared_gb = 9.0                 # GPU memory of the shared SAM 3 + GroundingDINO: held once per run
                                    # with share_tool_servers and workers > 1, else by every worker
    mps = true                      # bench: with workers > 1, share the GPU through a private MPS daemon
    share_tool_servers = true       # bench: with workers > 1, one SAM 3 / GroundingDINO per GPU, not per worker
    bus_timeout = 0                 # seconds; 0 keeps the cell's (bench) or the runner's (run)
    instances = 15                  # bench: a cell's first N instances, 0 = all of them; --limit wins
    env = { NAME = "value" }        # extra environment for the run and its workers

    [suites.tool_hanging]           # or [suites."tool_hanging/wrenches_level1"] for one variant
    physics = "tool_hanging_960hz@v1"     # a registered arrangement name, for every variant ...
    physics = { "tool_hanging_480hz@v1" = "tool_hanging_fast@v1" }   # ... or only where it replaces old
    cameras = "tool_hanging_views@v1"     # a registered camera rig name (same two forms)
    max_steps = 6000
    num_worlds = 1                  # e.g. a suite whose scenes refuse to replicate
    workers = 4                     # bench: processes for this suite's cells (a ceiling, or "auto")
    worker_gb = 6.0                 # GPU memory one of its workers holds; caps workers on a smaller GPU
    shared_gb = 0.0                 # this suite's shared tool servers (a suite without SAM 3 / DINO)
    variants = { wrenches_level1 = "wrenches_level1_fast" }   # suite tables only
    env = { NAME = "value" }
    flags = ["--settle-steps", "30"]      # extra `vabar run` flags; a flag the cell sets is replaced

``physics``, ``cameras`` and ``max_steps`` change the registered variants
themselves (:func:`apply_overrides`), so they hold in every process that loads
the suites -- including runs that never pass through this CLI.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CONFIGS",
    "Config",
    "ConfigError",
    "activate",
    "active",
    "apply_env",
    "apply_overrides",
    "from_argv",
    "from_env",
    "load",
    "names",
    "provenance",
    "resolve",
    "resolve_workers",
    "write_provenance",
]

CONFIGS = Path(__file__).resolve().parent / "configs"
"""The built-in configs: ``<name>.toml``."""

DEFAULT = "default"
ENV_CONFIG = "VABAR_CONFIG"
"""Names the config (a built-in name or an absolute path); exported for workers."""
ENV_SET = "VABAR_SET"
"""The ``--set`` overrides as a JSON list of ``key=value``; exported for workers."""

LINGER_ENV = "GAP_TOOL_BUNDLE_LINGER"
PLANNER_CACHE_ENV = "ROBOSIMSTUDIO_PLANNER_CACHE"
TRACE_ASSETS_ENV = "GAP_TRACE_ASSETS"

CLOCKS = ("", "free", "lockstep")
OBS_RENDERS = ("", "per-step", "on-demand")
RENDERERS = ("", "auto", "rtx", "gl", "tiled")

RUNTIME_DEFAULTS: dict[str, Any] = {
    "clock": "",
    "obs_render": "",
    "render_every": 0,
    "renderer": "",
    "keep_tool_servers": False,
    "cache_planner": False,
    "trace_assets": True,
    "num_worlds": 1,
    "workers": 1,
    "worker_gb": 20.0,
    "shared_gb": 0.0,
    "mps": False,
    "share_tool_servers": False,
    "bus_timeout": 0.0,
    "instances": 0,
    "env": {},
}
"""Every runtime key, at the value that changes nothing about a v1.0 run."""

SUITE_KEYS = (
    "physics",
    "cameras",
    "max_steps",
    "num_worlds",
    "workers",
    "worker_gb",
    "shared_gb",
    "variants",
    "env",
    "flags",
    "renderer",
)
REGISTRY_KEYS = ("physics", "cameras", "max_steps")
"""The suite keys that rewrite a registered variant, rather than a run's flags or environment."""

TOP_KEYS = ("inherits", "description", "runtime", "suites")


class ConfigError(ValueError):
    """A config that does not parse, names something unknown, or states a value of the wrong kind."""


# ----------------------------------------------------------------------------- the config
@dataclass(frozen=True)
class Config:
    """One resolved config: inheritance applied, ``--set`` applied, validated.

    Attributes:
        name: The file's stem (``default``, ``fast``, ``paper`` or a user file's).
        ref: What reproduces it in another process: the built-in name, or the file's absolute path.
        description: The file's own one-liner.
        runtime: Every :data:`RUNTIME_DEFAULTS` key, filled.
        suites: ``"<suite>"`` or ``"<suite>/<variant>"`` -> its table.
        chain: The files read, base first.
        overrides: The ``--set`` expressions applied on top, in order.
    """

    name: str
    ref: str
    description: str
    runtime: dict[str, Any]
    suites: dict[str, dict[str, Any]]
    chain: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()

    # ---------------------------------------------------------------- identity
    def resolved(self) -> dict[str, Any]:
        """What the config means: runtime and suites, canonical. Hashed for provenance."""
        return json.loads(json.dumps({"runtime": self.runtime, "suites": self.suites}, sort_keys=True))

    @functools.cached_property
    def sha256(self) -> str:
        blob = json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---------------------------------------------------------------- per task
    def table(self, task: str) -> dict[str, Any]:
        """The suite table merged with the variant's own, for ``suite/variant`` (or a bare suite)."""
        suite, _, variant = task.partition("/")
        out = copy.deepcopy(self.suites.get(suite, {}))
        if variant:
            for key, value in self.suites.get(f"{suite}/{variant}", {}).items():
                if key == "env":
                    out["env"] = {**out.get("env", {}), **value}
                else:
                    out[key] = copy.deepcopy(value)
        return out

    def map_task(self, task: str) -> str:
        """``suite/variant`` after the suite's ``variants`` table renames it."""
        suite, _, variant = task.partition("/")
        renamed = self.suites.get(suite, {}).get("variants", {}).get(variant)
        return f"{suite}/{renamed}" if renamed else task

    def num_worlds(self, task: str) -> int:
        """Worlds per wave this config allows for ``task``: its suite's override, else the runtime's."""
        return int(self.table(task).get("num_worlds", self.runtime["num_worlds"]))

    def renderer(self, task: str) -> str:
        """The renderer ``task`` runs on: its suite's, unless ``--set renderer=`` says otherwise, else the runtime's.

        A suite states one when its perception was tuned on it. The command line
        beats the file, and ``--set suites.<suite>.renderer=`` beats both, as for ``workers``.
        """
        suite = task.partition("/")[0]
        stated = self.table(task).get("renderer", "")
        overruled = _set_on_command_line(self, "renderer") and not _set_on_command_line(
            self, f"suites.{suite}.renderer"
        )
        return stated if stated and not overruled else self.runtime["renderer"]

    def env(self, task: str = "") -> dict[str, str]:
        """The environment this config adds: runtime ``env``, the two switches, then the task's suite."""
        out: dict[str, str] = dict(self.runtime["env"])
        if self.runtime["keep_tool_servers"]:
            out.setdefault(LINGER_ENV, "1")
        if self.runtime["cache_planner"]:
            out.setdefault(PLANNER_CACHE_ENV, "1")
        if not self.runtime["trace_assets"]:
            out.setdefault(TRACE_ASSETS_ENV, "0")
        if task:
            out.update(self.table(task).get("env", {}))
        return out

    def registry_overrides(self) -> dict[str, dict[str, Any]]:
        """Suite/variant keys -> the part of their table that rewrites registered variants."""
        out = {}
        for key, table in self.suites.items():
            picked = {k: table[k] for k in REGISTRY_KEYS if k in table}
            if picked:
                out[key] = picked
        return out

    def rewrites(self, task: str) -> bool:
        """Whether this config changes ``task``'s registered variant (physics, cameras, budget)."""
        return any(k in self.table(task) for k in REGISTRY_KEYS)


# ----------------------------------------------------------------------------- loading
def names() -> list[str]:
    """The built-in configs."""
    return sorted(p.stem for p in CONFIGS.glob("*.toml"))


def _is_path(ref: str) -> bool:
    return ref.endswith(".toml") or "/" in ref or os.sep in ref


def _locate(ref: str, *, relative_to: Path | None = None) -> Path:
    """A config reference to its file: a built-in name, or a path (relative to ``relative_to`` if given)."""
    if not ref:
        raise ConfigError("an empty config name")
    if _is_path(ref):
        path = Path(ref).expanduser()
        if not path.is_absolute() and relative_to is not None:
            path = relative_to / path
        if not path.is_file():
            raise ConfigError(f"no config file at {path}")
        return path.resolve()
    builtin = CONFIGS / f"{ref}.toml"
    if builtin.is_file():
        return builtin
    raise ConfigError(f"unknown config {ref!r}: the built-in ones are {names()}; or name a .toml file")


def _merge(base: dict, over: dict) -> dict:
    """``over`` on ``base``: tables merge key by key, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path, seen: tuple[Path, ...] = ()) -> tuple[dict, tuple[Path, ...]]:
    """One file with its ``inherits`` chain merged under it. Returns the data and the files, base first."""
    if path in seen:
        raise ConfigError(f"config inheritance cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    unknown = sorted(set(data) - set(TOP_KEYS))
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {unknown}; a config has {list(TOP_KEYS)}")
    parent = data.pop("inherits", "")
    if not parent:
        return data, (path,)
    if not isinstance(parent, str):
        raise ConfigError(f"{path}: inherits must name one config, got {parent!r}")
    base, chain = _read(_locate(parent, relative_to=path.parent), (*seen, path))
    description = data.get("description", "")
    merged = _merge(base, data)
    merged["description"] = description
    return merged, (*chain, path)


def _literal(text: str) -> Any:
    """A ``--set`` value: a TOML literal when it is one (8, true, 1.5, "x", [..], {..}), else the bare text."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _apply_set(data: dict, expression: str) -> None:
    """Apply one ``key=value``. ``clock=lockstep`` is short for ``runtime.clock=lockstep``."""
    key, sep, raw = expression.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"--set {expression!r}: expected key=value, e.g. runtime.clock=lockstep")
    parts = key.split(".")
    if parts[0] in RUNTIME_DEFAULTS:
        parts = ["runtime", *parts]
    if parts[0] not in ("runtime", "suites") or len(parts) < 2:
        raise ConfigError(
            f"--set {key!r}: name a runtime key ({', '.join(RUNTIME_DEFAULTS)}), runtime.<key>, "
            "runtime.env.<NAME> or suites.<suite>[/<variant>].<key>"
        )
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"--set {key!r}: {part!r} is a value, not a table")
        node = child
    node[parts[-1]] = _literal(raw.strip())


def _suite_dirs() -> set[str]:
    from . import TASKS

    return {p.name for p in TASKS.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}


def _check_str_map(where: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a table of NAME = value, got {value!r}")
    out = {}
    for k, v in value.items():
        if isinstance(v, dict | list):
            raise ConfigError(f"{where}.{k} must be a string or number, got {v!r}")
        out[str(k)] = ("1" if v else "0") if isinstance(v, bool) else str(v)
    return out


def _validate(where: str, data: dict) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    runtime_in = data.get("runtime", {}) or {}
    if not isinstance(runtime_in, dict):
        raise ConfigError(f"{where}: [runtime] must be a table")
    unknown = sorted(set(runtime_in) - set(RUNTIME_DEFAULTS))
    if unknown:
        raise ConfigError(f"{where}: unknown runtime key(s) {unknown}; known: {list(RUNTIME_DEFAULTS)}")
    runtime = {**copy.deepcopy(RUNTIME_DEFAULTS), **copy.deepcopy(runtime_in)}

    def choice(key: str, allowed: tuple[str, ...]) -> None:
        if runtime[key] not in allowed:
            raise ConfigError(f"{where}: runtime.{key} = {runtime[key]!r}; one of {[a for a in allowed if a]} or ''")

    def integer(key: str, low: int, table: dict = runtime, prefix: str = "runtime") -> None:
        value = table[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ConfigError(f"{where}: {prefix}.{key} = {value!r}; an integer >= {low}")

    def number(key: str, low: float) -> None:
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value < low:
            raise ConfigError(f"{where}: runtime.{key} = {value!r}; a number >= {low}")
        runtime[key] = float(value)

    choice("clock", CLOCKS)
    choice("obs_render", OBS_RENDERS)
    choice("renderer", RENDERERS)
    integer("render_every", 0)
    integer("num_worlds", 1)
    integer("instances", 0)
    if runtime["workers"] != "auto":
        integer("workers", 1)
    number("worker_gb", 0.5)
    number("shared_gb", 0.0)
    number("bus_timeout", 0.0)
    for key in ("keep_tool_servers", "cache_planner", "trace_assets", "mps", "share_tool_servers"):
        if not isinstance(runtime[key], bool):
            raise ConfigError(f"{where}: runtime.{key} = {runtime[key]!r}; true or false")
    runtime["env"] = _check_str_map(f"{where}: runtime.env", runtime["env"])

    suites_in = data.get("suites", {}) or {}
    if not isinstance(suites_in, dict):
        raise ConfigError(f"{where}: [suites] must be a table of [suites.<suite>] tables")
    known = _suite_dirs()
    suites: dict[str, dict[str, Any]] = {}
    for key, table in suites_in.items():
        suite, _, variant = key.partition("/")
        if suite not in known:
            raise ConfigError(f"{where}: [suites.{key}] names no suite; the suites are {sorted(known)}")
        if not isinstance(table, dict):
            raise ConfigError(f"{where}: [suites.{key}] must be a table")
        bad = sorted(set(table) - set(SUITE_KEYS))
        if bad:
            raise ConfigError(f"{where}: [suites.{key}] unknown key(s) {bad}; known: {list(SUITE_KEYS)}")
        out = copy.deepcopy(table)
        for name in ("physics", "cameras"):
            if name not in out:
                continue
            value = out[name]
            if isinstance(value, dict):
                # A replacement map: swap only the variants staged on the old name.
                out[name] = _check_str_map(f"{where}: suites.{key}.{name}", value)
                names = [n for pair in out[name].items() for n in pair]
            else:
                names = [value]
            if not all(isinstance(n, str) and "@" in n for n in names):
                raise ConfigError(
                    f"{where}: suites.{key}.{name} = {value!r}; a registered versioned name (x@v1), "
                    'or a map of them ({ "old@v1" = "new@v1" })'
                )
        if "max_steps" in out:
            integer("max_steps", 1, out, f"suites.{key}")
        if "num_worlds" in out:
            integer("num_worlds", 1, out, f"suites.{key}")
        if "workers" in out:
            if variant:
                raise ConfigError(f"{where}: [suites.{key}] sets workers; a cell is one suite, so [suites.{suite}]")
            if out["workers"] != "auto":
                integer("workers", 1, out, f"suites.{key}")
        for name, low in (("worker_gb", 0.5), ("shared_gb", 0.0)):
            if name not in out:
                continue
            value = out[name]
            if variant:
                raise ConfigError(f"{where}: [suites.{key}] sets {name}; a cell is one suite, so [suites.{suite}]")
            if isinstance(value, bool) or not isinstance(value, int | float) or value < low:
                raise ConfigError(f"{where}: suites.{key}.{name} = {value!r}; a number >= {low}")
            out[name] = float(value)
        if "variants" in out:
            if variant:
                raise ConfigError(f"{where}: [suites.{key}] renames variants; that belongs in [suites.{suite}]")
            out["variants"] = _check_str_map(f"{where}: suites.{key}.variants", out["variants"])
        if "env" in out:
            out["env"] = _check_str_map(f"{where}: suites.{key}.env", out["env"])
        if "renderer" in out and out["renderer"] not in RENDERERS:
            raise ConfigError(f"{where}: suites.{key}.renderer = {out['renderer']!r}; one of {RENDERERS}")
        if "flags" in out:
            flags = out["flags"]
            if not isinstance(flags, list) or not all(isinstance(f, str | int | float) for f in flags):
                raise ConfigError(f"{where}: suites.{key}.flags must be a list of strings, got {flags!r}")
            out["flags"] = [str(f) for f in flags]
        if out:
            suites[key] = out
    return runtime, suites


def load(ref: str = DEFAULT, overrides: tuple[str, ...] | list[str] = ()) -> Config:
    """Resolve a config: its file, what it ``inherits``, then ``overrides`` (``key=value``) on top."""
    path = _locate(ref)
    data, chain = _read(path)
    overrides = tuple(overrides)
    for expression in overrides:
        _apply_set(data, expression)
    runtime, suites = _validate(str(path.name), data)
    builtin = path.parent == CONFIGS
    return Config(
        name=path.stem,
        ref=path.stem if builtin else str(path),
        description=str(data.get("description", "")),
        runtime=runtime,
        suites=suites,
        chain=tuple(str(p) for p in chain),
        overrides=overrides,
    )


def from_env() -> Config:
    """The config ``$VABAR_CONFIG`` and ``$VABAR_SET`` name, else ``default``."""
    return resolve()


def resolve(ref: str | None = None, sets: list[str] | tuple[str, ...] = ()) -> Config:
    """The config a command line names: ``--config ref`` with ``--set`` on top.

    Without ``ref``, ``$VABAR_CONFIG`` (and its exported ``$VABAR_SET``) name the
    base -- ``default`` when unset -- and ``sets`` apply on top of those.
    """
    if ref:
        return load(ref, sets)
    try:
        inherited = json.loads(os.environ.get(ENV_SET) or "[]")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"${ENV_SET} is not a JSON list: {exc}") from None
    return load(os.environ.get(ENV_CONFIG) or DEFAULT, [*inherited, *sets])


def from_argv(argv: list[str]) -> tuple[Config, list[str]]:
    """Take ``--config X`` and ``--set k=v`` out of ``argv``; the config they name (:func:`resolve`) and the rest."""
    rest: list[str] = []
    ref = ""
    sets: list[str] = []
    it = iter(argv)
    for token in it:
        if token in ("--config", "--set"):
            value = next(it, None)
            if value is None:
                raise ConfigError(f"{token} needs a value")
        elif token.startswith(("--config=", "--set=")):
            token, _, value = token.partition("=")
        else:
            rest.append(token)
            continue
        if token == "--config":
            ref = value
        else:
            sets.append(value)
    return resolve(ref, sets), rest


# ----------------------------------------------------------------------------- activation
_ACTIVE: Config | None = None


def active() -> Config:
    """The config this process runs under: the one :func:`activate` set, else the environment's."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = from_env()
    return _ACTIVE


def activate(config: Config, *, export: bool = True) -> Config:
    """Make ``config`` this process's, export it for workers, and hook it into the suite registry."""
    global _ACTIVE
    _ACTIVE = config
    if export:
        os.environ[ENV_CONFIG] = config.ref
        if config.overrides:
            os.environ[ENV_SET] = json.dumps(list(config.overrides))
        else:
            os.environ.pop(ENV_SET, None)
    if config.registry_overrides() or _APPLIED:
        _install_hook()
    return config


def activate_from_env() -> None:
    """What ``import vabar`` does: hook the environment's config into the registry if it rewrites anything.

    Cheap when it does not -- one small TOML read, no registry import -- which is
    every process under ``default`` and ``paper``.
    """
    config = active()
    if config.registry_overrides():
        _install_hook()


def apply_env(
    config: Config, task: str = "", extra: dict[str, str] | None = None, tasks: tuple[str, ...] | list[str] = ()
) -> dict[str, str]:
    """Put the config's environment (and ``extra``) into ``os.environ``.

    A variable already set wins over ``runtime.env`` and a cell's own, but not
    over a suite table's ``env`` (or a ``--set`` of one): those pin what the
    suite measures -- the VLM a graph asks, for one -- and an export left in a
    shell must not change it unseen. Each value replaced that way is printed.

    Args:
        task: The task run (``vabar run``), whose suite table pins.
        tasks: The tasks a bench plan runs, when there is no single ``task``.

    Returns:
        Each variable as it now stands, which is what provenance records.
    """
    wanted = {**(extra or {}), **config.env(task)}
    pinned: dict[str, str] = {}
    for name in (task, *tasks):
        if name:
            pinned.update(config.table(name).get("env", {}))
    for key, value in wanted.items():
        if key in pinned:
            shell = os.environ.get(key)
            if shell is not None and shell != pinned[key]:
                print(
                    f"[config] {key}={pinned[key]} (the suite pins it; the shell's {shell!r} is not used)",
                    file=sys.stderr,
                )
            os.environ[key] = pinned[key]
        else:
            os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in wanted}


# ----------------------------------------------------------------------------- the registry hook
_HOOKED = False
_ORIGINALS: dict[str, Any] = {}
"""Suite name -> the suite as discovered, before any config rewrote it."""
_APPLIED: dict[str, tuple[Any, str]] = {}
"""Suite name -> (the suite object this module registered, the config sha it was built from)."""


def _install_hook() -> None:
    """Run :func:`apply_overrides` after the suite registry discovers its suites. Idempotent.

    The hook is a populator on ``SUITES`` itself (``Registry.add_discover``):
    the registry runs its populators in order on every load, and the one that
    scans the task roots is installed first, by the registry's own module. So
    ours runs once every suite is registered, in every process -- the parent,
    and each pool or fleet worker, where ``import vabar`` from a suite's
    ``staging.py`` installs it while the scan is in progress (the registry's
    load loop picks up a populator appended mid-iteration). Adding a populator
    re-arms the load, so a registry that had already loaded runs it on the next
    lookup.
    """
    global _HOOKED
    from robosimstudio_bench.taskbuild.registry import SUITES

    if not _HOOKED:
        SUITES.add_discover(apply_overrides)
        _HOOKED = True
    elif getattr(SUITES, "_loaded", False):
        # Already loaded, perhaps under another config: rewrite now (a no-op
        # when nothing changed). Only when loaded, so this never triggers a
        # scan -- which could run before the task root is declared.
        apply_overrides()


def _rewrite(spec: Any, table: dict[str, Any], config: Config) -> Any:
    from robosimstudio_bench.taskbuild.arrangements import CAMERA_RIGS, PHYSICS

    changes: dict[str, Any] = {}
    for key, registry, what in (("physics", PHYSICS, "physics arrangement"), ("cameras", CAMERA_RIGS, "camera rig")):
        if key not in table:
            continue
        wanted = table[key]
        pairs = wanted.items() if isinstance(wanted, dict) else [(None, wanted)]
        for old, name in pairs:
            for n in (old, name):
                if n is not None and n not in registry:
                    raise ConfigError(
                        f"config {config.name}: {spec.full_name} {key} names {n!r}, which is not a registered "
                        f"{what}; registered: {registry.names()}"
                    )
            # A map entry applies only to a variant staged on its old name; a
            # plain name applies to every variant the table covers.
            if old is not None and registry.get(old) != getattr(spec, key):
                continue
            value = registry.get(name)
            if value != getattr(spec, key):
                changes[key] = value
            break
    if "max_steps" in table and int(table["max_steps"]) != spec.max_steps:
        changes["max_steps"] = int(table["max_steps"])
    return dataclasses.replace(spec, **changes) if changes else spec


def apply_overrides() -> None:
    """Rewrite the registered suites the active config names: physics, cameras, budget. Idempotent.

    Each variant is rebuilt with ``dataclasses.replace`` from the suite as it was
    discovered and the suite re-registered with ``SUITES.replace``, so every
    ``get_task`` from then on -- the runner's, the pool's, a fleet's -- sees the
    configured variant. Physics and cameras are stored resolved on a lowered
    ``TaskSpec`` (an ``ArrangementSpec``, a tuple of ``CameraSpec``), so the
    names are looked up in the ``PHYSICS`` and ``CAMERA_RIGS`` registries the
    suites' ``staging.py`` populate.

    Re-applying under the same config is a no-op; under another, each suite is
    rebuilt from its original, so a suite the new config leaves alone is put back.
    """
    from robosimstudio_bench.taskbuild.registry import SUITES

    config = active()
    wanted = {key.partition("/")[0] for key in config.registry_overrides()}
    registered = set(SUITES.names())
    for name in sorted((wanted | set(_APPLIED)) & registered):
        current = SUITES.get(name)
        ours = _APPLIED.get(name)
        if ours is not None and ours[0] is current:
            if ours[1] == config.sha256:
                continue
            original = _ORIGINALS[name]
        else:
            # First time, or the suite was re-registered (extended) since: what
            # is registered now is the base to rewrite.
            original = _ORIGINALS[name] = current
        variants = tuple(_rewrite(v, config.table(f"{name}/{v.name}"), config) for v in original.variants)
        if all(a is b for a, b in zip(variants, original.variants, strict=True)):
            rebuilt = original
        else:
            rebuilt = dataclasses.replace(original, variants=variants)
        if rebuilt is not current:
            SUITES.replace(name, rebuilt, origin=f"vabar config {config.name}")
        _APPLIED[name] = (rebuilt, config.sha256)


# ----------------------------------------------------------------------------- workers
def free_gpu_gb() -> tuple[float, str] | None:
    """Free memory on the first visible GPU, in GB, and which GPU that is; ``None`` without ``nvidia-smi``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
    for index, uuid, free in rows:
        if not visible or visible in (index, uuid) or (visible.startswith("GPU-") and uuid.startswith(visible)):
            return float(free) / 1024.0, f"GPU {index}"
    return None


def _set_on_command_line(config: Config, key: str) -> bool:
    """Whether ``--set`` named ``key`` (a runtime key also matches its ``runtime.`` spelling)."""
    names = {key, f"runtime.{key}"} if "." not in key else {key}
    return any(o.partition("=")[0].strip() in names for o in config.overrides)


def resolve_workers(config: Config, units: int, suite: str = "") -> tuple[int, str]:
    """How many processes a run gets, and why: the config's count, or ``auto`` from free GPU memory.

    ``[suites.<suite>] workers`` and ``worker_gb`` win over the ``runtime`` ones for
    that suite's cells, unless the runtime key came from ``--set`` (the command
    line beats the file; ``--set suites.<suite>.workers=N`` beats both). A suite's
    count is a ceiling measured on a large GPU: it is also held to what fits in
    free memory, so the same config fits a smaller card. ``auto`` is what fits,
    at most one per four CPU cores (a graph worker starved of CPU scores
    differently). Never more than there are units of work. ``units`` is the
    run's waves -- episodes, for a serial job.

    **What fits:** ``n`` workers hold ``shared_gb + n * worker_gb`` when they share
    one SAM 3 / GroundingDINO (``share_tool_servers``), and ``n * (worker_gb +
    shared_gb)`` when each starts its own. ``worker_gb`` is the measured cost of
    one more worker with the servers shared. Counting the servers per worker
    instead would hold a 48 GB card to half the gear_mesh workers it can run.
    """
    units = max(1, int(units))
    table = dict(config.suites.get(suite, {})) if suite else {}
    for key in ("workers", "worker_gb", "shared_gb"):
        if _set_on_command_line(config, key) and not _set_on_command_line(config, f"suites.{suite}.{key}"):
            table.pop(key, None)
    workers = table.get("workers", config.runtime["workers"])
    per = float(table.get("worker_gb", config.runtime["worker_gb"]))
    shared = float(table.get("shared_gb", config.runtime["shared_gb"]))

    def fits(gb: float) -> int:
        if config.runtime["share_tool_servers"]:
            return int(max(0.0, gb - shared) // per)
        return int(gb // (per + shared))

    def per_text() -> str:
        if config.runtime["share_tool_servers"] and shared:
            return f"{shared:g} GB of shared tool servers + {per:g} GB per worker"
        return f"{per + shared:g} GB per worker"

    if workers != "auto":
        n = max(1, min(int(workers), units))
        if "workers" not in table:
            return n, f"config {config.name}"
        why = f"config {config.name} [suites.{suite}]"
        free = free_gpu_gb() if n > 1 else None
        if free is not None and fits(free[0]) < n:
            n = max(1, fits(free[0]))
            why += f", held to {n} by {free[0]:.0f} GB free on {free[1]} ({per_text()})"
        return n, why
    free = free_gpu_gb()
    if free is None:
        return 1, "auto: nvidia-smi unavailable, so one"
    gb, gpu = free
    by_cpu = max(1, (os.cpu_count() or 4) // 4)
    n = max(1, min(fits(gb), by_cpu, units))
    return n, f"auto: {gb:.0f} GB free on {gpu} ({per_text()}), {units} units of work"


# ----------------------------------------------------------------------------- provenance
@functools.cache
def git_rev() -> tuple[str, bool]:
    """The VABAR checkout's commit and whether its tracked files differ from it; ``("unknown", False)`` if not git."""
    from . import REPO

    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    return rev, bool(dirty)


def provenance(config: Config, **extra: Any) -> dict[str, Any]:
    """What ``vabar_config.json`` holds: the resolved config, its sha256, the VABAR revision."""
    rev, dirty = git_rev()
    return {
        "schema": "vabar.config.v1",
        "config": config.name,
        "ref": config.ref,
        "files": list(config.chain),
        "set": list(config.overrides),
        "sha256": config.sha256,
        "resolved": config.resolved(),
        "vabar_rev": rev,
        "vabar_dirty": dirty,
        "argv": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def write_provenance(run_dir: str | Path, config: Config, **extra: Any) -> Path:
    """Write ``vabar_config.json`` into ``run_dir`` (created if absent)."""
    path = Path(run_dir) / "vabar_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance(config, **extra), indent=1, sort_keys=False) + "\n")
    return path
