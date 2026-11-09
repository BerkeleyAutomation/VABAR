# SPDX-License-Identifier: Apache-2.0
"""Does every VABAR variant load on the installed RoboSimStudio?

One suite per subprocess, so a suite that fails to import cannot hide the state
of the others (RoboSimStudio scans a root as a whole and stops at its first
broken suite). For each suite the probe reports:

* ``load``: the suite's own Python ran and every document lowered;
* ``variants``: how many registered;
* ``env``: for the first ``--envs`` variants, an ``Env`` compiled and reset.

    python -m vabar probe                    # every suite, one env each
    python -m vabar probe --suite tool_sort --envs 0
    python -m vabar probe --markdown docs/probe.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback

from . import TASKS

__all__ = ["main", "probe_suite"]


def _suites() -> list[str]:
    from robosimstudio.roots import is_task_dir

    return [p.name for p in sorted(TASKS.iterdir()) if is_task_dir(p)]


def _load_one(suite: str) -> None:
    """Scan one suite of the VABAR root and nothing else."""
    from robosimstudio_bench.taskbuild import discovery

    from . import install

    install()
    root = TASKS
    # Mark the root loaded so a registry lookup does not scan the other suites.
    discovery._LOADED.add(str(root.resolve()))
    package = discovery.package_for_root(root)
    discovery._namespace(discovery.PARENT_PACKAGE, [])
    discovery._namespace(package, [str(root)])
    child = root / suite
    loader = child / "task.py"
    if loader.is_file():
        discovery._exec(f"{package}.{suite}", loader, is_package=True)
    else:
        discovery._generic(child, f"{package}.{suite}")


def _short(exc: BaseException) -> str:
    """The innermost cause, on one line."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:400]


def probe_suite(suite: str, envs: int = 1, steps: int = 0) -> dict:
    """The in-process half: load one suite, then compile and reset ``envs`` variants."""
    row: dict = {"suite": suite, "load": "", "variants": 0, "envs": []}
    t0 = time.time()
    try:
        _load_one(suite)
        row["load"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- a probe reports, it does not stop
        row["load"] = _short(exc)
        row["trace"] = traceback.format_exc()[-4000:]
        return row
    row["load_s"] = round(time.time() - t0, 1)

    from robosimstudio_bench.taskbuild.registry import get_suite

    variants = list(get_suite(suite).variant_names)
    row["variants"] = len(variants)
    from robosimstudio_bench.envs.env import Env

    for name in variants[:envs]:
        entry = {"task": f"{suite}/{name}"}
        t0 = time.time()
        env = None
        try:
            env = Env(f"{suite}/{name}")
            env.reset(0)
            for _ in range(steps):
                env.step(env.idle_action())
            entry["result"] = "ok"
        except Exception as exc:  # noqa: BLE001
            entry["result"] = _short(exc)
            entry["trace"] = traceback.format_exc()[-4000:]
        finally:
            if env is not None and hasattr(env, "close"):
                with contextlib.suppress(Exception):
                    env.close()
        entry["s"] = round(time.time() - t0, 1)
        row["envs"].append(entry)
    return row


def _markdown(rows: list[dict]) -> str:
    lines = ["| Suite | Load | Variants | Env compile + reset |", "| --- | --- | --- | --- |"]
    for r in rows:
        envs = "; ".join(f"`{e['task']}` {e['result']}" for e in r["envs"]) or "-"
        lines.append(f"| `{r['suite']}` | {r['load']} | {r['variants']} | {envs} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vabar probe", description=__doc__.splitlines()[0])
    parser.add_argument("--suite", action="append", help="only these suites (repeatable)")
    parser.add_argument("--envs", type=int, default=1, help="variants per suite to compile and reset")
    parser.add_argument("--steps", type=int, default=0, help="idle steps after each reset")
    parser.add_argument("--json", help="write the rows here")
    parser.add_argument("--markdown", help="write a table here")
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.in_process:
        print(json.dumps(probe_suite(args.suite[0], args.envs, args.steps)))
        return 0

    rows = []
    for suite in args.suite or _suites():
        cmd = [sys.executable, "-m", "vabar", "probe", "--in-process", "--suite", suite, "--envs", str(args.envs)]
        cmd += ["--steps", str(args.steps)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), check=False)
        last = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if last:
            row = json.loads(last[-1])
        else:
            row = {"suite": suite, "load": f"crashed (exit {proc.returncode})", "variants": 0, "envs": []}
            row["trace"] = proc.stderr[-4000:]
        rows.append(row)
        status = row["load"] if row["load"] != "ok" else f"ok, {row['variants']} variants"
        print(f"{suite:18s} {status}")
        for e in row["envs"]:
            print(f"  {e['task']:50s} {e['result']} ({e['s']} s)")
        if row.get("trace") and row["load"] != "ok":
            print("  " + row["trace"].strip().splitlines()[-1])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(_markdown(rows))
    ok = all(r["load"] == "ok" and all(e["result"] == "ok" for e in r["envs"]) for r in rows)
    return 0 if ok else 1
