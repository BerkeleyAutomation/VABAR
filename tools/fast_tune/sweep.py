# SPDX-License-Identifier: Apache-2.0
"""Run a list of fast_tune probes, a few at a time, one log per run.

    python tools/fast_tune/sweep.py --out DIR --parallel 4 \\
        "tool_hanging th_ss4 idle_w2 :: --task tool_hanging/wrenches_level2 --solve idle --max-steps 600" ...

Each job is ``"<suite> <candidate> <label> :: <python -m vabar run args>"``; the run is
named ``<candidate>__<label>`` under ``--out`` and its log goes beside it. A job whose
run directory already holds ``episode_results.jsonl`` is skipped, so a sweep resumes.
``--file`` reads jobs one per line (``#`` comments allowed).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parse(line: str) -> tuple[str, str, str, list[str]]:
    head, _, tail = line.partition("::")
    suite, candidate, label = head.split()
    return suite, candidate, label, shlex.split(tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--peaks", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument(
        "--pythonpath",
        default="",
        help="a directory put first on sys.path of every run (e.g. a torchvision overlay for SAM3); this interpreter only, not its subprocesses",
    )
    ap.add_argument("--hf-token-file", default="", help="read HF_TOKEN from this file for every run")
    ap.add_argument(
        "--open-robot-skills",
        default="",
        help=(
            "VABAR_OPEN_ROBOT_SKILLS for every run. From a git worktree the default (../open-robot-skills "
            "beside the checkout) does not exist, and a graph then finds none of its geometry/motion bundles"
        ),
    )
    args = ap.parse_args()
    lines = list(args.jobs)
    if args.file:
        lines += [
            ln for ln in Path(args.file).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")
        ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "6")
    if args.pythonpath:
        # Read by _common.main into its own sys.path; deliberately not PYTHONPATH, which
        # the graph's tool-bundle subprocesses (own venvs, own torch) would inherit.
        env["FAST_TUNE_SYSPATH"] = str(Path(args.pythonpath).resolve())
    if args.open_robot_skills:
        env["VABAR_OPEN_ROBOT_SKILLS"] = str(Path(args.open_robot_skills).resolve())
    if args.hf_token_file:
        env["HF_TOKEN"] = Path(args.hf_token_file).expanduser().read_text().strip()
    pending = [parse(ln) for ln in lines]
    running: list[tuple[subprocess.Popen, str, float]] = []
    while pending or running:
        while pending and len(running) < args.parallel:
            suite, candidate, label, run_args = pending.pop(0)
            name = f"{candidate}__{label}"
            if (out / name / "episode_results.jsonl").exists():
                print(f"[sweep] skip {name} (done)", flush=True)
                continue
            cmd = [sys.executable, str(HERE / f"{suite}.py"), candidate]
            if args.peaks:
                cmd.append("--peaks")
            if args.drift:
                cmd.append("--drift")
            cmd += ["--", *run_args, "--out", str(out), "--name", name]
            log = open(out / f"{name}.log", "w")  # noqa: SIM115
            print(f"[sweep] start {name}", flush=True)
            running.append(
                (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), name, time.time())
            )
        time.sleep(2.0)
        for item in list(running):
            proc, name, t0 = item
            if proc.poll() is None and time.time() - t0 < args.timeout:
                continue
            if proc.poll() is None:
                proc.kill()
                print(f"[sweep] TIMEOUT {name}", flush=True)
            else:
                print(f"[sweep] done {name} rc={proc.returncode} {time.time() - t0:.0f}s", flush=True)
            running.remove(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
