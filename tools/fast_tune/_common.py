# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for the ``fast`` config tuning: candidates, in-place overrides, provenance.

A candidate is a physics arrangement, a camera rig and/or an entity rewrite applied to
the variants of one suite that stage the suite's reference arrangement. It is applied
*in this process only*, by re-registering the suite with ``SUITES.replace`` and
``dataclasses.replace`` on each lowered :class:`TaskSpec`. Variant names are kept as
they are: ``tool_hanging``'s Level-2 sampler keys its fixtures on ``task.name``, so a
renamed variant would not reset.

Nothing on disk changes, no document is touched, and every run directory gets a
``fast_tune.json`` stating what was applied and what the solver was actually
constructed with (the ``SolverMuJoCo`` kwargs, captured at construction), so a run can
be checked against the candidate it claims to be.

    python tools/fast_tune/tool_hanging.py <candidate> [--peaks] [--drift] -- <python -m vabar run args>
    python tools/fast_tune/sweep.py --file tools/fast_tune/jobs/th_speed.jobs --out DIR   # many, logged
    python tools/fast_tune/report.py DIR    # one table row per run; show.py DIR... for metrics
    python tools/fast_tune/tool_hanging.py --list
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

__all__ = ["Candidate", "apply", "main", "scaled_rig"]


@dataclass(frozen=True)
class Candidate:
    """One knob setting to measure.

    Attributes:
        name: What the run directories and the tables call it.
        note: The knobs, in words.
        physics: Base arrangement -> candidate arrangement, or ``None`` to keep it.
        cameras: Base rig -> candidate rig, or ``None`` to keep it.
        task: Task -> task rewrite for anything else (colliders), or ``None``.
    """

    name: str
    note: str = ""
    physics: Callable[[Any], Any] | None = None
    cameras: Callable[[tuple], tuple] | None = None
    task: Callable[[Any], Any] | None = None


def scaled_rig(rig: Sequence[Any], factor: float = 0.5) -> tuple:
    """Every camera at ``factor`` of its resolution, intrinsics scaled to keep the picture."""
    return tuple(cam.scaled(max(1, round(cam.width * factor)), max(1, round(cam.height * factor))) for cam in rig)


def _describe_physics(spec: Any) -> dict[str, Any]:
    return {
        "control_hz": spec.control_hz,
        "substeps": spec.substeps,
        "backends": {d: {"solver": b.solver, **dict(b.params)} for d, b in spec.backends},
        "collision": dataclasses.asdict(spec.collision),
    }


def _describe_rig(rig: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"name": c.name, "width": c.width, "height": c.height} for c in rig]


def apply(suite_name: str, candidate: Candidate, *, base_physics: Any, base_rig: tuple | None) -> list[str]:
    """Re-register ``suite_name`` with ``candidate`` applied to the variants it governs.

    A variant is governed when it stages ``base_physics`` (compared by value); the rig is
    swapped only where the variant stages ``base_rig``. Returns the variants touched.
    """
    from robosimstudio_bench.taskbuild.registry import SUITES, get_suite

    suite = get_suite(suite_name)
    touched: list[str] = []
    variants = []
    for task in suite.variants:
        if task.physics != base_physics:
            variants.append(task)
            continue
        new = task
        if candidate.physics is not None:
            new = replace(new, physics=candidate.physics(task.physics))
        if candidate.cameras is not None and base_rig is not None and tuple(task.cameras) == tuple(base_rig):
            new = replace(new, cameras=candidate.cameras(tuple(task.cameras)))
        if candidate.task is not None:
            new = candidate.task(new)
        if new is not task:
            touched.append(task.name)
        variants.append(new)
    SUITES.replace(suite_name, replace(suite, variants=tuple(variants)), origin=f"fast_tune:{candidate.name}")
    return touched


class _Probe:
    """Records the MuJoCo solver's constructor kwargs and, optionally, per-step contact peaks
    and object drift (pose change from step ``DRIFT_FROM`` to the episode's last step)."""

    DRIFT_FROM = 60

    def __init__(self, peaks: bool, drift: bool = False) -> None:
        self.solver_kwargs: list[dict[str, Any]] = []
        self.peaks = peaks
        self.drift = drift
        self.peak: dict[str, Any] = {}
        self.samples = 0
        self.count = 0
        self.first: dict[str, Any] = {}
        self.last: dict[str, Any] = {}
        self.drifts: list[dict[str, Any]] = []

    def install(self) -> None:
        import newton.solvers

        cls = newton.solvers.SolverMuJoCo
        original = cls.__init__
        probe = self

        def init(this, model, **kwargs):
            probe.solver_kwargs.append(
                {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in kwargs.items()}
            )
            return original(this, model, **kwargs)

        cls.__init__ = init
        if not (self.peaks or self.drift):
            return
        from robosimstudio_bench.envs.env import Env

        step, reset = Env.step, Env.reset

        def stepped(this, *args, **kwargs):
            out = step(this, *args, **kwargs)
            probe.count += 1
            if probe.peaks:
                probe._sample(this)
            if probe.drift:
                probe._pose(this)
            return out

        def resetting(this, *args, **kwargs):
            probe.close_drift()
            probe.count = 0
            return reset(this, *args, **kwargs)

        # Serial runner only: the fleet steps its worlds through its own stepper.
        Env.step = stepped
        Env.reset = resetting

    def _pose(self, env: Any) -> None:
        try:
            poses = {k: [float(x) for x in v] for k, v in env._observe_objects().items()}
        except Exception as exc:  # noqa: BLE001
            self.peak.setdefault("drift_error", repr(exc))
            return
        if self.count == self.DRIFT_FROM:
            self.first = poses
        if self.count >= self.DRIFT_FROM:
            self.last = poses

    def close_drift(self) -> None:
        import numpy as np

        if not self.first or not self.last:
            return
        per = {}
        for name, a in self.first.items():
            b = self.last.get(name)
            if b is None:
                continue
            dp = float(np.linalg.norm(np.subtract(b[:3], a[:3]))) * 1000.0
            dot = abs(float(np.dot(a[3:7], b[3:7])))
            ang = float(np.degrees(2.0 * np.arccos(min(1.0, dot))))
            per[name] = (round(dp, 3), round(ang, 3))
        worst = max(per.items(), key=lambda kv: kv[1][0]) if per else ("", (0.0, 0.0))
        self.drifts.append(
            {
                "steps": self.count - self.DRIFT_FROM,
                "worst_mm": worst[1][0],
                "worst_obj": worst[0],
                "worst_deg": max((v[1] for v in per.values()), default=0.0),
                "per_object": per,
            }
        )
        self.first, self.last = {}, {}

    def _sample(self, env: Any) -> None:
        try:
            program = env.scene.program
            solver = program.mujoco_solver if program is not None else None
            if solver is None:
                return
            data = solver.mjw_data
            vals = {
                "nacon_total": int(data.nacon.numpy()[0]),
                "nefc_max_world": int(data.nefc.numpy().max()),
            }
            contacts = getattr(env.scene.pipeline, "contacts", None)
            count = getattr(contacts, "rigid_contact_count", None)
            if count is not None:
                vals["newton_rigid_contacts"] = int(count.numpy()[0])
            for k, v in vals.items():
                self.peak[k] = max(self.peak.get(k, 0), v)
            self.samples += 1
        except Exception as exc:  # noqa: BLE001
            self.peak.setdefault("error", repr(exc))


def _arg(argv: list[str], flag: str, default: str = "") -> str:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def main(
    suite: str,
    candidates: dict[str, Candidate],
    *,
    base_physics: Callable[[], Any],
    base_rig: Callable[[], tuple | None],
    extra_suites: tuple[str, ...] = (),
    argv: list[str] | None = None,
) -> int:
    """``<candidate> [--peaks] [--drift] -- <python -m vabar run args>``.

    ``extra_suites`` are other suites staging the same base arrangement that the
    candidate is applied to as well (the 500 Hz suites share ``vab_500hz@v1``).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print(__doc__.strip())
        print(f"\ncandidates for {suite}:")
        for c in candidates.values():
            print(f"  {c.name:24s} {c.note}")
        return 0
    name, rest = argv[0], argv[1:]
    head = rest[: rest.index("--")] if "--" in rest else rest
    peaks, drift = "--peaks" in head, "--drift" in head
    if "--" in rest:
        rest = rest[rest.index("--") + 1 :]
    if name not in candidates:
        raise SystemExit(f"unknown candidate {name!r}; have {sorted(candidates)}")
    candidate = candidates[name]

    # FAST_TUNE_SYSPATH: extra import directories for THIS interpreter only (sweep.py
    # --pythonpath). Not PYTHONPATH: the graph's tool bundles are subprocesses with venvs of
    # their own, and a torchvision built for this venv's torch, inherited through
    # PYTHONPATH, breaks theirs ("operator torchvision::nms does not exist").
    import os

    for extra in reversed([p for p in os.environ.get("FAST_TUNE_SYSPATH", "").split(os.pathsep) if p]):
        sys.path.insert(0, extra)
    # The fleet's graph workers are threads; import torchvision once, here, before any of
    # them reaches SAM3.
    try:
        import torchvision
        import torchvision.ops  # noqa: F401
    except ImportError:
        pass

    from vabar import install

    install()
    from robosimstudio_bench.taskbuild.registry import get_suite

    get_suite(suite)  # loads the task roots, which register the arrangements and rigs
    probe = _Probe(peaks, drift)
    probe.install()
    base_p, base_r = base_physics(), base_rig()
    touched: list[str] = []
    for target in (suite, *extra_suites):
        touched += [f"{target}/{v}" for v in apply(target, candidate, base_physics=base_p, base_rig=base_r)]

    task_name = _arg(rest, "--task")
    effective: dict[str, Any] = {}
    if "/" in task_name:
        task = get_suite(task_name.split("/", 1)[0]).variant(task_name.split("/", 1)[1])
        effective = {"physics": _describe_physics(task.physics), "cameras": _describe_rig(task.cameras)}
    print(f"[fast_tune] {suite}:{candidate.name} -> {len(touched)} variants; {candidate.note}", flush=True)

    from vabar.run import main as run_main

    t0 = time.time()
    code = 1
    try:
        code = run_main(rest)
    finally:
        probe.close_drift()
        out = Path(_arg(rest, "--out", "outputs")) / _arg(rest, "--name", "")
        record = {
            "suite": suite,
            "candidate": candidate.name,
            "note": candidate.note,
            "variants_touched": touched,
            "effective": effective,
            "solver_kwargs": probe.solver_kwargs,
            "peaks": probe.peak if peaks else None,
            "peak_samples": probe.samples,
            "drift": probe.drifts if drift else None,
            "process_wall_s": round(time.time() - t0, 2),
            "argv": rest,
        }
        if out.is_dir():
            (out / "fast_tune.json").write_text(json.dumps(record, indent=1, default=str))
        print(f"[fast_tune] solver kwargs: {probe.solver_kwargs}", flush=True)
        if peaks:
            print(f"[fast_tune] peaks over {probe.samples} steps: {probe.peak}", flush=True)
        for d in probe.drifts if drift else ():
            print(
                f"[fast_tune] drift over {d['steps']} steps: worst {d['worst_mm']} mm ({d['worst_obj']}), "
                f"{d['worst_deg']} deg",
                flush=True,
            )
    return code
