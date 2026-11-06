# SPDX-License-Identifier: Apache-2.0
"""Tabulate fast_tune run directories: one markdown row per run.

    python tools/fast_tune/report.py DIR [DIR ...]

Reads each run's ``run_meta.json``, ``episode_results.jsonl`` and ``fast_tune.json``
and prints candidate, probe, outcome (successes/episodes, mean reward, steps per
episode), rate (steps/s end to end and while stepping), build_s, substeps, the
solver kwargs actually constructed, and the run's wall time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(run: Path) -> dict | None:
    meta_p, res_p = run / "run_meta.json", run / "episode_results.jsonl"
    if not meta_p.exists() or not res_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    eps = [json.loads(ln) for ln in res_p.read_text().splitlines() if ln.strip()]
    ft = json.loads((run / "fast_tune.json").read_text()) if (run / "fast_tune.json").exists() else {}
    prov = meta.get("provenance", {})
    builds = prov.get("builds", [{}])
    kw = (ft.get("solver_kwargs") or [{}])[0]
    cams = (ft.get("effective") or {}).get("cameras") or []
    phys = (ft.get("effective") or {}).get("physics") or {}
    return {
        "run": run.name,
        "candidate": ft.get("candidate", "?"),
        "task": meta.get("task", ""),
        "success": f"{sum(bool(e.get('success')) for e in eps)}/{len(eps)}",
        "reward": round(sum(float(e.get("reward", 0.0)) for e in eps) / max(1, len(eps)), 3),
        "steps": [int(e.get("episode_length", 0)) for e in eps],
        "close": sorted({e.get("close_kind", "") for e in eps}),
        "steps_per_s": prov.get("steps_per_s"),
        "stepping": prov.get("steps_per_s_stepping"),
        "build_s": [b.get("build_s") for b in builds],
        "substeps": builds[0].get("substeps") if builds else None,
        "captured": prov.get("cuda_graph", {}),
        "wall_s": prov.get("wall_s"),
        "run_wall_s": prov.get("run_wall_s"),
        "renderer": meta.get("renderer", ""),
        "solver": {
            k: kw.get(k)
            for k in ("iterations", "ls_iterations", "cone", "enable_multiccd", "nconmax", "njmax")
            if k in kw
        },
        "collide_every": (phys.get("collision") or {}).get("collide_every"),
        "cams": "x".join(f"{c['width']}x{c['height']}" for c in cams[:1]) + (f" (x{len(cams)})" if cams else ""),
        "peaks": ft.get("peaks"),
        "drift": [(d["worst_mm"], d["worst_obj"], d["worst_deg"]) for d in (ft.get("drift") or [])],
        "final": [
            {
                k: round(v, 4)
                for k, v in (e.get("metrics") or {}).items()
                if isinstance(v, float) and not k.startswith(("goal.", "observe")) and "/" not in k
            }
            for e in eps
        ],
    }


def main(argv: list[str]) -> int:
    rows = []
    for d in argv:
        for run in sorted(Path(d).iterdir()):
            if run.is_dir():
                r = row(run)
                if r:
                    rows.append(r)
    cols = [
        "run",
        "task",
        "success",
        "reward",
        "steps",
        "steps_per_s",
        "stepping",
        "build_s",
        "substeps",
        "wall_s",
        "run_wall_s",
        "solver",
        "collide_every",
        "cams",
        "close",
        "peaks",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
