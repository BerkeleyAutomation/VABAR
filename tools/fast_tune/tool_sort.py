# SPDX-License-Identifier: Apache-2.0
"""``fast`` config candidates for tool_sort, measured against ``vab_500hz@v1`` / ``tool_sort_agentview@v3``.

tool_sort stages the same ``vab_500hz@v1`` as syringe_packing (the Level-1 documents
through ``vab_industrial_robotiq@v1``), so the physics candidates are the syringe
suite's, re-measured here on the sorting scenes: free tools that collide as the
``drawn_hull`` of their (6k-168k face) visual meshes. Newton already caps a convex
hull at ``Mesh.MAX_HULL_VERTICES`` = 64 vertices, which MuJoCo receives as
``maxhullvert``, so the hull-cap candidate goes below that.

    CUDA_VISIBLE_DEVICES=6 python tools/fast_tune/tool_sort.py ts_ss6 -- \\
        --task tool_sort/level1_1 --solve routine --out OUT --name NAME
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# This checkout first: `import vabar` must resolve to its vabar/ and tasks/ (as
# `python -m vabar` run from the checkout does), not to the one the venv installed.
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

from _common import Candidate, main, scaled_rig

SUITE = "tool_sort"


def _base_physics():
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement("vab_500hz@v1")


def _base_rig():
    from robosimstudio_bench.taskbuild.arrangements import get_camera_rig

    return get_camera_rig("tool_sort_agentview@v3")


def _registered_physics(name: str):
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement(name)


def _mj(spec, **params):
    return spec.with_backend("*", spec.backend_for("*").with_(**params))


def _hull_cap(n: int):
    """Cap every convex hull built from here on at ``n`` vertices (process-wide; probe only)."""

    def rewrite(task):
        import newton

        newton.Mesh.MAX_HULL_VERTICES = int(n)
        return task

    return rewrite


C = Candidate
CANDIDATES = {
    c.name: c
    for c in (
        C("ts_ref", "reference: vab_500hz@v1, tool_sort_agentview@v3"),
        C("ts_ss8", "substeps 10 -> 8 (400 Hz)", physics=lambda s: replace(s, substeps=8)),
        C("ts_ss6", "substeps 10 -> 6 (300 Hz)", physics=lambda s: replace(s, substeps=6)),
        C("ts_ss4", "substeps 10 -> 4 (200 Hz)", physics=lambda s: replace(s, substeps=4)),
        C(
            "ts_it10",
            "MuJoCo iterations/ls_iterations 20/20 -> 10/10",
            physics=lambda s: _mj(s, iterations=10, ls_iterations=10),
        ),
        C("ts_pyr", "cone elliptic -> pyramidal", physics=lambda s: _mj(s, cone="pyramidal")),
        C("ts_noccd", "enable_multiccd True -> False", physics=lambda s: _mj(s, enable_multiccd=False)),
        C("ts_buf", "nconmax/njmax 1024/1024 -> 384/512", physics=lambda s: _mj(s, nconmax=384, njmax=512)),
        C("ts_hull32", "convex hulls capped at 32 vertices (Newton default 64)", task=_hull_cap(32)),
        C(
            "ts_cam_half",
            "rig halved (overhead 1280x960 -> 640x480; agentview, wrist halved too)",
            cameras=lambda r: scaled_rig(r, 0.5),
        ),
        C(
            "ts_fast",
            "substeps 10 -> 6 (300 Hz); rig and everything else unchanged (= ts_ss6)",
            physics=lambda s: replace(s, substeps=6),
        ),
        C(
            "ts_winner",
            "the registered winner: vab_tool_sort_fast@v1, tool_sort_agentview@v3 unchanged",
            physics=lambda s: _registered_physics("vab_tool_sort_fast@v1"),
        ),
    )
}


if __name__ == "__main__":
    raise SystemExit(
        main(SUITE, CANDIDATES, base_physics=_base_physics, base_rig=_base_rig, extra_suites=("syringe_packing",))
    )
