# SPDX-License-Identifier: Apache-2.0
"""``fast`` config candidates for syringe_packing, measured against ``vab_500hz@v1`` / ``agentview@v2``.

The reference is what ``vab_industrial@v2`` stages: 50 Hz x 10, MuJoCo-Warp with its own
contacts and ``enable_multiccd``, nconmax/njmax 1024, contact ke/kd 6e4/500, and three
1280x960 cameras. ``vab_500hz@v1`` is also tool_sort's arrangement, so every physics
candidate is applied to tool_sort's variants too: ``tool_sort/syringe`` (the single
syringe posted through the same sharps box, by the scripted routine) is the cheap
manipulation probe for this suite, since its own ``gt`` graph does not run the tray
levels (docs/validation/syringe_packing.md).

    CUDA_VISIBLE_DEVICES=6 python tools/fast_tune/syringe_packing.py sp_ss6 --drift -- \\
        --task syringe_packing/tray_pair_both_t00 --solve idle --max-steps 1500 --out OUT --name NAME
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# This checkout first: `import vabar` must resolve to its vabar/ and tasks/ (as
# `python -m vabar` run from the checkout does), not to the one the venv installed.
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

from _common import Candidate, main, scaled_rig

SUITE = "syringe_packing"


def _base_physics():
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement("vab_500hz@v1")


def _base_rig():
    from robosimstudio_bench.taskbuild.arrangements import get_camera_rig

    return get_camera_rig("agentview@v2")


def _registered_physics(name: str):
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement(name)


def _mj(spec, **params):
    return spec.with_backend("*", spec.backend_for("*").with_(**params))


C = Candidate
CANDIDATES = {
    c.name: c
    for c in (
        C("sp_ref", "reference: vab_500hz@v1, agentview@v2"),
        C("sp_ss8", "substeps 10 -> 8 (400 Hz)", physics=lambda s: replace(s, substeps=8)),
        C("sp_ss6", "substeps 10 -> 6 (300 Hz)", physics=lambda s: replace(s, substeps=6)),
        C("sp_ss4", "substeps 10 -> 4 (200 Hz)", physics=lambda s: replace(s, substeps=4)),
        C(
            "sp_it10",
            "MuJoCo iterations/ls_iterations 20/20 -> 10/10",
            physics=lambda s: _mj(s, iterations=10, ls_iterations=10),
        ),
        C(
            "sp_it5",
            "MuJoCo iterations/ls_iterations 20/20 -> 5/5",
            physics=lambda s: _mj(s, iterations=5, ls_iterations=5),
        ),
        C("sp_pyr", "cone elliptic -> pyramidal", physics=lambda s: _mj(s, cone="pyramidal")),
        C("sp_noccd", "enable_multiccd True -> False", physics=lambda s: _mj(s, enable_multiccd=False)),
        C(
            "sp_buf",
            "nconmax/njmax 1024/1024 -> 384/512 (still 4-10x the measured peaks: 33 contacts, nefc 95-106)",
            physics=lambda s: _mj(s, nconmax=384, njmax=512),
        ),
        C("sp_cam_half", "rig 3 x 1280x960 -> 3 x 640x480 (fov kept)", cameras=lambda r: scaled_rig(r, 0.5)),
        C(
            "sp_ss8_noccd",
            "substeps 8 + enable_multiccd off",
            physics=lambda s: _mj(replace(s, substeps=8), enable_multiccd=False),
        ),
        C(
            "sp_fast",
            "substeps 8 (400 Hz) + rig halved to 3 x 640x480; multiccd kept (turning it off bought nothing)",
            physics=lambda s: replace(s, substeps=8),
            cameras=lambda r: scaled_rig(r, 0.5),
        ),
        C(
            "sp_winner",
            "the registered winner: vab_syringe_fast@v1, agentview@v2 unchanged (= sp_ss8)",
            physics=lambda s: _registered_physics("vab_syringe_fast@v1"),
        ),
    )
}


if __name__ == "__main__":
    raise SystemExit(
        main(SUITE, CANDIDATES, base_physics=_base_physics, base_rig=_base_rig, extra_suites=("tool_sort",))
    )
