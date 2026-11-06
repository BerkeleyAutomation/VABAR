# SPDX-License-Identifier: Apache-2.0
"""``fast`` config candidates for tool_hanging, measured against ``tool_hanging_480hz@v1``.

The reference is what the Level-1/Level-2 documents stage: 60 Hz x 8, MuJoCo-Warp
integrating, Newton generating the contacts, ``yam_station_tool_hanging@v1`` (4 x
1280x720 RGB-D). Each candidate changes one knob (then the survivors combine); see
``docs/validation/fast_tool_hanging.md`` for what each measured.

    cd <checkout> && CUDA_VISIBLE_DEVICES=6 python tools/fast_tune/tool_hanging.py th_ss4 -- \\
        --task tool_hanging/wrenches_level1 --solve routine --out OUT --name NAME
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# This checkout first: `import vabar` must resolve to its vabar/ and tasks/ (as
# `python -m vabar` run from the checkout does), not to the one the venv installed.
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

from _common import Candidate, main, scaled_rig

SUITE = "tool_hanging"


def _base_physics():
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement("tool_hanging_480hz@v1")


def _base_rig():
    from robosimstudio_bench.taskbuild.arrangements import get_camera_rig

    return get_camera_rig("yam_station_tool_hanging@v1")


def _registered_physics(name: str):
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    return get_arrangement(name)


def _registered_rig(name: str):
    from robosimstudio_bench.taskbuild.arrangements import get_camera_rig

    return get_camera_rig(name)


def _mj(spec, **params):
    """``spec`` with the MuJoCo backend's parameters changed."""
    return spec.with_backend("*", spec.backend_for("*").with_(**params))


def _coll(spec, **changes):
    return replace(spec, collision=replace(spec.collision, **changes))


def _support_collider(kind: str):
    """Swap the plier support's raw 13.3k-triangle collider for ``kind`` (pliers variants only)."""

    def rewrite(task):
        statics = []
        changed = False
        for spec in task.statics:
            asset = getattr(spec, "asset", None)
            if asset is not None and str(asset.path).endswith("plier_holders/triangular_support.stl"):
                if kind == "coacd":
                    root = Path(asset.path).parent / "triangular_support_collision_coacd"
                    paths = tuple(str(p) for p in sorted(root.glob("hull_*.obj")))
                    asset = replace(asset, collision_paths=paths)
                elif kind == "decimated":
                    asset = replace(asset, collision_path=_decimated(Path(asset.path), 2000))
                elif kind == "hull":
                    asset = replace(asset, approximate="convex_hull", keep_visual=True)
                spec = replace(spec, asset=asset)
                changed = True
            statics.append(spec)
        return task.with_entities(statics=tuple(statics)) if changed else task

    return rewrite


def _decimated(source: Path, faces: int) -> str:
    """A quadric decimation of ``source`` to ``faces`` triangles, cached beside this file's
    scratch (not under tasks/: the candidate lost, so nothing ships it). Measured on the
    support: 13,290 -> 2,000 triangles, watertight, volume within 0.1%, worst surface
    deviation 0.10 mm before the 1.2x mount scale."""
    import trimesh

    out = Path.home() / ".cache" / "vabar_fast_tune" / f"{source.stem}_dec{faces}.stl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        trimesh.load(source).simplify_quadric_decimation(face_count=faces).export(out)
    return str(out)


C = Candidate
CANDIDATES = {
    c.name: c
    for c in (
        C("th_ref", "reference: tool_hanging_480hz@v1, yam_station_tool_hanging@v1"),
        C("th_ss4", "substeps 8 -> 4 (240 Hz)", physics=lambda s: replace(s, substeps=4)),
        C(
            "th_it10",
            "MuJoCo iterations/ls_iterations 20/20 -> 10/10",
            physics=lambda s: _mj(s, iterations=10, ls_iterations=10),
        ),
        C(
            "th_it5",
            "MuJoCo iterations/ls_iterations 20/20 -> 5/5",
            physics=lambda s: _mj(s, iterations=5, ls_iterations=5),
        ),
        C("th_pyr", "cone elliptic -> pyramidal", physics=lambda s: _mj(s, cone="pyramidal")),
        C(
            "th_ce2",
            "collide_every 1 -> 2 (Newton narrow phase every other substep)",
            physics=lambda s: _coll(s, collide_every=2),
        ),
        C(
            "th_sap",
            "Newton broad phase explicit pair list -> sweep-and-prune",
            physics=lambda s: _coll(s, broad_phase="sap"),
        ),
        C("th_nxn", "Newton broad phase explicit pair list -> nxn", physics=lambda s: _coll(s, broad_phase="nxn")),
        C(
            "th_buf2",
            "nconmax 4096 -> 2048, njmax 8192 -> 6656 (the documented L2 nefc peak 4,394 x 1.5)",
            physics=lambda s: _mj(s, nconmax=2048, njmax=6656),
        ),
        C("th_cam_half", "rig 4 x 1280x720 -> 4 x 640x360 (fov kept)", cameras=lambda r: scaled_rig(r, 0.5)),
        C(
            "th_sup_coacd",
            "plier support collides on its 31-hull CoACD set instead of the raw mesh",
            task=_support_collider("coacd"),
        ),
        C(
            "th_sup_dec",
            "plier support collides on a 2k-triangle decimation of the raw mesh",
            task=_support_collider("decimated"),
        ),
        # combinations, and the next steps up from the survivors
        C(
            "th_ss4_ce2",
            "substeps 4 + collide_every 2",
            physics=lambda s: _coll(replace(s, substeps=4), collide_every=2),
        ),
        C(
            "th_ce4",
            "collide_every 1 -> 4 (Newton narrow phase every fourth substep)",
            physics=lambda s: _coll(s, collide_every=4),
        ),
        C(
            "th_ss4_ce2_pyr",
            "substeps 4 + collide_every 2 + pyramidal cone",
            physics=lambda s: _mj(_coll(replace(s, substeps=4), collide_every=2), cone="pyramidal"),
        ),
        C(
            "th_ce2_half",
            "collide_every 2 + rig halved to 4 x 640x360",
            physics=lambda s: _coll(s, collide_every=2),
            cameras=lambda r: scaled_rig(r, 0.5),
        ),
        C(
            "th_fast",
            "the registered winner: tool_hanging_fast@v1 + yam_station_tool_hanging_fast@v1 (= th_ce2_half)",
            physics=lambda s: _registered_physics("tool_hanging_fast@v1"),
            cameras=lambda r: _registered_rig("yam_station_tool_hanging_fast@v1"),
        ),
    )
}


if __name__ == "__main__":
    raise SystemExit(main(SUITE, CANDIDATES, base_physics=_base_physics, base_rig=_base_rig))
