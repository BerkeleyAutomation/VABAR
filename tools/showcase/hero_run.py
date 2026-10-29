# SPDX-License-Identifier: Apache-2.0
"""``python -m vabar run`` with presentation cameras added to the task's own rig.

    HERO_CAMS='hero:1.2,-0.6,1.3:0.5,0,0.8:40;close:...' \
        uv run python tools/showcase/hero_run.py --task gear_mesh/pair --solve routine \
        --record --record-cameras hero --renderer rtx

Each camera is ``name:eye:look_at[:fov_y[:WxH]]`` (world metres). The other
switches, all presentation-only and all off by default:

``HERO_ONLY=1``
    Drop the task's own cameras so the path tracer only pays for the new ones.
    Leave it off for a perception graph, which observes through those cameras.
``HERO_OVERLAY=1``
    Keep the graph-stage caption a graph run burns into every frame.

Nothing in the task or in RoboSimStudio is edited: each is a patch applied where
the environment resolves it. ``render_offline.py`` imports this module, so a
re-render gets the same cameras. The backdrops and the tool-sort materials
are part of the tasks' own looks (``vabar/backdrop.py``), so every run draws them.
"""

from __future__ import annotations

import os
import sys

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.sim import sim as sim_module


def parse(text: str) -> tuple[CameraSpec, ...]:
    cams = []
    for item in filter(None, (s.strip() for s in text.split(";"))):
        parts = item.split(":")
        name, eye, look = parts[0], parts[1], parts[2]
        fov = float(parts[3]) if len(parts) > 3 and parts[3] else 40.0
        w, h = (int(v) for v in parts[4].split("x")) if len(parts) > 4 else (1280, 720)
        cams.append(
            CameraSpec(
                name=name,
                width=w,
                height=h,
                fov_y=fov,
                eye=tuple(float(v) for v in eye.split(",")),
                look_at=tuple(float(v) for v in look.split(",")),
                near=0.01,
                far=8.0,
                channels=("rgb",),
            )
        )
    return tuple(cams)


HERO = parse(os.environ.get("HERO_CAMS", ""))
_resolve = sim_module.Sim._resolve_cameras


def _with_hero(self):
    own = () if os.environ.get("HERO_ONLY") == "1" else _resolve(self)
    return (*own, *HERO)


sim_module.Sim._resolve_cameras = _with_hero

if os.environ.get("HERO_OVERLAY") != "1":
    # A graph run burns its active node into every frame; a presentation clip does not want it.
    from robosimstudio.record import recorder

    recorder._overlay_stage = lambda frames, stage: frames

if __name__ == "__main__":
    from vabar.run import main

    sys.exit(main(sys.argv[1:]))
