# SPDX-License-Identifier: Apache-2.0
"""The bench, and the parts-tray cradle the free receptacle rests in.

Three plain boxes, declared so the document can name them. The table is not
here: it is the *cell's*, and ``bench_075`` declares it. The bench the plug
rests on carries the reference sim's own world numbers, put through
:func:`~robosimstudio.scene.frames.point_from_surface` so the quoted figures
stay legible beside the offsets they become. Both variants stand on it.

The two cradle steps are the bulkhead's alone -- computed by
:func:`~tasks/usbc_insert/assets/stage.py::bulkhead_fixture_boxes` off the
asserted barrel and flange radii, so the cradle cannot drift from the mesh the
loader verifies. ``port`` has a bolted receptacle standing where they do
instead, and that receptacle is
:data:`~tasks/usbc_insert/assets/parts.py.USB_C_PORT`'s, not this module's: a
part with a declared bore is not scenery.

Nothing scores against any of them. They are records anyway, because a fixture
a document names has to be a thing the catalog knows: a type nothing declares
is a permanent ``undeclared-type`` note in the lint, and a box the loader used
to conjure is a box no document could place on its own.
"""

from __future__ import annotations

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.core.render.palette import FIXTURE
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import SimRequirements, StaticSpec, contacts_with

from .stage import CELL, bulkhead_fixture_boxes

__all__ = [
    "BENCH",
    "BENCH_CENTER",
    "BENCH_HALF",
    "CRADLE_FRONT",
    "CRADLE_REAR",
    "cradle_front",
    "cradle_rear",
    "usbc_bench",
]

BENCH_HALF = (0.06, 0.07115, 0.025)
"""The reference's bench box: from the plug's rear face to 2 mm short of the
mouth, 25 mm thick under a 0.800 pad top."""
BENCH_CENTER = (0.44, -0.06285, 0.775)
"""Its centre, in the reference cell's world coordinates."""

_SCENERY_SIM = SimRequirements(domain="static", contacts=contacts_with("objects", "robot"))
"""Welded boxes the parts rest on and the hands may brush."""


def _box(name: str, center, half) -> StaticSpec:
    """One welded box, its reference-world centre restated in the surface frame."""
    return StaticSpec(
        name=name,
        half_extents=half,
        xform=(*point_from_surface(CELL, center), 0.0, 0.0, 0.0, 1.0),
        color=FIXTURE,
    )


def _cradle_step(name: str) -> StaticSpec:
    """One of the cradle's two steps, by the name ``bulkhead_fixture_boxes`` gives it."""
    for box_name, center, half in bulkhead_fixture_boxes():
        if box_name == name:
            return _box(box_name, center, half)
    raise KeyError(f"bulkhead_fixture_boxes() builds no {name!r}")


def usbc_bench() -> StaticSpec:
    """The bench the plug starts on."""
    return _box("bench", BENCH_CENTER, BENCH_HALF)


def cradle_front() -> StaticSpec:
    """The front step: top at the flange rim's underside, under the nose that carries the CoM."""
    return _cradle_step("cradle_front")


def cradle_rear() -> StaticSpec:
    """The rear step: top at the smooth barrel's underside, holding the axis level with the front."""
    return _cradle_step("cradle_rear")


# The records last, because each names the factory that places it.
BENCH = register_asset(
    AssetRecord(
        name="usbc_bench",
        summary="The 120 x 142 x 50 mm bench block the plug starts on, at bore height.",
        tags=("bench", "block", "fixture"),
        features=(),
        sim=_SCENERY_SIM,
        provenance="The reference MJC sim's bench, from the plug's rear face to 2 mm short of the mouth.",
        build=usbc_bench,
    )
)

CRADLE_FRONT = register_asset(
    AssetRecord(
        name="usbc_cradle_front",
        summary="The cradle's front step: a rest under the bulkhead's flange rim, with no side cheeks.",
        tags=("rest", "cradle", "fixture"),
        features=(),
        sim=_SCENERY_SIM,
        provenance=(
            "Computed off the asserted flange radius by stage.bulkhead_fixture_boxes; deliberately "
            "no cheeks and no axial stop, because the support jaws close where a cheek would stand."
        ),
        build=cradle_front,
    )
)

CRADLE_REAR = register_asset(
    AssetRecord(
        name="usbc_cradle_rear",
        summary="The cradle's rear step: a rest under the bulkhead's smooth barrel band.",
        tags=("rest", "cradle", "fixture"),
        features=(),
        sim=_SCENERY_SIM,
        provenance="Computed off the asserted barrel radius by stage.bulkhead_fixture_boxes, as the front step.",
        build=cradle_rear,
    )
)
