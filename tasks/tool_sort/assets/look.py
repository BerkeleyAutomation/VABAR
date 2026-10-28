# SPDX-License-Identifier: Apache-2.0
"""How the VAB workcell looks: the export's own materials, made visible again.

Every surface in this cell is *authored* -- ``assets/vab_tool_sort/scene.xml``
gives its bins a wood material, its bench a steel plate, its hammer a wooden
handle and a scratched-steel head, and each tool a diffuse map beside its mesh.
None of that reached a picture, and the scene rendered as a uniform grey mass.
Two separate losses, both measured on ``tool_sort/sort_all`` before this module
existed:

**One: a textured MJCF material states no colour, so the import states none.**
``import_mjcf`` threads a material's *image* through only on its mesh branch;
on every primitive branch it passes the material's ``rgba`` and drops the image.
``light-wood``, ``dark-wood``, ``floorplane``, ``table_legs_metal``,
``hammer_0_wood_mat``, ``hammer_0_metal_mat`` and ``round_nut_0_smetal`` all
carry a texture and no ``rgba``, so those shapes arrive with neither -- they
fall to ``ModelBuilder``'s build-order palette, and
:meth:`~robosimstudio.scene.entity.Entity.apply_display` then repaints the lot
with the fixture default. 30 of the 93 drawn shapes in ``sort_all``, every one
of them ``(0.40, 0.40, 0.43)``: both bins, all four bench legs, the whole
hammer and all nine faces of the round nut.

**Two: the shapes that *did* keep a texture had it overwritten at render time.**
The path tracer binds one ``UsdPreviewSurface`` per finish to each drawn
instance (:func:`~robosimstudio.solvers.render.rtx.decorate_stage`), and that
instance-level binding wins over the textured material Newton put on the
prototype. A shape carrying an image gets
``finish_for(None, role)`` -- a finish with no texture -- so its diffuse comes
from ``displayColor``, which
:func:`~robosimstudio.scene.build.imports.clear_texture_tints` has by then set
to white. Measured: the drill (a 4096-pixel red-and-black map), the clamp, and
all four printed compartment plates rendered as flat white slabs.

So the fix for both is the same one, and it is a *statement about this cell*
rather than machinery: :class:`EntityToolSortWorkcell` walks the shapes its own
import added and gives each one the colour and the :class:`Finish` its material
already claimed -- the finish naming the export's own PNG, so the tracer draws
the image instead of erasing it. A shape whose texture survived the import is
handed *that* texture; a shape whose texture was dropped is handed the file the
material named. Nothing here invents an appearance: every image path and every
repeat count below is transcribed from ``scene.xml``, which is named beside each.

**Opt-in, per document, through** ``(:export ... (:entity tool_sort_workcell))``
-- the same seam ``astro`` uses over the same export
(``tasks/astro/assets/workcell.py``, which paints the *sorting palette* onto
this cell rather than its own materials, because that suite is about reading a
heap and not about reproducing a benchmark's picture).

**Colours are the textures' own means, in linear RGB.** They matter to every
backend that cannot sample an image -- the tiled tracer a policy is evaluated on
included -- and each is the mean of the PNG the finish beside it names, converted
out of sRGB, so the flat version of a surface is the colour the textured version
averages to. Measured, not chosen::

    from PIL import Image
    import numpy as np

    a = np.asarray(Image.open("assets/vab_tool_sort/assets/light-wood.png").convert("RGB")) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    print(lin.reshape(-1, 3).mean(0))          # -> WOOD_LIGHT

**Repeat counts are MuJoCo's own.** A material with ``texuniform="true"`` states
``texrepeat`` in repetitions per metre, which is exactly what
:attr:`~robosimstudio.core.render.finish.Finish.uv_scale` means for a primitive
-- so ``light-wood``'s ``texrepeat="15 15"`` is ``uv_scale=(15, 15)`` with
nothing reinterpreted. The two ``cube`` materials (the hammer's and the nut's
steel, the bench legs') have no per-metre statement to copy; theirs are chosen
to put the scratches at a plausible size on parts that are centimetres across,
and are the only numbers in this module that are not transcribed.

**Only the path tracer sees the finishes and the scenery.** Colours are shared by
every backend; a tiled observation changes only by the colours above, which is
the change from "one grey" to "wood, steel and plaster".

**One recorded baseline moved with this.**
``tests/fixtures/before_documents_only.json`` is each variant's spec as its
deleted ``task.py`` produced it, and it recorded ``tool_sort``'s workcell as a
plain ``static`` -- which it was, and is no longer. The eleven rows now say
``tool_sort_workcell``, the way ``astro``'s three have always said
``vab_workcell``. The gate that fixture serves is "converting a suite to
documents changed nothing"; declaring an entity afterwards is a change the
suite *means*, and the baseline is the only place that claim is written down.
"""

from __future__ import annotations

from dataclasses import replace

import newton

from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.look import STUDIO
from robosimstudio.core.render.room import house_room
from robosimstudio.core.render.scenery import register_scenery
from robosimstudio.core.render.spec import RenderSpec
from robosimstudio.domains.static import EntityStatic
from robosimstudio.paths import resolve_path
from robosimstudio.scene.registry import register_entity

__all__ = [
    "BENCH_GREY",
    "BENCH_STEEL",
    "CONCRETE",
    "DARK_WOOD",
    "HANDLE_WOOD",
    "LEG_STEEL",
    "LIGHT_WOOD",
    "PLASTER",
    "PRINTED",
    "SCENERY_NAME",
    "SHARPS_YELLOW",
    "SHOP_FLOOR",
    "SHOP_WALL",
    "STEEL_BRUSHED",
    "STEEL_PLATE",
    "TOOL_SHELL",
    "TOOL_SORT_LOOK",
    "TOOL_STEEL",
    "WOOD_DARK",
    "WOOD_LIGHT",
    "EntityToolSortWorkcell",
    "surface_for",
    "tool_sort_shop",
]

# ------------------------------------------------------------------ colours
#
# Linear RGB in [0, 1], each the mean of the PNG the finish beside it names.
# None is within the palette test's 2e-3 of a palette entry: these are the
# export's materials, not this corpus's roles.

WOOD_LIGHT = (0.6313, 0.4101, 0.2351)
"""``assets/light-wood.png``: bin1, and the hammer's handle."""

WOOD_DARK = (0.0735, 0.0495, 0.0374)
"""``assets/dark-wood.png``: bin2. Genuinely dark -- a walnut-stained tray, and
the reason its four printed plates are the pale thing in that bin."""

STEEL_PLATE = (0.4004, 0.4460, 0.4365)
"""``assets/steel-scratched.png``: the bench top, the hammer's head and the nut."""

STEEL_BRUSHED = (0.4750, 0.4750, 0.4750)
"""``assets/steel-brushed.png``: the four bench legs."""

CONCRETE = (0.5501, 0.4938, 0.4441)
"""``assets/light-gray-floor-tile.png``: the shop floor the bench stands on.

The export's own ``floorplane`` geom is imported invisible, so this colour and
its texture reach a picture through the scenery below rather than through a
shape."""

PLASTER = (0.7345, 0.7057, 0.6579)
"""``assets/light-gray-plaster.png``: the walls. The export declares the texture
and hangs nothing on it; the scenery below is what puts it on a surface."""

# ------------------------------------------------------------------ finishes


def _map(name: str) -> str:
    """One of the export's images, as an absolute path a finish can name."""
    return str(resolve_path(f"vab_tool_sort/assets/{name}"))


LIGHT_WOOD = Finish(roughness=0.65, texture=_map("light-wood.png"), uv_scale=(15.0, 15.0))
"""bin1's material, at ``texrepeat="15 15"`` with ``texuniform="true"``."""

DARK_WOOD = Finish(roughness=0.6, texture=_map("dark-wood.png"), uv_scale=(5.0, 5.0))
"""bin2's, at ``texrepeat="5 5"``."""

BENCH_STEEL = Finish(roughness=0.4, metallic=0.7, texture=_map("steel-scratched.png"), uv_scale=(2.0, 4.0))
"""The bench top's ``steel_bench_mat``, at ``texrepeat="2 4"``. This one already
carried an ``rgba`` -- it is the one material in the file somebody added a colour
to after noticing the import dropped the image -- and the colour it states,
``0.38 0.41 0.46``, is kept."""

LEG_STEEL = Finish(roughness=0.35, metallic=0.85, texture=_map("steel-brushed.png"), uv_scale=(3.0, 3.0))
"""``table_legs_metal``. A cube map, so there is no per-metre repeat to copy: 3
per metre puts the brush lines a hand's width apart on a 0.75 m leg."""

TOOL_STEEL = Finish(roughness=0.30, metallic=0.9, texture=_map("steel-scratched.png"), uv_scale=(30.0, 30.0))
"""``hammer_0_metal_mat`` and ``round_nut_0_smetal``, both cube maps of the same
image.

Nearly a conductor, and that is what the export says: ``round_nut_0_smetal``
carries ``reflectance="1.0" shininess="1.0" specular="1.0"``, which is a MuJoCo
material asking to be a mirror. Drawn as a dielectric it is not merely duller --
the scratch map becomes an *albedo*, and a hammer head comes out as a mottled
grey stone rather than as forged steel. Measured on the ``overhead`` still at
``metallic=0.6, roughness=0.42``: chalk.

30 repeats per metre, against the 12 that reading was taken at. A 42 mm hammer
face spans a third of this image at 12 and a metre and a quarter of it at 30, and
the difference is one blotch across the whole head against a scratch pattern."""

HANDLE_WOOD = replace(LIGHT_WOOD, uv_scale=(6.0, 6.0))
"""``hammer_0_wood_mat``: the same image as bin1's, on a 200 mm handle. Its own
cube map states ``texrepeat="3 3"`` per face, and 6 per metre is that count over
the handle's length."""

SHOP_FLOOR = Finish(roughness=0.8, texture=_map("light-gray-floor-tile.png"), uv_scale=(4.0, 4.0))
SHOP_WALL = Finish(roughness=0.9, texture=_map("light-gray-plaster.png"), uv_scale=(0.8, 0.8))

PRINTED = Finish(roughness=0.55)
"""What a compartment plate is: matte print on a flat sheet. Its image is the
one the shape already carries, so this states no texture of its own -- see
:meth:`EntityToolSortWorkcell.apply_display`."""

TOOL_SHELL = Finish(roughness=0.45)
"""What a tool mesh is, before its own diffuse map is put on it. Slightly glossier
than the print: these are moulded plastic shells and painted castings."""


# ---------------------------------------------------------------- the cell

BENCH_GREY = (0.38, 0.41, 0.46)
"""The bench top's colour, and the only one here that is *not* measured.

``steel_bench_mat`` is the one textured material in the file that carries an
``rgba`` of its own, which is what let the bench top come through this whole
failure as the only coloured surface in the frame. It stays the export's
number rather than becoming the mean of the plate image beside it."""
SHARPS_YELLOW = (0.95, 0.80, 0.10)
"""The sharps box's colour: the export's own number, kept verbatim like the
bench top's. ``syringe`` and ``sort_power`` weld the box into ``bin2`` as
geometry -- fourteen geoms named ``syringe_box_*``, every one carrying
``rgba="0.950 0.800 0.100 1"`` and no material -- so the body rule below would
otherwise paint it walnut. Measured 2026-09-03 on ``tool_sort/syringe``'s
``agentview``: the box rendered ``(0.07, 0.05, 0.04)`` under this module and
yellow at ``origin/main``, and ``graphs/gap_perception_syringe`` could no longer
find "the yellow sharps disposal box" it is written to look for."""


def surface_for(label: str) -> tuple[tuple[float, float, float], Finish] | None:
    """The colour and finish for one imported shape, or ``None`` for one nobody named.

    Matched on the shape's label, which is where the export's own body and geom
    names survive an import: a shape arrives as
    ``base/worldbody/<body>/<geom>``. Fragments rather than exact names, because
    a bin has five to seven drawn boxes wearing one material and the export's
    generator numbers them -- and because ``sort_power`` stands two of several
    tools in the same cell, so nothing here may depend on an instance index.

    The bins go by body rather than by geom: every drawn box of ``bin1`` is
    ``light-wood`` and every one of ``bin2`` is ``dark-wood``, floors included,
    which is what the export states and what makes the four pale printed plates
    the thing your eye finds in the dark bin. The one thing in ``bin2`` that is
    not the bin is the sharps box the ``syringe`` and ``sort_power`` exports weld
    into it, so that is matched first, by its own geom name -- see
    :data:`SHARPS_YELLOW`.

    The hammer is the only object the export builds out of *primitives* with two
    materials, and it is the reason this is a function and not a mapping: a claw
    hammer is a wooden shaft and four steel castings, in five separate geoms
    that all begin ``hammer_``.
    """
    if "syringe_box" in label:
        return SHARPS_YELLOW, TOOL_SHELL
    if "/bin1/" in label:
        return WOOD_LIGHT, LIGHT_WOOD
    if "/bin2/" in label:
        return WOOD_DARK, DARK_WOOD
    if "workbench_top" in label:
        return BENCH_GREY, BENCH_STEEL
    if "workbench_leg" in label:
        return STEEL_BRUSHED, LEG_STEEL
    if "hammer_" in label:
        return (WOOD_LIGHT, HANDLE_WOOD) if "_handle" in label else (STEEL_PLATE, TOOL_STEEL)
    if "round_nut_" in label:
        return STEEL_PLATE, TOOL_STEEL
    return None


@register_entity("tool_sort_workcell")
class EntityToolSortWorkcell(EntityStatic):
    """The workcell as imported, wearing the materials its MJCF already states."""

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """Colour and finish this import's drawn shapes from the export's materials.

        Runs *after* the base pass, not instead of it: that pass is what replaces
        a build-order colour with a role default and what fills
        :attr:`finishes` for everything this does not name, and both are still
        wanted for the shapes the export leaves plain (the robot mount, the
        pedestal, the collision boxes nothing draws).

        Two cases, and the order matters. A shape that **carries** a texture --
        a tool mesh, a printed plate -- keeps it: its own image is copied onto
        its finish, which is the only way the tracer draws it, and its colour is
        left alone because
        :func:`~robosimstudio.scene.build.imports.clear_texture_tints` will white
        it out and should. A shape whose texture the importer **dropped** gets
        the file its material named, plus the colour that image averages to, so
        that the backends which cannot sample it still show wood as wood.
        """
        super().apply_display(builder)
        drawn = int(newton.ShapeFlags.VISIBLE)
        dressed = 0
        for shape in range(*self.range.shapes):
            if not int(builder.shape_flags[shape]) & drawn:
                continue
            label = str(builder.shape_label[shape])
            carried = getattr(builder.shape_source[shape], "texture", None)
            if carried is not None:
                base = PRINTED if "_label" in label else TOOL_SHELL
                self.finishes[shape] = replace(base, texture=str(carried))
                dressed += 1
                continue
            surface = surface_for(label)
            if surface is None:
                continue
            color, finish = surface
            builder.shape_color[shape] = color
            self.finishes[shape] = finish
            dressed += 1
        if not dressed:
            raise ValueError(
                f"static {self.spec.name!r}: dressed nothing. The export's shape labels are not the "
                "bin1/bin2/workbench ones this expects; see tasks/tool_sort/assets/look.py."
            )
        self.dressed = dressed


# ------------------------------------------------------------------ scenery

SCENERY_NAME = "tool_sort_shop"

WALL_SETBACK = 2.2
"""How far from the cell centre the shop walls stand [m].

Far enough that the arm, its pedestal and the bench are nowhere near one, and
close enough that the ``agentview``'s 45-degree frame is filled by a lit room
rather than by the void the tracer clears to. The camera sits at
``(1.30, -0.80, 1.60)`` looking back across the bench, so what it sees past the
work is the ``-x`` wall and the ``+y`` one; the third is drawn because a light
bouncing off only two of them is a room with a hole in it."""

WALL_HEIGHT = 2.6
"""Ceiling height [m]. Above the ``lab`` panel light at ``surface_z + 1.7``, so
the panel hangs inside the room rather than outside its walls."""


@register_scenery(SCENERY_NAME)
def tool_sort_shop(kit) -> None:
    """A tiled floor and three plaster walls around whatever cell this is.

    Placed against the cell rather than at literals, so the two rigs -- and any
    host pairing that stands another robot here -- get the same room.

    The floor is drawn rather than left to the export's own ``floorplane``: that
    geom is imported invisible, and Newton's substitute ground plane is a flat
    dark grey, which is most of the ``agentview`` frame and all of what "the
    scene is dark" meant.

    The shape is :func:`robosimstudio.core.render.room.house_room` -- this is
    where that shape was first measured, and it is now the library's, checked
    against every camera in the corpus rather than against this suite's two.
    What stays here is the *materials*, which are the export's own tile and
    plaster maps and belong to this cell rather than to rooms in general.
    """
    house_room(
        kit,
        setback=WALL_SETBACK,
        wall_height=WALL_HEIGHT,
        floor_color=CONCRETE,
        wall_color=PLASTER,
        floor_finish=SHOP_FLOOR,
        wall_finish=SHOP_WALL,
        prefix="shop",
    )


# ------------------------------------------------------------------ the look

TOOL_SORT_LOOK: RenderSpec = replace(STUDIO, environment="lab", scenery=SCENERY_NAME, quality="preview")
"""What every ``tool_sort`` variant renders under.

The house look plus two things the path tracer alone honours: the ``lab``
preset -- a soft panel over a neutral dome, which is what a workshop bench is
lit by and several stops brighter than Newton's ``default`` key -- and the shop
above. Stating a :class:`RenderSpec` at all replaces the house look wholesale,
which is why this is ``replace(STUDIO, ...)`` and not a fresh one: an all-``None``
spec would silently drop the tiled tracer's fill light and background.
"""
