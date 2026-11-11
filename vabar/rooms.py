# SPDX-License-Identifier: Apache-2.0
"""Real Isaac Sim rooms around every suite's bench, in every path-traced picture.

Each suite stands in a 3D environment from NVIDIA's public Isaac Sim assets:

* ``tool_sort``, ``tool_hanging``, ``gear_mesh``: the full warehouse, in an aisle;
* ``syringe_packing``: the hospital supply room;
* ``usbc_insert``, ``cable_route``: rows of data-hall server pods inside the
  emptied simple warehouse.

The room is a scenery builder, so it is drawn by the RTX path tracer and by
nothing else: it has no collider, no mass and no effect on the physics. It is on
by default for every renderer this module sees built through
:meth:`robosimstudio.sim.Sim.make_renderer` -- the observation rig, the recorded
video and the Studio alike -- with the task's own lighting brightened to the
level the rooms were tuned at. The tiled renderer draws colliders alone and so
shows no room.

The USD files are not in this repository (about 2 GB, under NVIDIA's Isaac Sim
asset terms). They are read from ``$VABAR_ROOMS_DIR`` (default
``~/.cache/vabar/isaac_rooms``), laid out as ``assets/Environments/...`` and
``dc/Assets/DigitalTwin/...`` exactly as on the Omniverse content bucket.
``VABAR_ROOMS=0`` turns the rooms off; a suite whose room file is missing keeps
its own scenery and says so once.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

__all__ = ["ROOMS", "enabled", "install", "room_for"]

ROOT = Path(os.environ.get("VABAR_ROOMS_DIR", Path.home() / ".cache" / "vabar" / "isaac_rooms"))
_ENV = ROOT / "assets" / "Environments"
FULL_WAREHOUSE = _ENV / "Simple_Warehouse" / "full_warehouse.usd"
WAREHOUSE = _ENV / "Simple_Warehouse" / "warehouse.usd"
HOSPITAL = _ENV / "Hospital" / "hospital.usd"
DATA_HALL = ROOT / "dc/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"

# The emptied warehouse the server pods stand in: its racks, bins, boxes and signs, and its own lights.
_DC_HIDE = (r"^(SM_RackFrame|SM_RackShelf|SM_Rackshield|SM_PaletteA|KLT_Bins|SM_CardBox|SM_SignCVer|S_AisleSign"
            r"|S_Barcode|SM_FloorDecal|SM_Pushcart|DistantLight|RectLight)")


def _pods(x0: float) -> list[tuple[str, float, float, float, float, float]]:
    """Two facing rows of three data-hall pods (centimetre USD, so scale 0.01), the bench in the aisle."""
    out = []
    for y in (-3.34, 1.06, 5.46):
        out.append((str(DATA_HALL), x0, y, 0.0, 0.0, 0.01))
        out.append((str(DATA_HALL), x0 + 3.0, y - 2.3, 0.0, 180.0, 0.01))
    return out


# Lighting the rooms were tuned at (RenderSpec fields). Brighter rooms need a stronger key.
_LIT = {"enable_textures": True, "environment": "lab", "light_direction": (-0.4532, 0.7704, -0.4226), "fill_lights": (((0.9, -0.36, -0.25), 0.35),)}
_LOOK = {
    "warehouse": {**_LIT, "ambient": (1.5, 1.5, 1.5), "key_intensity": 1.0},
    "hospital": {**_LIT, "ambient": (2.2, 2.2, 2.2), "key_intensity": 1.5},
    "datahall": {**_LIT, "ambient": (5.5, 5.5, 5.5), "key_intensity": 3.6},
}

# The cable benches' dark grey table and plate lightened, as in the X video, so the dark data hall stays readable.
_LIGHT_TABLE = [(r"^table$", (0.66, 0.66, 0.67)), (r"^bench$", (0.76, 0.76, 0.78))]

# Per suite: the room, where it stands (x, y, z [m], yaw [deg], turned in 90 deg steps), which of its top-level
# prims to switch off, a world xy box [x0, x1, y0, y1] kept clear of its floor clutter, extra rooms and boxes.
# ``shapes_hide`` and ``recolor`` act on the scene's own shapes, by label: a hidden shape must be display-only
# (no collider), so the physics never notices; a recolour changes the picture alone.
ROOMS: dict[str, dict] = {
    "tool_sort": dict(usd=FULL_WAREHOUSE, xf=(2.85, -16.9, 0.0, 0.0), clear=(-1.3, 1.3, -1.6, 1.6), look="warehouse",
                      # the pedestal a hand-free arm import leaves out
                      boxes=[("pedestal", (-0.5, -0.1, 0.456), (0.10, 0.10, 0.456), (0.72, 0.72, 0.70))]),
    "tool_hanging": dict(usd=FULL_WAREHOUSE, xf=(-17.76, -8.0, 0.0, 270.0), clear=(-0.5, 2.0, -1.5, 1.5), look="warehouse"),
    "gear_mesh": dict(usd=FULL_WAREHOUSE, xf=(-17.99, -8.03, 0.0, 270.0), clear=(-0.7, 1.7, -1.5, 1.5), look="warehouse",
                      # the arm's base stands 0.882 m up at (0, -0.115) with nothing under it
                      boxes=[("pedestal", (0.0, -0.115, 0.441), (0.10, 0.10, 0.441), (0.72, 0.72, 0.70))]),
    "syringe_packing": dict(usd=HOSPITAL, xf=(19.03, 26.26, 0.0, 90.0), look="hospital",
                            hide=r"(Rooff|Roof|Ceiling|TopTrim|Light_test|Light2_low|^DomeLight$)",
                            # the suite's own plain backdrop walls and floor stand between the bench and the room
                            shapes_hide=r"^dressing_(floor|wall_)", recolor=[(r"^dressing_pedestal$", (0.86, 0.86, 0.84))]),
    "usbc_insert": dict(usd=WAREHOUSE, xf=(0.0, 0.0, 0.0, 0.0), hide=_DC_HIDE, extra=_pods(-1.06), look="datahall",
                        recolor=_LIGHT_TABLE),
    "cable_route": dict(usd=WAREHOUSE, xf=(0.0, 0.0, 0.0, 0.0), hide=_DC_HIDE, extra=_pods(-0.97), look="datahall",
                        recolor=_LIGHT_TABLE),
}


def enabled() -> bool:
    return os.environ.get("VABAR_ROOMS", "1").strip().lower() not in ("0", "false", "no", "off")


def _suite_of(sim) -> str | None:
    for obj in (getattr(sim, "task", None), getattr(sim, "request", None)):
        suite = getattr(obj, "suite", None)
        if suite:
            return str(suite).split("/")[0]
        name = getattr(obj, "full_name", None) or getattr(obj, "name", None)
        if name and "/" in str(name):
            return str(name).split("/")[0]
    return None


_missing_said: set[str] = set()


def room_for(suite: str | None) -> dict | None:
    """The room for ``suite``, or ``None`` when it has none or its files are not on this machine."""
    room = ROOMS.get(suite or "")
    if room is None:
        return None
    files = [Path(room["usd"])] + [Path(e[0]) for e in room.get("extra", ())]
    if not all(f.is_file() for f in files):
        if suite not in _missing_said:
            _missing_said.add(suite)
            print(f"[rooms] {suite}: room files not found under {ROOT}; keeping the suite's own scenery")
        return None
    return room


def _builder(suite: str, room: dict):
    def build(kit) -> None:
        from pxr import Gf, Usd, UsdGeom

        stage = kit.stage
        x = UsdGeom.Xform.Define(stage, "/root/scenery/room")
        x.ClearXformOpOrder()
        tx, ty, tz, yaw = room["xf"]
        x.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        x.AddRotateZOp().Set(yaw)
        x.GetPrim().GetReferences().AddReference(str(room["usd"]))
        hidden = 0
        children = list(stage.GetPrimAtPath("/root/scenery/room").GetChildren())
        if room.get("hide"):
            for prim in children:
                if re.search(room["hide"], prim.GetName()):
                    prim.SetActive(False)
                    hidden += 1
        for k, (usd, ex, ey, ez, eyaw, scale) in enumerate(room.get("extra", ())):
            e = UsdGeom.Xform.Define(stage, f"/root/scenery/extra_{k}")
            e.ClearXformOpOrder()
            e.AddTranslateOp().Set(Gf.Vec3d(ex, ey, ez))
            e.AddRotateZOp().Set(eyaw)
            e.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            e.GetPrim().GetReferences().AddReference(usd)
            for c in list(e.GetPrim().GetChildren()):  # the pod files carry their own sky and cameras
                if c.GetName().startswith(("Environment", "Camera")):
                    c.SetActive(False)
        if room.get("clear"):  # furniture standing where the bench is goes
            x0, x1, y0, y1 = room["clear"]
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            for prim in children:
                if not prim.IsActive():
                    continue
                r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if r.IsEmpty():
                    continue
                lo, hi = r.GetMin(), r.GetMax()
                small = (hi[0] - lo[0]) < 8 and (hi[1] - lo[1]) < 8
                if lo[0] < x1 and hi[0] > x0 and lo[1] < y1 and hi[1] > y0 and lo[2] < 2.5 and hi[2] > 0.05 and small:
                    prim.SetActive(False)
                    hidden += 1
        if room.get("boxes"):
            from robosimstudio.core.render.finish import Finish

            for name, c, h, color in room["boxes"]:
                kit.box(name, c, h, color=color, finish=Finish(roughness=0.55))
        print(f"[rooms] {suite}: {Path(room['usd']).name}, {hidden} prims hidden", flush=True)

    return build


_dressed: set[int] = set()


def _dress_model(model, room: dict) -> None:
    """Hide the room's display-only stand-ins and recolour shapes, once per model."""
    if id(model) in _dressed or not (room.get("shapes_hide") or room.get("recolor")):
        return
    _dressed.add(id(model))
    labels = list(model.shape_label)
    if room.get("shapes_hide"):
        xf, flags, n = model.shape_transform.numpy().copy(), model.shape_flags.numpy(), 0
        for i, label in enumerate(labels):
            if re.search(room["shapes_hide"], label) and not (int(flags[i]) & 2):  # display-only shapes alone
                xf[i][2] -= 50.0
                n += 1
        model.shape_transform.assign(xf)
        print(f"[rooms] {n} backdrop shapes hidden", flush=True)
    if room.get("recolor"):
        color = model.shape_color.numpy().copy()
        for i, label in enumerate(labels):
            for rx, rgb in room["recolor"]:
                if re.search(rx, label):
                    color[i] = rgb
        model.shape_color.assign(color)


def _hide_ground(renderer) -> None:
    """The room has its own floor; Newton's ground plane would sit in it."""
    seen: set[int] = set()

    def walk(o, depth=0):
        if id(o) in seen or depth > 4:
            return
        seen.add(id(o))
        if hasattr(o, "show_ground") and hasattr(o, "_last_show_ground"):
            o.show_ground = False
            return
        for v in list(getattr(o, "__dict__", {}).values()):
            if hasattr(v, "__dict__") and not isinstance(v, type):
                walk(v, depth + 1)

    walk(renderer)


_installed = False


def install() -> None:
    """Draw each suite's room in every RTX renderer a scene builds. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    from robosimstudio.core.render.scenery import SCENERY, register_scenery
    from robosimstudio.sim.sim import Sim

    for suite, room in ROOMS.items():
        name = f"vabar_room_{suite}"
        if name not in SCENERY.names():
            register_scenery(name, _builder(suite, room))

    original = Sim.make_renderer

    def make_renderer(self, backend, *, cameras=None, render=None, **kwargs):
        if not enabled():
            return original(self, backend, cameras=cameras, render=render, **kwargs)
        resolved = self.platform.renderer_for(backend)[0]
        suite = _suite_of(self)
        room = room_for(suite) if resolved == "rtx" else None
        if room is None:
            return original(self, backend, cameras=cameras, render=render, **kwargs)
        import robosimstudio.solvers.render.rtx as rtx

        rtx.DOME_TEXTURE_POLE_FIX = (0.0, 0.0, 0.0)  # the house tilt of -90 deg leans the sky in a room
        base = self.look if render is None else render
        render = replace(base, scenery=f"vabar_room_{suite}", **_LOOK[room["look"]])
        _dress_model(self.model, room)
        out = original(self, backend, cameras=cameras, render=render, **kwargs)
        _hide_ground(out)
        return out

    Sim.make_renderer = make_renderer
