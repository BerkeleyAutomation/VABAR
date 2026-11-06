# SPDX-License-Identifier: Apache-2.0
"""Write the ``syringe_packing`` layout sweeps as documents of the suite.

The task language samples nothing per episode -- its only sampler is ``fixed``
-- so a sweep over where things start is a set of variants, each a document of
its own with its own ``poses/`` layout, the way every other variant in the
corpus is stated. They are the suite's variants, beside the two documents they
are drawn from::

    uv run python tools/bench_run.py --list | grep syringe_packing/

Three families, each written from one of the suite's documents:

* ``tray-pair`` and ``tray-pair-both``: ``tray_pair.task`` / ``tray_pair_both.task``
  with the whole tray arrangement (goal syringe, spare, tray) shifted together
  in the bench plane (``--range``) so the goal stays in its tray and the spare
  keeps its spacing, yawed about the goal syringe (``--yaw-range``) so the
  barrel's heading -- and with it the grasp, the wrist turn and the carry --
  differs from trial to trial, and the sharps box shifted on the bench
  (``--box-range``). The two families share their layouts: run both with the
  same seed. ``tray-pair`` alone is then *widened*: a second seeded draw
  (``--widen-seed``) gives each trial 2, 3 or 4 syringes at 50/30/20 %, and a
  widened trial's extra spares extend the goal/spare row at its own 90 mm
  pitch -- ``syringe_pack_1`` one pitch beyond the spare, ``syringe_pack_2``
  one pitch behind the goal -- parallel and separated, the goal and the first
  spare untouched. Fourteen of the thirty are widened at the recorded seed.
* ``tray-clutter``: ``tray_clutter.template.task`` with 5-8 syringes packed
  into the tray at ``tray_pair``'s own tray pose -- the *population* of the
  tray rather than its placement, so each trial is a fresh packing rather
  than a transform of a shared template (the placement rule is below) -- and
  the same small box shift.

A box shift is three things the document states and the suite's loader used
to do at build: the ``syringe_box_region`` ranges carry the move (the
aperture's frame is the export's, so a moved box is a moved region), each
foreign-robot host stands in a copy of the robot-stripped workcell whose
``syringe_box_*`` geoms moved with it
(``vabar/assets/vab_syringe_packing/tray_pair/scene_norobot_box<dx><dy>.xml``, named
by the shift to the millimetre and shared by trials whose shifts agree that
closely -- the first writer's geometry, the loader's own rule), and the trial's
header says what was drawn.

**Clutter placement rule.** The tray's cavity is 200x400 mm; a syringe is a
150x23.6 mm capsule that only fits the 200 mm axis lying dead straight across
it, so every syringe's long axis runs along that short axis and the tray's
400 mm axis is the one row has to fit them in. ``tray_pair`` used a fixed
90 mm pitch there (barrel width 23.6 mm, so mostly clear air, by design --
see its own header). This sweep instead:

1. spaces N syringes on an even grid across the row (pitch shrinks as N
   grows, 5 -> ~94 mm, 8 -> ~53 mm -- more syringes reads as more cluttered
   on its own before any jitter is added);
2. jitters each one's row position, cross-axis position and heading
   independently, generously enough (``Y_JITTER_FRAC`` of its own pitch,
   ``X_JITTER_M``, ``YAW_JITTER_DEG``) that neighbours' jitter ranges
   overlap -- barrels actually crossing is the cluttered look asked for,
   not a printed grid nudged off-true;
3. checks every pair with a 2-D oriented-box separating-axis test and, only
   for a pair overlapping by more than ``ALLOWED_PENETRATION_M`` (deep
   enough that Newton's contact solver would eject one of them on the first
   settle step rather than just letting them rest across each other),
   halves both syringes' jitter and rechecks -- a bounded local repair
   rather than resampling the whole trial.

Usage::

    uv run python tasks/syringe_packing/tools/make_sweeps.py tray-pair      --count 30 --seed 0 --range 0.06 --yaw-range 25 --box-range 0.03 --widen-seed 3
    uv run python tasks/syringe_packing/tools/make_sweeps.py tray-pair-both --count 30 --seed 0 --range 0.06 --yaw-range 25 --box-range 0.03
    uv run python tasks/syringe_packing/tools/make_sweeps.py tray-clutter   --count 30 --seed 5
    uv run python tasks/syringe_packing/tools/make_sweeps.py clean

Re-running with the same seed rewrites identical files.

**This tool cannot write anything today, and the documents have moved past it.**
Two things, both worth knowing before trusting the header a trial carries:

* ``write_clutter_trials`` and ``write_pair_trials`` are called by ``main`` and
  defined nowhere. Commit ``8b0cc9ae`` rewrote this module and dropped roughly
  325 lines with them; the last complete copy is
  ``1cc402d5:vabar/tools/make_sharps_sweeps.py``, and restoring it needs
  ``task_module("syringe_packing", "assets.records")`` changed to ``"records"``,
  since ``records.py`` moved out of ``assets/``.
* The ``tray-pair`` family is gone: that single-syringe sweep was dropped on
  2026-09-15 as superseded by the both-posted and clutter ones.
* Twenty ``tray_clutter`` trials were edited after they were drawn, to remove
  syringes whose footprint crossed the tray walls the tray gained a collider for
  on 2026-09-15 (each says so in its own header). A restored generator has to
  filter the packing against the cavity -- ``_pack_row`` already has the OBB and
  the tray constants -- or it will write those syringes straight back in.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from robosimstudio.roots import suite_dir
from robosimstudio_bench.taskbuild.discovery import task_module
from robosimstudio_bench.taskbuild.registry import load_all

SUITE = suite_dir("syringe_packing")
DOCS = SUITE
"""Where the trials go: the suite's own directory, as its variants."""
EXPORT = SUITE.parents[1] / "assets" / "vab_tool_sort" / "syringe"
"""The export whose robot-free workcell each trial's hosts stand in, shifted with the box."""
STRIPPED = "scene_norobot.xml"
"""The workcell with the Panda cut out, which every host row of the base documents names."""

FAMILIES = ("tray_pair", "tray_pair_both", "tray_clutter")
"""Variant prefixes."""
CLUTTER_TEMPLATE = SUITE / "assets" / "tray_clutter.template.task"
"""The clutter family's base: a document of the suite in every respect except
that it is never registered, since a trial's population is the trial's own."""
REFERENCE_LAYOUT = "tray_pair"
"""Whose tray pose and "parallel" heading the clutter packing starts from."""
STEPS_PER_GOAL = 11400
"""``tray_pair``'s own budget: 2400 for one goal plus a flat 9000 container retry."""

MIN_SPACING_M = 0.010
"""Least distance between any two pair trials' offsets [m], so the sweep is
thirty different places and not thirty draws that happen to coincide."""
WIDEN_COUNTS = (2, 3, 4)
WIDEN_WEIGHTS = (0.5, 0.3, 0.2)
"""How many syringes a widened ``tray_pair`` trial holds, and how often: one
``choice`` per trial from the widening seed's own generator."""
WIDEN_ROW_PITCHES = {"syringe_pack_1": 2.0, "syringe_pack_2": -1.0}
"""Where a widened trial's extra spares stand, in goal-to-spare pitches along the row."""

MIN_SYRINGES = 5
MAX_SYRINGES = 8
ROW_MARGIN_M = 0.010
"""Inset from the cavity's long-axis wall the row itself stays inside of."""
CROSS_MARGIN_M = 0.008
"""Inset from the cavity's short-axis wall a barrel's tip must clear."""
Y_JITTER_FRAC = 0.55
"""Row-position jitter, as a fraction of that trial's own pitch -- large
enough that neighbours' jitter ranges overlap, which is the point: a pile
reads as cluttered when barrels actually cross, not just crowd."""
X_JITTER_M = 0.018
"""Cross-axis jitter. 18 mm plus a 35 deg-yawed barrel's own footprint
(~68 mm half-length along this axis) still clears the cavity's ~89 mm."""
YAW_JITTER_DEG = 35.0
ALLOWED_PENETRATION_M = 0.016
"""How deep two syringes may actually overlap before the repair loop eases
them apart -- most of a 23.6 mm body's width, so barrels crossing is the
normal look and only a near-coincident pair (a placement physics would
throw across the tray on the first settle step) gets pulled back. Passed
to :func:`_obb_overlap` as a *negative* clearance, which shrinks the test
rectangles rather than growing them."""
MAX_REPAIR_ROUNDS = 6
CLUTTER_BOX_RANGE_M = 0.03
"""The clutter family's box shift half-width [m], ``tray_pair``'s own."""


def _scenery():
    """The suite's own numbers and its recorded-entry writer, by suite name."""
    load_all()
    return task_module("syringe_packing", "scenery")


def _rounded(value: Any, places: int = 6) -> Any:
    """Every float in a record to a micrometre: what a recorded layout states."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [_rounded(v, places) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v, places) for k, v in value.items()}
    return value


def _quat_yaw(q_xyzw, yaw_rad: float) -> list[float]:
    """``Rz(yaw) * q`` for an xyzw quaternion."""
    x, y, z, w = q_xyzw
    c, s = np.cos(yaw_rad / 2.0), np.sin(yaw_rad / 2.0)
    return [
        round(float(c * x - s * y), 6),
        round(float(c * y + s * x), 6),
        round(float(c * z + s * w), 6),
        round(float(c * w - s * z), 6),
    ]


# ----------------------------------------------------------------------------- the box shift


def shifted_workcell(shift: tuple[float, float]) -> str:
    """A copy of the robot-stripped workcell with the sharps box moved [m], by file name.

    The box is welded into ``bin2`` as the ``syringe_box_c*`` geom family, so
    moving it is a rewrite of those geoms' ``pos`` -- nothing else in the file
    changes. Written beside the original (same directory, so its ``../assets``
    references still resolve), named by the shift to the millimetre, and
    reused when it already exists.
    """
    dx, dy = shift
    stem = f"scene_norobot_box{dx:+.3f}{dy:+.3f}".replace(".", "p").replace("-", "m").replace("+", "")
    dst = EXPORT / f"{stem}.xml"
    if not dst.exists():
        tree = ET.parse(EXPORT / STRIPPED)
        moved = 0
        for geom in tree.getroot().iter("geom"):
            if (geom.get("name") or "").startswith("syringe_box_"):
                px, py, pz = (float(v) for v in geom.get("pos", "0 0 0").split())
                geom.set("pos", f"{px + dx:.6f} {py + dy:.6f} {pz:.6f}")
                moved += 1
        if not moved:
            raise ValueError(f"{EXPORT / STRIPPED}: no syringe_box_* geoms to shift")
        tree.write(dst, encoding="unicode")
    return dst.name


_BOX_RANGES = re.compile(r"(\(syringe_box_region\n(?:.*\n)*?)(\s*)\(:ranges \(\((\S+) (\S+) (\S+) (\S+)\)\)\)\n")


def with_box_shift(doc: str, shift: tuple[float, float]) -> str:
    """The document with its box region and its hosts' workcell following a box shift."""
    bx, by = shift
    if shift == (0.0, 0.0):
        return doc

    def moved(m: re.Match) -> str:
        x0, y0, x1, y1 = (float(v) for v in m.group(3, 4, 5, 6))
        pad = m.group(2)
        return (
            f"{m.group(1)}{pad}(:ranges (({x0 + bx:.6f} {y0 + by:.6f} {x1 + bx:.6f} {y1 + by:.6f})))\n"
            f"{pad}; The box this trial moved by ({bx:+.6f}, {by:+.6f}) m in bin2; the aperture's\n"
            f"{pad}; frame is the export's, so the ranges carry the move.\n"
        )

    doc, n = _BOX_RANGES.subn(moved, doc, count=1)
    if n != 1:
        raise ValueError("the base document has no syringe_box_region ranges to shift")
    return doc.replace(
        f"(:static-override workcell {STRIPPED})", f"(:static-override workcell {shifted_workcell(shift)})"
    )


# ----------------------------------------------------------------------------- the documents


def _trial_document(base: str, *, index: int, header: str, variant: str) -> str:
    """``base`` as trial ``index``: renamed, headed, and reading its own poses file."""
    m = re.search(r"\(define \(problem (\w+)\)", base)
    if m is None:
        raise ValueError("the base document defines no problem")
    doc = base.replace(m.group(0), f"(define (problem {m.group(1)}_T{index:02d})", 1)
    doc = re.sub(r"^; -{20,}\n", header, doc, count=1, flags=re.M)
    doc, n = re.subn(r"\(:start-poses poses/\S+\.json\)", f"(:start-poses poses/{variant}.json)", doc, count=1)
    if n != 1:
        raise ValueError("the base document names no (:start-poses ...)")
    return doc


def _write(variant: str, doc: str, layout: dict[str, Any]) -> None:
    (DOCS / f"{variant}.task").write_text(doc)
    (DOCS / "poses" / f"{variant}.json").write_text(json.dumps(_rounded(layout), indent=2) + "\n")


# ----------------------------------------------------------------------------- the command


def clean() -> int:
    """Every generated file: the trials, their poses and the shifted workcells."""
    removed = 0
    for path in [
        *DOCS.glob("*.task"),
        *(DOCS / "poses").glob("*.json"),
        *EXPORT.glob("scene_norobot_box*.xml"),
    ]:
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="family", required=True)
    for family in ("tray-pair", "tray-pair-both"):
        p = sub.add_parser(family, help=f"the {family.replace('-', '_')} family")
        p.add_argument("--count", type=int, default=30)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--range", dest="extent", type=float, default=0.05, help="half-width of the uniform xy shift [m]"
        )
        p.add_argument(
            "--yaw-range", type=float, default=0.0, help="half-width of the arrangement yaw about the goal [deg]"
        )
        p.add_argument("--box-range", type=float, default=0.0, help="half-width of the sharps box shift [m]")
        p.add_argument(
            "--widen-seed", type=int, default=None, help="widen trials to 3-4 syringes from this seed's draw"
        )
    p = sub.add_parser("tray-clutter", help="the tray_clutter family")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--seed", type=int, default=5)
    sub.add_parser("clean", help="remove every generated file")
    args = parser.parse_args()
    if args.family == "clean":
        print(f"removed {clean()} files")
        return 0
    (DOCS / "poses").mkdir(parents=True, exist_ok=True)
    if args.family == "tray-clutter":
        names = write_clutter_trials(count=args.count, seed=args.seed)
    else:
        names = write_pair_trials(
            args.family.replace("-", "_"),
            count=args.count,
            seed=args.seed,
            extent=args.extent,
            yaw_extent_deg=args.yaw_range,
            box_extent=args.box_range,
            widen_seed=args.widen_seed,
        )
    listed = sorted(p.stem for family in FAMILIES for p in DOCS.glob(f"{family}_t??.task"))
    print(f"wrote {len(names)} trial documents: {names[0]} .. {names[-1]}; the suite holds {len(listed)} trials")
    for name in names:
        trial = json.loads((DOCS / "poses" / f"{name}.json").read_text())["trial"]
        box = trial["box_shift_m"]
        what = (
            f"{trial['n_syringes']} syringes"
            if "n_syringes" in trial
            else f"shift ({trial['offset_m'][0]:+.4f}, {trial['offset_m'][1]:+.4f}) m, yaw {trial['yaw_deg']:+6.1f} deg"
        )
        print(f"  {name}: {what}, box ({box[0]:+.4f}, {box[1]:+.4f}) m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
