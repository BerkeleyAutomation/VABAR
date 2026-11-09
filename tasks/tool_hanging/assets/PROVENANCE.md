# Provenance — `tasks/tool_hanging/assets/`

Every mesh in this family was supplied for the Tool Hanging suite. **Licence
status: pending confirmation** for the items below — do not redistribute this
directory outside the repository until their sources are confirmed and
recorded here (the same standing note `tasks/astro/assets/libero/PROVENANCE.md`
carries for its NC-licensed items).

| Path | What it is | Source / licence |
| --- | --- | --- |
| `pegboards/Pegboard_240X320.stl` | legacy 320 × 240 mm pegboard panel | IKEA SKÅDIS-derived CAD; **licence pending confirmation** |
| `pegboards/skadis_whole/` | active single-piece OBJ, MTL and diffuse map | supplied SKÅDIS-derived export from the tool-hanging branch; **licence pending confirmation** |
| `wrench_holders/halterung_*.stp.stl` | SKÅDIS holder modules (unused by the shipped variants; kept with the set) | German CAD assemblies; **licence pending confirmation** |
| `hammer_holders/enforce_1250_screws.stl` | open-fronted hammer holder | supplied CAD; **licence pending confirmation** |
| `scissor_holders/hook_2.stl` | 516-triangle extruded J-hook | supplied CAD; **licence pending confirmation** |
| `plier_holders/triangular_support.stl` | triangular support | supplied CAD; **licence pending confirmation** |
| `screwdriver_holders/screwdriver_holder.stl` | 155 mm graduated rack — `screwdriver_rack`, the full-size variant's fixture | supplied CAD; **licence pending confirmation** |
| `screwdriver_holders/screwdriver_box.stl` | 110 mm open pegboard cup with two mounting straps — `screwdriver_box`, the level-1 variant's fixture | authored in-repo by `generate_screwdriver_box.py`; same licence as this repository |
| `wrenches/wrench_{1,2}/` | 150 mm and 200 mm task wrenches with matching CoACD hulls | scaled from supplied CAD; **licence pending confirmation** |
| `hammers/*.stl` | two hand hammers, authored in centimetres | supplied CAD; **licence pending confirmation** |
| `scissors/scissor_{1..4}/*.obj` | four scanned scissor pairs (variants 3–4 unused by the shipped tasks) | supplied scans; **licence pending confirmation** |
| `screwdrivers/screwdriver_{1,2}/*.obj` | two screwdrivers, `screwdriver_2` with an authored diffuse texture | supplied scans; **licence pending confirmation** |
| `pliers/plier_{1,2}/*.obj` | combination pliers and diagonal cutters | supplied scans; **licence pending confirmation** |

## Derived geometry (built in-repo, same licence as its source)

- `*/collision_coacd/hull_*.obj` + `manifest.json` — CoACD approximate convex
  decompositions, built by `tools/build_coacd_collisions.py` with one shared
  parameter set. Each manifest records those parameters and the generated hulls.
- `scissor_holders/hook_2_collision_coacd/`,
  `screwdriver_holders/screwdriver_holder_collision_coacd/`, and
  `plier_holders/triangular_support_collision_coacd/` — validated holder hull
  sets built by `tools/build_coacd_collisions.py`. The shared hook
  has eight pieces; the rack and triangular support each have 31.
- `scissors/scissor_1/collision_compound/` — the hand-authored 30-piece convex
  proxy, kept as the A/B alternative to the CoACD set
  (`tools/build_scissor_1_compound_collision.py`); `assets/records.py`'s
  `SCISSOR_1_COLLISION_METHOD` knob selects between them.
- `scissors/*_collision_decimated.obj`, `scissors/decimation_report/` — the
  earlier decimation route and its before/after report
  (`tools/decimate_scissors.py`), superseded by the hull sets but kept with
  the set they document.

The routines' landmark constants and the records' feature payloads
(`records.py`) are measurements of these meshes; the wrench thickening
(1.35× / 1.50×, recorded in the wrench manifests) is the one geometric
modification, made so the thin stamped rings survive convex decomposition.
