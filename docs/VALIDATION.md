# Validation of this release

This records what was checked on the release stack, measured 2026-10-28 on RTX PRO 6000 GPUs:

| Component | Revision |
| --- | --- |
| RoboSimStudio | `d9b2e44`, installed unmodified |
| graph-as-policy | `09e45f8` |
| open-robot-skills | `752dfbd` |
| cuRobo | `v0.8.0` |

Each suite has a detailed record under [`validation/`](validation/), with the
file-by-file port and every run:
- [tool_sort](validation/tool_sort.md)
- [tool_hanging](validation/tool_hanging.md)
- [gear_mesh](validation/gear_mesh.md)
- [syringe_packing](validation/syringe_packing.md)
- [usbc_insert and cable_route](validation/cable.md)

Single episodes are reported as single episodes. They show that a policy runs
end to end on this stack. The paper's per-cell rates come from 30 instances.

## What porting onto upstream took

VABAR used to run on its own reimplementation of RoboSimStudio (tool, gear and
syringe suites) and on a vendored fork of it (the cable suites). Both are gone.
The suites now load as an external task root on the pinned upstream release.
Everything upstream lacks is supplied on the VABAR side.

| Gap | Where it is bridged |
| --- | --- |
| Scripted routines are not in the public RoboSimStudio release, so documents that declare `(:routine)` are refused | `vabar/routines.py`, installed where upstream looks for the module by `import vabar` in each suite's `staging.py` |
| Renamed modules (`robosimstudio_gap` → `robosimstudio_policy_agent`, `robosimstudio.domains.rod` → `robosimstudio_materials.rod`, `robosimstudio.taskbuild` → `robosimstudio_bench.taskbuild`, …) | import lines in the suites |
| Upstream's loader runs `records.py`, not the `assets/` package | `tasks/{tool_hanging,gear_mesh,usbc_insert,cable_route}/records.py` |
| Robots upstream does not ship: `yam_tool_hanging`, `ur5`, `linear_4310_attached` | `vabar/ext/` |
| `D405_WRIST_XFORM`, which upstream's YAM hand no longer carries | stated in `tasks/usbc_insert/staging.py`, value unchanged |
| Newton 1.7 rejects a cable chain that crosses articulations | `tasks/usbc_insert/assets/entity.py`: each chain joins the articulation of the body it hangs from |
| The cable graphs' `curve.*` tools (TrackDLO skeleton, curve fitting) | vendored with their MIT licence in `vabar/cable/`, registered by `tasks/cable_route/tools.py` |
| Six shared graph node scripts | `tasks/_shared/scripts/` (upstream's own lookup path) |
| Upstream stamps a body's friction onto all its shapes, overwriting the gear hub's | `tasks/gear_mesh/assets/gear.py` restores per-part friction after the build |
| graph-as-policy's skill contract (SK1/SK2) rejects subgraphs that run graph-local scripts under a bundle's name | relabelled `generic` in five workflows, with the original names in `meta.composes` |
| `scripts/bench_run.py` is not in RoboSimStudio's wheel | copied unmodified to `vabar/_upstream/` |

Every one of the 15 shipped graphs passes `gap run --validate-only` against the
pinned runtime and the skill library.

## Loading

`python -m vabar probe` loads every suite on the pinned RoboSimStudio, and
`python -m vabar list` registers 371 variants:

| Suite | Variants |
| --- | --- |
| `cable_route` | 16 |
| `gear_mesh` | 204 |
| `syringe_packing` | 93 |
| `tool_hanging` | 13 |
| `tool_sort` | 41 |
| `usbc_insert` | 4 |

Compiling and resetting an environment passes on every variant that was probed;
the per-suite records list them. The two exceptions are `cable_route/panel_l2`
and `port3`, which raise the source's own `KeyError` on the original runtime as
well (see [cable](validation/cable.md)).

## The frozen instances

`python -m vabar bench --cell <cell> --verify` draws each instance's episode and
compares it with the transforms its frozen document recorded. **All 180 benchmark
instances that carry recorded transforms reproduce exactly** (|d| ≤ 1e-6; for the cable
variants, all 400 shipped files do):
- tool_hanging L1 and L2;
- cable insertion L1 and L2;
- cable routing L1 and L2.

The other cells are registered variants, one document per instance:
- tool sorting L1;
- gear meshing L1 and L2;
- syringe packing L1 and L2.

## Scripted groundtruth against the recorded parity table

| Task | Recorded (original runtime) | This release |
| --- | --- | --- |
| `tool_sort/hammer` | success, 786 steps | success, 786 |
| `tool_sort/syringe` | success, 1,958 | success, 1,958 |
| `tool_sort/sort_all`, `sort_all_neat` | fail 0.25, 4,801 | fail 0.25, 4,801 |
| `tool_hanging/{pliers,scissors,wrenches}_level1` | success | success, same step counts |
| `tool_hanging/screwdrivers_level1` | fail 0.667 | fail 0.667 |
| `tool_hanging/pliers` | fail 0.000 | fail 0.000 |
| `tool_hanging/wrenches` | fail, reward 1.000, 2,330 | success, reward 1.000, 2,330 (upstream's evaluator scores the same end state as success) |
| `gear_mesh` | routine raises (no finger bodies) | raises the same way; the graph is the suite's policy |
| `syringe_packing/*` | routine fails or raises | fails or raises identically on the original runtime |

## Reference graphs

Rows naming a cell were run with `python -m vabar bench --cell <cell> --limit 1 --solve graph`, i.e. the cell's first frozen instance under its recorded graph and flags.

| Cell / task | Graph, flags | Result |
| --- | --- | --- |
| tool hanging L1, `toolhanging-easy-hangwrench-001` (`wrenches_level1`, ep. 1) | `gap_perception`, cell flags (unprivileged, vision, cuRobo) | **success**, 3,498 steps |
| `tool_hanging/{scissors,pliers}_level1`, `wrenches` | `gt` (privileged) | success |
| tool hanging L2, `toolhanging-medium-hangwrench-001` (`wrenches_level2`, ep. 1) | `gap_perception`, cell flags | 1 of 2 tools hung (reward 0.50, 6,935 steps); counted per object as in the paper |
| tool sorting L1, `level1_1` | `gap_perception_loop`, cell flags (20,000-step budget) | 2 of 4 tools sorted (reward 0.50), truncated; counted per object as in the paper |
| gear meshing L1, `gearmesh-easy-pair-01` | `gap_perception`, cell flags | **success**, 6,903 steps |
| gear meshing L2, `gearmesh-medium-train-001` | `gap_perception`, cell flags | **success**, 8,816 steps |
| syringe packing L1, `tray_pair_both_t00` | `gap_perception`, cell flags | 1 of 2 syringes posted (reward 0.50) |
| `gear_mesh/gearmesh-easy-single-01` | `gap_perception`, unprivileged, vision | success, 2,511 steps |
| `syringe_packing/tray_pair_t00` | `gap_perception`, unprivileged, vision | success, 2,831 steps |
| syringe packing L2, `tray_clutter_t00` | `gap_perception` | fail 0.00: the graph reported done with nothing contained (single episode; the paper reports 63% over 30) |
| `tool_sort/hammer` | `gt` (privileged) | success, 604 steps |
| `gear_mesh` gate (`rigs/gear_bench.py spin`) | n/a | the train drives; ratio 0.967 (0.968 on the original runtime) |
| `usbc_insert/insert_easy` | `gt` (privileged) | **success**, 3,711 steps (depth 12.5 mm) |
| cable insertion L2, instance -86 (seed 10085) | `mate`, cell protocol | **4 of 6 runs succeed** on identical code and packages (4,393–4,447 steps). One failure inserted the plug fully, then aborted at the `stroke_sg.again_vision` re-check ("grasp leaves no socket clearance") without releasing. The other stopped at the mouth (1 mm deep, 23° tilt). The cable perception is not repeatable run to run. |
| cable insertion L1, seven selected instances | `mate`, cell protocol | 0/7: the plug stops 14–17 mm off the bore. The one scene also run on the original cable runtime fails the same way there. |
| cable routing L1, five selected instances | `gt` (`graphs/weave`), cell protocol | 0/5 (1/5 on the original runtime): the seats are woven but the terminal block is never detected (palette masks 2.1–2.6k px against the graph's 3,400 px minimum, on both stacks) |
| `cable_route/weave3` | `routine` | **success**, 2,732 steps (the source README recorded 0.67 on the original runtime) |

**Cable.** Under the delivered protocol (palette segmentation), the reference graphs do
not reach the paper's USB-C Easy and Routing Easy rates on this release, nor on the cable
suites' original runtime rebuilt from its lock. The paper's 86.7 % / 46.7 % were measured
under a configuration that is still being confirmed with the suite owners (SAM 3
segmentation is the leading candidate). See [cable](validation/cable.md).

**Since resolved.** It was not the segmentation. Both cells' graphs had been tuned on a
different camera rig from the one the benchmark stages; with that corrected, cable insertion
L1 scores 30/30 and cable routing L1 15-23/30 over full cells, under the shipped protocol.
The README's cell tables carry the current figures, and [cable](validation/cable.md) the fixes.

## A fresh install

The pushed commit was cloned from GitHub into an empty workspace beside the two sibling
repositories, and installed with a plain `uv sync`. From that clone:
- the README's quick-start commands and the tests pass;
- `vabar list` registers 371 variants;
- the Studio serves its page;
- the reference graphs reproduce the outcomes above, one instance per cell.

## The Studio

`python -m vabar studio` serves upstream's Studio page over the VABAR suites:
- **Menus:** the Task menu lists all six suites and their 371 variants.
- **Switching:** a session opened on `tool_hanging/wrenches_level1` or `tool_sort/hammer`
  switched from the menu to `gear_mesh`, `usbc_insert`, `cable_route`,
  `syringe_packing` and `tool_sort` tasks.
- **Policies:** the Policy panel offers `routine` plus each task's declared graphs.

## Not validated here

- Full 30-instance runs of each cell (12 × 30 graph episodes). Since run: the README's
  cell tables, measured 2026-11-05.
- Cable routing Level 2, the `weave_vision` graph, SAM 3 segmentation for the cable
  graphs, and multi-world Studio fleets.
- The Isaac Lab Arena implementation, which is not part of this repository.
