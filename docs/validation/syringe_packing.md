# `syringe_packing` on upstream RoboSimStudio

Measured 2026-10-28 on one RTX PRO 6000 (`CUDA_VISIBLE_DEVICES=2`):
- RoboSimStudio pinned at `d9b2e44`, installed unmodified;
- graph-as-policy at `09e45f8`;
- open-robot-skills at `main@752dfbd`.

## What changed, and why

The suite still builds through its own `task.py` loader. Upstream's discovery runs a suite's `task.py` when it has
one. The documents, numbers, assets and routine logic are as delivered.

| File | Change | Why |
| --- | --- | --- |
| `staging.py` | `import vabar` | Installs the scripted-routine module upstream leaves out, so `(:routine ...)` lowers. Every suite has this line. |
| `staging.py` | `robosimstudio.robot.registry` → `robosimstudio.robots.registry`; `robosimstudio.taskbuild[.staging]` → `robosimstudio_bench.taskbuild[.staging]` | Upstream package moves. Same names, same signatures. |
| `scenes.py` | `SceneSpec` / `register_scene` → `CellSpec` / `register_cell` | Upstream renamed them. The fields are identical. |
| `task.py` | `robosimstudio.assetlib.evaluate` → `robosimstudio_bench.goals.evaluate`; `robosimstudio.taskbuild.*` / `robosimstudio.taskdoc.*` → `robosimstudio_bench.*` | Upstream package moves. |
| `task.py` | `get_table(BINDINGS)` → local `_binding_table(BINDINGS)` | Upstream has no binding-table lookup. It exports the one table, `VAB_INDUSTRIAL_V1` (`name == "vab_industrial@v1"`). The helper returns it and still raises on any other name. |
| `task.py` | Four `dataclasses.replace(spec, objects=/statics=...)` → `restage(...)` (from `robosimstudio.specs`) | Upstream `TaskSpec` keeps one `entities` tuple, and `objects`/`statics` are now views over it. `restage` is upstream's documented replacement for exactly this call. |
| `tools.py` | `robosimstudio.policies.graph.sim_tools` → `robosimstudio_policy_agent.sim_tools` | Upstream package rename. |
| `routine.py` | `robosimstudio.assetlib.goal` → `robosimstudio_bench.goals.goal`; `robosimstudio.robot` → `robosimstudio.solvers` | Upstream package moves. Same names. |
| `graphs/gap_perception/workflow.json` | 10 subgraphs: `"skill"` relabelled to `"generic"`. The original labels are recorded in `meta.composes` | See below. |

Comments that still cite old module paths (e.g. `robosimstudio.robot.motion.osc`) were left as written.

No `vabar/ext/syringe_packing.py` was needed.

**The graph relabel is a metadata change; nothing it runs changed.** The pinned graph-as-policy rejects
`gap_perception` before any node runs: 12 errors under its skill contract.
- SK2: a subgraph names a bundle but runs none of that bundle's canonical scripts.
- SK1: two exit values `remaining_sg` returns are outside `perceiving-objects`' vocabulary.

Every one of those subgraphs runs the graph's *own* scripts: all 17 are under `graphs/gap_perception/scripts/`. So the
bundle label claimed a script that never ran. `generic` is the label for that case: it consults no bundle, and scripts
resolve next to the graph first. The `sam3.*` / `geometry.*` / `motion.*` tool bundles still boot, because gap boots
them from the tool names the scripts call. The library skill each subgraph implements, which the paper's skill-reuse
analysis counts, is kept in `meta.composes`.

Two subgraphs pass the contract as delivered and keep their labels: `mate_sg` (`computing-feature-mating-poses`) and
`insert_sg` (`executing-feature-mating`). After the relabel, both `gt` and `gap_perception` validate with 0 errors.

## Probe

`python -m vabar probe --suite syringe_packing --envs 1`: **ok, 93 variants**. `syringe_packing/syringe` compiles and
resets in 4.2 s.

93 is the expected count for 94 `.task` files. `dispose_the_syringe_from_the_cluttered_tray.task` is the template the
`tray_clutter_tNN` trials were generated from, and the loader registers no bare variant for it (see `task.py`'s
`VARIANTS`). That matches the original checkout.

`Env` compile + `reset(0)` + 5 idle steps:

| Variant | Objects | Result |
| --- | --- | --- |
| `tray_pair_both_t00` (L1) | 4 | ok |
| `tray_clutter_t00` (L2) | 9 | ok |
| `tray_pair_t00` | 4 | ok |

The suite's own `tests/test_perception.py` passes (2/2).

## Variant names ↔ paper levels

The documents are named for what they ask. The loader registers them under shorter variant names:

| Document | Variant | Benchmark set |
| --- | --- | --- |
| `put_the_syringe_in_the_sharps_box.task` | `syringe` | — |
| `dispose_the_syringe_from_the_tray[_tNN].task` | `tray_pair[_tNN]` | — |
| `dispose_both_syringes_from_the_tray[_tNN].task` | `tray_pair_both[_tNN]` | **Level 1**: `benchmark/syringe_packing/level1/tasks` (base + t00–t29) |
| `dispose_the_syringe_from_the_cluttered_tray_tNN.task` | `tray_clutter_tNN` | **Level 2**: `benchmark/syringe_packing/level2/tasks` (template + t00–t29) |

The benchmark copies differ from the suite's documents in one line only: `(:profile vab_industrial@v1)` where the suite
says `@v2`. The loader passes `profile="vab_industrial@v2"` explicitly, so both lower identically.

The old run notes (`GRAPH_RUN_COMMANDS.md`) also counted `tray_pair_tNN` as Level 1. The frozen benchmark set does not.

## Scripted groundtruth

Run with `python -m vabar run --task syringe_packing/<v> --solve routine`. The baseline is the original checkout
(`run.py --gt routine`, its own locked environment), on the same GPU:

| Variant | Here (upstream) | Original checkout |
| --- | --- | --- |
| `syringe` | fail, reward 0.000, 1,201 steps (`policy_request`: "routine finished: 1 goal(s) attempted") | **identical**: fail, 0.000, 1,201 |
| `tray_pair_both_t00` | raises `ValueError: object 'syringe_pack_0' has no OBB; its spec sets obb_grasp=False` | **identical** raise |
| `tray_clutter_t00` | same `ValueError` | **identical** raise |

The port is faithful, and the suite ships no working scripted groundtruth.
- `RoutineSharpsDisposal` is the tool-sort routine. Its docstring's "1.0 at step 907" predates the suite split.
  `graphs/gt/README.md` records it at 0/9 since 2026-08-17.
- The tray loader deliberately turns off the OBB grasp box on `syringe_pack_*`, which the routine needs.

## Reference graphs

| Graph | Variant | Flags | Result |
| --- | --- | --- | --- |
| `gap_perception` | `tray_pair_t00` | `--no-privileged --perception vision --motion curobo --obs-renderer rtx` | **success, reward 1.000, 2,831 steps** (the old notes record success in 4,864) |
| `gap_perception` | `tray_pair_both_t00` (L1) | same | fail, reward 0.500, 7,495 steps: one of two posted; `verify_sg`: "syringe remains visible outside or above the sharps-box aperture" |
| `gap_perception` | `tray_clutter_t00` (L2) | same | fail, reward 0.000, 5,455 steps: one pick-and-insert cycle ran, then `count_remaining` saw 0 syringes left while none were contained |
| `gt` | `syringe`, `tray_pair_t00`, `tray_pair_both_t00` | `--motion curobo` (± `--roll held`), and `--roll held` with OSC | fail, reward 0.000 every time |

These are single episodes. The paper's GaP (Human–Newton) numbers are 87.0% (L1) and 63.0% (L2) over N = 30, so one
episode per level says the graph runs end to end, not what its rate is.

Notes on the `gt` failures:
- **`--obs-renderer rtx` is required for any perception graph.** Upstream's runner builds the observation renderer only
  when asked; the empty default is fastest. Without it `obs['cameras']` is empty, and `perceive_target` dies with
  `StopIteration` looking for the `overhead` camera.
- `gt` is `tool_sort`'s single-syringe graph, copied whole (see its README). It reads `sim.get_object_obb`, which the
  tray loader withholds for `syringe_pack_*` ("no grasp box"). So it is not a reference for the tray levels.
- On `syringe`, its `post_upright` needs wrist-roll authority:
  - **default (roll free):** it declines up front, as designed;
  - **`--roll held`:** the stand-up pose is `unreachable` (with cuRobo and with OSC), or the syringe stays flat.

  The old checkout got there with `--ik pyroki`. Upstream has no `--ik` flag: PyRoKi is its default solver, and roll
  authority is `--roll`. Recorded, not fixed: it needs graph or motion work, not a port.

## Not validated

- `gap_perception` over the full N = 30 Level-1 / Level-2 sets: one episode each above.
- `gap_perception` is the only reference graph that runs the tray levels. `gt` was never one for them.
- The scripted routine on the tray families: it cannot run them by construction (as before).

## Requested shared changes

None needed for loading or running. Two items for the README / run docs:
1. Document `--obs-renderer rtx` for every `--perception vision` graph run. It's not syringe-specific: any camera-based
   graph has the same empty-camera failure.
2. The syringe Level-1 set is `tray_pair_both_tNN` and Level 2 is `tray_clutter_tNN` (table above). This is for the
   benchmark manifest.
