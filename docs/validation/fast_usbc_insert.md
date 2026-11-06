# `usbc_insert` for the `fast` config: what was measured

The two cable-insertion cells -- `cable_insertion/L1` (`usbc_insert/insert_easy`) and
`cable_insertion/L2` (`usbc_insert/insert_medium`) -- were the slowest physics in the
benchmark: the privileged `gt` graph on L1 ep 10009 ran at 14 steps/s (3,203 steps,
3.8 min). This page is the sweep that picked their `fast` setup. It is re-runnable with
`tools/fast_tune/usbc_insert.py` (`list`, `run`, `sweep --probe speed|gt|mate`, `table`).
Everything below ran on GPU 4 of the 8x RTX PRO 6000 box, RoboSimStudio d9b2e44
(Newton 1.7, mujoco-warp), 2026-11-05; scratch runs are under
`agent_scratch/d2/{speed_iso,speed_agg,gt,gt_rep,mate,mate_rep,iso_episode,fleet}` (not
part of the release).

## Result

**Winner: the shipped `insert_easy` / `insert_medium` documents on `usbc_fast@v1`** --
`usbc_bulkhead_960hz@v1` with CUDA graph capture back on and nothing else changed (the
`graph` candidate). The scene is untouched, so every outcome measured matches the
reference: the `gt` graph per instance (2/5 L1, and on L2 1/3 against the reference's
0/3, the reference's L2 10085 loss being run-to-run noise), and the cell protocol on all
six instances (L1 all fail identically, 14-17 mm off the bore; L2 4/4 runs, the reference
3/3).

* Idle stepping, one process alone on GPU 4: **8.9 -> 47.5 steps/s on L1 (5.3x), 11.4 ->
  45.2 on L2 (4.0x)**, on the same loaded host. Against the 14 steps/s the brief's
  reference `gt` episode ran at: 3.4x / 3.2x.
* Whole episodes, alone on GPU 4: the brief's `gt` episode (L1 ep 10009) succeeds in
  3,206 steps at 33.5 steps/s end to end, **1.75 min including the build, against 3.8 min
  (2.2x the wall time, 2.4x the steps/s)** -- the graph's own tool and planning time is now
  a visible share. Under the cell protocol an L1 instance takes 1.9 min and an L2 instance
  2.8 min, build included; four processes on the one GPU give ~2.5x one, so a 30-instance
  cell is ~21 min (L1) / ~30 min (L2) on four workers (estimate, below).

**Faster, not chosen: `insert_easy_nocable` / `insert_medium_nocable`** (registered, staging
`usbc_fast@v1`): the same minus both cables, and on L2 re-posed instead of rebuilt per
episode: 16-22 % more steps/s (55 / 55) and no L2 rebuild, and L1 matches the reference
everywhere it was run. **But the cable-free plug is measurably less steady:** under the
cell protocol L2 scored **6/10 runs without the cable against 7/7 with it** (reference 3/3,
captured-cabled 4/4; one-sided Fisher p ~ 0.09), all four losses in the protocol's known
shallow-cocked and "no socket clearance" modes; and on L1 10084 the privileged graph
missed the grasp in 2/3 cable-free runs against 0/4 cabled. Small numbers, one direction.
It is the right call only if ~20 % matters more than that risk, and then L2 should be
re-measured on all 30.

## Why capture is legal again

`usbc_bulkhead_960hz@v1` (both cells' arrangement) runs every substep eager. Its comment
gives the reason: Newton's hydroelastic narrow phase read its hash-table overflow counter
back to the host on every launch, and a device-to-host copy inside a captured block is
illegal (CUDA error 906, then a poisoned context). That is no longer what the pinned
Newton does, for two independent reasons:

* **Newton 1.7 polls; it does not read every launch.** `HydroelasticSDF.launch` reads
  `ht_insert_failures` once every 120 launches (`_host_warning_poll_interval = 120`,
  `newton/_src/geometry/sdf_hydroelastic.py`). `Schedule.warmup_and_capture` steps two
  control periods eagerly (launches 1-32 at 16 substeps) and captures the third
  (33-48), so the captured block never contains the poll, and a replay runs no Python,
  so the counter never advances again.
* **L1 has no hydroelastic pair at all.** The welded `port` is a plain mesh (`PortUsbc`
  never sets `is_hydroelastic`), and `HydroelasticSDF._from_model` returns `None` unless
  some pair is hydroelastic on both ends. Only the plug declares it, so on L1 the narrow
  phase that motivated eager is never built. (L2's plug-bulkhead pair is a real pair.)

Every captured run in this sweep -- 79 runs, L1 and L2, cabled and not, idle and full
`gt`/`mate` episodes up to 4,720 steps -- reports `steps_eager 0, recapture_failures 0,
capture_error ""` in `run_meta.json`.

## Candidates

Derived in-process from the lowered shipped `TaskSpec` with `dataclasses.replace` and
added to the suite as `usbc_insert/<base>__<candidate>` (nothing shipped is renamed or
changed). The sweep's `fast_tune.json` sidecar reads back what the live env built --
entities, substeps, backend params, capture, `model_key_fields`, peak `nacon`/`nefc`
sampled every 50 steps -- which is how each override was verified.

| Candidate | Knobs vs the shipped document |
| --- | --- |
| `ref` | shipped `insert_easy` / `insert_medium` (`usbc_bulkhead_960hz@v1`: 60 Hz x 16, it 100 / ls 50, nconmax 2048 / njmax 4096, impratio 10, eager) |
| **`graph`** | `use_cuda_graph=True` -- i.e. `usbc_fast@v1` |
| `nocable` | drop the `cables` entity (L1: 16-link plug cable; L2: plus the 8-link bulkhead tail) |
| `nc_graph` | `nocable` + `graph` |
| `nc_graph_con` | + nconmax / njmax 512 / 1024 |
| `nc_graph_it` | + iterations 50, ls_iterations 20 |
| `nc_graph_sub8` | + 8 substeps |
| `nc_graph_vox` | + plug / bulkhead `sdf_voxel` 0.5 mm (from 0.25) |
| `nc_graph_point` | + plug / bulkhead `hydroelastic=False` (point contacts) |
| `doc` | the registered `insert_*_nocable` documents: `nc_graph`, plus L2 re-posed between episodes (`staging.USBC_MEDIUM_REPOSE`). Its protocol runs used nconmax / njmax 1024 / 2048 (the arrangement's first cut); the final `usbc_fast@v1` keeps 2048 / 4096. No run's sampled peak came within 3x of either (nacon 225, nefc 580), so no overflow and the same physics |

The knobs were first swept one at a time from `ref` (eager), but on the eager path a step
is bound by host launches and the solver's per-iteration `capture_while` read-back, and
under the other agents' CPU load those numbers were noise (`agent_scratch/d2/speed_contended`).
So the solver/contact knobs are reported on top of `nc_graph`, where the GPU work is what
is measured.

## Speed (idle 600 steps, no renderer, one process alone on GPU 4)

`stepping` is control steps / s with the per-episode reset taken out; `reset_s` is that
reset for the requested episode (the runner first builds episode 0, then the instance's
layout -- a rebuild, except on the re-posable `doc` L2). Host load average was 85-155 from
other agents throughout, which mostly penalises the eager rows.

| Candidate | L1 stepping | L1 build_s / reset_s | L2 stepping | L2 build_s / reset_s | Captured |
| --- | ---: | --- | ---: | --- | --- |
| `ref` | 8.9 | 11.3 / 8.6 | 11.4 | 10.2 / 9.8 | no |
| `nocable` | 16.4 | 14.4 / 8.7 | 13.4 | 12.1 / 7.3 | no |
| **`graph`** | **47.5** | 10.8 / 7.5 | **45.2** | 14.7 / 7.9 | yes |
| `nc_graph` | 58.0 | 9.3 / 6.7 | 54.5 | 10.7 / 7.5 | yes |
| `nc_graph_con` | 56.1 | 11.0 / 7.1 | 55.3 | 9.1 / 7.2 | yes |
| `nc_graph_it` | 56.4 | 9.0 / 6.9 | 50.8 | 9.8 / 7.7 | yes |
| `nc_graph_sub8` | 70.2 | 13.2 / 8.2 | 46.9 | 19.1 / 10.5 | yes |
| `nc_graph_vox` | 56.0 | 9.8 / 7.9 | 44.0 | 11.6 / 11.6 | yes |
| `nc_graph_point` | 58.1 | 9.4 / 7.8 | 52.7 | 10.5 / 7.6 | yes |
| `doc` | 55.4 | 9.2 / 7.0 | 55.0 | 10.0 / 0.0 | yes |

Reading it: capture is the whole win (4-5x). The cable costs ~20 % once captured. Ceilings,
iterations (the solver already exits on convergence, `graph_conditional`), voxel size and
point contacts buy nothing an idle step can see; only `sub8` is faster, on L1. An idle step
has no plug-port contact, so it cannot see a contact-path cost; the whole-episode runs
below can.

Whole episodes, each alone on GPU 4 (`agent_scratch/d2/iso_episode`):

| Run | Outcome (depth / lat mm) | Steps | Episode wall | Run wall (build incl.) | steps/s end to end |
| --- | --- | ---: | ---: | ---: | ---: |
| reference: `gt`, L1 10009, `insert_easy` (the brief's) | success | 3,203 | -- | 3.8 min | 14 |
| `gt`, L1 10009, **`insert_easy` on `usbc_fast@v1`** | success, 12.62 / 0.18 | 3,206 | 95.6 s | 104.9 s | 33.5 |
| `mate` protocol, L1 10009, **cabled on `usbc_fast@v1`** | failure, -0.25 / 16.58 (as `ref`) | 3,773 | 93.3 s | 116.5 s | 40.4 |
| `mate` protocol, L2 10085, **cabled on `usbc_fast@v1`** | success, 10.84 / 0.67 | 4,602 | 149.9 s | 167.7 s | 30.7 |
| `gt`, L1 10009, `insert_easy_nocable` | success, 12.39 / 0.20 | 3,226 | 86.7 s | 96.6 s | 37.2 |
| `mate` protocol, L1 10009, `insert_easy_nocable` | failure, -0.26 / 16.15 (as `ref`) | 3,822 | 87.2 s | 104.0 s | 43.8 |
| `mate` protocol, L2 10085, `insert_medium_nocable` | success, 13.24 / 0.09 | 4,720 | 122.7 s | 138.5 s | 38.5 |

Several processes on one GPU (`sweep --probe speed --cands doc --repeat 4 -j 4`, 2 L1 +
2 L2 at a time): 33-44 steps/s each, **~135 steps/s aggregate, 2.5x one process**
(independent processes; the `--workers` pool is the same shape and was not run). Processes
are the way to fill the GPU here, not fleets (below). So a 30-instance cell on the winner
is ~53 min (L1) / ~75 min (L2) in one process and, at the measured 2.5x, **~21 / ~30 min
on four processes** (estimate; not run as a pool). The shipped documents need ~3.5 h
serial (estimate: 3,500-4,400 steps at 8.9-11.4 steps/s plus a 10-40 s rebuild per
instance).

## Cheap fidelity: privileged `gt` graph, no renderer

First five L1 instances and first three L2 instances (benchmark/vabar_v1.json order).
`depth / lat` are the goal's insertion depth and lateral offset at the end [mm]. These
ran up to ~25 processes deep on GPU 4, so their steps/s are not speed numbers.

| Instance | `ref` | **`graph`** | `nc_graph` |
| --- | --- | --- | --- |
| L1 10009 | **Y** 3,142, 12.54 / 0.19 | **Y** 3,203, 12.60 / 0.18 | **Y** 3,203, 12.46 / 0.18 |
| L1 10024 | n 3,286, -1.80 / 13.37 | n 3,285, -2.99 / 14.98 | n 3,285, -2.23 / 14.21 |
| L1 10031 | **Y** 3,091, 12.43 / 0.21 | **Y** 3,203, 12.54 / 0.23 | **Y** 3,203, 12.47 / 0.25 |
| L1 10032 | n 3,619, 0.04 / 2.54 | n 3,618, -0.53 / 3.19 | n 3,285, 4.13 / 0.24 (tilt 16 deg) |
| L1 10084 | n 2,702, -2.40 / 14.04 | n x3: 2,693 / 2,704 / 2,712, about -2 / 14-15 | n x3: missed grasp (1,253); 2,721, -1.18 / 14.99; missed grasp (1,253) |
| L2 10085 | n 3,381, -4.55 / 30.45 | **Y** 2,999, 12.03 / 0.22 | **Y** 2,999, 14.04 / 0.45 |
| L2 10032 | n 1,057, missed grasp | n 1,057, missed grasp | n 1,049, missed grasp |
| L2 10010 | n 1,109, missed grasp | n 1,089, missed grasp | n 1,049, missed grasp |
| **Total** | **2/5, 0/3** | **2/5, 1/3** | **2/5, 1/3** |

`graph` follows `ref` step for step where the graph takes the same route (10032: 3,619 vs
3,618 steps; 10084: 2,702 vs 2,693-2,712) -- capture is the same kernels. The eager
`nocable` on L1 10084 grasped and ended 8.0 / 0.22 mm, still a failure.

Stopped at their first failure: `nc_graph_sub8` and `nc_graph_it` each **missed the grasp
on L1 10031** (jaw gap 8.0 mm, closed on nothing), which all three columns above seat.
`nc_graph_vox`, `nc_graph_point` and `nc_graph_con` bought no speed, so were not run (and
the 0.5 mm field is the one `assets/parts.py` `MATE_SDF_VOXEL` records seating 0/13 on the
bulkhead). The `gt` graph on L2 is not a validated reference (docs/validation/cable.md:
"Not validated: `usbc_insert` `gt` on medium"): two of three L2 instances miss the grasp in
every column.

## Real fidelity: the cell protocol

`ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`, `--solve graph --graph mate --no-privileged
--perception vision --renderer rtx --bus-timeout 600 --settle-steps 600`, first three
instances of each cell. `doc` = the cable-free documents.

| Instance | `ref` | **`graph`** | `doc` (runs) |
| --- | --- | --- | --- |
| L1 10009 | n 3,484, -0.29 / 16.66 | n 3,773, -0.25 / 16.58 (alone on the GPU) | n 3,498, -0.30 / 16.12; n 3,822, -0.26 / 16.15 |
| L1 10024 | n 3,469, -0.44 / 14.45 | -- | n 3,484, -0.39 / 14.50 |
| L1 10032 | n 3,495, -0.45 / 14.65 | -- | n 3,522, -0.39 / 14.32 |
| L2 10010 | **Y** 4,278, 13.50 / 0.68 | **Y** 4,301, 12.81 / 0.49 | **Y** 13.04 / 0.06; n 0.76 / 4.68 (tilt 28 deg); n 1.64 / 3.76 |
| L2 10032 | **Y** 4,287, 12.72 / 0.50 | **Y** 4,313, 12.89 / 0.59 | n 4.85 / 1.93 ("no socket clearance" abort); **Y** 12.69 / 0.75; n 4.73 / 2.08 (same abort) |
| L2 10085 | **Y** 4,311, 14.00 / 0.30 | **Y** 4,348, 12.59 / 0.49; **Y** 4,602, 10.84 / 0.67 (alone) | **Y** x4: 13.14 / 0.53, 12.42 / 0.61, 12.35 / 0.60, 13.24 / 0.09 |
| **L2 runs** | **3/3** | **4/4** | **6/10** |

L1 reproduces the validated failure to the tenth of a millimetre in every column: the plug
stops at the mouth 14-17 mm off the bore (docs/validation/cable.md: 0/7, "lat 14-17 mm at
the mouth"; 10009 there: -0.24 / 16.4). On L2 the cabled scene seats every run, eager or
captured; the cable-free scene loses 4 of 10 -- 10085 4/4, but 10010 and 10032 1/3 each --
in the two failure modes the validation already recorded for the reference on 10085 (4/6
there): stopped at the mouth cocked ~25 deg, and the `again_vision` "grasp leaves no socket
clearance" abort. Those modes are the protocol's; what the cable changes is how often they
fire, which is the measured reason the cabled scene is the winner.

## Per-episode rebuild

Measured in the sidecar (`resets[].reset_s`, `rebuilt`). Both shipped samplers declare
`object_xforms` and `static_xforms` model-changing, so every episode rebuilds: 7-10 s
headless, and the rebuild also re-creates the RTX renderer (21-55 s per episode in the
protocol runs above, under load; CoACD also re-decomposes the YAM fingertips on every
build, 1-1.5 s per mesh, upstream).

* **L2 without the cable: avoided.** The rebuild exists for the bulkhead's rear cable,
  whose joint anchor `EntityUsbcCables.build` computes off the drawn bulkhead pose. With
  no cable the draw moves only free bodies, which `EntityRigid` re-places on reset.
  `USBC_MEDIUM_REPOSE` is `USBC_MEDIUM`'s function with `model_key_fields =
  ("static_xforms",)` (always empty on L2): same draws (identical overrides for
  10085/10032/10010), one model key, reset 0.2-2.7 s. Re-posed episodes land in their own
  layouts (idle 10085 re-posed reads -65.12 / 12.51 mm, as rebuilt).
* **L2 with the cable, and L1: kept.** The cabled scene needs it (the anchor). L1's
  welded port moves per episode (`static_xforms`) and `PortUsbc` builds it world-hung
  (`body=-1`), so re-posing it would need a kinematic port body -- the fleet's
  `PLACEABLE_FIELDS` machinery refuses this class for the same reason. Not cheap; left for
  the suite owners.

## Fleets (`--num-worlds 2`)

* Idle: L2 `insert_medium_nocable` runs 10032 + 10085 as **one wave of two**, captured,
  each in its own layout. L1 runs **two waves of one** (50 % utilization): its port pose is
  model-changing, so a fleet buys L1 nothing; nor does the cabled L2, for the same reason.
* `gt` graph (L2 `insert_medium_nocable`, 10085 + 10032), with the lead's `_wait_for_step`
  fix applied in `tasks/usbc_insert/tools.py` (blocking `bus.wait_for_step`, as upstream
  `robot_tools._wait`): **fails before any wait matters**, at the first planning node,
  `brace_sg.plan` -> `RuntimeError: Copy source and destination must be arrays` (both
  worlds, step 91). `ToolsUsbc.plan_brace` (tools.py ~889) and the insertion planner
  (~1545) do `wp.copy(ik.joint_q_ik..., env.state.joint_q)` and step `ik.solver` directly;
  in a fleet `env.state.joint_q` is a world slice (`_ArraySlice`, not a `wp.array`) and
  `env.arm()` is the shared template solver behind a `_WorldArm` proxy whose device calls
  have to be offloaded. So the suite's planners are not fleet-safe independently of the
  wait fix (the `mate` graph's planner is the same code); `num_worlds = 1`.

## Risks

* **Re-capture with a hydroelastic pair (L2).** A re-capture after `invalidate_graph`
  starts wherever the eager launch counter stands; if that capture contains a multiple of
  120 the overflow poll lands inside it. No run here re-captured. A future Newton that
  polls more often, or a hook that invalidates mid-episode, would show up as a CUDA 906 /
  eager fallback -- `run_meta.json` `cuda_graph_steps` says which, so check it on the
  first fast-config cell run.
* **Small samples.** 5+3 `gt` instances and 3+3 protocol instances, mostly single runs;
  the graphs are not run-to-run deterministic (thread timing). The winner's claim rests on
  its scene being the reference's and capture replaying the same kernels; the cable-free
  verdict rests on 10 vs 6 runs.
* **Speed numbers carry other agents' CPU load** (load average 85-155). The eager rows are
  the most affected, so the speedups are if anything conservative.
* `python -m vabar list` now reports 6 `usbc_insert` variants (docs/VALIDATION.md says 4).

## For `vabar/configs/fast.toml`

```toml
[suites.usbc_insert]
variants = { insert_easy = "insert_easy", insert_medium = "insert_medium" }   # the cabled scenes, unchanged
physics = "usbc_fast@v1"   # usbc_bulkhead_960hz@v1 + CUDA graph capture
num_worlds = 1             # L1/L2 rebuild per episode; the suite's IK planners are not fleet-safe
# throughput comes from processes: 4 on one GPU measured ~2.5x one
```

Faster (16-22 % more steps/s, no L2 rebuild), with the L2 caveat above -- the documents
stage `usbc_fast@v1` themselves:

```toml
[suites.usbc_insert]
variants = { insert_easy = "insert_easy_nocable", insert_medium = "insert_medium_nocable" }
num_worlds = 1
```
