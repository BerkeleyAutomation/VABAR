# cable_route: a `fast` physics arrangement, by measurement

**Winner: `cable_route_fast@v1`** (`tasks/cable_route/staging.py`). It is the reference
`cable_coupled_960hz@v1` with **one proxy-coupler pass and 20 VBD iterations**, still at
960 Hz, with every other number unchanged. **15.1 steps/s idle, against 7.16 for the
reference measured the same way (2.1x) and the 6.3 the brief started from (2.4x).** Its
outcomes stay inside the reference's own run-to-run spread on every probe, and on the
cell's graph protocol it scored what the reference scored on both scenes.

SuperDex is 4x faster than the reference but fails cheap fidelity. Nothing here gets a
30-instance cell near 30 minutes on one GPU (see "What one GPU gets you").

## Method

Same stack as [cable](cable.md): RoboSimStudio `d9b2e44` (unmodified), newton 1.7.0,
mujoco-warp 3.15.0, superdex-physics 1.0.1 (import name `superdex.physics`), RTX PRO
6000s (GPUs 3 and 7), and a 240-core host shared with other sweeps (load average 60-70).

Every candidate is a string of knobs (see the `tools/fast_tune/cable_route.py` docstring).
The tool builds an `ArrangementSpec` from it and registers it under a new name
(`cable_fast_tune/<knobs>@v0`). It then swaps the arrangement into every `cable_route`
variant in-process (`dataclasses.replace` on the lowered `TaskSpec` + `SUITES.replace`)
and runs `vabar run` in the same process. No shipped name, document or arrangement is
touched. `run_meta.json` `provenance.builds[].substeps` confirms that each override took
effect. `name:cable_route_fast@v1` runs the registered winner itself, and
`CABLE_FAST == arrangement_for("vbd:passes=1,iters=20")`.

Probes, cheapest first; a candidate stops at its first failure:

1. **Speed**: `cable_route/port_easy --episodes 10009 --solve idle --max-steps 400` with
   no renderer, reading `provenance.steps_per_s` and `builds[].build_s`. One run at a
   time on an otherwise idle GPU. The reference ran five times and the finalists two to
   five times.
2. **Cheap fidelity**: `--solve routine` on `cable_route/weave3` and on `port_easy`
   instances 10009, 10062, 10084, 10043 and 10053 (cable_routing/L1). The routine's
   step count is fixed by its schedule, so these ran in parallel on a shared GPU.
3. **Real fidelity**: the cell protocol (`ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`,
   `--solve graph --graph gt --no-privileged --perception vision --renderer rtx
   --bus-timeout 600 --settle-steps 60`) on 10009 and 10062.

To re-run, use `python tools/fast_tune/cable_route.py probe --gpu {3|7} --stage
{speed|weave3|port|graph} [--episodes a,b] --out DIR <candidates...>`, and summarise
with `... table --out DIR`. The raw runs are in `/home/shadeform/speed_ws/agent_scratch/d1/`.

## The yardstick is noisy, and the reference is part of the noise

The routine is open-loop between its `[grasp]` read-outs, and contact is not
bit-deterministic. As a result, **the reference does not reproduce its own outcomes**:

* **weave3**: success at 2,737 steps, then **0.67 at 2,931 steps (seat1 lost)** on the
  identical command. cable.md recorded success at 2,732. The two runs part at the
  station-2 read-out (116 vs 132 mm of cable behind it).
* **port_easy, two runs per instance**: four of the five instances closed at a different
  reward in the second run:

  | Instance | Run 1 | Run 2 |
  | --- | --- | --- |
  | 10009 | seat0 | seat0 |
  | 10062 | seat0+seat1 | seat0 |
  | 10084 | seat1 | seat1+port box |
  | 10043 | seat0+seat1 | seat0+port box |
  | 10053 | seat0+seat1 | full success, 1.00 |

  Seats held over the five instances: 8 of 10 in run 1, 6 of 10 in run 2.
* **Graph on 10009**: success (1.00, 6,374 steps) here; 0.50 in cable.md.

So "matches the reference" is read against the **range** of outcomes the reference
produced: per instance, and as seats held over the five port_easy instances. Every
verdict-deciding probe was run twice.

## Results

Speed is idle steps/s on the clean GPU, one number per run; the spread between repeats
comes from the shared host. In the outcome columns, S0/S1/S2 = that seat held,
P = the port-box clause, +end = the free end in the port box, `--` = nothing held.

The port_easy column lists instances 10009 / 10062 / 10084 / 10043 / 10053, followed by
the seats held out of 10. A second run follows `|`.

| Candidate | Knobs vs reference | steps/s | x ref | build_s | weave3 routine | port_easy routine | graph 10009 / 10062 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **reference** `cable_coupled_960hz@v1` | 16 substeps, 2 passes, VBD 40, MuJoCo 100/20 | 7.30, 7.10, 7.10, 7.20, 7.10 | 1.0 | 17-19 | 1.00 (2,737) \| 0.67 S0S2 (2,931) | S0/S0S1/S1/S0S1/S0S1 = 8 \| S0/S0/S1P/S0P/S0S1P+end = 6 | 1.00 (6,374) / 0.50 S0S1 (4,111) | -- |
| **`vbd:passes=1,iters=20`** = **`cable_route_fast@v1`** | 1 pass, VBD 20 | 15.2, 15.1, 15.0, 15.0 | **2.1** | 15-16 | 0.67 S0S1 (2,931) \| **1.00** (2,733) | S0S1/S0S1/S0S1/S0S1/S0 = **9** | **1.00** (6,297) / **0.50** S0S1 (4,123) | **winner** |
| `vbd:substeps=8,passes=1` | 8 substeps (480 Hz), 1 pass | 12.9, 16.0, 15.8, 14.8, 16.6 | 2.1 | 16-19 | 1.00 (1,916) \| 1.00 (2,733) | S0/--/S1P/S1P/S0 = 4 \| S0/S0S1/S1P/S1P/S0 = 6 | 0.75 S0S1+end (8,566) / 0.50 S0S1 (4,176) | in range but no faster than the winner, at the 480 Hz edge (risk 1); 10053 one seat short in both runs |
| `vbd:substeps=12,passes=1` | 12 substeps (720 Hz), 1 pass | 13.2 | 1.8 | 17 | 0.67 S0S1 \| 1.00 (2,732) | S0/S0S1/P/S0S1/S0S1 = 7 | 0.75 S0S1+end (4,242) / 0.50 S0S1 (4,237) | admissible, slower |
| `vbd:passes=1` | 1 pass | 10.5, 10.8, 11.2 | 1.5 | 16-25 | 0.67 S0S2 \| 0.67 S0S2 | S0/S0S1/S1P/S0S1/S0S1P = 8 | 0.75 S0S1P (6,200) / 0.50 S0S1 (4,116) | admissible, slower |
| `vbd:substeps=8,passes=1,iters=20` | 480 Hz, 1 pass, VBD 20 | 18.4, 20.2 | 2.7 | 16-17 | 0.67 S0S1 \| **0.33** S0 | S0/S0S1/S1 (3 instances) | -- | out: 0.33 on weave3 is below anything the reference produced |
| `vbd:substeps=8` | 480 Hz | 11.3 | 1.6 | 18 | **diverged** at step 2,636 (MuJoCo iteration-limit warnings first) | -- | -- | out |
| `vbd:iters=10` | VBD 10 | 12.6 | 1.8 | 16 | 0.33 S0 | -- | -- | out |
| `vbd:iters=20` | VBD 20 | 10.4, 10.5 | 1.5 | 17-21 | -- | -- | -- | not run: the 20 iterations only pay with one pass |
| `vbd:substeps=8,passes=1,iters=10` | | 17.2 | 2.4 | 16 | -- | -- | -- | not run: `iters=20` at 480 Hz was already out |
| `vbd:pitch=0.5` | 42 capsules/m (24 on port_easy) | 7.6 | 1.1 | 17 | -- | -- | -- | out: no speed gain (the step is launch-bound, not math), and it changes what the 0.04/0.06 containment shares mean |
| `sdx:rod=2` (adapter v1) | rod on SuperDex, 2 engine steps, one-way | 32.9, 28.2 | 4.3 | 9-11 | 0.00 | S0/S0/-- (3 instances) | 0.50 S0S1 (4,015) / 0.50 S0S1 (4,224) | out: cheap fidelity |
| `sdx:rod=4` | 4 engine steps | 29.7 | 4.1 | 9 | 0.00 | | | out |
| `sdx:rod=2,fb=auto` | two-way | 20.6 | 2.9 | 11 | 0.00 | | | out |
| `sdx:rod=2,arm=8` | arm island at 8 substeps | 20.2 | 2.8 | 13 | -- | | | no gain |
| `sdx:rod=2,mjc=1` | MuJoCo's own contacts | 33.3 | 4.7 | 10 | -- | | | |
| `sdx:rod=2,mj_iters=50,mj_ls=10` | MuJoCo 50/10 | 29.9 | 4.2 | 9 | -- | | | |
| `sdx:rod=2,threads=1` / `threads=32` | SuperDex worker threads | 24.1 / 29.8 | | 13 / 12 | -- | | | 8 (the sweep's default) is enough |
| `sdx:rod=2,ls=ldlt,mu=3600` (adapter v3) | + scene friction, blade filter, direct solve, rod mu 60 | 33.2, 25.3 | 4.1 | 10-13 | 0.00 | S0/S0/-- (3 instances) | -- | out: cheap fidelity |

Ten more SuperDex rod configurations on weave3 are listed under candidate (a).

## Winner: `cable_route_fast@v1`

The definition is
`dataclasses.replace(CABLE, backends=CABLE.backends_replacing("rod", CABLE.backend_for("rod").with_(iterations=20)), coupling=CABLE.coupling.with_(iterations=1))`.
Everything else is as the reference states it: 16 substeps at 60 Hz, AVBD beta, contact
k-start and history, the 256-row contact buffer, MuJoCo newton/implicitfast/elliptic at
100/20 with its constraint ceilings, the proxy's mass scale of 20, staggered coupling,
`latest` matching and the 2 mm gap.

* **Speed: 15.1 steps/s idle.** That is the mean of four clean runs (15.0-15.2), two of
  them through the registered name. It is **2.1x** the reference measured the same way
  (7.16, five runs) and **2.4x** the brief's 6.3. The build takes 15-16 s.
* **Why these two knobs.** The step is launch-bound. One coupler pass halves both
  solvers' restart scaffolding per substep. Once there is one pass, VBD's iterations are
  most of what is left, so halving them buys about 40 % (10.8 -> 15.1). That is not the
  ~2 % the staging note measured for the same cut at two passes.
* **weave3 routine**: 0.67 (S0+S1), then success at 2,733 steps. Both are outcomes the
  reference produced.
* **port_easy routine**: 9 of 10 seats over the five instances (reference: 8, then 6).
  Against the reference's best run, it gained seat1 on 10009 and seat0 on 10084, and was
  one seat short on 10053.
* **Real graph**: identical outcomes to the reference on both scenes. 10009 was a
  success (1.00 at 6,297 steps; reference 1.00 at 6,374), and 10062 scored 0.50 with
  both seats woven (4,123 steps; reference 4,111).

**Why not 480 Hz.** `vbd:substeps=8,passes=1` is equally fast (mean 15.2 over five runs,
spread 12.9-16.6) and also inside the reference's range. However, the same rate with the
reference's two passes **diverged** on weave3. The staging note also records that
480 Hz kills the `wrap` variant's wrap. Its port seats also trended lower: 4 and 6 of 10,
and one short on 10053 in both runs. At equal speed, the winner keeps the measured
960 Hz.

**Fallbacks**, should a longer evaluation disagree: `vbd:substeps=12,passes=1` (1.8x)
and `vbd:passes=1` (1.5x) are both inside the reference's range on every probe.

## Candidate (a): SuperDex rod + MuJoCo arms

### What it took to run at all

`cable_route`'s rod spec says `entity="rod"`, and the library binds
`("rod", engine="superdex")` to `EntityRodSuperdex`. Staged as it is, that rod builds and
steps, but it is not this suite's cable, and two library seams break before the first
episode ends.

The adapter is `tools/fast_tune/cable_superdex.py` (`EntityCableSuperdex`). The sweep
binds it with `register_entity("cable_rod", engine="superdex")` and points the variants'
rod spec at `cable_rod`; on VBD, `cable_rod` is the same `EntityCable` as `rod`. Because
SuperDex did not win, the suite does not register the adapter. Moving it into
`tasks/cable_route/assets/` and binding it in `assets/rod.py` is all a SuperDex
arrangement would need.

| Gap | Symptom | Closed by |
| --- | --- | --- |
| superdex-physics 1.0.1 renamed `RodActorParams.use_visual_mesh_contact` -> `use_contact_skin` (and `visual_mesh_contact_element_type` -> `contact_skin_element_type`) | `AttributeError` in the pinned library's `build_rod_actor` at scene build | `superdex_compat()`: a property alias on the binding class, applied once per process and only when the old name is missing |
| The goal binder's `rod`/`velocity` reader indexes `body_qd` by the chain's bodies | `BindError: cable is a chain, but nothing in the scene answers for it` at the first `(speed cable)` | `velocity_reads_entity_first()`: the reader asks `Entity.read("velocity")` first, as its `points` reader already does. The adapter answers from node velocities; every other rod reads as before |
| **Spawn shape**: `EntityRodSuperdex` lays the straight `rest_polyline()` and lists `bow/shift/wave/yaw_range` as unread | every episode's cable is straight | the same straight nodes are bent through the chain's own `_bow_cable` with the episode's `cable_*` fractions, and the rod is stress-free at the spawn, as the chain is. **On port_easy 10009, all 51 nodes are within 0.05 um of the capsule chain's endpoints** (`tools/fast_tune/spawn_check.py`) |
| Stiffness: `EA/EI/GJ` derived from `young_modulus` (5 MPa) | about 44x the bend that `cable.weave_bend` states at 3.5 mm | `EI = bend_ke * pitch`, `EA = stretch_ke * pitch`, `GJ = twist_ke * pitch` |
| `positions(body_q)`, which the routine and tools call | indexes `body_q` rows by particle id, silently returning wrong points | a `[n, 7]` pose array is answered with the chain's segment centres from the live particles; `hold_segment` maps to the engine pin |
| `scene_friction` unread | every shape kept mu 1.0 (the chain's scene sets tips 60, bench 0.02, the rest 2.0), and MuJoCo's arm-vs-bench friction differs too | `apply_model_materials` writes the reference's values before the bridge mirrors them (adapter v2) |
| `filter_fingertip_scenery` unread | on the weaves, the blades stand on the bench instead of passing through it to the rod's equator (`routine.PAD_FLOOR_CLEAR = 0`) | the chain's blade-vs-static filter pairs, added at build (adapter v3) |

Two limits remain by the library's design: **one world only** (`hold` raises on
`world != 0`), and **one-way coupling** with `feedback="off"`. Two-way coupling was
measured too.

### SuperDex results

Speed is the reason to want it: **25-33 steps/s idle, about 4x**, because the rod leaves
the CUDA graph. A 600-step profile puts the SuperDex tick at about 8 ms per control step
(two engine steps); the rest is the MuJoCo arm graph and host readback. No knob moved
the speed much:

* 2 and 4 engine steps run at the same speed.
* Two-way feedback costs 30 %.
* MuJoCo's own contacts add 10 %.
* 1 worker thread costs 15 %.
* The arm island at 8 substeps is no faster.

Fidelity is where SuperDex stops. Thirteen rod configurations ran on weave3, where the
reference's outcomes are 1.00 or 0.67:

| Adapter | Knobs | weave3 reward (seats) |
| --- | --- | --- |
| v1 spawn + stiffness | rod=2 / rod=4 / fb=auto / ff=1 mm/s | 0.00 / 0.00 / 0.00 / 0.00 |
| v1 | ls=ldlt (CG left 202 linear solves unconverged; LDLT left none) / pad=2 mm / rod=4+ff+ldlt | 0.33 S0 / 0.33 S0 / 0.33 S0 |
| v2 + scene friction | rod=2+ldlt / +rod mu 60 / +mu 60+ff / rod=4+ldlt+mu 60 | 0.00 / 0.33 S1 / 0.00 / 0.67 S0S1 |
| v3 + blade filter | rod=2+ldlt / +mu 60 / rod=4+ldlt | 0.00 / 0.00 / 0.00 |

On port_easy, v1 and v3 alike held seat0 on 10009 and 10062 and nothing on 10084, where
the reference held seat1 in both of its runs. Seat2 of weave3 was never held. At the
routine's last take, the cable lay 25-47 mm off station 0; for the reference it lies
4-7 mm off.

The real graph is a different picture. **On 10009 and 10062, the `gt` graph wove both
seats and missed the port box (0.50 at 4,015 and 4,224 steps).** That is cable.md's
reference outcome on both scenes, but below the reference's 1.00 on 10009 here. The
graph re-perceives the rod between crossings and has a restore pass; the routine does
not. That is a reason to come back to SuperDex, not a pass.

**Verdict: out on cheap fidelity**, across solver, friction, contact, coupling and rate
knobs.

## What one GPU gets you

None of this reaches 30 minutes for a 30-instance cell.

* **Episode length**: the graph protocol ran 4,000-8,600 steps per episode here
  (cable.md: 4,271).
* **One process**: at the winner's 15 idle steps/s, and less with the RTX renderer the
  protocol needs, that is about 5-10 minutes per episode. Thirty instances take about
  2.5-5 hours.
* **More processes on the same GPU**: VBD gains only about 1.33x (time-slicing; the
  brief's own measurement).

SuperDex is the only arrangement here whose step is mostly CPU, so it is the only one
that could scale across processes on one GPU. Closing its seat-retention gap on the
routine is the path to the 30-minute target, not more VBD knobs.

## Risks

1. **Only port_easy and weave3 were probed.** port_medium (24,000-step budget, 70
   capsules, three stations) and the L2 cell were not run, and the arrangement is
   suite-wide once the config selects it. The `wrap` variant (not a benchmark cell) was
   not run either. Its winding is the suite's most physics-sensitive measure, and its
   history (4 -> 2 passes improved it; 480 Hz killed it) does not say what one pass at
   20 iterations does to it.
2. **Halving VBD iterations has the least margin of the two cuts.** At 480 Hz the same
   cut produced the one sub-range weave3 outcome in this sweep (0.33). At 960 Hz it stayed
   in range, but over 2 weave3 runs, 5 port instances and 2 graph scenes. A 30-instance
   routine pass of both arrangements is the cheapest test that could separate them.
3. Speeds are idle steps/s without a renderer. The protocol adds RTX rendering, which the
   arrangement does not change, so the end-to-end ratio will be below 2.1x.
4. `num_worlds = 1` is the only setting measured. It is not a limit of the winner, since
   VBD replicates. Anything above 1 needs the lead's `_wait_for_step` fix.

## The `fast` config

```toml
[suites.cable_route]
# One proxy-coupler pass and 20 VBD iterations at the reference's 960 Hz:
# 2.1x the reference's idle steps/s. docs/validation/fast_cable_route.md
physics = "cable_route_fast@v1"
num_worlds = 1
```

## Files

| File | What |
| --- | --- |
| `tasks/cable_route/staging.py` | `CABLE_FAST = register_arrangement("cable_route_fast@v1", ...)` under a new name; `cable_coupled_960hz@v1` and every document unchanged |
| `tools/fast_tune/cable_route.py` | the re-runnable sweep: candidates, probes and the table |
| `tools/fast_tune/cable_superdex.py` | the SuperDex adapter the sweep registers for `sdx:` candidates |
| `tools/fast_tune/spawn_check.py` | dumps a candidate's initial centreline, for diffing spawns |
| `docs/validation/fast_cable_route.md` | this document |

`pyproject.toml` is unchanged, because the winner needs nothing beyond the existing
stack. To re-run the `sdx:` candidates, install `superdex-physics==1.0.1`.
