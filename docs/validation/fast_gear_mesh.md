# `gear_mesh` under the `fast` config

Measured 2026-11-05 on one RTX PRO 6000 Blackwell (`CUDA_VISIBLE_DEVICES=5`), RoboSimStudio `d9b2e44`
(newton 1.7.0, mujoco / mujoco-warp 3.15.0), with `tools/fast_tune/gear_mesh.py`. Raw runs are in
`/home/shadeform/speed_ws/agent_scratch/d3/sweep/` (the sweep) and `.../d3/final/` (the re-measurement
by registered name and the cell-protocol runs). `summary.json` in each is the table as data.

**Winner: `gear_mesh_fast@v1` (the shipped arrangement at 8 substeps instead of 16) with
`gear_mesh_views_fast@v1` (the shipped rig at half resolution per side).** It runs at **1.90x the
shipped arrangement's idle rate: 130.3 / 133.6 steps/s on L1 / L2, against 68.8 / 69.8**. It passes the gear-train
gate on all five boards measured, within 0.002 of the reference ratio. With the graph runner given
strict turns (see below), `gap_perception` solved **5 of 5** probe instances (3 L1, 2 L2). The reference
solved 4 of 5 under the same protocol. Wall time per episode was 0.63x the reference's on the four
instances both solved. Under the unchanged cell protocol (shipped graph and runner, every camera
rendered every step), the fast pair solved both instances, as the reference did. It took **11.1 min against 19.8 min** on L1
and **18.1 against 24.7 min** on L2: 0.66x the wall time over the two.

Every graph result here is the **first and only episode of its process**. Each probe is one
`python -m vabar run ... --episodes 0` in a fresh interpreter, and all 29 run directories hold one
row. So `sight.py`'s module-level `_CACHE["geometry"]` cannot carry a previous board's layout into
any of them.

```toml
# vabar/configs/fast.toml
[suites.gear_mesh]
physics = "gear_mesh_fast@v1"
cameras = "gear_mesh_views_fast@v1"
```

Both names are new registrations in `tasks/gear_mesh/staging.py`. `gear_mesh_960hz@v1` and
`gear_mesh_views@v2` are unchanged, and so is every document that names them.

## How it was measured

Each candidate is the shipped arrangement and rig with some knobs turned. A candidate is named by its
knobs: `ss8` is 8 substeps, `ss6+imp1` is 6 substeps at impratio 1, `s50` is every camera at 50% per
side. The tool lowers the shipped variant and copies it with `dataclasses.replace(task, name=...,
physics=..., cameras=...)`. It adds the copy with `SUITES.replace("gear_mesh", ...)`, in the probing
process only, and runs it through `python -m vabar run`. So every probe takes the cell's own code path
with one document swapped.

The override is checked twice. `run_meta.json`'s `builds[]` records the substeps. The tool also wraps
`SolverMuJoCo.__init__` and `ScenePipeline.__init__` in its own process and writes `tune_probe.json`:
the options read back from the compiled `mj_model.opt`, the contact buffers, and whether a Newton
collision pipeline exists. Every knob in the table below was confirmed there: iterations, line search,
cone, impratio, tolerance, ccd_iterations, solver, Jacobian, multiccd flag, nconmax/njmax and
collide_every.

The three probes, cheapest first:

1. **Speed.** 600 idle control steps, no renderer, on `gearmesh-easy-pair-01` (L1) and
   `gearmesh-medium-train-001` (L2). Reported: `provenance.steps_per_s` and `builds[0].build_s`.
   They ran one at a time on an otherwise idle GPU 5.
2. **Gate.** `rigs/gear_bench.py`'s `build` + `press_and_run` (the `spin` command) on the candidate
   arrangement. The pass rule is that every gear's mean ratio over the last quarter of the drive lies
   in 0.95-0.99, every gear is seated and the train started. The boards are the default z16-20-24 and
   the hard tier's z16-20-24-20. The finalists also ran z16-20-16 (the L2-001 board), z24-24-24 and
   z24-16-20-24, plus the `broken` negative control. The tool also records the 5th-percentile ratio
   and the peak contact and constraint-row counts.
3. **Graph.** `gap_perception` under the cell's flags (`--motion curobo --no-privileged
   --perception vision --bus-timeout 180`, `--renderer rtx`). It ran on `gearmesh-easy-pair-01..03`
   (L1) and `gearmesh-medium-train-001..002` (L2), episode 0, with peak contacts watched every step.

```bash
python tools/fast_tune/gear_mesh.py speed --cands ref,ss8,ss12,it50,it20,ls20,ls10,pyr,noccd,imp1,nc1408+nj4096,ce2,gap0,tol6,ccd10,cg,dense,ss6,ss4,ss2 --out OUT
python tools/fast_tune/gear_mesh.py gate --cands <same> --teeth-sets "16,20,24;16,20,24,20" --parallel 6 --out OUT
python tools/fast_tune/gear_mesh.py graph --paced 0.25 --cands ref --rigs views --parallel 3 --out OUT
python tools/fast_tune/gear_mesh.py graph --paced 0.25 --cands ss8 --rigs s50 --parallel 3 --out OUT
python tools/fast_tune/gear_mesh.py table --out OUT
```

### The fleet trick is refused for this suite

`--num-worlds 2 --episodes 0:2` raises before building:

```
robosimstudio.scene.compiler.CannotReplicate: gear_mesh/gearmesh-easy-pair-01 cannot run replicated
(2 worlds) yet: these entities contribute substep physics written for one world: board
```

The board's motor and button latch are per-world substep hooks. So every graph probe is single-world.

### Why the graph probe gives the runner strict turns

The single-world runner calls `SolutionGraph.get_action` once per loop, and that call steps the
simulation whatever the graph is doing. The seconds that SAM 3, cuRobo and the node scripts take are
therefore paid in **control steps, at whatever rate the physics runs**. A 1.9x faster arrangement
burns 1.9x more of the step budget per second of thought. On a shared, CPU-loaded machine that alone
fails episodes:

- `ss8`, unpaced, **failed `gearmesh-easy-pair-02` and `-03`**. Both were truncated at the 12,000-step
  budget with every gear seated: one inside `watch_sg` right after the press, one at the last
  `next_station_sg`. The reference ran the same subgraph sequence in 9,296 and 8,192 steps.
- The same `ss8` episode on pair-02 with `--max-steps 24000` **succeeded at 15,810 steps**. The physics
  did the task. The budget ran out on thinking.

So the graph probe of record runs `--paced 0.25`. The tool patches, in its own process only,
`SolutionGraph.get_action`, so that each loop waits until the graph has something for the simulation:
a request on the bus, or a thread blocked in `wait_for_step`. A loop also proceeds when the graph has
finished, or after 0.25 s, so no unknown wait can deadlock. This is the rule the fleet's barrier
states for a wave (`robosimstudio_policy_agent/fleet/barrier.py`: "while a node script is thinking,
nothing steps"). It is applied equally to the reference and every candidate. The 0.25 s ceiling leaks
about 4 steps per second of thought: 529 timed-out holds in a 6,452-step episode. That is about the
5 steps/s the documented reference runs leaked when they rendered every step. Without the ceiling,
the leak would scale with the physics rate.

Two more protocol notes:

- **The graph reads on demand.** A staged copy of `gap_perception` sets `meta.observation_mode:
  "on_demand"`, the runtime's own switch. The per-step frames and gap's 10 Hz observation stream
  stop. `robot.get_observation`, the only way `sight.py` reads the camera, still renders fresh RGB-D.
  The trial reference run under this mode, `gearmesh-easy-pair-01`, succeeded in 6,971 steps.
  `docs/VALIDATION.md` records 6,903 steps with per-step rendering.
- **`--perception oracle` buys nothing here.** `gap_perception` calls `sam3.segment_text` itself and
  has no oracle body. The flag is ignored, so it is not a cheaper probe.

## Every candidate

Speed is idle steps/s on an otherwise idle GPU. The gate column is the default board; the boards column
counts every board each candidate ran. "x ref" is the mean of the L1 and L2 ratios.

| candidate | knob(s) | steps/s L1 | steps/s L2 | x ref | build_s | spin ratio z16-20-24 (p5) | boards passed (worst) | graph | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `ref` | `gear_mesh_960hz@v1` | 68.9 | 69.9 | 1.00 | 5.5 / 4.8 | 0.967 (0.95) | 5/5 (0.961) | 4/5 paced; 4/5 unpaced | reference |
| `ss12` | substeps 12 | 89.6 | 90.1 | 1.29 | 4.4 / 6.8 | 0.966 (0.95) | 2/2 (0.961) | - | survives, dominated by ss8 |
| **`ss8`** | **substeps 8** | **130.7** | **132.1** | **1.89** | 4.9 / 4.4 | **0.968 (0.95)** | **5/5 (0.963)** | 5/5 paced (s50 rig); 1/3 unpaced (budget) | **winner (physics)** |
| `ss6` | substeps 6 | 169.3 | 169.6 | 2.44 | 4.3 / 7.6 | 0.974 (0.95) | 1/2: z16-20-24-20 **0.801** (p5 0.56) | - | fails the hard-tier board |
| `ss4` | substeps 4 | 239.3 | 243.5 | 3.48 | 5.0 / 4.5 | **0.507** (0.37) | 0/2 | - | fails |
| `ss2` | substeps 2 | 412.8 | 421.0 | 6.01 | 4.2 / 4.2 | **0.038** (0.00) | 0/2 | - | fails |
| `it50` | iterations 50 | 64.3 | 69.8 | 0.97 | 7.2 / 6.6 | 0.967 (0.95) | 1/1 | - | no gain |
| `it20` | iterations 20 | 66.2 | 63.2 | 0.93 | 9.1 / 7.8 | 0.967 (0.95) | 1/1 | - | no gain |
| `ls20` | ls_iterations 20 | 67.1 | 68.1 | 0.97 | 6.9 / 6.2 | 0.966 (0.95) | 1/1 | - | no gain |
| `ls10` | ls_iterations 10 | 68.2 | 68.6 | 0.99 | 6.1 / 5.6 | 0.966 (0.95) | 1/1 | - | no gain |
| `tol6` | tolerance 1e-6 | 67.9 | 70.0 | 0.99 | 5.5 / 4.5 | 0.966 (0.95) | 1/1 | - | no gain |
| `ccd10` | ccd_iterations 10 | 69.0 | 70.3 | 1.00 | 13.2 / 4.4 | 0.966 (0.95) | 1/1 | - | no gain |
| `pyr` | cone pyramidal | 74.5 | 81.0 | 1.12 | 9.6 / 8.1 | **0.924** (0.92) | 0/1 | - | fails the gate |
| `noccd` | multiccd off | 80.4 | 84.5 | 1.19 | 4.8 / 6.2 | **0.205** (0.05) | 0/1 | - | fails: the train does not drive |
| `imp1` | impratio 1 | 68.5 | 69.3 | 0.99 | 4.7 / 6.3 | 0.980 (0.97) | 1/1 | - | no gain alone; shifts the ratio +0.013 |
| `nc1408+nj4096` | nconmax/njmax = 1.5x peak | 69.6 | 69.5 | 1.00 | 4.3 / 6.2 | 0.966 (0.95) | 1/1 | - | no gain, so not worth the overflow risk |
| `ce2` | collide_every 2 | 68.0 | 70.2 | 1.00 | 4.9 / 6.5 | 0.967 (0.95) | 1/1 | - | inert (see below) |
| `gap0` | rigid_gap 0 | 69.0 | 69.0 | 0.99 | 4.5 / 6.0 | 0.966 (0.95) | 1/1 | - | no gain |
| `cg` | solver CG | 23.9 | 22.4 | 0.33 | 5.6 / 8.8 | 0.967 (0.95) | 1/1 | - | 3x slower |
| `dense` | jacobian dense | 31.0 | 24.2 | 0.40 | 6.0 / 6.0 | 0.967 (0.95) | 1/1 | - | 2.5x slower |
| `ss8+imp1` | combination | (= ss8) | | | | 0.981 (0.96) | 2/2 (0.979) | - | impratio buys no speed; not needed at 8 |
| `ss6+imp3` | combination | (= ss6) | | | | 0.978 (0.95) | 1/2: z16-20-24-20 0.946 (p5 0.75) | - | fails the hard-tier board |
| `ss6+imp1` | combination | 171.0 | 167.2 | 2.44 | 5.0 | 0.985 (0.96) | 5/5 (0.982) | 3/5 paced (4/7 with repeats) | runner-up; see below |
| `ss4+imp1` | combination | (= ss4) | | | | **0.691** (0.52) | 0/2 | - | fails |
| `ss4+imp3` | combination | (= ss4) | | | | **0.462** (0.34) | 0/2 | - | fails |

The combinations change only impratio, which costs nothing (`imp1` runs at 0.99x). So their speed is
their substep count's. `build_s` is dominated by Warp's kernel cache: a knob that is a compile-time
constant (line search, cone) compiles new kernels on first use. It does not separate the candidates.

**Gate boards for the finalists** (mean ratio, 5th percentile):

| board | `ref` | `ss8` | `ss6+imp1` |
|---|---|---|---|
| z16-20-24 (the default gate) | 0.967 (0.95) | 0.968 (0.95) | 0.985 (0.96) |
| z16-20-24-20 (hard tier, `quad`) | 0.962 (0.94) | 0.963 (0.94) | 0.982 (0.96) |
| z16-20-16 (`gearmesh-medium-train-001`) | 0.968 (0.95) | 0.970 (0.95) | 0.986 (0.96) |
| z24-24-24 | 0.966 (0.95) | 0.967 (0.95) | 0.984 (0.96) |
| z24-16-20-24 | 0.961 (0.94) | 0.963 (0.94) | 0.983 (0.96) |
| gear_b removed (`broken`) | gear_c 0.000 ([VALIDATION](gear_mesh.md)) | gear_c 0.000, gear_a 0.973 | gear_c 0.000, gear_a 0.988 |
| peak contacts / rows, z16-20-24-20 | 858 / 2,568 | 856 / 2,568 | 856 / 2,568 |

## Graph fidelity

`gap_perception` ran under the cell flags, on-demand observations and strict turns (`--paced 0.25`),
single world, `--renderer rtx`, episode 0. Each cell is success, then steps, then episode wall time.
"Seated" counts the gears on their pegs at the end.

| instance | `ref` (`gear_mesh_960hz@v1`, `views@v2`) | **winner: `ss8` + `s50`** | `ss6+imp1` + `s50` |
|---|---|---|---|
| `gearmesh-easy-pair-01` (L1) | yes, 4,154, 589 s | **yes, 3,922, 268 s** | yes, 6,452, 384 s |
| `gearmesh-easy-pair-02` (L1) | yes, 6,437, 800 s | **yes, 6,055, 465 s** | **no**, 6,224: 1 of 2 seated. Repeat: **no**, 6,234, same state |
| `gearmesh-easy-pair-03` (L1) | yes, 5,822, 512 s | **yes, 5,462, 440 s** | yes, 3,762, 282 s |
| `gearmesh-medium-train-001` (L2) | yes, 6,018, 803 s | **yes, 5,795, 519 s** | yes, 8,033, 533 s |
| `gearmesh-medium-train-002` (L2) | no, 9,328: 2 of 3 seated, `watch` stalled, abort | **yes, 7,773, 660 s** | no, 9,027: 2 of 3 seated. Repeat: yes, 8,020 |
| **solved** | **4/5** | **5/5** | **3/5** (4/7 with repeats) |
| peak contacts / rows in any episode | 693 / 2,080 | 720 / 2,164 | 731 / 2,188 |

- **The winner solved every instance the reference solved, plus `medium-train-002`.** The reference
  also failed that one in its unpaced run, in the same state: 2 of 3 gears seated. Steps on the four
  common successes: 21,234 against 22,431. Wall time: 1,692 s against 2,704 s (0.63x).
- **`ss6+imp1` failed `gearmesh-easy-pair-02` twice**, in the same way both times: 1 of 2 seated,
  6,224 and 6,234 steps. The reference and `ss8` both seated that board. Its gate numbers are good, but
  the gate starts from an assembled train. A gear landing on a peg and finding its neighbour's tooth
  spaces is the part the gate does not exercise. At 6 substeps and impratio 1 that landing goes
  differently. So it misses the rule "success at least the reference's", and it is not the winner.
- **Unpaced, as the shipped runner steps** (on-demand observations, `views@v2`):

  | instance | `ref` | `ss8` |
  |---|---|---|
  | pair-01 | yes, 5,566 | yes, 6,904 |
  | pair-02 | yes, 9,296 | **no**: truncated at 12,000 in `watch_sg`, both gears seated. With `--max-steps 24000`: **yes**, 15,810 |
  | pair-03 | yes, 8,192 | **no**: truncated at 12,000 at the last `next_station_sg`, both gears seated |
  | medium-001 | yes, 11,796 | (stopped; superseded by the paced probe) |
  | medium-002 | no: abort, 2 of 3 seated | (stopped) |

  Those ran 7-10 at a time on GPU 5, on a host at load average 100-240. That is the condition under
  which the step budget pays for thinking.
- **Step counts vary from run to run by more than the arrangements differ.** The reference took
  these step counts on `gearmesh-easy-pair-01`:
  - 4,154 paced;
  - 4,230 under the cell protocol;
  - 5,566 unpaced, under load;
  - 6,971 unpaced, run alone.

  `docs/VALIDATION.md` records 6,903. Compare success first and steps second.

### The shipped cell protocol, exactly

This is the shipped `gap_perception` with no pacing and every camera rendered every step: what
`python -m vabar bench --cell gear_meshing/L1` runs today, with only the two staged names swapped. Two
episodes ran at a time on GPU 5. Each cell is success, then steps, then wall time, then end-to-end
steps/s.

| instance | `gear_mesh_fast@v1` + `gear_mesh_views_fast@v1` | `ref` (`gear_mesh_960hz@v1` + `views@v2`) |
|---|---|---|
| `gearmesh-easy-pair-01` (L1) | **yes, 4,001, 663 s (11.1 min), 6.0/s** | yes, 4,230, 1,187 s (19.8 min), 3.6/s |
| `gearmesh-medium-train-001` (L2) | **yes, 8,134, 1,085 s (18.1 min), 7.5/s** | yes, 6,161, 1,480 s (24.7 min), 4.2/s |

`docs/VALIDATION.md` records 6,903 and 8,816 steps for these two reference episodes; it does not
record their wall time.

## What the sweep found

- **The substep count is the only knob that buys speed.** Steps/s goes as 1/substeps: 1.29x at 12,
  1.89x at 8, 2.44x at 6. Every solver cap is slack. MuJoCo-Warp runs the Newton iterations inside a
  CUDA-graph conditional loop (`opt.graph_conditional`) that exits when every world converges, so
  `iterations` 100 -> 20, `ls_iterations` 50 -> 10, `tolerance` 1e-8 -> 1e-6 and `ccd_iterations`
  35 -> 10 each moved steps/s by at most 7%, and never upward.
- **`collide_every`, `rigid_contact_max` and `contact_matching` are inert under this arrangement.**
  With `use_mujoco_contacts=True`, MuJoCo detects its own contacts, and the composer builds no Newton
  collision pipeline: `tune_probe.json` records `newton_collision_pipeline: false` for every build.
  The schedule only consults `collide_every` when that pipeline exists. `ce2` is 1.00x. The
  CollisionSpec fields that do reach MuJoCo are the contact stiffness and `rigid_gap`. The gears and
  the board state their own gap of 5e-5 and stiffness of 6e4/500, so `rigid_gap` only touches the
  robot, the bench and the mat (`gap0`: 0.99x).
- **Buffers.** The largest count seen anywhere was 936 contacts and 2,788 rows (the failing `ss2` on
  the four-gear board). The reference peaks at 886 on z24-16-20-24. With the arm in the scene, graph episodes peaked
  at 693-731 contacts and 2,080-2,188 rows (L2). Trimming nconmax/njmax to 1408/4096 (1.5x) changed nothing measurable.
  An overflow is an illegal memory access rather than a warning, so the fast arrangement keeps
  16384/32768.
- **multiccd and the elliptic cone hold the task up.** `noccd` is 1.19x faster and the train runs at
  0.205 of its ratio. `pyr` is 1.12x faster and the train runs at 0.924.
- **impratio trades against the substep.** At 6 substeps and impratio 10, the three-gear gate drives
  (0.974) and the four-gear train slips to 0.80. impratio 3 gets it to 0.946 and impratio 1 to 0.982.
  impratio 1 is MuJoCo's default; with it the long train runs *closer* to ideal than the reference
  does, at every substep count tried (16, 8, 6). Nothing tried rescued 4 substeps.
- **Hub sectors and tooth prisms are not exposed.** `SpecGear` calls `gear_frame` without
  `hub_sectors`, so every gear gets `HUB_SECTORS = 24`. The tooth count is the gear's identity. Neither
  can be swept without changing the asset code, which this sweep does not do.
- **Cameras.** The graph reads only `overhead`, and `sight.py` measures its own metres per pixel, so
  every threshold is in metres. At `s50` (overhead 640x480, about 1 mm/px) it read every board. The
  winner's 5/5 and `ss6+imp1`'s successes were all at `s50`. `ss6+imp1`'s failures were seating failures, not
  perception failures: the gears were found and carried. The price of the rig, with every camera path
  traced every step (idle, 150 steps, run alone):

  | physics | `views@v2` | `s75` | `s50` |
  |---|---|---|---|
  | `ref` | 6.7 steps/s | - | 9.9 |
  | `gear_mesh_fast@v1` | 6.4 | 8.7 | 9.1 |

  Under per-step rendering, rendering is the whole cost (about 110 ms a step against 7.6 ms of
  physics). So halving the resolution buys 1.4-1.5x and the physics buys nothing there. On demand,
  the rig costs only the few dozen frames the graph asks for.

## Speedup

- **Physics: 1.90x.** `gear_mesh_fast@v1` re-measured by its registered name, on an idle GPU, two
  repeats: 130.3 (L1) and 133.6 (L2) idle steps/s. The reference in the same session: 68.8 and 69.8.
  The task called this rate "70 steps/s".
- **Graph episodes: 0.63x the wall time** under strict turns on the four instances both solved
  (1,692 s against 2,704 s). These are single episodes taken 7-10 at a time on one GPU, so the
  absolute times carry contention. Both arms ran in the same window, so the ratio is the fairer
  number.
- **The cell protocol as shipped renders every step.** There the rig is the lever, at 1.4-1.5x (see
  Cameras), and the physics is not. Measured with the shipped graph and runner, the fast pair took
  11.1 min for an L1 episode and 18.1 min for an L2 episode, two at a time on one GPU. The reference took
  19.8 and 24.7 min, so the pair runs at 0.66x the wall time. **A
  30-instance cell in 30 min therefore needs several episodes on the GPU at once, and on-demand
  rendering helps most.** The paced probe ran 7-10 graph episodes at a time on one GPU, at 4-11 min
  each. That suggests 30-40 min for 30 instances. It is an extrapolation from those runs, not a
  measured cell.

## Risks

- **Step budgets pay for thinking in the single-world runner.** Faster physics burns more of a
  task's `:max-steps` per second of SAM 3 or cuRobo. Unpaced and under load, `ss8` truncated 2 of 3
  L1 episodes that its physics completed (the doubled budget succeeded). Under per-step rendering the
  runner only steps 6-7.5 times a second, and the fast pair finished both cell-protocol episodes well
  inside budget (4,001 of 12,000 and 8,134 of 18,000). The risk grows as rendering gets cheaper. The
  `fast` config should run where the simulation waits for the graph: the fleet's barrier, the future default runtime, or the
  `--paced` rule this probe used. Otherwise the budgets need scaling by the speedup. The fleet itself
  refuses this suite today (`CannotReplicate ... board`).
- **Five instances, one episode each.** Graph timing is not pinned, so 5/5 against 4/5 is evidence of
  "no worse", not proof of "better". The gate is the deterministic part, and on it `ss8` sits within
  0.002 of the reference on all five boards. The margin below it is real: 6 substeps already drops
  the hard-tier board to 0.80.
- **The contact time constant is 4% softer.** MuJoCo clamps the stated 4 ms solref to 2*dt =
  4.17 ms at 8 substeps. No measured quantity moved, but a fit tighter than this suite's 0.3 mm would
  see it first.
- **Graph runs never reached the hard tier.** The `quad` / four-station boards were gated (0.963) but
  not run under the graph.
- **Contact buffers are unchanged.** That is deliberate. Trimming them bought nothing, and an
  overflow is an illegal memory access. The largest count seen was 936 contacts, against nconmax 16384.
- **Not swept:** hub-sector and tooth-prism counts (not exposed by `SpecGear`). Also the `panda` host
  row.
