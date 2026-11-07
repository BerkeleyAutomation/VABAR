# Where a cable-routing fleet wave's wall time goes

`cable_routing/L2`, `--config fast`, 15 worlds on each of 2 MPS workers, GPU 2 of a shared host (load average
80-130), measured 2026-11-05/06.

## Result

**The cell is bound by GPU physics, not by the host.** Before any change, physics was 63 % of a wave's wall time.
The per-step barrier wait for graph threads was 7 %, so the "host-bound" reading this work started from is wrong.
6.2 GB of GPU memory says nothing here, because the time goes to the number of kernels launched, not to memory.

**Four changes take the mean step from 118.7 ms to 86.4 ms (1.37x), and from 140.6 ms to 87.8 ms (1.60x) while all
15 worlds are live.** None of them changes the physics. The last full cell took 24.2 min against 44.3 min for the
baseline, but that 1.83x overstates the gain: the baseline's longest episode happened to run 23,161 steps and the
fixed cell's 16,547. The fair figures are the wave rates on the longest episode: 8.88 to 11.77 steps/s (1.33x) on one
wave, 7.32 to 10.80 (1.48x) on the other. So the gain is about 1.3-1.5x, around the 1.5x aim and not clearly past it.

**On the shipped code that is 31 min to 23 min on L2 at the same score.** See "Validation on the shipped code".

## The breakdown

From per-step timers on the fleet thread, ms per control step, all steps of both waves. "GPU" is the enqueue plus the
wait for the device. "Think" is the barrier waiting on graph threads. "Render" is the barrier held for on-demand
frames. "Pump" is marshalled calls, mostly `plan_station`'s build. "Host" is snapshot, publish, observations, motion
bookkeeping and flush.

| Run | GPU | Render | Think | Pump | Score | Host | Total | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (`base1`) | 74.6 (63 %) | 12.9 (11 %) | 8.0 (7 %) | 0.6 | 20.6 (17 %) | 2.1 | 118.7 | 8.42 |
| Render fix, IK gate (`fix15x2a`) | 67.7 (65 %) | 6.5 | 7.3 | 0.6 | 20.3 (19 %) | 2.4 | 104.8 | 9.54 |
| + score on close (`fix15x2b`) | 67.4 (74 %) | 10.1 | 7.4 | 0.9 | 2.8 | 2.4 | 91.0 | 10.99 |
| + GPU turn (`turn15x2a`) | 71.6 (83 %) | 3.6 | 6.6 | 0.6 | 2.4 | 1.7 | 86.4 | 11.57 |

The same by live worlds, baseline against all four changes:

| Live worlds | Baseline: GPU / render / think / score, total | All four: GPU / render / think / score, total |
| --- | --- | --- |
| 15 | 75.7 / 18.7 / 9.3 / 33.4, **140.6** | 72.6 / 4.1 / 4.8 / 3.5, **87.8** |
| 5-9 | 81.6 / 10.6 / 10.1 / 15.8, **120.6** | 78.9 / 2.7 / 12.0 / 1.6, **97.0** |
| 1 | 67.0 / 1.0 / 1.0 / 2.5, **73.1** | 59.3 / 0.3 / 1.0 / 0.4, **62.4** |

Other figures:

* SAM 3 costs nothing here. The cell runs palette segmentation.
* cuRobo does not show up as a cost of its own. `plan_station` and `plan_support` builds are the pump column, under 1 %.
* GPU utilisation while busy went from 84 % to 92 % with the three changes of `fix15x2b`.

## The four changes

All four are exact by construction, or are scheduling only. No fidelity change, so they belong upstream rather than
in `fast`.

1. **Score on close** (`run/fleet.py`, `goals/reward.py`, `goals/combinators.py`). A goal-tree pass was about 2 ms of
   Python per world per step: 29 ms of a 78 ms idle step at 15 worlds, 99 ms of 152 ms at 60. It decides nothing unless
   a world is closing. A world skips it when its document does not end on success, its goal and reward hold no history,
   and nothing reads the verdict (no privileged tools, no recorder, no live view). Stateful `observe` entries, such as
   L2's `once`, still advance every step, so row metrics are unchanged. Any condition false scores every step as before.
   Its remaining cost is 3.5 ms a step.
2. **No shape rebuild per render** (`solvers/render/rtx.py`). Newton's `set_visible_worlds` re-populates every shape
   batch in Python, and the RTX viewer does not read those batches after the build phase. It was 631 ms of a 775 ms
   render at 15 worlds, against 83 ms in the tracer. A render alone at 15 worlds is now 143 ms.
3. **IK solve only when a world servos** (`envs/fleet.py`, `solvers/ik/batched.py`). The captured control stage ran the
   batched LM solve every step, even with every world's write gate shut: 7.9 ms a step. Seeds and damping restart on each
   solve, so skipping it changes no target. It runs on about 8 % of steps (only the weave's terminal stage servos).
4. **GPU turn** (`core/gpu_turn.py`, `run/pool.py`, `envs/fleet.py`). Under MPS a render beside another worker's physics
   slows both. Workers now share the GPU for physics and take it alone for a render, using `flock`. In a full cell a
   render went from 700 ms to 240 ms, and a step waits about 3 ms for its turn. Under the lockstep clock the simulation
   does not see the wall clock. A worker that waits past 60 s stops taking turns. `ROBOSIMSTUDIO_GPU_TURN=off` disables it.

| Probe: one worker rendering, one stepping idle physics | Render | Cost to the stepper per render |
| --- | --- | --- |
| Render alone | 143 ms | -- |
| Together | 340 ms | 128 ms |
| Taking turns | 165 ms | 138 ms |

Only one bug surfaced on the way, and it was mine. `fix15x2a` was meant to include score on close, but the pool's
watchdog stamp sets `on_wave_tick`, which my gate read as a live view, so scoring stayed on. The timers showed it. The
fix is `tick_reads_verdicts`, and that run is kept above as the ablation of changes 2 and 3.

## Outcomes

All on the VABAR copy from the start of this work (no servo declaration, 24,000-step budget). "Timed" means the
run's tree carried my scratch timers.

| Run | Changes | Wall (all in) | Success | Reward | seat0-3 / box / end | Mean / max steps | Longest-episode rates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base1`, timed | none | 44.3 min | 16/30 | 0.878 | 29/30/28/27/24/20 | 10,953 / 23,161 | 7.32, 8.88 |
| `fix15x2a`, timed | 2, 3 | 34.7 min | 14/30 | 0.856 | 29/28/26/27/24/20 | 10,566 / 19,604 | 9.04, 9.66 |
| `fix15x2b`, timed | 1, 2, 3 | 30.6 min | 17/30 | 0.883 | 29/28/27/29/23/23 | 9,939 / 19,399 | 9.97, 11.05 |
| `turn15x2a`, timed | all four | 24.2 min | 14/30 | 0.883 | 30/27/29/30/24/19 | 9,631 / 16,547 | 10.80, 11.77 |

* Every run is inside the stated baseline of 13-18/30 and reward 0.86-0.89, except `fix15x2a`'s reward of 0.856, which
  equals the earlier unpatched `L2f_final` (0.856).
* Same success per instance against `base1`: 16-18 of 30 for the patched runs. Earlier unpatched cells give 18-19
  against `base1`, and fleet against fleet was 12-19 in `cable_routing_l2.md`. So this is inside the known noise, but
  the sample is small and could not detect a small shift.
* Step distribution: p10 8,183-8,190 patched against 8,100-8,186 unpatched; p90 12,417-14,682 against 12,696-14,823.
  The median is bimodal, 8,741-10,040 patched against 8,762-10,406 unpatched.
* Frames, before and after change 2, mean absolute difference in grey levels of 255, 4 worlds. Cable scene: 0.24-0.27
  across the patch, against 0.18 between two unpatched runs and 0.5-0.9 between two renders in one run. Depth is the
  same to 0.01 mm and palette IoU matches. A wrong world differs by 2.6-3.1. The 0.24-0.27 is a little above the
  0.18; I did not find out why. `tool_hanging/wrenches_level1`: patched matches unpatched within noise on all four
  cameras.

No watchdog line in any of these four runs.

## Validation on the shipped code

These replace the earlier gaps: L1 and the current VABAR had not been run with the patch, and all four changes had only
one full cell. All are full cells under `--config fast` on the declared servo speed and 14,000-step budget.

| Cell | Code | Wall | Solved | Reward |
| --- | --- | --- | --- | --- |
| `cable_routing/L2` | no patch | 31.3 min, 31.4 min | 20/30, 21/30 | 0.917, 0.922 |
| `cable_routing/L2` | patch | 23.0 min, 22.3 min | 19/30, 14/30 | 0.889, 0.856 |
| `cable_routing/L2` | patch, on colour baking and the wave guard | 22.9 min, 23.3 min | 21/30, 21/30 | 0.917, 0.917 |
| `cable_routing/L1` | patch | 16.4 min | 23/30 | 0.917 |
| `tool_hanging/L1` | patch | 16.9 min, 15 of it one build hang | 29/30 | 0.978 |

- **Speed:** 1.35x on L2, measured side by side: the last baseline and the last two patched cells ran on one host at
  the same time.
- **Outcomes:** 75/120 with the patch against 41/60 without on L2 (Fisher p = 0.51), and 21/30 three times running
  side by side. The 14/30 stayed inside what the runs allow, but a small shift cannot be ruled out at this sample size.
- **Other suites:** changes 2-4 reach every fleet run. Only `cable_routing` and `tool_hanging` have been run.
- **Not run:** the medium and cpu test tiers.

## The barrier and per-world clocks

Yes, the barrier stalls every world while one thinks. It is not where the time goes.

* A holder is present at barrier entry on 24-27 % of steps, mostly for a few milliseconds (median think 6 ms).
* In one baseline wave of 1,666 s, some world was thinking for 220 s (13 %), and exactly one world for 123 s. The
  barrier's own wait was 123 s.
* **Ceiling for ideal per-world clocks: 1.18-1.31x at 15 x 2.** That is a replay of the logs where each world waits only
  for its own holds, with a free and exact freeze, thinking treated as parallel, and renders uncontended (`tools/clock_ceiling.py`,
  run before the GPU turn). The renders would really run beside stepping, which the probe above shows is slow.
* A per-world clock needs a world to stop while the others step. I found no per-world step mask in MuJoCo-Warp or VBD,
  so the world's whole state would have to be restored after each shared step. MuJoCo's contacts sit in one list tagged
  by world, and VBD keeps contact history. I read this code only far enough to judge the cost and did not build or test
  a freeze. Given the ceiling and the risk to the lockstep guarantee, I would not attempt it.

## Checked and not worth changing

* **`eval_every`, `.numpy()` copies, per-step Python loops, logging:** snapshot 0.8 ms, publish and observations 0.9 ms,
  flush 0.45 ms, about 2 % of wall together.
* **GIL:** py-spy (GIL holders only, non-blocking) over a baseline wave's first 1,500 steps: fleet thread 37 s and graph
  threads 3.5 s of a 138 s wave, 30 s of it scoring. That window has little thinking, so it understates graph threads,
  but over whole waves two or more worlds think at once for only about 5 % of wall.
* **MuJoCo solver iterations:** 1-3 per substep, so no world forces long solves on the others.
* **Fusing VBD's 5 per-iteration memsets into one kernel:** 1,280 fewer graph nodes, no gain (39.5 ms against 40.2 ms).
* **A cable-only camera rig:** no gain once change 2 is in (the tracer took 87 ms on one camera, 83 ms on three). Dropped.

## Wave shape

A wave's wall time is its longest episode's steps at the wave's rate, so the longest episode sets the cell. Finished
worlds cost nearly what live ones do (GPU 54-59 ms with one live world, 39.5 ms idle at reset), but the idle GPU cost
hardly depends on width, so re-packing has little to gain:

| Worlds | 2 | 8 | 15 | 30 | 60 |
| --- | --- | --- | --- | --- | --- |
| Idle GPU, ms a step (with the IK solve) | 41.5 | 44.8 | 47.5 | 50.2 | 51.7 |
| Scoring, ms a step | 4.0 | 19.8 | 29.4 | 51.6 | 99.2 |
| Steps/s, as found | 21.8 | 15.4 | 12.9 | 9.8 | 6.6 |

So the note's "9-12 steps/s whatever its width" came largely from scoring, and the "21 steps/s physics alone" still
included the idle IK solve (25 steps/s without it). Real contact costs more than idle: 55 ms a step alone at 15 worlds.

Width, with changes 1-3 and no GPU turn, which only matters with more than one worker:

| Shape | Wall | Success / reward | Longest-episode rates | GPU peak |
| --- | --- | --- | --- | --- |
| 15 x 2 | 30.6 min | 17/30, 0.883 | 9.97, 11.05 | 11.6 GB |
| 30 x 1 | 35.1 min of episodes | 18/30, 0.900 | 9.72 overall | 6.4 GB |
| 8 x 4 | 32.5 min | 22/30, 0.944 | 8.73-12.11 | 21.8 GB |

15 x 2 stays the setting. The 8 x 4 wall is not clean: `[pool] watchdog: w2 made no progress for 901 s while building`,
so that worker was killed and retried. Its GPU was 98 ms a step at 5-9 live worlds, and its 22/30 is one cell, so I would
not read it as a gain. None of these shapes was re-measured with the turn.

## What's left

GPU physics is now 83 % of the wall, at 54-72 ms a step, and needs a physics or Newton change to move:

* **VBD** is about half of kernel activity at 20 iterations; collision about a fifth. Any cut changes physics, so it belongs in `fast` with a full-cell check.
* **Finished worlds:** returning them to a contact-free state could save up to about 14 ms a step in a wave's tail, roughly 7-12 % of a long wave. Estimated, not tried; it needs a per-world reset mid-wave.
* **Thinking**, 7 %: a fifth of holder samples are `skimage` skeletonize in `open-robot-skills`' `curve` tools.
* **Docs to correct** (not edited): "host-bound" in `cable_fleet.md`'s Speed section and in `default.toml`'s comment.
