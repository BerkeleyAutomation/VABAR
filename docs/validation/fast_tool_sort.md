# `fast` config: tool_sort

Measured 2026-11-05 on GPU 6 (one RTX PRO 6000 Blackwell). RoboSimStudio was pinned at `d9b2e44` and left
unmodified. Every run went through `tools/fast_tune/tool_sort.py`, the same harness as the other two suites. The job
lists are `tools/fast_tune/jobs/ts_*.jobs` and `clean_speed.jobs`. Raw output is under
`/home/shadeform/speed_ws/agent_scratch/d4/ts/`.

**Reference** = what `vab_industrial_robotiq@v1` (the Level-1 profile) stages:
- physics `vab_500hz@v1`: 50 Hz × 10, MuJoCo contacts, `enable_multiccd`, nconmax/njmax 1024;
- cameras `tool_sort_agentview@v3`: agentview 640×480, overhead 1280×960, wrist 320×240.

The free tools collide as the `drawn_hull` of their visual meshes.

## Winner

```toml
[suites.tool_sort]
physics = "vab_tool_sort_fast@v1"   # vab_500hz@v1 with 6 substeps (300 Hz); nothing else changed
# cameras: unchanged (tool_sort_agentview@v3)
```

`vab_tool_sort_fast@v1` is registered in `tasks/tool_sort/staging.py`, and no document names it. Run by name
(`ts_winner`, `jobs/winners.jobs`), it builds 6 substeps and sorts `level1_1` at 2,183 steps (the reference: 2,181).
- **Speedup:** 1.59× on a Level-1 bin's idle and 1.40× on its scripted sort, which is host-bound like every routine.
- **Graph protocol:** the cell's `gap_perception_loop` sorted all four goals in 2/2 worlds on both the reference and
  the candidate.
- **Same cell as syringe_packing, different answer:** tool_sort takes 300 Hz where syringe_packing stops at 400 Hz. The
  syringe suite's clutter tray let a syringe roll at 300 Hz; this suite's bins showed no comparable change at that rate.

## Probes

- **Speed**: 600 idle steps on `level1_1` and the Level-1 routine on `level1_1`, from the serial clean sweep.
- **Scripted routine** (`RoutineSortFaces`, 4 goals) on `level1_1..3`. The reference succeeds at 2,181 / 2,053 /
  2,068 steps.
- **Settle drift**: 1,500 idle steps on `level1_1`, `level1_7`, `level1_12` and `level1_20`, measuring each object's move from step 60 to the
  end. The reference already rolls `level1_1`'s measuring tape 35 mm / 21° in 30 s; `level1_7` moves at most 1.2 mm.
- **`gt` graph** on `level1_1` and `level1_2`. It is not a success reference here: the reference itself scores 0.50 on
  both. Candidates were held to "no worse than the reference".

## Candidates

Speeds are steps/s from the serial clean sweep, with the ratio to the reference in parentheses.
- Routines: ✓ = success, at the step counts shown for l1_1 / l1_2 / l1_3.
- Drift: worst object over 30 s, in mm, on l1_1 / l1_7 / l1_12 / l1_20.
- gt: reward and steps on l1_1 / l1_2.

| Candidate | Knobs | idle l1_1 | routine l1_1 | build_s | Routines l1_1 / 2 / 3 | Drift | gt | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ts_ref` | `vab_500hz@v1` | 117.8 | 78.2 | 5.1 | ✓ 2,181 / 2,053 / 2,068 | 34.7 (tape) / 1.2 / 1.8 / 34.8 (tape) | 0.50, 6,018 / 0.50, 6,155 | reference |
| `ts_ss8` | 8 substeps | 143.6 (1.22×) | — | 6.2 | — | 32.9 / 3.0 | — | dominated by `ts_ss6` |
| **`ts_ss6`** | **6 substeps (300 Hz)** | **187.8 (1.59×)** | **109.2 (1.40×)** | 5.7 | **✓ 2,183 / 2,127 / 2,063** | **31.7 / 2.2 / 1.0 / 31.6** | **0.75, 5,585 / 0.50, 6,841** | **winner** |
| `ts_ss4` | 4 substeps (200 Hz) | 255.5 (2.17×) | 125.3 (1.60×) | 10.2 | ✓ 2,182 / 2,124 / 2,076 | 29.1 / 2.1 / 1.5 / 22.8 | 0.75, 5,674 / 0.50, 6,303 | passed the cheap probes; held back, see Risks |
| `ts_it10` | iterations/ls 10/10 | 113.3 (0.96×) | — | 8.0 | — | 34.7 | — | no gain |
| `ts_noccd` | multiccd off | 107.8 (0.92×) | ✗ 0.75 at 2,348 | 13.9 | — | **62.6 (a multimeter turns 171°) / 99.9** | — | rejected |
| `ts_buf` | nconmax/njmax 384/512 | 116.6 (0.99×) | — | 12.6 | — | — | — | no gain (peaks: 33 contacts, nefc 106) |
| `ts_hull32` | convex hulls capped at 32 vertices (Newton's default is 64) | 152.7 (1.30×) | — | 5.7 | ✓ 2,180 / 2,127 / 2,064 | **0.006** / — / 0.1 | — | passes, but not configurable: see below |
| `ts_pyr` | cone pyramidal | — | — | — | — | — | — | not run: it drops the syringe on this same arrangement (`fast_syringe_packing.md`) |
| `ts_cam_half` | rig halved | — | — | — | — | — | — | not run: the 1280×960 overhead is what resolves the Level-1 small parts, and halving the syringe rig broke that suite's perception |

The `level1_12` routine fails identically for the reference and every substep candidate (reward 0.25 at about 427
steps). It is the routine's limit, not the physics'.

**The hull cap is a real gain that the config cannot express.** Capping every convex hull at 32 vertices made the
idle 1.30× faster and kept all three routines. It also changes the settled scene: `level1_1`'s measuring tape stops
rolling, so drift drops from 35 mm to 0.006 mm. The scene is steadier, but it is no longer the reference's scene.
- It is a property of the collider meshes, not of the arrangement or the rig. The probe set
  `newton.Mesh.MAX_HULL_VERTICES` process-wide, so the robot's meshes were capped too.
- Shipping it needs a per-object cap (`maxhullvert` on the `drawn_hull` meshes) and a config knob that selects it.
  It is left out of the winner for that reason, not for fidelity.
- Combined with `ts_ss6` it was not measured.

## Real fidelity: the cell's graph protocol

Run on `level1_6 --episodes 0:2 --num-worlds 2` with `gap_perception_loop --no-privileged --perception vision
--renderer rtx --max-steps 20000`. The two runs were started together, with `--bus-timeout 600` on both: the machine's
CPUs were saturated by other work, and the first render of a freshly started process took 4-6 minutes. The document's
sampler is fixed, so the two worlds share one layout.

| Run | Physics | Success | Steps (ep 0 / ep 1) | Wall |
| --- | --- | --- | --- | --- |
| `ts_ref` | `vab_500hz@v1` | **2/2** | 13,404 / 16,171 | 15.2 min |
| `ts_fast` | `vab_tool_sort_fast@v1` | **2/2** | 14,980 / 16,801 (+12% / +4%) | 20.2 min, of which about 5 min was the first render |

Both sort all four goals in both worlds. The subgraph traffic is nearly the same:
- 21 / 20 `pair_sg`;
- 21 / 20 `grasp_sg`;
- 15 / 14 `recover_attempt_sg`;
- the same cuRobo "no reachable point inside the destination OBB" retries in both.

Episode 0's +12% is just outside the ±10% the cheap probes used. It is one episode of a VLM-driven closed loop, and
it is in the direction of more retries rather than a different outcome. The wall times are not comparable here: the
candidate waited about 5 minutes longer for its first render. Physics is a few percent of this graph's wall time
either way: about 15k steps against 15 minutes.

## Risks

- **`ts_ss4` passed every cheap probe here and was still not taken.** 200 Hz is 2.17× on idle and 1.60× on the
  routine. But `vab_500hz@v1`'s own docstring measured 200 Hz bins of ten thin tools (`sort_all`, `sort_bench`,
  `sort_power`) that never come to rest. The Level-1 bins hold four objects, and at 200 Hz they moved no more in 30 s than at 500 Hz. The
  syringe suite's clutter tray did not: a syringe rolled 66°. It is the next step if a full `fast` cell at 300 Hz
  matches the default, and it should be decided on that cell, not on these probes.
- **The measuring tape is a roller at every rate.** Its 23-35 mm in 30 s dominates the drift column for `level1_1`
  and `level1_20`, and 300 Hz did not change it. The other objects moved 1-3 mm at every rate.
- The `gt` graph is not a success reference on Level 1: the reference scores 0.50. "No worse than the reference" is a
  weak gate. The scripted routine, which succeeds, carried the decision.
- The contact timeconst is clamped from 4 ms to 6.7 ms at 300 Hz (about 40% softer contact). Neither the routines
  nor the drift moved because of it.
