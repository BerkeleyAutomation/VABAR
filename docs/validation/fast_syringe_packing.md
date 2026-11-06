# `fast` config: syringe_packing

**Superseded.** This was measured on the scene before `docs/validation/syringe_working_scene.md`, and `fast` no longer
uses `vab_syringe_fast@v1`. It is kept as the record of what was tried.

Measured 2026-11-05 on GPU 6 (one RTX PRO 6000 Blackwell). RoboSimStudio was pinned at `d9b2e44` and left
unmodified. Every run went through `tools/fast_tune/syringe_packing.py`, which works the same way as the tool_hanging
harness (see `docs/validation/fast_tool_hanging.md`): an in-process override of the lowered `TaskSpec`, plus a
`fast_tune.json` per run that records the effective arrangement and rig and the constructed `SolverMuJoCo` kwargs. The
job lists are `tools/fast_tune/jobs/sp_*.jobs` and `clean_speed.jobs`. Raw output is under
`/home/shadeform/speed_ws/agent_scratch/d4/sp/`.

**Reference** = what `vab_industrial@v2` stages:
- physics `vab_500hz@v1`: 50 Hz × 10, MuJoCo-Warp with its own contacts, `enable_multiccd`, nconmax/njmax 1024,
  contact ke/kd 6e4/500;
- cameras `agentview@v2`: 3 × 1280×960.

`vab_500hz@v1` is also tool_sort's arrangement, so the harness applies every physics candidate to tool_sort's variants
as well.

## Winner

```toml
[suites.syringe_packing]
physics = "vab_syringe_fast@v1"   # vab_500hz@v1 with 8 substeps (400 Hz); nothing else changed
# cameras: unchanged (agentview@v2)
```

`vab_syringe_fast@v1` is registered in `tasks/syringe_packing/staging.py`, and no document names it. Run by name
(`sp_winner`, `jobs/winners.jobs`), it builds 8 substeps with multiccd on and posts the syringe at the reference's
1,958 steps. The renderer stays RTX.
- **Speedup:** 1.20× on the Level-1 tray idle and 1.25× on the Level-2 clutter tray idle.
- **Ceiling:** physics is a small share of a graph episode's wall time (see the tool_hanging numbers), so expect much
  less than that end to end.
- **No faster setting survived:**
  - fewer iterations, multiccd off and smaller buffers bought nothing, and the pyramidal cone dropped the syringe;
  - 6 and 4 substeps fail the settle-drift probe;
  - the half-resolution rig loses the syringe in the perception graph.


**Full-cell check (syringe_packing/L1, 30 instances, lockstep, one GPU).** The physics stays in `fast`. Outcomes
cannot be told apart from `default`, and it saves 3-6% of wall time:

| Physics | Workers | Wall | Success | Rewards |
| --- | --- | --- | --- | --- |
| `default` (`vab_500hz@v1`) | 2 | 1,832 s | 2/30 | 0.5 × 26, 0.0 × 2 |
| `fast` (`vab_syringe_fast@v1`) | 2 | 1,772 s | 1/30 | 0.5 × 27, 0.0 × 2 |
| `default` | 4, MPS | 1,168 s | 0/30 | mean 0.467 |
| `fast` | 4, MPS | 1,091 s | 2/30 | mean 0.500 |
| `fast` (final validation) | 4, MPS | 1,104 s | 2/30 | 0.5 × 27, 0.0 × 1 |

The cell's time goes to perception, VLM calls and retries, so a 1.2× faster physics step moves it little. The
gain grows slightly when 4 workers share the GPU.

## How fidelity was probed here

The suite's own `gt` graph does not run the tray levels: `docs/validation/syringe_packing.md` records reward 0.000 on
every tray variant, because the tray loader withholds the OBB that `gt` reads. So the cheap probes were:

1. **`tool_sort/syringe` scripted routine**: one syringe, picked, stood up and posted through the same sharps box, in
   the same cell, on the same arrangement. The reference succeeds at 1,958 steps and ends on `success`, so the step
   count is a sensitive check. `tool_sort/hammer` (786) adds a second manipulation.
2. **Settle drift**: 1,500 idle steps (30 s) on the Level-1 tray (`tray_pair_both_t00`) and two Level-2 clutter trays
   (`tray_clutter_t00`, `t03`). The measure is how far each object moves from step 60 to the end. This is the failure
   the arrangement's own docstring was written against: piles that never come to rest at 200 Hz.

Then the cell's graph protocol (`gap_perception`) on one Level-1 document, 2 worlds per run, reference and candidate
close together in time (see below).

## Candidates

Speeds are steps/s over 600 idle steps with no renderer, from the serial clean sweep, with the ratio to the reference
in parentheses. L1 = `tray_pair_both_t00`, L2 = `tray_clutter_t00`. Drift is the worst object over 30 s on
`tray_clutter_t00`, from step 60. On `tray_pair_both_t00` and `tray_clutter_t03` every candidate measured 0.0 mm,
the reference included.

| Candidate | Knobs | L1 idle | L2 idle | build_s | syringe routine | hammer routine | L2 drift | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sp_ref` | `vab_500hz@v1` | 215.4 | 175.5 | 4.4 | ✓ 1,958 | ✓ 786 | 0.91 mm, 0.65° | reference |
| **`sp_ss8`** | **8 substeps (400 Hz)** | **257.8 (1.20×)** | **219.4 (1.25×)** | 5.3 | **✓ 1,958** | **✓ 786** | **0.00 mm, 0.05°** | **winner** |
| `sp_ss6` | 6 substeps (300 Hz) | 337.4 (1.57×) | 274.8 (1.57×) | 3.8 | ✓ 1,958 | ✓ 786 | **2.73 mm, 12.0°** | rejected: 3× drift; a syringe rolls 12° at rest |
| `sp_ss4` | 4 substeps (200 Hz) | 460.7 (2.14×) | 337.2 (1.92×) | 6.2 | ✓ 1,958 | — | **7.71 mm, 66.3°** | rejected: drift |
| `sp_it10` | iterations/ls 10/10 | 213.9 (0.99×) | 157.5 (0.90×) | 6.5 | ✓ 1,958 | — | 0.91 mm | no gain |
| `sp_it5` | iterations/ls 5/5 | — | 177.5 (1.01×) | 5.4 | ✓ 1,958 | — | — | no gain |
| `sp_pyr` | cone pyramidal | — | 179.5 (1.02×) | 6.4 | **✗ 2,251, reward 0** | ✓ 785 | 0.00 mm | rejected: drops the syringe |
| `sp_noccd` | multiccd off | 178.6 (0.83×) | 180.5 (1.03×) | 8.8 | ✓ 1,958 | ✓ 787 | 0.91 mm | no gain |
| `sp_ss8_noccd` | 8 substeps + multiccd off | 255.0 (1.18×) | 218.0 (1.24×) | 4.4 | ✓ 1,958 | ✓ 787 | 0.00 mm | no gain over `sp_ss8` |
| `sp_buf` | nconmax/njmax 384/512 | — | 178.8 (1.02×) | 5.1 | — | — | — | no gain |
| `sp_fast` = `sp_ss8` + rig 3 × 640×480 | half-resolution `agentview@v2` | (no renderer) | | | | | | **rejected in the graph**: both worlds abort at `target_sg.perceive`, "could not perceive 'syringe'" |

Measured peaks (`--peaks`): 33 contacts and nefc 95 on the L2 tray at rest, 22 / 59 through the scripted posting.
The 1024 ceilings are 10-30× those peaks, and trimming them changes nothing.

The scripted routine's own rate barely moves (112.3 → 111.1 steps/s for `sp_ss8`). Its per-step host work (the OSC
law and the evaluator) bounds it, not physics. The idle numbers are the physics cost.

## Real fidelity: the cell's graph protocol

`tray_pair_both_t05 --episodes 0:2 --num-worlds 2`, `gap_perception --motion curobo --no-privileged --perception
vision --renderer rtx`. The document's sampler is fixed, so the two worlds share one layout.

| Run | Physics / rig | Success | Reward | Steps (ep 0 / ep 1) | Wall | What happened |
| --- | --- | --- | --- | --- | --- | --- |
| `sp_ref` | reference | 0/2 | 0.50, 0.50 | 2,887 / 6,549 | 11.6 min | one syringe posted per world. Then `count_remaining` saw none left (ep 0), or `verify_sg` saw the second still outside the aperture (ep 1) |
| `sp_ss8` | **`vab_syringe_fast@v1`** | 0/2 | 0.50, 0.50 | 2,890 / 15,050 | 12.0 min | the same: one posted per world, ep 0 ended 3 steps from the reference's, ep 1 retried longer |
| `sp_fast` | 400 Hz + rig 640×480 | 0/2 | — | aborted | — | both worlds: `target_sg.perceive` "could not perceive 'syringe'" |
| `sp_cam_half` | reference + rig 640×480 | 0/2 | — | aborted | — | the same abort on reference physics, so the rig alone does it |

The graph posts one of the two syringes in every world, with or without the candidate. That matches this suite's own
validation run on `tray_pair_both_t00` (reward 0.500), so the comparison is outcome against outcome, not success
against success. The two arms did not overlap exactly in time: `sp_ss8` ran with `--bus-timeout 180`, `sp_ref` with
600 after two reference attempts lost their first render on the starved machine. The wall times are therefore
indicative only. Physics is well under a tenth of an episode here: 4,700-9,000 steps at about 200 steps/s, against 12 minutes.

## Risks

- **The cheap manipulation probe is borrowed.** `tool_sort/syringe` stands in for the tray levels because the suite
  has no reference policy that completes them. It shares the cell, the Panda, the sharps box and the arrangement, but
  not the tray or the grasp-from-clutter.
- **The settle-drift probe is coarse.** It covers one 30 s idle on three layouts, and two of those layouts do not move
  at all at any rate. 400 Hz was as still as 500 Hz on the one layout that does move.
- **The contact stiffness is now clamped.** `vab_500hz@v1` chose ke/kd 6e4/500 as the stiffest contact a 2 ms step
  admits. At 400 Hz MuJoCo clamps its 4 ms timeconst to 5 ms, so contacts are about 25% softer. Nothing measured here
  moved because of it.
- **The camera result is a perception-threshold fact.** `perceive_target.py` scores SAM 3 detections and red-cap
  evidence fitted at 1280×960. A rig that renders fewer pixels would need its thresholds refitted, which is graph work.
- `vab_syringe_fast@v1` is deliberately not shared with tool_sort. The two suites stand in the same cell, but tool_sort
  takes 300 Hz (`vab_tool_sort_fast@v1`) and this suite does not.
