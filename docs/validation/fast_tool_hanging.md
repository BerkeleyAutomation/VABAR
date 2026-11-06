# `fast` config: tool_hanging

Measured 2026-11-05 on GPU 6 (one RTX PRO 6000 Blackwell). RoboSimStudio was pinned at `d9b2e44` in the shared
venv; nothing in it was changed. Every run went through `tools/fast_tune/tool_hanging.py`. That script re-registers the
suite in-process with one candidate applied (`SUITES.replace` + `dataclasses.replace` on the lowered `TaskSpec`), then
calls `python -m vabar run`. Variant names are left alone, because `layouts.sample_level2_layout` keys on `task.name`.
Each run directory gets a `fast_tune.json`, which holds the effective arrangement and rig and the `SolverMuJoCo` kwargs
actually constructed. Every row below was checked against it, so the override took effect. The job lists are
`tools/fast_tune/jobs/th_*.jobs`. Raw output is under `/home/shadeform/speed_ws/agent_scratch/d4/th/`.

**Reference** = what the L1/L2 documents stage:
- physics `tool_hanging_480hz@v1`: 60 Hz × 8, MuJoCo-Warp integrating, Newton contacts, nconmax/njmax 4096/8192,
  rigid_contact_max 8192;
- cameras `yam_station_tool_hanging@v1`: 4 × 1280×720 RGB-D.

## Winner

```toml
[suites.tool_hanging]
physics = { "tool_hanging_480hz@v1" = "tool_hanging_fast@v1" }  # + collide_every = 2
```

**The 640x360 rig was dropped after the full-cell check.** The sweep below picked
`yam_station_tool_hanging_fast@v1` (4 x 640x360) on 7 Level-1 episodes. On the whole
`tool_hanging/L1` cell (30 instances, 4 fleet waves, lockstep, one GPU) it failed:

| Run (2026-11-05) | wrenches | scissors | pliers | screwdrivers | total |
| --- | --- | --- | --- | --- | --- |
| `default` | 8/8 | 8/8 | 7/7 | 7/7 | **30/30** |
| `fast`, physics + 640x360 rig | 8/8 | 7/8 | 1/7 | 4/7 | 20/30 |
| the same, repeated | 7/8 | 8/8 | 1/7 | 4/7 | 20/30 |
| physics only (`collide_every 2`) | 8/8 | 8/8 | 7/7 | 7/7 | **30/30** |
| 640x360 rig only | 7/8 | 8/8 | 2/7 | 5/7 | 22/30 |

The failing pliers episodes run every node of the graph, insertion included. They end at
about 1,000 steps with reward 0.33: the plier is not contained by the left hook, and there
is no support contact. In the two episodes compared (ep 3 failing, ep 11 passing, same
run), `final_realign_sg.register` puts the plier's loop feature 16.4 mm along the TCP's z
in ep 3, against 11.5 mm in ep 11. So `final_mate_sg` aims the insertion about 5 mm short
(mate x 0.8538 against 0.8588), and the plier is released off the hook. Registration at
640x360 does not have the precision the pliers' loop needs. `fast` keeps the reference rig.
The rig stays registered so the measurement can be repeated.

Both names are registered in `tasks/tool_hanging/staging.py`. No document names them. The renderer stays RTX. Run by
name (`tools/fast_tune/tool_hanging.py th_fast`), the registered pair builds 8 substeps with collide_every 2 and hangs
the plier at the reference's 1,090 steps (`jobs/winners.jobs`), which makes 6/6 pliers runs on this physics.

Read both entries as *in place of* what was measured:
- `physics` replaces `tool_hanging_480hz@v1`, which every variant stages except the hammers, whose
  `tool_hanging_plain_960hz@v1` was not measured.
- `cameras` replaces `yam_station_tool_hanging@v1`, which only the eight Level-1/2 variants stage. The full variants use
  `tool_hanging_views@v1`, a different rig.

A runtime that swapped these names in suite-wide would change scenes that nothing here measured.

| | Reference | `fast` | |
| --- | --- | --- | --- |
| `wrenches_level2` idle, no renderer | 55.6 steps/s | 82.6 steps/s | **1.49×** |
| `wrenches_level1` / `pliers_level1` routine | 138.2 / 135.5 steps/s | 164.4 / 168.9 steps/s | 1.19× / 1.25× |
| L1 graph protocol (4-world fleet, RTX), wall | 5.5 / 4.9 min | 4.5 / 4.1 min | **−17%** (two rounds, run side by side) |
| L1 graph protocol, success | 7/7 (+1 infra abort) | 6/7 (+1 infra abort) | within noise |
| L2 graph protocol (2-world fleet), wall | 10.7 min | 10.5 min | −2% |
| L2 graph protocol, success | 0/2 (reward 0.50, 0.50) | 1/2 (reward 0.50, 1.00) | no worse |

Physics is a small part of a graph episode's wall time.
- **Level 1:** `collide_every 2` alone made the graph no faster (+2%). The half-resolution rig made it 16-18% faster.
- **Level 2:** even there, where a physics step costs 4× a Level-1 step, the end-to-end difference was 2%. The graph's
  perception, VLM and planning calls dominate.
- `collide_every 2` is in the winner because it is free on every probe and pays wherever the simulation itself is the
  bottleneck: scripted routines, idle settling, and any future runtime that makes the graph cheaper.

## Candidates

Speed = 600 idle steps on `wrenches_level2` (the heaviest scene: 1,546 contacts and nefc 4,313 at rest) and the
`wrenches_level1` routine. No renderer. Rates come from the serial clean sweep (`clean_speed.jobs`), run with
nothing else of ours on the GPU. Fidelity probes:
- the three Level-1 routines (wrench, scissors, pliers);
- the privileged `gt` graph on `scissors_level1` and `pliers_level1`.

The gate is success with steps within ±10% of the reference. A candidate stopped at its first failure. The routines
end on a fixed script, so their step counts never move. The pliers routine is the probe that separates candidates, so
it was repeated.

Speeds are steps/s, with the ratio to the reference in parentheses. build_s is the scene build on that run. "w1 / s1 /
p1" are the wrenches / scissors / pliers Level-1 routines (✓ = success at the reference's 1,165 / 915 / 1,090 steps).
"gt s1 / p1" give success and steps against the reference's 2,121 / 2,032.

| Candidate | Knobs | L2 idle | w1 routine | build_s | w1 | s1 | p1 (repeats) | gt s1 | gt p1 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `th_ref` | `tool_hanging_480hz@v1` | 55.6 | 138.2 | 5.7 | ✓ | ✓ | ✓ 5/5 | ✓ 2,121 | ✓ 2,032 | reference |
| **`th_ce2`** | **collide_every 2** | **82.6 (1.49×)** | **164.4 (1.19×)** | 5.3 | ✓ ✓ | ✓ ✓ | **✓ 5/5** | ✓ 2,110 | ✓ 2,037 | **winner** |
| `th_ce4` | collide_every 4 | 114.3 (2.06×) | — | 5.3 | ✓ | ✓ | ✗ 1/2 | ✓ 2,132 | ✓ 2,050 | rejected: pliers |
| `th_ss4` | 4 substeps (240 Hz) | 105.5 (1.90×) | 194.3 (1.35×)¹ | 7.7 | ✓ | ✓ | ✗ 1/3 | ✓ 2,084 | — | rejected: pliers |
| `th_ss4_ce2` | 4 substeps + collide_every 2 | 114.4² | — | 9.7 | — | ✓ | ✗ 1/2 | ✓ 2,073 | — | rejected: pliers |
| `th_pyr` | cone pyramidal | 72.0 (1.29×) | 152.3¹ | 6.4 | ✓ | ✗ | ✗ | ✗ 1,323, 0.33 | — | rejected |
| `th_ss4_ce2_pyr` | all three | — | — | — | ✓ | — | — | — | — | dropped (pyramidal and 4 substeps both failed) |
| `th_it10` | iterations/ls 10/10 | 56.8 (1.02×) | 142.8¹ | 7.3 | ✓ | — | — | — | — | no gain |
| `th_it5` | iterations/ls 5/5 | 60.1¹ (1.04×) | 143.8¹ | 8.2 | ✓ | — | — | — | — | no gain |
| `th_buf2` | nconmax/njmax 2048/6656 | 55.9 (1.00×) | — | 7.1 | — | — | — | — | — | no gain |
| `th_sap` | broad phase SAP | 54.7 (0.98×) | 78.4² | 7.4 | ✓ | — | — | — | — | no gain |
| `th_nxn` | broad phase N×N | 25.8² | — | 8.4 | — | — | — | — | — | slower |
| `th_sup_coacd` | support on its 31 CoACD hulls | p2: 75.0 vs 76.5¹ | p1: 149.9² | 6.7 | — | — | ✓ | — | — | no gain |
| `th_sup_dec` | support decimated to 2k tris | — | p1: 134.2 vs 133.9² | 3.7 | — | — | ✓ | — | — | no gain |
| `th_cam_half` | rig 4 × 640×360 | (no renderer) | | | | | | | | only the graph can tell; see below |

¹ From the first serial sweep, before any other run of ours shared the GPU. The other rates are from the clean
sweep.
² Measured while other runs shared the GPU: screening only.

Where the reference spends its time: `wrenches_level2` at rest carries **1,546 contacts and nefc 4,313** (measured with
`--peaks`). Every L2 tool lies on the table as 24-39 CoACD hulls, and Newton's narrow phase over that pair list is most
of each substep. That is why the collision cadence pays and the solver knobs do not. For comparison, the
`wrenches_level1` idle is 231 → 246 steps/s (1.06×) under `th_ce2`; the routines gain 1.19-1.25×.

Notes on the rejections:
- **4 substeps (240 Hz)** is 1.9× on the L2 idle, but it fails the pliers routine in 2 of 3 runs. The plier never
  lands on its support: reward 0.33, `containment` false, contact force 0.
- **collide_every 4** fails the pliers routine 1 run in 2.
- **Pyramidal cone** fails the scissors routine, the pliers routine and the scissors `gt` graph (1,323 steps, reward
  0.33).
- **Fewer MuJoCo iterations** (10/10, 5/5) bought nothing. Collision dominates this suite, not the solve.
- **SAP and N×N broad phase** never beat the explicit pair list. SAP measured 0.98× in the clean sweep. N×N measured 0.45×, but only in contended screening.
- **Buffers trimmed** to the measured peaks ×1.5 (nconmax 2048, njmax 6656; the L2 idle peak is nefc 4,313): no
  measurable gain, and less headroom.
- **Plier-support collider**:
  - the CoACD set already beside the mesh (31 hulls) and a 2k-triangle quadric decimation (0.10 mm worst deviation)
    both keep the pliers routine (1,090 steps, contact force 0.440 vs 0.442);
  - neither speeds anything up: the CoACD set idles `pliers_level2` at 75.0 vs 76.5 steps/s, and the decimation runs
    the pliers routine at 134.2 vs 133.9;
  - the 13.3k-triangle support is not where the time goes, so no collider asset was registered.

## Real fidelity: the cell's graph protocol

`gap_perception`, `--motion curobo --no-privileged --perception vision --renderer rtx --bus-timeout 180`, run with the
fleet trick. The reference and the candidates ran **concurrently**, so they shared the machine's load. That makes
their wall times comparable with each other, though not with the lead's 4.0 min solo reference.

**Level 1**: `wrenches_level1 --episodes 1,5,9,13 --num-worlds 4 --max-steps 8500`. The "wall" column is the run's
minutes of episodes. "Steps" lists the successful episodes. Infra = the torch CUDA-graph race described at the end
(`register_sg`, "Offset increment outside graph capture").

| Round | Run | Success | Steps (successes) | Wall | Failures |
| --- | --- | --- | --- | --- | --- |
| 1 (2 fleets) | `th_ref` | 4/4 | 1,822 / 1,824 / 1,829 / 1,948 | 5.5 min | — |
| 1 | `th_ce2_half` | 3/4 | 874 / 1,828 / 2,312 | **4.5 min** | 1 infra (ep 1, step 555) |
| 2 (3 fleets) | `th_ref` | 3/4 | 1,821 / 1,822 / 2,433 | 4.9 min | 1 infra (ep 5, step 557) |
| 2 | `th_ce2` | 3/4 | 1,821 / 1,825 / 2,311 | 5.0 min | 1 placement miss (ep 1: wrench 168 mm from the hook) |
| 2 | `th_ce2_half` | 3/4 | 1,813 / 1,824 / 1,949 | **4.1 min** | 1 placement miss (ep 9: wrench 459 mm away) |

Pooled, leaving out the two infra aborts:

| | Success | Wall vs the reference in the same round |
| --- | --- | --- |
| reference | 7/7 | — |
| `th_ce2_half` (the winner) | 6/7 | −18% (round 1), −16% (round 2) |
| `th_ce2` (physics only) | 3/4 | +2% |

Steps per successful episode are unchanged: the median is 1,824-1,828 in every arm. These are small samples, and the
Wilson intervals overlap almost entirely. The lead's solo reference was 4/4.

**Level 2**: `wrenches_level2 --episodes 1,5 --num-worlds 2 --max-steps 22000`, side by side, `--bus-timeout 600`
on both (see the environment notes).

| Run | Success | Reward (ep 1 / ep 5) | Steps (ep 1 / ep 5) | Wall | Failures |
| --- | --- | --- | --- | --- | --- |
| `th_ref` | 0/2 | 0.50 / 0.50 | 4,106 / 3,611 | 10.7 min | one of the two wrenches hung in each episode |
| `th_ce2_half` | 1/2 | 0.50 / **1.00** | 4,101 / 3,689 | 10.5 min | ep 1 hung one of two |

Two of the four episodes logged `grasp_sg_2.rotate_align: pre-grasp tracking error too large` (23-25 mm), a graph
behaviour shared by both arms. The candidate is no worse, and the wall time is the same: at Level 2 the graph's own
time dominates as well.

**Renderer: GL instead of RTX.** This was one world, because fleets have no GL: `wrenches_level1` episode 1 with
`--renderer gl`, reference physics. It succeeded, but took **6,329 steps and 3.6 min for the one episode**. A single
world keeps stepping while the graph thinks, which inflates the step count to 3.4× the fleet's; that is also 74% of the
8,500-step budget. The RTX fleet runs four episodes in 4.1-5.5 min, about 1.0-1.4 min each. RTX with the fleet stays.

## Risks

- **The graph's wall time is not the physics.**
  - At Level 1, `collide_every 2` alone moved the fleet's wall time by nothing measurable (+2%), though it is 1.5× on an
    L2 idle and 1.2× on a routine.
  - What moved the wall time is the camera rig: −16 to −18% from rendering a quarter of the pixels on demand.
  - Perception, VLM calls and cuRobo planning make up the rest. A 30-instance cell under 30 min needs wider fleets and
    a faster graph, not a faster arrangement.
- **The pliers routine is close to its margin.** 4 substeps and collide_every 4 each pass it sometimes and fail it
  sometimes. collide_every 2 passed it 6/6, counting the registered `th_fast` run, and the reference passed 5/5. A plier settling onto the 13.3k-triangle support
  is the contact this suite is most sensitive to; watch `pliers_level1/2` in the first full `fast` cell.
- **Two placement misses in the candidate arms against none in the reference** (6/7 + 3/4 vs 7/7). Both missed by tens
  of centimetres, which looks more like a graph decision than a contact detail, and the samples cannot separate them. A
  full 30-instance L1 cell, `fast` against the default, is the check this budget could not afford.
- **Half-resolution cameras are a perception change.** tool_hanging's `gap_perception` held up at 640×360. The syringe
  suite's graph did not hold up at 640×480 (`fast_syringe_packing.md`). The `fast` rig is for this suite only.
- `collide_every` keeps 480 Hz integration, so contacts are up to one 2 ms substep stale. A fast-moving fingertip moves
  about 1 mm in that time against `rigid_gap` 0.5 mm. That is the price, and the pliers repeats are its measurement.

## Environment problems found on the way (affect every perception run, not just this suite)

None of these changes a measured number, but each one cost a graph run before it was found. All four apply to any
VABAR checkout run as a git worktree, or in today's shared venv:

1. **From a git worktree no tool bundles are found.** `vabar.skill_registries()` looks for `../open-robot-skills`
   beside the checkout. For `.claude/worktrees/<name>/` that path does not exist. The graph then boots without the
   geometry, SAM 3 and grounding bundles and aborts at its first perception node: ``Tool
   'geometry.mask_to_world_points' not found``. Fix: set `VABAR_OPEN_ROBOT_SKILLS=/home/shadeform/speed_ws/open-robot-skills`
   (`sweep.py --open-robot-skills`).
2. **The shared venv has no torchvision.** torch 2.10.0+cu128 was reinstalled at 01:41 today, and torchvision went with
   it. The in-process SAM 3 path then reports "SAM3 weights are unavailable ... provide HF_TOKEN", which is misleading:
   the real error is `Sam3ImageProcessor requires the Torchvision library`. The runs here put torchvision 0.25.0+cu128
   (copied from the uv cache) on `sys.path` of the runner process only (`sweep.py --pythonpath`, read through
   `FAST_TUNE_SYSPATH`).
   - Do not use `PYTHONPATH` for this. The tool-bundle subprocesses inherit it, and their own torch then fails with
     `operator torchvision::nms does not exist`.
   - With item 1 fixed, open-robot-skills' `sam3` bundle server (its own venv) also runs. Whether the overlay is still
     needed then was not isolated.
   - The shared venv was not modified.
3. **`HF_TOKEN` is not exported** in this environment. The token is in `~/.cache/huggingface/token`
   (`sweep.py --hf-token-file`).
4. **Two runtime hazards that look like policy failures:**
   - a fleet run sometimes loses one episode to `Offset increment outside graph capture encountered unexpectedly` in
     `register_sg`: torch's RNG racing a CUDA-graph capture across the fleet's worker threads. This hit the
     reference and the candidate once each;
   - the on-demand renderer can miss the 180 s bus timeout (`render was not answered within 180 s`). It happened with
     seven graph processes on the GPU. Later it happened with four, at the first observation of every process that
     had just started, while the GPU sat at 0% and the machine's load average was 160-200 from other work on its
     CPUs: the renderer's host side was starved, not the GPU. Those runs were discarded. The Level-2 and syringe
     reference re-runs use `--bus-timeout 600` on both arms.
