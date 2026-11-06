# `fast` config: camera resolution

Can a lower camera resolution speed up the GaP-graph episodes without breaking perception?
This was measured per suite with the cell's own graph protocol, scaled rig against reference
rig.

Measured 2026-11-05 on GPU 6 (one RTX PRO 6000 Blackwell, 96 GB), from a checkout of `main`
(`18b974e`). RoboSimStudio was pinned at `d9b2e44` and left unmodified. The 240-core host was
shared with other sweeps, and for most of the window this GPU ran 6-9 of these graph processes
at once (GPU at 97-98 %).

## Result

| Suite | Reference rig | Verdict | `fast` rig |
| --- | --- | --- | --- |
| syringe_packing | `agentview@v2`, 3 × 1280×960 | **0.625 passes** (and 0.75). Same reward as the reference on t00 and t05. On t01 it beats the reference, which cannot perceive the syringe there. | **`agentview_fast@v1`**: 3 × 800×600 |
| tool_sort | `tool_sort_agentview@v3`: 640×480 + **1280×960 overhead** + 320×240 | **Halving the overhead fails.** Grasp verification loops; 0/2 after 2× the reference's time. A 0.75 overhead sorts 2/2, but takes +24 % steps and +12 % wall time (more re-grasps). | none: keep `@v3` |
| tool_hanging L2 | `yam_station_tool_hanging@v1`, 4 × 1280×720 | **Half resolution holds** (`yam_station_tool_hanging_fast@v1`, already on the integration branch). Over 4 episodes under `gap_perception_level2` there was no perception failure, and 3 of 4 episodes had the same outcome. The fourth (ep 5) is one placement miss, the miss class already seen on L1. Wall time −5 % and −23 % in two concurrent pairs. | `yam_station_tool_hanging_fast@v1` (4 × 640×360), unchanged |
| usbc_insert | `yam_station_usbc@v1`, 3 × 1280×720 | **0.75 and 0.5 both insert** on L2 10085 and 10032, same depth and lateral error. At 0.5, one instance's plug-axis estimate shifted (8.6° vs 3.9°). **No wall-time gain measurable** (single-world, physics-bound). | `yam_station_usbc_fast@v1`: 3 × 960×540 (optional) |
| cable_route | `yam_station_cable@v1`, 3 × 1280×720 | **0.75 passes** on L1 10009 and 10062: both stations woven, TrackDLO and `restore` spans within a few mm, and on 10062 the reference's outcome seat for seat. **0.5 is shakier**: on 10062 it left station 0 unseated after its check read it as seated. The terminal is undetected at every scale (pixel gate). **No wall-time gain measurable** (physics-bound). | `yam_station_cable_fast@v1`: 3 × 960×540 (optional) |

## Full-cell check: no camera change ships in `fast`

The verdicts above come from 2-world probes. Both candidates that reached `fast` were then run
on their full L1 cell (30 instances, lockstep, one GPU), and both were dropped:

| Cell | Rig | Outcome | Wall | Verdict |
| --- | --- | --- | --- | --- |
| tool_hanging/L1 | reference, 4 × 1280×720 | 30/30 (`default`) | 431 s | -- |
| tool_hanging/L1 | `yam_station_tool_hanging_fast@v1`, 4 × 640×360 | 20/30 twice; pliers 1/7 | 379-487 s | **dropped** |
| tool_hanging/L1 | the same rig, reference physics | 22/30 | 547 s | the rig alone does it |
| syringe_packing/L1 | reference, 3 × 1280×960, 2 workers | 1/30, reward 0.5 on 27, 0.0 on 2 | 1,772 s | -- |
| syringe_packing/L1 | `agentview_fast@v1`, 3 × 800×600, 2 workers | 1/30, reward 0.5 on 26, 0.0 on 3 ("could not perceive 'syringe'") | 1,725 s | **dropped**: −3 % wall |

On tool_hanging the half-resolution registration put the plier's loop about 5 mm off, and the
plier was released beside the hook (`fast_tool_hanging.md`). On syringe, perception at
800×600 was no better than at the reference and the cell barely got faster: the time goes to
retries, VLM calls and planning, not rendering. The rigs stay registered (`agentview_fast@v1`
in `tasks/syringe_packing/staging.py`, `yam_station_tool_hanging_fast@v1` in
`tasks/tool_hanging/staging.py`) so the measurement can be repeated. The optional usbc and
cable rigs were not merged.

## Method

`python -m vabar run` with the suite's camera rig swapped in-process. A small harness
(`res.py`, kept with the raw output under `agent_scratch/r/`) does the swap and then calls
`vabar.run.main` in the same process:

- `get_camera_rig(<reference rig>)`, then `CameraSpec.scaled(round(w·f), round(h·f))` per
  camera, or only for the cameras named;
- `dataclasses.replace(task, cameras=...)` on every lowered `TaskSpec` that stages the
  reference rig, then `SUITES.replace`;
- a `res.json` in each run directory records the effective rig.

No document, and no registered name, changes. The registered `fast` rigs were then checked to
equal the tested ones by value.

**Runner.**
- The SAM 3 suites (syringe_packing, tool_sort, tool_hanging) ran in fleet mode
  (`--num-worlds 2 --episodes <2 ids>`), which gives lockstep stepping and on-demand
  observation rendering on `main`.
- usbc_insert and cable_route ran single-world. `ToolsUsbc`'s planners are not fleet-safe
  (see "Runtime problems"). cable_route's `fast` physics was only ever measured at
  `num_worlds = 1`.
- Flags were the cell's, except `--bus-timeout 600` (CPU-loaded host) and an explicit
  `--renderer rtx`.

**Environment.**
- `VABAR_OPEN_ROBOT_SKILLS=/home/shadeform/speed_ws/open-robot-skills`.
- torchvision 0.25.0 from the uv cache on the runner's `sys.path` only, not `PYTHONPATH`.
- `HF_TOKEN` read from `~/.cache/huggingface/token`.
- `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette` for the two cable suites.

**Timing.** Wall times below are per process, and reference and candidate were started
together where the table says "concurrent". Two things drown out a resolution effect in the
end-to-end time:
- the load on the GPU and host changed by 2× over the window;
- the graphs branch: a retry for the second syringe or wrench adds minutes.

So the tables also give the two parts of an episode that resolution can move, read from each
episode's `dag_trace.json`:
- **observe**: the `*.observe` nodes, which render on demand and ship the frames;
- **perception**: nodes that call SAM 3 / GroundingDINO / `mask_to_world_points`, or that
  perceive, map, count or verify.

## Per-call costs (measured by the lead, for context)

- SAM 3 resizes every input to 1008², so its model compute is fixed: about 65 ms per
  text-segmentation call.
- Through the bundle RPC, one call costs 142 ms at 1280×720 and 74 ms at 640×360 on a real
  tool_hanging frame. The difference is per-pixel work outside the model. Top score went
  0.80 → 0.71, still detected.
- Mask → point cloud (`geometry.mask_to_world_points`): 6 ms vs 1 ms.
- Trace assets: each tool call writes PNG/npy. An 8-world tool_hanging wave writes 3.5 GB,
  and one RGB PNG encode is 91 ms at 720p. Both scale with pixel count.
  - Measured here, tool_hanging L2 (2 worlds): 1,663 MB of traces at 1280×720 vs 432 MB at
    640×360.
  - syringe t00: 650 MB at 1280×960, 266 MB at 800×600 (same number of observations).

## syringe_packing (`agentview@v2` → 960×720 and 800×600)

Cell protocol: `gap_perception --motion curobo --no-privileged --perception vision`, 2 worlds
of one document per run (`--episodes 0:2`). The tray documents use a fixed sampler, so both
worlds share one layout. Reward 0.5 = one of the two syringes posted, which is the
reference's usual outcome.

| Layout | Rig | Reward (ep 0 / ep 1) | Steps | Episode wall | observe s/call | Perception failures | What happened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t00 | ref 1280×960 | 0.50 / 0.50 | 2,887 / 3,861 | 8.0 / 8.9 min | 3.1 / 3.4 | — | one posted per world; `count_remaining` then saw none left |
| t00 | 0.75 960×720 (concurrent with ref) | 0.50 / 0.50 | 3,178 / 7,827 | 8.5 / 13.4 min | 2.1 / 1.5 | — | one posted; `count_remaining` **saw the second syringe** and the graph went back for it: ep 0 aborted at `grasp_sg.execute_descend` (no trajectory), ep 1 retried, then counted none |
| t00 | 0.625 800×600 | 0.50 / 0.50 | 2,887 / 2,890 | 5.6 / 5.6 min | 2.6 / 2.6 | — | as the reference, to within 3 steps |
| t01 | ref 1280×960 (run twice) | 0.00 / 0.00 | 179 / 179 | 1.0 min | — | **both worlds, both runs: `target_sg.perceive` "could not perceive 'syringe'"** | SAM 3 top scores 0.099 ("red needle cap of syringe") and 0.012 ("syringe") |
| t01 | 0.75 960×720 | 0.50 / 0.50 | 2,958 / 2,958 | 5.4 / 5.4 min | 2.6 / 2.6 | — | one posted per world |
| t01 | 0.625 800×600 | 0.50 / 0.50 | 3,757 / 2,962 | 5.6 / 5.0 min | 2.9 / 2.6 | — | one posted per world; ep 0 ended at `verify_sg.check` (second syringe outside the aperture) |
| t05 | ref 1280×960 | 0.50 / 0.50 | 11,661 / 6,612 | 26.8 / 13.3 min | 2.4 / 2.8 | — | one posted; both worlds went back for the second |
| t05 | 0.75 960×720 (concurrent with ref) | 0.50 / 0.50 | 5,627 / 9,006 | 11.1 / 19.7 min | 2.0 / 1.9 | — | one posted; both retried the second and ended at `verify_sg.check` |
| t05 | 0.625 800×600 (started 6 min after) | 0.50 / 0.50 | 7,583 / 5,840 | 26.2 / 20.7 min | 1.8 / 2.0 | — | one posted; retried the second, counted none |
| t05 | 0.5 640×480 (other agent, `fast_syringe_packing.md`) | — | aborted | — | — | **both worlds, twice: "could not perceive 'syringe'"** | |

**Verdict: 800×600 passes.**
- Over three layouts and twelve worlds, 0.75 and 0.625 scored exactly what the reference
  scores where the reference can score: 0.5 in every world.
- No scaled run lost the syringe. 0.5 (640×480) does, on t05.
- The non-monotonic result is real and reproducible. On t01 the 1280×960 reference fails
  perception in both worlds of two separate runs, while 960×720 and 800×600 perceive and
  post.
  - `perceive_target.py` gates on SAM 3 scores fitted at 1280×960. Its inputs are resized to
    1008² whatever their size, so the path tracer's render at each resolution, not the pixel
    count, decides which side of the gate a marginal layout lands on.
  - Read this as "these thresholds have little margin on some layouts at any resolution",
    not as "800×600 is better".

**Speed.**
- Per observation, the scaled rigs shipped frames in 1.5-2.9 s against 2.4-3.4 s for the
  reference. Those are on-demand renders on a saturated GPU, so the absolute values are
  inflated.
- Trace volume fell 2.4× at 800×600.
- End-to-end episode time is dominated by planning (`carry_plan_sg.plan` alone is about
  150 s) and by whether the graph retries the second syringe. Neither depends on the rig,
  so no end-to-end speedup can be claimed from these samples.
- The concurrent t05 pair finished in 19.7 min (0.75) against 26.8 min (reference), but the
  episodes differ by thousands of steps.

## tool_sort (`tool_sort_agentview@v3`, overhead only)

Cell protocol: `gap_perception_loop --no-privileged --perception vision --max-steps 20000`.
`level1_6` (screwdriver, two compression springs, tape measure, glue), `--episodes 0:2`, 2
worlds.

| Rig (overhead) | Success | Steps | Wall | observe s/call | Perception failures | What happened |
| --- | --- | --- | --- | --- | --- | --- |
| ref 1280×960 | **2/2** | 16,127 / 13,514 | 24.5 / 21.3 min | 1.5 / 1.6 | — | all four objects sorted in both worlds |
| 640×480 (concurrent with ref) | **0/2 (stopped at 49 min, still looping)** | — | > 2× the reference | — | no "could not perceive"; **grasp verification rejects the grasps** | 3 transports in 49 min; 43 `verify_reach` misses, 31 of them on the tape measure (the reference missed it none) |
| ref 1280×960 (second run) | **2/2** | 12,694 / 16,731 | 32.7 / 38.9 min | 2.1 / 1.8 | — | all sorted; 24 pair selections, 17 recovery attempts, 7 reach misses |
| 960×720 (concurrent with second ref) | **2/2** | 19,312 / 17,033 (**+24 %**; ep 0 at 97 % of the budget) | 43.4 / 40.7 min (**+12 %**) | 1.5 / 1.6 | — | all sorted, but 28 selections, 22 recovery attempts, 10 reach misses |

**Why 640×480 fails.** `verifying-grasps/verify_reach.py` lets the hand miss its grasp pose by
at most the smallest extent of the perceived object box: `target_obb`, fitted to the overhead
depth points, clamped to 3-20 mm. At half resolution those boxes come out much thinner:

| Object | `tol_mm` at 1280×960 (reference) | `tol_mm` at 640×480 | `tol_mm` at 960×720 |
| --- | --- | --- | --- |
| tape measure | 8.8-13.6 | **3.2-3.4** | 11.0-18.6, but also **4.9, 5.0** |
| screwdriver | 13.5-13.9 | **6.8-7.1** | 13.5-13.6 |
| compression spring | 8.1-20.0 | **3.0** | 7.2-20.0, but also **3.5** |
| cyanoacrylate glue | 9.7-18.5 | 8.8-11.8 | 9.7-13.0 |

(Reference ranges pool both reference runs.)

- The servo lands 3.7 mm off on every tape-measure grasp, and the reference forgives that.
- At 640×480 it is 0.3-0.5 mm over the 3.3 mm allowance. So the graph opens, re-perceives and
  tries again, 31 times on the tape measure alone in one run.
- This is the failure the rig's own docstring warns about: the overhead depth is the only 3D
  measurement these graphs get. Here it shows up as box thickness rather than as a missed
  object.
- `level1_6` has no small parts (smallest: a 37 × 27 mm spring). Eleven of the thirty L1
  documents carry the aa_battery, screws or nuts that `@v3` doubled its overhead for, so a
  rig that fails here would fail there too.

**960×720 sorts everything, but the episodes get slower, not faster.**
- Most boxes come out as thick as the reference's. A few do not (tape measure 4.9-5.0 mm,
  spring 3.5 mm), and each of those costs a re-grasp.
- The run needed 3 more reach misses, 4 more pair selections and 5 more recovery attempts than
  the concurrent reference, so +24 % control steps and +12 % wall time. One world finished at
  19,312 of its 20,000-step budget.
- Each observation was cheaper (1.5-1.6 s vs 1.8-2.1 s), but the overhead is one camera of
  three and the graph's time is in its loop.

**Verdict: no `fast` rig for tool_sort.** Keep `tool_sort_agentview@v3`.
- 640×480 breaks grasp verification outright.
- 960×720 passes but is a net loss on the one document tried. It would be worse on the
  eleven L1 documents with small parts, since the overhead was doubled for exactly those.
- (`tool_sort_agentview@v2` *is* `@v3` with a 640×480 overhead, so the failing candidate is
  the rig this suite already retired.)

## tool_hanging Level 2 (`yam_station_tool_hanging@v1` → `yam_station_tool_hanging_fast@v1`)

`wrenches_level2`, graph `gap_perception_level2` (the two-tool unrolled graph; passed as a
directory, because the variant ships only `gap_perception` under its name), `--motion curobo
--no-privileged --perception vision --max-steps 22000`, 2 worlds. Each pair ran concurrently.

| Episodes | Rig | Success | Reward | Steps | Wall (process) | observe s/call | perception nodes | Perception failures | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1, 5 | ref 4 × 1280×720 | 1/2 | 0.50 / **1.00** | 4,106 / 3,820 | 7.6 min | 2.2-2.3 | 66-81 s | — | ep 1: `grasp_sg_2.rotate_align` tracking error 26.9 mm (second wrench) |
| 1, 5 | 4 × 640×360 | 0/2 | 0.50 / 0.50 | 4,101 / 3,702 | 7.2 min (−5 %) | 1.2-1.5 | 53-57 s | — | ep 1: the same tracking error (27.4 mm); ep 5: wrench 1 released 0.44 m from its hook (placement miss), wrench 2 hung |
| 9, 13 | ref 4 × 1280×720 | 1/2 | 0.50 / **1.00** | 4,552 / 3,823 | 19.9 min | 4.6 | 123-128 s | — | ep 9: wrench 1 not hung (0.44 m off), wrench 2 hung |
| 9, 13 | 4 × 640×360 | 1/2 | 0.50 / **1.00** | 4,286 / 3,583 | **15.3 min (−23 %)** | 2.6-2.7 | 101-105 s | — | ep 9: wrench 1 not hung (1.2 m off), wrench 2 hung; the graph's second insertion aborted ("Cartesian mating waypoint not reached") |

Graph span per episode:
- pair 1: 7.2-7.3 min (reference) against 6.2-6.9 min (half), −5 to −14 %;
- pair 2: 17.4-19.5 min against 13.1-14.9 min, about −24 %. That pair ran while the GPU was
  saturated (98 %), and on-demand renders cost 4.6 s each at 1280×720 against 2.6 s at
  640×360.

The trace volume fell from 1,663 MB to 432 MB.

- **Perception holds.** Every SAM 3 / GroundingDINO node completed in both arms, in all four
  episodes. `register_held`'s CAD ICP accepted the same way, with one wrist refinement
  rejected (`rmse=inf`) in each arm of pair 1.
- **Outcomes match episode for episode, except ep 5:**

  | Rig | ep 1 | ep 5 | ep 9 | ep 13 | Success | Mean reward |
  | --- | --- | --- | --- | --- | --- | --- |
  | reference | 0.5 | 1.0 | 0.5 | 1.0 | 2/4 | 0.75 |
  | half | 0.5 | 0.5 | 0.5 | 1.0 | 1/4 | 0.625 |

  - Ep 9 loses wrench 1 in both arms; ep 13 succeeds in both.
  - On ep 5 the half-resolution arm released wrench 1 0.44 m from its hook after a completed
    insertion: a placement miss.
- **Pooled with the other agent's results:**
  - L1 (`gap_perception`, 7 + 7 episodes): 6/7 vs 7/7, with two placement misses in
    candidate arms;
  - L2 under `gap_perception`, eps 1 and 5: rewards 0.5 / 1.0 for the candidate vs 0.5 / 0.5
    for the reference.
- Over the six L2 episodes on both graphs, the candidate's mean reward is 0.67 and the
  reference's 0.67. The samples cannot separate the arms. The misses are tens of
  centimetres, which looks like a graph decision rather than a degraded estimate.
- **Keep the half-resolution rig.** It is −5 to −23 % wall time on L2 in these concurrent
  pairs, the same order as the other agent's −16..−18 % on L1. Watch the placement-miss
  rate in the first full `fast` cell.

## usbc_insert (`yam_station_usbc@v1`, palette segmentation)

Cell protocol: `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`, graph `mate`, `--no-privileged
--perception vision --renderer rtx --bus-timeout 600 --settle-steps 600`.
- The runs are Level-2 `insert_medium`, the tier on which the reference succeeds. USB-C Easy
  under `mate` fails at the reference (`cable.md`: 7/7, about 15 mm lateral), so it cannot
  show a regression.
- Single-world, one instance per process, the three rigs concurrent per instance. The fleet
  form aborts at `hold_sg.plan`; see "Runtime problems".

| Instance | Rig | Success | depth / lat / tilt | Steps | Episode wall | Perception nodes | Graph geometry (in-jaw correction; stroke direction z) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10085 | ref 3 × 1280×720 | **yes** | 13.2 mm / 0.2 mm / 0.043 rad | 4,351 | 43.6 min | 14.1 s | 3.90°; −0.056 |
| 10085 | 0.75 960×540 | **yes** | 12.8 mm / 0.3 mm / 0.062 rad | 4,394 | 44.2 min | 12.1 s | 2.45°; −0.042 |
| 10085 | 0.5 640×360 | **yes** | 13.4 mm / 0.2 mm / 0.058 rad | 4,384 | 43.4 min | 12.1 s | **8.60°; −0.149** |
| 10032 | ref 3 × 1280×720 | **yes** | 12.9 mm / 0.6 mm / 0.118 rad | 4,308 | 27.8 min | 14.1 s | 6.06°; −0.067 |
| 10032 | 0.75 960×540 | **yes** | 12.7 mm / 0.5 mm / 0.102 rad | 4,249 | 26.4 min | 10.9 s | 3.75°; +0.015 |
| 10032 | 0.5 640×360 | **yes** | 12.7 mm / 0.6 mm / 0.116 rad | 4,235 | 28.7 min | 12.3 s | 5.08°; +0.009 |

- Every perception step returned at every scale: `ground_vision` / bore CV, the brace grasp
  point, and the plug's in-jaw pose. All six episodes inserted, with depth, lateral error and
  tilt within the reference's spread.
- On 10085 at 0.5 the plug-axis estimate moved, though. The stroke was corrected by 8.6°
  instead of 3.9°, and aimed 0.149 down instead of 0.056.
  - On 10032 both scaled rigs differed from the reference by similar amounts, so it is one
    instance, not a pattern.
  - `bore_cv.py` and `ground_vision.py` carry pixel constants: `SEARCH_PX = 34`,
    `MIN_FACE_PX = 600`, `MIN_SLOT_PX = 40`, `RING_PX = 7`, `SUPPORT_RING_PX = 12`. These
    were sized at 1280×720 and cover twice the physical size at 0.5.
- **Speed: none measurable.** Wall time was the same across rigs per instance (43.4-44.2 and
  26.4-28.7 min).
  - These single-world episodes ran eager (no CUDA graph) at 1.7-2.6 steps/s on a
    shared GPU, so physics bounds them, not rendering.
  - The graph's perception nodes are 11-14 s of each episode.
  - Trace volume: 63 → 41 → 26 MB.
  - On the integration branch's `usbc_fast@v1` (captured, 30-40 steps/s, 2-2.5 min per
    episode), those 11-14 s become a larger share. That regime was not measured here.

**Verdict: 0.75 passes and 0.5 passes outcomes.** Registered **`yam_station_usbc_fast@v1`**
(3 × 960×540): the conservative choice, given the pixel constants above. It is optional in
`fast`, since no gain was measured.

## cable_route (`yam_station_cable@v1`, palette segmentation)

> **Superseded for the terminal:** the "orange block" gate below was a fixed 3,400 px floor
> measured on a 1280x960 view. It now scales with the frame's `fx * fy`
> ([cable_routing_fix](cable_routing_fix.md)): 1,912 px at 720 rows, 1,076 at 540 and 478 at
> 360. That is under the port's 2,132 / 1,248 / 568 px measured here, so these runs
> predate the endpoint insertion running at all.

Cell protocol: `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`, `port_easy`, `gt` (weave)
graph, `--no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps
60`. Single-world, one instance per process, the three rigs concurrent per instance.
Physics is the reference `cable_coupled_960hz@v1`.

| Instance | Rig | Reward | seat0 / seat1 / port box | Steps | Episode wall | Perception nodes | Perception notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10009 | ref 3 × 1280×720 | 0.50 | yes / yes / no | 6,392 | 60.4 min | 1,117 s | TrackDLO 579-586 mm (true 588); `restore` saw 0.0 mm at station 0 and re-seated it; orange block 2,132 px, under the 3,400 px gate |
| 10009 | 0.75 960×540 | **1.00** | yes / yes / **yes** | 4,090 | 47.4 min | 307 s | TrackDLO 577-587 mm; `restore` saw both seated (46 / 71 mm visible); orange block 1,248 px |
| 10009 | 0.5 640×360 | 0.50 | yes / yes / no | 4,090 | 47.0 min | 296 s | TrackDLO 580-586 mm; `restore` saw both seated (48 / 43 mm visible); orange block 568 px |
| 10062 | ref 3 × 1280×720 | 0.50 | yes / yes / no | 4,144 | 29.3 min | 199 s | TrackDLO 586-590 mm; `restore` 46.9 / 46.5 mm visible; verify: woven, worst margin 11.7 mm |
| 10062 | 0.75 960×540 | 0.50 | yes / yes / no | 4,113 | 29.2 min | 185 s | TrackDLO 587-589 mm; `restore` 46.9 / 45.6 mm visible; verify: woven, 12.4 mm. **Same as the reference.** |
| 10062 | 0.5 640×360 | 0.50 | **no** / yes / yes | 4,256 | 29.5 min | 224 s | `restore` saw **34.1** / 48.1 mm visible and left station 0; verify said woven (18.5 mm), but **station 0 ended unseated**; orange block 549 px |

- **0.75 holds.** The tracked cable length, the station takes, the `restore` spans and the
  `verify_vision` weave check agree with the reference to within a few millimetres. On 10062
  the 0.75 run reproduces the reference outcome seat for seat.
- **0.5 is shakier.** On 10062 it read 34 mm of station 0's span against the reference's 47
  mm. It left the station alone and verified the weave, and the evaluator then found station 0
  unseated. Its 0.5 reward came from the free end landing in the port box instead.
  - One instance cannot say whether the cable slipped after the check or the check misread it.
  - It is the one place a scaled rig's perception and the evaluator disagreed.
- **The terminal stage is unaffected because it never runs.** Its `BLOCK_MIN_PX = 3400` gate
  is already missed at 1280×720 (2,132 px, the known defect in `cable.md`), so no rig attempts
  the endpoint insertion.
  - That gate is in pixels. Anyone who lowers it to make the terminal work at 1280×720 has to
    scale it by the area factor for a scaled rig: 0.56 at 0.75, 0.25 at 0.5.
- **On 10009 the outcome differences are not perception.**
  - The 0.75 arm's 1.0 is the free cable end settling inside the port box. That is the same
    luck that gave the old runtime its 1.0 on this scene.
  - The reference's extra 2,300 steps are a re-seat of station 0 that the scaled arms did not
    need.
  - The runs are single-world and free-running (no lockstep), so contact timing differs run
    to run.
- **Speed: none measurable.**
  - On 10009 the 0.75 and 0.5 arms took the same path and the same time (47.4 vs 47.0 min).
  - On 10062 all three rigs took 29.2-29.5 min.
  - Physics bounds these episodes, not rendering: about 1.5-2.4 steps/s here, against 7
    steps/s idle on a quiet GPU.
  - Trace volume: 91 / 54 / 27 MB on 10062.

**Verdict: 0.75 passes; 0.5 is not recommended.** Registered **`yam_station_cable_fast@v1`**
(3 × 960×540). It is optional in `fast`, since no gain was measured: the three rigs took
29.2-29.5 min on 10062, because the weave graph's perception is a handful of TrackDLO /
palette calls in a physics-bound episode.

## Runtime problems found on the way

- **usbc_insert is not fleet-safe on `main`, whatever the `_wait_for_step` fix.**
  `insert_medium` with `--num-worlds 2` (10085 + 10032) aborted both worlds at
  `hold_sg.plan` with `RuntimeError: Copy source and destination must be arrays`. This
  happened at every resolution, the reference included.
  - `ToolsUsbc._hold` steps the template IK solver against `env.state.joint_q`, which in a
    fleet is a per-world slice rather than a `wp.array`. `fast_usbc_insert.md` reports the
    same failure for `brace_sg.plan` and the insertion planner.
  - The usbc runs here are therefore single-world, one episode per process.
- The fleet's GPU memory is mostly in the tool-bundle servers. Each SAM 3 graph process
  starts its own `sam3` (5-6.5 GB) and `grounding-dino` (3 GB) servers next to 4-7 GB of its
  own.
  - Twelve concurrent processes on one 96 GB GPU ran it out of memory (`disengage_sg_1.plan`:
    CUDA OOM). Those runs were discarded and repeated.
  - Plan on about 13 GB per SAM 3 graph process.
- `tasks/tool_hanging/graphs/gap_perception_level2` is not registered as a graph of
  `wrenches_level2` (the variant ships only `gap_perception`). It has to be passed as a
  directory.

## For `vabar/configs/fast.toml`

Nothing. See the full-cell check above: neither camera entry survived it, and the usbc and
cable rigs showed no wall-time gain to trade for.

## Files

- `tasks/syringe_packing/staging.py`: `agentview_fast@v1` (= `agentview@v2` at 800×600).
- `tasks/usbc_insert/staging.py`: `yam_station_usbc_fast@v1` (= `yam_station_usbc@v1` at
  960×540).
- `tasks/cable_route/staging.py`: `yam_station_cable_fast@v1` (= `yam_station_cable@v1` at
  960×540).
- No change to `tasks/tool_hanging/staging.py`. `yam_station_tool_hanging_fast@v1` is on
  the integration branch, and was tested here in-process at the same value: every camera
  `scaled(w // 2, h // 2)`.
- Every registered rig was checked to equal the tested rig by value
  (`get_camera_rig(name) == tuple(c.scaled(round(w·f), round(h·f)) for c in base)`).
- `tasks/usbc_insert/tools.py`, `tasks/cable_route/tools.py`: `_wait_for_step` waits on
  `bus.wait_for_step` (catching `BusTimeout`) instead of polling `snapshot.step`, the
  lead's integration-branch fix. It is applied here so the two suites could be tried in
  fleets, and is identical to the integration branch's version.
- Raw output: `/home/shadeform/speed_ws/agent_scratch/r/runs/<run>/` (`res.json`,
  `episode_results.jsonl`, `traces/*/dag_trace.json`) and `<run>.log`. OOM and fleet-bug
  casualties are under `runs/failed/`.
