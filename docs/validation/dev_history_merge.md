# Fixes the v1.0 export missed: syringe packing, tool sorting, tool hanging, gear meshing, cable insertion

VABAR v1.0 and public RoboSimStudio are squashed exports of two private development repositories. This note records
which graph, skill and task-definition fixes in those repositories never reached VABAR. It covers five task classes:
syringe packing, tool sorting, tool hanging, gear meshing and cable insertion. For each fix it says whether it is in,
was ported, or was rejected, and what porting it measured on full 30-instance cells.

Measured 2026-11-05 on GPUs 3 and 7 (RTX PRO 6000 Blackwell). The stack was the checkouts under
`/home/shadeform/speed_ws/` as they stood that day, each with uncommitted work: RoboSimStudio `d9b2e44`, graph-as-policy
`09e45f8`, open-robot-skills `752dfbd`. Raw output is under `/home/shadeform/speed_ws/agent_scratch/devmerge/runs/`.

## Answer

Three fixes are ported, one patch each. All three are graph or task-definition fixes; none changes physics, instances
or scoring. Fix 3 does change how fast the robot is commanded to move, through the gear suite's staging, so it is worth a
deliberate decision.

| Cell | Before | After | Paper | Fix |
| --- | --- | --- | --- | --- |
| Syringe packing L1 | 2/60 (3%), reward 0.483 | **22/60 (37%), reward 0.650** | 87.0% | 2 |
| Syringe packing L2 | 0/30, reward 0.028, 5/198 syringes | 0/30, **reward 0.272, 51/198 syringes** | 63.0% | 2 |
| Tool sorting L1 | 29/120 (24%), reward 0.595, 60% of objects | **35/90 (39%), reward 0.742, 75% of objects** | 47.1% | 1 |
| Gear meshing L1 | 157/180 (87%), reward 0.933 | **57/60 (95%), reward 0.983** | 100% | 3 |
| Gear meshing L2, straight boards (as exported) | 40/60 (67%), reward 0.789 | **80/90 (89%), reward 0.944** | | 3 |
| Gear meshing L2, bent boards (the live tree, as the documents state) | 15/30 (50%), reward 0.767 | **22/30 (73%), reward 0.889** | 90.0% | 3 |
| Tool hanging L1 | 29/30 | unchanged | 86.7% | |
| Tool hanging L2 | 49/60 (82%), reward 0.883 | unchanged | 81.7% | |
| Cable insertion L1, L2 | 30/30, 21/30 | unchanged | 86.7%, 30.0% | |

Each fix came from a repository the export skipped:
- **Fix 1, tool sorting:** the "grasp and loop fixes behind the 85-90% level-1 run". The jaws close 6 mm lower, and grasps
  that reach transport double, from 20% to 41%.
- **Fix 2, syringe packing:** the multi-syringe graph `gap_perception_v2`, which AUTOLab's unmerged `syringe_packing`
  branch and RoboSimStudio-dev main both carry. VABAR shipped the graph from three days before it.
- **Fix 3, gear meshing:** the servo speed the graph was tuned at, 0.12 m/s instead of the pinned runtime's 0.31, with
  its contact windows restated in control steps. **It is already in the live tree:** the same change landed there at
  20:49 today, with the SAM 3 cap and the bent-board port (74e0cfd8), which are not this note's. So patch 03 is a no-op
  on the live tree and applies to the tree as it stood before.

**The syringe cells remain well below the paper.** v2's L1 failures concentrate on the spare syringe, and its L2
failures on box jams and short grasp descents. VABAR's colliders are the likely reason, but that is untested; see
"Scene-level leads". Dev's own record does not show 87% either: 538c49f has v2 at 0/1 on `tray_pair_both_t00` on dev's
scene, and I found no recorded L1 rate in either repository.

**The table's two measures are not the same.** "Before" and "After" lead with whole-instance success, where the paper's
rule is per object where a task's objects are independent. RoboSimStudio-dev's paper harness (`vabar_eval.py`, cb96425e)
states the rule: one trial per object's goal clause group, per goal region for containment, and per seated gear for gear
meshing. On that rule:
- tool sorting compares as 60% → 75% against 47.1%, already above the paper before this work;
- tool hanging L2 compares as 87-90% of tools against 81.7%;
- gear meshing L2 on the bent boards compares as about 0.77 → 0.89 against 90.0%, if mean reward tracks seated gears;
- syringe packing's goals share one region, so a whole tray is one trial, and the instance numbers above are the
  comparable ones.

**Tested and not kept:** the wrist roll held on gear meshing; servo speed 0.12 m/s on tool sorting, syringe packing and
tool hanging, so that change is per suite and not global; the older single-target syringe graph; and removing the beads
from the Level-2 scissors distractor. AUTOLab's scissors and screwdriver fixes are correct but within noise, and are an
optional patch.

Patches are in `/home/shadeform/speed_ws/agent_scratch/devmerge/patches/`, against the live VABAR tree. They touch
different suites and apply in any order. Patch 03 is already in the live tree.

## Sources and method

Three trees, compared by file content, since they share no history:

| Tree | Ref | What it is |
| --- | --- | --- |
| VABAR | v1.0 (`4d5e45e`) plus the uncommitted `fast`/fix work on disk | the baseline every "before" number ran on |
| AUTOLab-GaP-NVIDIA-Collaboration | `origin/main` f6057d7 (10-26), `vabar-robosimstudio` 18b974e, and 22 branches | what VABAR was cut from; same `tasks/<suite>/` layout |
| RoboSimStudio-dev | `origin/main` 1cd9f76a (10-05), `release/oss*`, and 58 other branches | where most suites were developed; the suites sit at `tasks/<suite>/` on older branches and `vabar/tasks/<suite>/` on main |

Path map: AUTOLab `tasks/<suite>/` = VABAR `tasks/<suite>/`; RoboSimStudio-dev `vabar/tasks/<suite>/` (or `tasks/<suite>/`)
= VABAR `tasks/<suite>/`, except `sharps_disposal` = `syringe_packing`. RoboSimStudio-dev main moved several graph
scripts into `vabar/tasks/_shared/scripts/` and into open-robot-skills bundles, so graphs were compared by the script
each node *resolves* to (graph-local, then `_shared`, then the subgraph's bundle), not by path.

Every "after" number is a full cell, `vabar bench --cell <cell> --solve graph --config fast`, run from a copy of the
VABAR tree with one change applied. Copies that shared a GPU ran with fewer workers (`--set
suites.<suite>.workers=N`); the clock is lockstep, so that should change wall time, not outcomes. "Before" is today's
runs of the same cells under `fast`. Outcomes still vary between runs (identical copies here gave 13, 12 and 10/30 on
tool sorting, and 13 and 9/30 on syringe L1), so cells are compared pooled over repeats wherever repeats exist.

## Ported

### Fix 1. Tool sorting: the loop-failure fixes (`01_tool_sort_loop_failure_fixes.patch`)

**Source.** RoboSimStudio-dev `gap_perception_loop` as it stands on origin/main (1cd9f76a). It is VABAR's graph plus
four later commits by the graph's author (Jaimyn Drake) and the merge that carried them:
- 067645c6 (`tool_perception`, 2026-09-13; the same graph lines are AUTOLab `tool-sorting` 69c8fcb and AUTOLab
  `lgulich/tool-sort-loop-failure-fixes` dca8679, "Modifications to resolve some failure cases"). RoboSimStudio-dev
  merged it as 48fcd8a5, "the grasp and loop fixes behind the 85-90% level-1 run";
- 0f1cde27 (09-10) and 4032b33d (09-11), two `select_pair.py` changes;
- e94a5826 (09-12), which drops a debug write.

VABAR's `select_pair.py` is byte-identical to RoboSimStudio-dev 5ec6e601 (09-10), from before all four. The two
grasp-script edits also reached open-robot-skills (ad8cf8f), but this graph runs its own copies, so they never reached
the cell.

**What it changes.** All of it is in `tasks/tool_sort/graphs/gap_perception_loop/`:
- **Grasp height.** `workflow.json` passed `grasp_sg.compute_grasp` a `grasp_z_offset` of 0.02. It is now 0.014, so
  the jaws close 6 mm lower on the object. The scripts' default goes from 0.015 to 0.014 too.
- **Lift before transport.** `transport_sg` now lifts the grasped object (`lift_grasped.py`, from the
  `transporting-objects` bundle) and goes home before the transport leg. It used to start the leg from the bench.
- **Reach check floor.** `verify_sg.verify_reach` gets `floor_m: 0.005`, the smallest reach tolerance it may demand.
- **Wrist limit.** A symmetric gripper allows two jaw orientations. `axis_grasp_top_down.py` now takes the one that
  keeps joint 7 inside its ±2.897 rad limit and nearest to where it is.
- **Retries per position.** In `select_pair.py`, a detection position is spent after `max_attempts_per_instance` failed
  attempts (now 3), not after the first.
- **Label prompt.** `select_pair.py` asks the VLM to pick the closest label from the list and always reply with one,
  where it used to accept "none".
- **Bin filter.** `select_pair.py` drops DINO boxes covering more than 30% of the source crop, since that box is the
  bin. A surviving box counts as a container only when it holds the centres of at least 3 other detections (was 2).
- **Debug write.** The selection overlay is no longer written to `Path.cwd()/outputs/debug`, one path every worker
  shared. The runtime already saves it into each episode's trace.

**Measured.** `tool_sorting/L1`, `fast`: Gemini 3.1 Flash-Lite, 14,000-step budget.

| Run | Graph | Success | Mean reward | Objects sorted | Grasps that reached transport |
| --- | --- | --- | --- | --- | --- |
| `VAL_sort` (today) | v1.0 | 8/30 | 0.597 | 85/140 | 87/466 (19%) |
| `FINAL2_sort_L1` (today) | v1.0 | 7/30 | 0.620 | 88/140 | 101/413 (24%) |
| `cell_g31fl_a` (`tool_sort_vlm.md`) | v1.0 | 5/30 | 0.525 | 74/140 | 77/480 (16%) |
| `cell_g31fl_v2` (`tool_sort_vlm.md`) | v1.0 | 9/30 | 0.638 | 90/140 | 97/429 (23%) |
| **v1.0, pooled** | | **29/120 (24%)** | **0.595** | **337/560 (60%)** | 362/1,788 (20%) |
| `ts_loopfix_a` | dca8679 only (an ablation) | 12/30 | 0.733 | 104/140 | 111/293 (38%) |
| `ts_rssmain_a` | **fix 1** | 13/30 | 0.758 | 107/140 | 113/272 (42%) |
| `ts_rssmain_b` | **fix 1** | 12/30 | 0.740 | 104/140 | 111/270 (41%) |
| `ts_rssmain_c` | **fix 1** | 10/30 | 0.728 | 102/140 | 108/273 (40%) |
| **fix 1, pooled** | | **35/90 (39%)** | **0.742** | **313/420 (75%)** | 332/815 (41%) |

Against v1.0's pooled 120 episodes, the change is significant:
- success: Fisher p = 0.024;
- mean reward: Welch p = 0.0002.

"Grasps that reached transport" counts `transport_sg` entries over `grasp_sg` entries in each episode's trace. It
doubles, and a cell makes about 270 grasp attempts instead of 410-480, so less of the budget goes on the select, grasp,
recover loop. The lower grasp is the likely cause. RoboSimStudio-dev d0fdd2c7 measured why: `grasp_z = bin_floor_z +
grasp_z_offset` ignores the object's height, so at 0.02 the fingertips stop 15.58 mm above the bin floor for every object,
and items shorter than that went 0 for 17. The grasp changes were not separated from one another.

The dca8679-only ablation (12/30, 0.733) cannot be told apart from the full fix. So the gain is the grasp and loop
changes; the `select_pair` additions are carried because they are the author's later state and cost nothing.

Most failures still end at the budget: 17 of 30 episodes in `ts_rssmain_a`. Raising it does not change that. With the
cell's own 20,000 steps instead of `fast`'s 14,000 (`ts_rssmain_20k`), fix 1 scored 12/30, 0.778 and 109/140 objects, and
14 of 30 episodes still ran out the budget. Its latest success took 14,293 steps, so `fast`'s budget costs little.

The paper counts tool sorting per object, and on that measure the cell now reaches 75% (paper: 47.1%).

### Fix 2. Syringe packing: the multi-syringe graph (`02_syringe_packing_gap_perception_v2.patch`)

**Source.** RoboSimStudio-dev `sharps_disposal/gap_perception_v2` at origin/main (1cd9f76a):
- 473d7f9 (Shuangyu Xie, 2026-09-13, "cluttered-tray perception graph"), carried to main by fbe20d4;
- ee4ad93 (2026-09-16), for a prismatic hand.

AUTOLab carried the same graph on its `syringe_packing` branch (028911d, 2026-09-14). That branch was never merged, so
VABAR shipped the 2026-09-10 graph (RoboSimStudio-dev 13b4322).

**What it changes.** v2 replaces `tasks/syringe_packing/graphs/gap_perception/`. It was written for these documents,
whose goal is every syringe on the tray; fbe20d4 declares it for the `tray_clutter` and both-posted documents.
Compared with the old graph:
- **The loop.** v2 perceives the box aperture once, then loops through these steps:
  - fresh RGB-D target perception;
  - the collision world;
  - a checked grasp;
  - a checked two-stage lift, then a check that a syringe moved with the hand;
  - registration;
  - a collision-aware carry and turn, planned together over a yaw grid;
  - staged insertion;
  - a seat check before release;
  - a two-view check after release;
  - an unjam push when a syringe is left standing on the aperture.
- **When it stops.** It ends only when two calibrated views find no syringe left in the tray. Weak evidence is
  "uncertain" (abort), never "done".
  - The old `count_remaining` asked SAM 3 for "syringe" with a 0.08 score floor.
  - In two v1.0 runs, 25 of 60 L1 episodes ended "graph finished" with the second syringe still in the tray (11 and 14).
    In all 25, every overhead mask in the last count scored under the floor (the best mask scored 0.012-0.080).
- **Target choice.** It ranks candidates by jaw clearance measured from depth, and vetoes masks whose red pixels lie off
  the mask's axis.

**Two edits to the source:**
- ee4ad93's fingertip-drop fix is included. This cell runs `franka_hand_robosuite`, a prismatic hand; without the fix,
  the grasp TCP is commanded 11 mm high.
- `carry_sg` is labelled `generic`. Its script is graph-local, which graph-as-policy's SK2 rule refuses under the
  `executing-held-object-motion` label. Every subgraph's library skill is kept under `meta.composes`, as VABAR did for
  the old graph.

The source README is kept, with a note on top. `tasks/syringe_packing/tests/test_perception.py` is deleted: it tested the
old `perceive_target.py`'s single-target identity rule, which v2 does not have.

**Not ported from the same commits:**
- f55eee5 and 18ab35f: the tray's wall and floor colliders, the 61-hull CoACD syringe collider, and the 20 re-drawn
  Level-2 layouts. These are physics and instances.
- 564bbd1's bare `RenderSpec()`, which changes what the cameras see.
- 5c561ad's grasp box, which is physics.

v2's own README numbers (16/30 Level-2 trials, 152/176 syringes) were measured with those changes, not on VABAR's scene.

**Measured.** `fast` config.

| Run | Graph | Success | Mean reward | Syringes posted |
| --- | --- | --- | --- | --- |
| L1 `VAL_syr` (today) | v1.0 | 0/30 | 0.450 | 27/60 |
| L1 `FINAL2_syr_L1` (today) | v1.0 | 2/30 | 0.517 | 31/60 |
| L1 `syr_v2_L1` | **fix 2** | 13/30 | 0.700 | 42/60 |
| L1 `syr_v2_L1_b` | **fix 2** | 9/30 | 0.600 | 36/60 |
| **L1, pooled** | v1.0 → fix 2 | **2/60 → 22/60** | **0.483 → 0.650** | **58/120 (48%) → 78/120 (65%)** |
| L2 `L2_syr` (today) | v1.0 | 0/30 | 0.028 | 5/198 (2.5%) |
| L2 `syr_v2_L2` | **fix 2** | 0/30 | 0.272 | 51/198 (25.8%) |

Significance:
- L1 success: Fisher p = 5e-6.
- L1 mean reward: Welch p = 0.0002.
- L2 mean reward: Welch p = 5e-8.

**What v2 still loses on L1 (60 episodes, 38 failures):**
- **The second syringe's grasp check: 22.** After the lift, the hand check (`hand_map_sg.verify_grasp` or
  `hand_map_sg.map`) finds no syringe with the hand, and the graph aborts.
  - In every one of these, the syringe posted was `syringe_0` and the one left was `syringe_pack_0`, the spare.
  - The task stages the spare differently: a white body, a `bounding_box` collider of the whole mesh, and friction 3.0
    (`task.py`, `_tray_pair_staging`). Either the appearance or the collider could explain it; the traces cannot tell a
    missed grasp from a missed detection, and this was not tested.
- **Seat check refusing a tilted syringe before release: 5**, 4 of them on the first syringe (reward 0).
- **The second post still blocked after the unjam push: 8.**
- **Budget: 1. Target perception: 2.**

The graph aborts rather than retrying a failed grasp. A retry, which neither repository has, is the obvious next step;
it is listed below.

**On L2,** no tray was emptied: the trays hold 5-8 syringes. v2 posts ten times as many syringes as the old graph. Its
episodes end on:
- a box jam that the unjam push does not clear: 12;
- a grasp descent that stops 18-19 mm short in the crowded tray (`grasp_sg.execute`, "trajectory tracking failed;
  release prohibited"): 10;
- the seat check: 4;
- an "uncertain" post-release view: 3;
- target perception: 1.

The box holds about five syringes lying flat (v2's README, VAB-industrial's measurement), so some trays may not fit.
The descent stops may be the collider difference: VABAR's syringes collide as boxes up to 23.6 mm wide, v2's clearance
is measured on the drawn 10.7 mm barrels, and the source sweep ran on CoACD hulls. That was not tested.

Cost: v2 is thorough. On L1 an episode takes 13,600 steps instead of 6,800-7,100 (budget 22,800), and the L1 cell ran
49-60 min on a shared GPU, against 16-18 min. L2 took 72 min with 6 workers.

### Fix 3. Gear meshing: the servo speed and contact windows the graph was tuned at (`03_gear_mesh_servo_speed.patch`)

**Source.** The gear_mesh part of RoboSimStudio-dev cb96425e (2026-09-16), "declared servo speed, step units":
- `gear_mesh.suite`: `(:servo-speed-mps 0.12)` in the suite's staging;
- `move_cartesian_until_contact` windows, in control steps:
  - `carry.py`: 200/15 → 480/36 and 120/15 → 300/36;
  - `press.py`: 240/12 → 600/30.

**Why.** The pinned runtime's servo counts every simulator step and cruises at the OSC law's 0.31 m/s unless the task
states a speed (`robosimstudio_policy_agent/servo.py`, docstring of `base_speed_scale`). The graph was written when a
servo turn spanned 2-3 simulator steps, which ran the tool at about 0.10-0.13 m/s, and its contact windows were counted
in those turns. The runtime's docstring says it directly: "gear_mesh's and cable_route's fine legs failed at the
nominal 0.31 m/s until their suites stated the speed they had always been run at". VABAR copied the gear graph and suite
before that commit.

**Scope.** This is a controller speed plus three graph inputs. Physics, documents and scoring are untouched. The speed
lives in the `.suite`, so it changes the task definition; there is no config key for it.

**Measured.** `fast` config.

| Run | Change | Success | Mean reward | Mean steps |
| --- | --- | --- | --- | --- |
| L2 `L2_gear` (today) | none | 22/30 | 0.833 | 5,678 |
| L2 `base_gear_L2` | none | 18/30 | 0.744 | 6,209 |
| **L2 baseline, pooled** | | **40/60 (67%)** | **0.789** | |
| L2 `gear_servo_L2` | **fix 3** | 28/30 | 0.967 | 5,923 |
| L2 `gear_servo_L2_b` | **fix 3** | 26/30 | 0.933 | 5,623 |
| L2 `final_gear_L2` | **fix 3**, with the SAM 3 cap | 26/30 | 0.933 | 5,722 |
| **L2 fix 3, pooled** | | **80/90 (89%)** | **0.944** | |
| L2 bent boards, `live_noservo_gear_L2` | SAM 3 cap, bends; fix 3 reversed | 15/30 | 0.767 | 5,976 |
| L2 bent boards, `live_gear_L2` | SAM 3 cap, bends, **fix 3** (the live tree) | 22/30 | 0.889 | 6,083 |
| L1, six runs today | none | 157/180 (87%) | 0.933 | about 3,900 |
| L1 `gear_servo_L1` | **fix 3** | 29/30 | 0.983 | 4,011 |
| L1 `final_gear_L1` | **fix 3**, with the SAM 3 cap | 28/30 | 0.983 | 4,218 |
| **L1 fix 3, pooled** | | **57/60 (95%)** | **0.983** | |

Significance:
- L2 success: Fisher p = 0.001.
- L2 mean reward: Welch p = 0.001.
- L1 success: Fisher p = 0.15, so not significant on its own. L1 mean reward: Welch p = 0.005.

**Read the two L2 scenes separately.** Everything down to the pooled row ran on the straight boards VABAR exported. The
L2 documents state bent chains and the live tree now builds them, which is harder: with fix 3 the cell falls from 89% to
73%. The head-to-head on that scene is one run each (success p = 0.11, mean reward p = 0.06), and the README's own
runs on it, 21/30 and 23/30, agree with `live_gear_L2`. The paper's L2 rate is 90.0%. Fix 3 reaches that only on the
straight boards, not on the bent ones that `layout.py` says the published numbers used.

Slowing the tool did not cost steps on L2 (a mean of 5,773 against 5,944 in the runs without the cap). L1 is 3-8% up.

The SAM 3 cap (f4bd50b6) landed in the live `sight.py` during this work, and its measurement belongs to another
note. The two `final_*` runs are fix 3 on top of it, from the copy that matches the live tree plus the patches, with 8
workers alone on a GPU: L2 in 12 min, L1 in 9 min. The two changes combine with no sign of interaction.


## Tested and not ported

### Syringe packing: the rewritten single-target graph (AUTOLab `syringe_packing` 028911d, `graphs/gap_perception`)

This is the same commit range as fix 2, keeping the old graph's shape with these additions:
- c44b813's SAM-3 checks: two-view "medical syringe" counting with crop recovery, and `uncertain` instead of `none`;
- 2211d9a's checked grasp and two-stage checked lift;
- 5cd4a73's median grasp height;
- a seat check before release.

It fixes the counting: none of its episodes ended "graph finished" with a syringe left. But it does not get the second
syringe in. Its first five L1 episodes (t00-t04) all posted one of two (reward 0.5), each aborting on the second pick:
- three times `grasp_sg.execute`: "trajectory tracking failed: 11.6-13.5 mm; release prohibited";
- once `carry_plan_sg`: "no wrist-only turn stands the syringe up clear of the tray with joint room".

v2 contains every one of these checks and adds what the second syringe needed, so the run was stopped (see
Incidents) and the graph not ported.

### Gear meshing: the wrist roll held (`(:embodiment panda_robotiq_roll)`, RoboSimStudio-dev 68690b04 / 78120209)

On the pinned runtime, `panda_robotiq` leaves the wrist roll free; `panda_robotiq_roll` is the pairing that holds it.
The gear graph was written when `panda_robotiq` held it. Restoring that alone scored 18/30 on L2 with mean reward 0.644,
against 40/60 and 0.789 for the two baselines, and it lost 8 instances the first baseline solved. No gain, so it was not
kept. It was not run with fix 3, which reaches the paper's rate without it, and the cause was not investigated.

### Tool sorting: servo speed 0.12 m/s (cb96425e's `tool_sort.suite` line)

On top of fix 1 it scored 9/30 (0.682), against 35/90 (0.742) for fix 1 alone.
- Grasps converted to transports at the same rate (99/243, 41%, against 42%).
- The slower legs ran 20 of 30 episodes into `fast`'s 14,000-step budget, against 17.

### Syringe packing: servo speed 0.12 m/s (cb96425e's `sharps_disposal.suite` line)

VABAR's syringe suite has no `.suite`, so the speed went into `task.py`'s spec (`servo_speed_mps=0.12`). On top of fix 2
it scored 7/30 (0.583), against 22/60 (0.650) for fix 2 alone: no gain, so it was not kept. v2 plans most of its motion
with cuRobo; its servo moves are the grasp correction, the unjam push, the turn's final pose and the retract fallback.

### Tool hanging: the scissors axis and screwdriver tip fixes (AUTOLab d7a29d0, ddda198)

**Patch:** `optional/tool_hanging_scissors_screwdriver_axis.patch`. Not counted as an accepted fix.

- d7a29d0 picks the sign of the scissors' PCA grasp axis that agrees with the registered CAD cross-axis. The commit's
  CAD-path fallback is a no-op here.
- ddda198 searches the screwdriver's distal tip on measured depth only, not on the cloud completed down to the support.

| Run | Change | Success | Mean reward | pliers | scissors | screwdrivers | wrenches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 `L2_hang` (today) | none | 24/30 | 0.867 | 4/7 | 8/8 | 6/7 | 6/8 |
| L2 `base_th_L2` | none | 25/30 | 0.900 | 5/7 | 8/8 | 6/7 | 6/8 |
| L2 `th_lgulich_L2` | both | 26/30 | 0.917 | 5/7 | 8/8 | 6/7 | 7/8 |
| L2 `th_lgulich_L2_b` | both | 26/30 | 0.933 | 4/7 | 8/8 | 7/7 | 7/8 |

Pooled, 52/60 against 49/60 (Fisher p = 0.62). The two tools it touches moved by one episode (scissors were already
16/16); the wrench gain is in code these commits do not touch, so it is noise. L1 scored 30/30 with both
(`th_lgulich_L1`), so there is no sign of harm. These are correct, low-risk fixes with no measurable effect on the cell.

Tool hanging L2 without any of this is 49/60 (82%) by instance and 106/120 (88%) by tool, against the paper's 81.7%.

### Tool hanging: servo speed 0.12 m/s (cb96425e's `tool_hanging.suite` line)

L2 scored 25/30 (0.917), against 49/60 (0.883) for the baselines: no gain in success. It cost 6,474 steps an episode
against about 4,560, 42% more. Not kept.

### Cable insertion: servo speed 0.12 m/s (cb96425e's `usbc_insert.suite` staging)

Not run. Two surveys disagree on whether it applies. The RoboSimStudio-dev archaeology notes say the `mate` graph
streams trajectories, so the servo speed does not reach it. The survey of these three suites says the cable cells run
the servo for every move. The run was skipped anyway: the cell's perception fix was tuned at today's speed, the
documents' 9,000-step budgets leave little room for slower legs, and L2 already scores above the paper (21/30 against
30%).


## Scene-level leads

These change what the benchmark stages: colliders, initial states, documents or what the cameras see. They are kept
apart from the graph fixes above. The only one tested, the bead removal, has its own patch under `patches/scene/`, and
it is not recommended.

### Syringe packing: the syringe and the tray collide as drawn (RoboSimStudio-dev 18ab35f6, with 5c561ad9 and d8a46cbe)

**How VABAR's scene differs from dev's:**
- **Syringe collider.** `syringe_0` collides as the export's authored 13.6 × 23.6 × 150 mm box around a 10.7 mm barrel.
  The spare `syringe_pack_*` collides as the mesh's bounding box, with friction 3.0.
- **Tray.** The tray's drawn walls do not collide. The floor slab sits 6 mm high and is 4 mm thick; dev's runs pushed a
  syringe 11 mm through it.
- **Start height.** Syringes start at z 0.795825 against dev's 0.789825.
- **Level 2 layouts.** 22 syringes in 20 of the 30 `tray_clutter` documents straddle the drawn walls. Dev removed
  them and re-derived each document's goal and step budget.

**What dev changed:** 61 CoACD hulls for the syringe, four wall colliders, and a 20 mm floor slab at the drawn floor;
re-derived layouts and documents; `obb_grasp` kept on the syringes (5c561ad9); and `njmax` about 3,209 for the hulls
(d8a46cbe's `vab_sharps_union@v1`).

**Evidence from these runs, which is suggestive only.** With fix 2, L1's remaining failures are dominated by the
spare: 22 of 38 are the hand check after lifting `syringe_pack_0`, and the spare's collider differs from what the
cameras see. On L2, 10 of 30 episodes end with a grasp descent stopped 18-19 mm short in the crowded tray. Neither was
traced to the collider.

**What porting would take:**
- VABAR's `task.py` was restructured for upstream RoboSimStudio, so AUTOLab 028911d's `task.py` diff does not apply;
  the collider and tray changes have to be ported by hand.
- 61 hull meshes.
- The 30 + 30 L1/L2 start-pose files.
- 20 Level-2 documents.
- A new physics arrangement for `njmax`.
- Re-freezing `benchmark/syringe_packing`.

Not attempted here. The L2 documents change, so it would be a new benchmark version, and the result would have to be
reported against the new scene, not today's.

### Tool hanging: the beads on the Level-2 scissors distractor (RoboSimStudio-dev fdbf4d11)

**How VABAR's scene differs from dev's.** `scissor_pair_1` draws 50 non-colliding ring beads marking its scored feature
(`visual_markers`), and the Level-2 documents use that record for their scissors distractor. So:
- `distractor_scissors_*` in `pliers_level2`, `wrenches_level2` and `screwdrivers_level2` carry 50 beads each;
- `scissors_level2` has no scissors distractor.

Dev made the distractor a marker-less copy: "painted on a tool nothing is asking about hands the policy the answer for
the tool it should ignore".

**Today's L2 failures looked as if they fit it.** All 6 failures in `L2_hang` are in the three documents with a beaded
distractor: pliers 3 of 7, wrenches 2 of 8, screwdrivers 1 of 7, scissors 0 of 8.

**Tested, scene change only.** `"visual_markers": []` on the `distractor_scissors_*` entries of the three
`poses/*_level2_start.json` files (`scene/tool_hanging_distractor_beads.patch`). The beads are visual: no collider and
no goal, so what changes is what the cameras see.

Result: 22/30 (0.867), pliers 3/7, scissors 7/8, screwdrivers 6/7, wrenches 6/8, against 49/60 (0.883) for the two
baselines. The failures stay on the same tools, so the beads do not explain them. Dev's reason for removing the beads
still stands as a matter of scene hygiene, but it is not a score fix, and it is not recommended on these numbers.

### Tool sorting Level 2: what it would take

The heap tier exists in RoboSimStudio-dev `tool-sorting` 99179e6a and `release/oss` 7ffc4017 (meadowshen, 2026-09-10,
"the heap tier arrives, already clear of the thirteen assets"). That commit is 137 files:
- 30 `level2_N.task` documents;
- their exported scenes (`scene.xml`, `scene_norobot.xml`, `scene_meta.json`);
- the label textures.

`release/oss` adds the `(:export ... scene_norobot.xml)` fix; without it the Robotiq profile welds a second arm into
the cell. AUTOLab `tool-sorting` fc83e27 carries the same documents: of those compared (`level2_1`, `_2`, `_9`, `_17`),
each differs only by that export and by two `(:graph ...)` lines declaring `gt` and `gap_perception_loop`.

Porting it into VABAR means:
- the documents and scenes;
- the suite's `(:variants)`;
- a frozen copy under `benchmark/tool_sort/level2`;
- the `tool_sorting/L2` cell's instance list in `vabar/benchmark.py`, which is empty today.

Adding instances is a benchmark decision, not a fix. Not ported, and fix 1 was not measured on them.

## Not tested, ranked

Ranked by expected effect on today's cells. None of these has a measured rate on VABAR's scene.

1. **Syringe packing: the collider and tray port** (18ab35f6 and its companions). The largest possible change to both
   syringe cells, and the only one that changes the scene. See "Scene-level leads".
2. **Syringe packing: retry a failed grasp instead of aborting.** This is not in either repository. 22 of v2's 38 L1
   failures are the post-lift check on the spare syringe, and each ends the episode at 0.5. Routing
   `hand_map` → `not_found` back to `target`, with an attempt cap, would turn those into retries. On L2, the 10
   short-descent aborts are the same shape.
3. **Tool hanging: the pliers registration first pass** (open-robot-skills 113261b, `_first_pass_loop`). It tries masks
   best-first and keeps the first within the carried bound. Pliers are 5 of the two baselines' 11 L2 failures. Not
   mechanical: VABAR's `register_held.py` inlines the circle fit that the bundle factored out.
4. **Tool hanging L2: loop back to `pair_2` after the second tool** (from RoboSimStudio-dev 145b9e07's loop). It only
   adds a retry when a released tool falls back into the source. Check that `completed_source_ids` is wired first.
5. **Tool hanging: support-disengagement margin 0.05 m** (a9d680d9 raised the runtime default from 0.04). Port it as a
   graph input on `disengage_sg*.plan`.
6. **Tool sorting Level 2.** See "Scene-level leads": adding it is a benchmark decision.

Porting instructions for 3-5 are in
`/home/shadeform/speed_ws/agent_scratch/devmerge/notes/three_suites.md`.


## Already in VABAR

Checked by content, not by merge status:

- **AUTOLab `codex/perception-eval-fixes` 6fe4ae7** (whole syringe chosen by score once a red marker identifies it):
  VABAR's `syringe_packing/graphs/gap_perception/scripts/perceive_target.py` is byte-identical to main's, which has it.
  Fix 2 below replaces that graph; the replacement selects syringes its own way (every syringe on the tray is a goal,
  and a red cap off a mask's axis vetoes it).
- **AUTOLab `tool-hanging-sk2-generic`** (bce011e, 82e414c): the `generic` relabelling of the tool-hanging and
  syringe subgraphs. VABAR made the same change, with the library skill kept under `meta.composes`.
- **AUTOLab `privileged-opt-in`, `gap-perception`, `tool-level-easy-medium`, `yam-station-cameras`,
  `codex/robot-adapter-boundary`, `upstream-gap-pin`**: ancestors of main for these suites. Their own commits are
  runtime adapters, cameras and dependency pins.
- **Tool hanging**: both reference graphs are RoboSimStudio-dev `gap_perception` (Xinyue Zhu, a9d680d9) byte for byte.
  Its goal speed threshold change (82178a1b) is scoring and is already in every VABAR document. RoboSimStudio-dev main
  re-expresses the same graph over open-robot-skills bundles (145b9e07), designed to behave the same.
- **Gear meshing**: the graph is RoboSimStudio-dev `feat/gear-mesh-levels` aa42d099 verbatim. 5ce70c64 (board memory
  per execution, not per process) is unnecessary on this runtime: `forget_graph_modules` re-imports a graph's helper
  modules at the start of every episode (`robosimstudio_policy_agent/graph_solution.py:398`).
- **Cable insertion**: VABAR's `mate` graph (AUTOLab bf985d0 = RoboSimStudio-dev `integrate/tool-hanging-levels`
  eeb2aa88, plus this tree's own perception fixes) is the newest in either repository. RoboSimStudio-dev main's copy
  is older (no `hold_sg.lift`, an older `relook_vision`). fdb64fdb's `bore_cv` fixes are in VABAR's rewritten
  `bore_cv.py`.
- **Tool sorting**: `gap_perception_loop`'s `discover_regions.py` and `hover_aside.py` equal RoboSimStudio-dev main's.
  `gap-perception-level1`'s loop graph equals VABAR's or is older.


## Branch by branch

### Syringe packing and tool sorting

#### AUTOLab-GaP-NVIDIA-Collaboration

| Branch | Tip | Syringe / tool-sort content against VABAR |
| --- | --- | --- |
| `main` | f6057d7, 10-26 | VABAR's source. `syringe_packing` is the 2026-09-10 port (a8f38a3, = RoboSimStudio-dev 13b4322) plus 6fe4ae7. `tool_sort/gap_perception_loop` is Jaimyn Drake's 8b934e1 (2026-09-10), byte-identical to RoboSimStudio-dev 5ec6e601. |
| `vabar-robosimstudio` | 18b974e, 10-29 | = VABAR v1.0 export. |
| `lgulich/tool-sort-loop-failure-fixes` | dca8679, 10-27 | **Missing. Ported (fix 1).** A rebase of `tool-sorting`'s 69c8fcb onto main, graph lines only. |
| `tool-sorting` | 69c8fcb, 09-12 | Its `gap_perception_loop` = RoboSimStudio-dev 067645c6: the core of fix 1, with the label prompt and the bin filter. Its 30 `level2_*` documents and scenes are instances (out of scope: the Level-2 cell is not frozen). |
| `syringe_packing` | 72560b0, 09-14 | **Missing. Ported (fix 2).** Re-port of RoboSimStudio-dev `sharps/port-v2-perception-checks` (7049239): `graphs/gap_perception` (rewritten checks), `graphs/gap_perception_v2`. The same commit also changes physics and instances (CoACD syringe hulls, tray wall and floor colliders, 20 Level-2 layouts, `RenderSpec()` look); none of that is ported. |
| `tool-hanging-sk2-generic` | 82e414c, 11-02 | 82e414c relabels the syringe graph's subgraphs `generic`; VABAR already did the same (with `meta.composes`). Fully in. |
| `codex/perception-eval-fixes` | 6fe4ae7, 10-06 | "Select a whole syringe using confidence after red-marker identification": in VABAR (`perceive_target.py` = main). Fully in. |
| `codex/robot-adapter-boundary`, `privileged-opt-in`, `gap-perception`, `tool-level-easy-medium`, `yam-station-cameras`, `upstream-gap-pin` | 09-10 to 10-19 | Ancestors of main for these two suites; their own changes are runtime, adapters or dependency pins. Fully in. |
| `lgulich/screwdriver-*`, `lgulich/scissors-*`, `lgulich/cable-port-floor`, `cable-*`, `isaac_gap`, `tool-hanging`, `vab-industrial-deliverable` | | Do not touch these two suites. |

#### RoboSimStudio-dev

| Branch | Tip | Syringe / tool-sort content against VABAR |
| --- | --- | --- |
| `syringe-tray-pair-graph` | 13b4322, 09-10 | The `sharps_disposal` suite VABAR's `syringe_packing` was copied from. Fully in. |
| `sharps/port-v2-perception-checks` | 7049239, 09-14 | = autolab `syringe_packing` (fix 2). c44b813 (SAM-3 checks), 2211d9a (checked grasp and lift), 5cd4a73 (median grasp height), 473d7f9 (`gap_perception_v2`). f55eee5 (colliders, layouts) is physics/instances. |
| `vabar/held-back-suite` | dd57aae, 09-16 | = ee4ad93 on main: v2's fingertip-drop table applies only to a geared hand. VABAR's syringe cell runs the prismatic Franka hand, so this is part of fix 2. |
| `main` | 1cd9f76a, 10-05 | Syringe: everything above, plus 18ab35f/5c561ad (collision, grasp box: physics), 114329f (the `gt` graph's posting fix: not a reference graph), d8a46cb/55f7a89 (fleet bases). `gap_perception` there = autolab `syringe_packing`'s; `gap_perception_v2` = that + ee4ad93. Tool sort: `gap_perception_loop` = fix 1, as shipped; the two grasp scripts moved into open-robot-skills bundles (60b23556), with fix 1's grasp edits forwarded there. cb96425e: `(:servo-speed-mps 0.12)` in every suite (tested on syringe and tool sort, not kept). |
| `tool_perception` | 067645c6, 09-13 | Jaimyn Drake, "Fixing the graph, add level2": the graph half is fix 1 (+ label prompt, bin filter). Merged to main as 48fcd8a5, "the grasp and loop fixes behind the 85-90% level-1 run". |
| `integrate/syringe` | 48fcd8a5, 09-13 | That merge. In main. |
| `gap-perception-level1` | d0fdd2c7, 09-10 | meadowshen's Level-1 work: its loop graph = VABAR's (select_pair, grasp scripts) or older (discover_regions, hover_aside); the rest is Level-1 assets/documents (instances). Nothing to port. |
| `tool-sorting` | 99179e6a, 09-10 | meadowshen's tier and asset work (instances, assets). Nothing to port. |
| `paper/instrumentation` | 8dc74824, 09-14 | Instrumentation (`sim_step` tags, step wall, scoring harness). Out of scope. |
| `vis/rtx`, `vis/show_asset`, `traj_example` | 09-14 to 09-25 | Backdrop textures, asset catalogue, showcase. Out of scope. |
| `release/oss`, `release/oss-bench`, `release/oss-r2s` | 10-10 to 10-12 | Same suite content as main (renames only). |
| everything else | | Does not touch these two suites' reference graphs. |

### Tool hanging, gear meshing and cable insertion

What each reference graph runs: tool hanging L1 `graphs/gap_perception`, L2 `graphs/gap_perception_level2` (whose
scripts are `../gap_perception/scripts/*`), gear meshing `graphs/gap_perception`, cable insertion `graphs/mate`. All
four resolve every script graph-locally, so no skill bundle and nothing in `tasks/_shared` affects these cells.

| Repository / branch | Tip | Content against VABAR |
| --- | --- | --- |
| AUTOLab `lgulich/scissors-grasp-axis-and-cad-paths` | d7a29d0, 10-26 | **Missing.** Picks the sign of the scissors' PCA grasp axis that agrees with the registered CAD cross-axis, so the wrist does not flip 180 degrees between looks; the CAD-path fallback is a no-op here. No rates in the commit. Tested: within noise, optional patch. |
| AUTOLab `lgulich/screwdriver-tip-from-measured-depth` | ddda198, 10-26 | **Missing.** The screwdriver's distal tip is searched on measured depth points only, not on the cloud completed down to the support. No rates. Tested: within noise, optional patch. |
| AUTOLab `tool-hanging-sk2-generic` | 82e414c, 11-02 | Labels only; in (see above). |
| AUTOLab `tool-hanging`, `tool-level-easy-medium`, `gap-perception`, `yam-station-cameras`, `cable-tasks`, `privileged-opt-in`, `codex/*` | | Ancestors of main; superseded or in. |
| RoboSimStudio-dev `feat/gear-mesh-levels` | aa42d099, 09-10 | Gear graph = VABAR's. On that branch `panda_robotiq` honoured the wrist roll (78120209); on the pinned runtime it does not, and the roll-honouring pairing is `panda_robotiq_roll`. Tested, not kept. |
| RoboSimStudio-dev `main` | 1cd9f76a, 10-05 | cb96425e: every suite declares `(:servo-speed-mps 0.12)`, the tool speed the graphs were written at, and gear meshing's contact windows are restated in 60 Hz steps (fix 3 for gear meshing; tested and not kept on tool hanging). f4bd50b6: gear perception asks SAM 3 for 48 masks, not 200 (wall time only). 5ce70c64: gear board memory per execution (unneeded, see above). 145b9e07: the tool-hanging graph over bundles, behaviour-equivalent; its Level-2 loop back to `pair` after the second tool (not tested). Its `mate` graph is older than VABAR's. |
| RoboSimStudio-dev `gap_perception` | a9d680d9, 09-30 | Tool-hanging graph = VABAR's. Raises the support-disengagement margin 0.04 to 0.05 m in the runtime (not tested). |
| RoboSimStudio-dev `integrate/tool-hanging-levels` | eeb2aa88, 09-13 | `mate` = VABAR's; nothing new. |
| RoboSimStudio-dev `fleet/gear-mesh-easy-per-world`, `az/force_feedback`, `paper/instrumentation`, cb96425e's usbc entity/jaws/staging | | Fleet, physics or instrumentation. Out of scope. |
| RoboSimStudio-dev `cable_tasks`, `cable-tasks-rtx`, `integrate/cable-gap`, `integrate/cable-rtx`, `vis-fix`, `integrate/vis-fix` | 09-04 to 09-12 | Older `mate` graphs, superseded by VABAR's. |
| RoboSimStudio-dev Kaiyuan-line tool-hanging graph (`cad_refine`, `target_landmarks`, ...) on `level1-rebuild`, `live-demo`, `tool_perception`, `refactor-*` and others | 08-31 to 09-13 | Deleted on main in favour of Xinyue Zhu's graph, which VABAR has. No rates favouring it. |
| open-robot-skills 113261b (`agentic-r2s2r`, not on main) | 09-16 | `register_held` first pass tries masks best-first within the carried bound; VABAR's tool-hanging copy reaches that path for pliers. Not mechanical to port (not tested). |

The full per-branch survey of these three suites, with the commands that reproduce it, is
`/home/shadeform/speed_ws/agent_scratch/devmerge/notes/three_suites.md`.


## Incidents

- **The live tree moved during this work.** At 19:23-20:49 the live VABAR gained the SAM 3 cap, fix 3 and the bent boards
  (not from this work). `ts_*` and `syr_*` are unaffected: those suites' files did not change. The gear runs named above
  say which scene and which cap they had.
- **RTX scene-build stall (the known one).** In `ts_loopfix_a`, `tool_sort/level1_26` waited 31 min in
  `StreamingStatusManager : Streaming client gpu.resourceManager timed out` before its episode started. The episode then
  ran normally; only the cell's wall time is affected.
- **Stopping a bench with SIGINT kills its shared SAM 3 server, not the bench.** `syr_v1dev_L1` was interrupted after
  its first five episodes. The workers carried on with no `sam3` server and failed every remaining episode at their
  first perception node, then hung at exit until terminated, leaving the private MPS daemon to be quit by hand. Its
  record keeps only the five valid episodes. Let a bench finish, or start it with `--limit`.

