# Syringe packing: perception GaP graph (v2)

> **In VABAR.** This is the `gap_perception` reference graph of the `syringe_packing/L1` and `/L2` cells. It is
> RoboSimStudio-dev's `syringe_packing/gap_perception_v2` (origin/main 1cd9f76a), which the VABAR v1.0 export missed;
> the 2026-09-10 graph it replaces posted one syringe of two (0/30 on Level 1). Two changes from the source: the
> fingertip-drop fix ee4ad93 is included (this cell runs the prismatic Franka hand, not the Robotiq), and `carry_sg` is
> labelled `generic` with its skill under `meta.composes`, for graph-as-policy's SK2 contract. The README below is the
> source's. Its sweep numbers were measured on RoboSimStudio-dev's tray colliders, CoACD syringe hulls and re-drawn
> Level-2 layouts, which VABAR does not stage; VABAR's own numbers are in `docs/validation/dev_history_merge.md`.

This graph packs every syringe visible on the tray. It detects the box aperture
first, then loops over fresh RGB-D target perception, collision geometry, a
checked grasp and lift, held-object registration, collision-aware transport,
staged insertion, release, and post-release verification. The loop ends only
after two calibrated views find no remaining tray syringe; benchmark success
independently requires every task syringe to satisfy its containment goal.

The grasp station is anchored to the detected red cap and the measured syringe
centre of mass. The held-object transform remains rigid after grasp, while a
post-lift multi-view observation verifies that the syringe actually moved with
the Robotiq hand. Insertion uses straight, orientation-locked 10 mm phases down
to the aperture plane. A seated check must pass before the graph opens the
gripper and retracts.

The graph uses no `sim.*` ground-truth tools.

## Running it

From the RoboSimStudio repository root, on `syringe_packing/tray_clutter_tNN`
(30 cluttered-tray trials, 4-8 syringes each) or `syringe_packing/tray_pair_both`:

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
uv run --extra gap --extra curobo --extra rtx python tools/bench_run.py \
  --task syringe_packing/tray_clutter_t03 \
  --scene robosuite_bench --embodiment panda_robotiq \
  --policy graph --graph tasks/syringe_packing/graphs/gap_perception_v2 --allow-undeclared-graph \
  --bus-timeout 300 --motion curobo --renderer tiled --device cuda \
  --episodes 0:1 --max-steps 60000 \
  --record --out outputs/tray_clutter_t03_v2
```

* `--motion curobo` is required. The insertion, the wrist-turn search and the
  retract all plan with cuRobo (`motion.plan_to_pose` / `motion.plan_linear`).
  Only `motion.plan_joint` goes to the Newton IK member.
* `--max-steps 60000`. An 8-syringe trial needs about 3,000 control steps per syringe.
* With `--renderer rtx`, select the GPU with `--device cuda:N` and leave
  `CUDA_VISIBLE_DEVICES` unset: OVRTX enumerates no GPU when it is set.
* The trace (`outputs/<run>/*/traces`) dumps every camera array a tool call
  sees, 3-6 GB per trial. On a shared disk, delete `traces/**/assets` after a
  run. The JSON requests and responses, `steps.npz` and the videos are enough
  to diagnose a failure.

### Environment this was run with

* `uv sync --extra gap --extra curobo --extra rtx --inexact`, then
  `torchvision==0.24.1` (matching the venv's torch 2.9.1+cu128).
* `graph-as-policy` and `open-robot-skills` cloned from github.com/graph-robots
  as siblings of the repository. The `geometry` bundle is installed with
  `gap skills install`.
* `sam3.segment_text` is served by RoboSimStudio's in-process SAM3
  (`facebook/sam3`, cached under `~/.cache/huggingface`). The upstream `sam3`
  bundle pins torch 2.12+cu130, which needs NVIDIA driver >= 580. On an older
  driver, hide that bundle (e.g. rename `open-robot-skills/tools/sam3`) so the
  in-process one is used.
* The graph declares only the tools it calls (`requires.tools` in
  `workflow.json`): `geometry.*` and `sam3.segment_text`. Its subgraphs use the
  `generic` skill, so no cuRobo, Grounding-DINO or VLM bundle is started.

## What the scripts do beyond the canonical skills

Each change below was driven by a recorded failure; the scripts' docstrings carry
the measurements.

| Stage | Script | Behaviour |
| --- | --- | --- |
| Target | `perceive_target.py` | Keeps only syringe-sized masks: at least 500 px, 80-200 mm long, at most 45 mm wide. That rejects cap slivers, lines on the bench and masks merging neighbours. A mask whose red pixels lie off its own axis (e.g. a strip along the tray wall beside a cap) goes last. The rest are ranked by room across the jaw, read from depth: the nearest surface standing 4 mm proud of the floor, neighbour barrel or tray wall, within +/-35 mm of the station. Then SAM's score. |
| Needle end | `perceive_target.py` | The needle direction comes from red cap pixels on the target's axis, then a cap detection on its axis, then the 23.6 mm flange's position along the barrel, and only then the narrower tail. A syringe posted plunger-first stands on its thumb press under the hole and blocks the next one. |
| Grasp | `compute_grasp.py`, `grasp_checked.py` | Descends with the jaw pre-closed. The width comes from the measured clearance, 16.7-30 mm, against the open Robotiq's +/-49 mm fingertip band. The TCP rises by the measured fingertip drop for that width. |
| Grasp check | `verify_grasp.py` | Considers every returned mask from the wrist and overhead cameras. With 6+ syringes in the tray, the held one is not among SAM's top 8. |
| Carry / turn | `plan_clearance_motion.py`, `execute_reorientation.py` | Plans the carry and the wrist yaw together before moving, over a 30 deg yaw grid. First: carry in the pick orientation, then turn in place, probed from the carry's planned end joints. If no yaw keeps 0.15 rad of joint margin: one free move carrying the syringe already upright. As a last resort: carry, then free turn. Each candidate is probed end to end, through the three insertion strokes and the 160 mm retract (`motion.plan_linear` with `seed_joints`). The graph executes the candidate with the largest margin over every planned waypoint, and turns play at 4x. The old smallest-turn rule left the stroke within 2-6 mrad of a joint limit. |
| Insert | `plan_visual_insert.py` | Aims with the hand map's registered tip offset, which was within 8 mm of the simulator's mesh end. A perceived leading end replaces it only when the hand map is implausible, and must lie on the barrel axis 45-110 mm ahead of the TCP. The side view under-reads by 15-33 mm. |
| Seat | `verify_seated.py` | Depth and centring come from TCP proprioception and the hand-map offset. Cameras cannot see below the rim, and the Robotiq fingers hide the barrel's top. Release is allowed up to 12 mm short of the rim plane. The cameras only veto a tilted syringe, measured from on-axis detection points above the rim; a syringe already leaning in the box under the hole can't trigger it. |
| Release | `release_and_retract.py` | Retracts with a cuRobo line (160 mm, then 80 mm) instead of the Cartesian servo. |
| Unjam | `unjam_aperture.py` (`unjam` → `verify_again`) | When the post-release check finds a syringe still above the rim, the graph tries a recovery before aborting. The syringe has usually landed standing on the syringes already under the hole. The closed gripper comes top-down over the hole, pushes the fingertips 25 mm below the rim, and sweeps +/-10 mm along the box's long axis. Then the check runs again, and only a second `blocked` aborts. |

`tools/sharps_placement_pretest.py` scores tray and box placements, and every
wrist yaw, with the same strict cuRobo IK. It is how the joint-limit problem of
the old yaw rule was found.

## Sweeping the 30 trials

```bash
for t in $(seq -w 0 29); do
  CUDA_VISIBLE_DEVICES=$((t % 7)) HF_HUB_OFFLINE=1 uv run --extra gap --extra curobo --extra rtx \
    python tools/bench_run.py --task syringe_packing/tray_clutter_t$t ... --out outputs/tray_clutter_t${t}_v2 &
done
```

Run results are summarised in the section below.

## Results

**Current graph** (sweep "s8", 2026-09-13; includes the three changes listed at
the end of this section):

| | |
| --- | --- |
| Trials fully solved | **16 / 30 (53%)**: t00, t03, t06, t07, t11, t13, t15, t16, t17, t18, t19, t20, t22, t26, t27, t28 |
| Syringes in the box | **152 / 176 (86%)** |
| 4-syringe trials solved | 4 / 5 (t03, t16, t17, t26) |

s8 failures:

| Count | Mode | Trials |
| --- | --- | --- |
| 8 | Box jam: a later syringe is still `blocked` after the unjam push | t01, t05, t08, t09, t10, t24, t25, t29 |
| 4 | Grasp stopped short, in tight corners | t02, t12, t21, t23 |
| 2 | Syringe slipped in or fell from the fingers before release; the seat check correctly refused | t04, t14 |

Ground truth at each jam shows only three are pile-ups under the hole (t01, t25,
t29). Three stuck syringes are standing on one or two leaning syringes (t05,
t10, t24). In two, the syringe ended lying flat on top of the lid (t08, t09).

Earlier sweep "s7", before those three changes:

| | |
| --- | --- |
| Trials fully solved | **13 / 30 (43%)**: t00, t03, t06, t13, t14, t15, t16, t17, t19, t20, t22, t26, t27 |
| Syringes in the box | **137 / 176 (78%)** |
| 4-syringe trials solved | 4 / 5 (t03, t16, t17, t26; t24 posted 3/4) |
| Largest trays solved | 7 syringes (t00, t22) |

The previous full sweep (s6), before the unjam recovery, the half-turn grasp and
the tilt-check fix, solved 8 / 30 (27%) with 64% of syringes posted.

Failures in s7:

| Count | Mode | Trials |
| --- | --- | --- |
| 12 | Box jam: a late syringe stays standing above the rim on the ones already inside, still `blocked` after the unjam push | t01, t04, t05, t07, t08, t12, t18, t21, t23, t24, t28, t29 |
| 3 | Grasp descent stopped short, in tight corners with 18-20 mm jaw clearance (t02, t11) or at a false target in the box's shadow (t09) | t02, t09, t11 |
| 2 | Seat check refused a syringe that really slipped in the fingers during a large free turn (t10), or a tilt reading (t25) | t10, t25 |

The box cavity (89 x 166 x 172 mm) holds about five syringes lying flat
(VAB-industrial's own measurement). Trays of 6-8 syringes fill it, and the posted
syringes pile up under the aperture, so the jam is partly a capacity limit.
Several of the jams were marginal: tops only +3-4 mm above the rim, with the
syringe's centre counted inside the box. Three changes made after s7 are
measured in sweep "s8":

* `blocked` requires points more than 10 mm above the rim.
* Candidates must stand at least 5 mm proud of their local floor.
* Reconfiguring turns play at 6x instead of 4x.
