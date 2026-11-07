# cable_routing/L2: why the reference graph rarely succeeded, and the fix

Cell `cable_routing/L2` = `cable_route/port_medium` (four pegs, 840 mm rod, 24,000-step
budget), graph `gt` (= `graphs/weave`, the graph L1 runs), `--config fast`, cell protocol
(`--no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps 60`,
palette segmentation). The same stack as [cable_routing_fix](cable_routing_fix.md) and
[cable_fleet](cable_fleet.md). Measured 2026-11-05 on RTX PRO 6000 GPUs 0, 1 and 5 of a
shared host.

## Result

| Run (`--config fast`, 30 instances) | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end | Wall |
| --- | --- | --- | --- | --- |
| As found, fleet 15 x 2 (two runs) | **2/30, 1/30** | 0.656, 0.661 | 20-22 / 9-11 / 23-25 / 30 / 30 / 3-4 | 20.8, 23.1 min |
| As found, single world, 6 MPS workers (stopped at 29) | **2/29** | 0.644 | 23 / 6 / 24 / 29 / 28 / 2 | -- |
| **Fixed, fleet 15 x 2 (three runs)** | **18/30, 14/30, 13/30** | 0.878, 0.856, 0.889 | 28-29 / 24-29 / 27-30 / 26-29 / 23-26 / 20-23 | 45.8, 34.1, 35.6 min |
| **Fixed, single world, 10 MPS workers** | **15/30** (50 %, CI 33-67 %) | 0.883 | 27 / 27 / 29 / 29 / 25 / 22 | 57.0 min |
| *L1 with the same code, fleet 15 x 2 (two pegs)* | 19/30 | 0.850 | 28 / 28 / -- / -- / 23 / 23 | 20.3 min |
| **Fixed + 0.12 m/s declared, fleet 15 x 2, budget 14,000** | **21/30** | 0.922 | 29 / 29 / 27 / 30 / 29 / 24 | 32.6 min |
| **Fixed + 0.12 m/s declared, single world, 10 workers, budget 14,000** | **21/30** | 0.922 | 30 / 28 / 29 / 29 / 27 / 23 | 53.5 min |
| *L1 with the 0.12 change (budgets only), fleet 15 x 2* | 16/30 | 0.825 | 28 / 26 / -- / -- / 21 / 24 | 21.5 min |

* **Two faults, both in the graph, neither in the task, its scoring or its physics.** The
  crossing's fully open jaws came down on the *next* spool (L2's pegs are 72-80 mm apart), and
  the Medium-only terminal branches let go of the end on top of the port and never corrected
  it. A third change re-enables the seat-repair pass that Medium had switched off.
* Over 120 fixed episodes (four cells) the graph succeeds 60 times, **50 %**, against
  5 of 89 as found and the paper's 46.7 %. Fleet and single world agree: per instance two fleet
  cells agree on success as often (12-19 of 30) as a fleet cell and the single-world cell do
  (13-16 of 30).
* L1's crossings are unchanged by construction (no L1 take comes near a peg); only the lower
  port floor reaches it. It measured 19/30, inside its earlier 16-20/30.
* The fix costs time: the repair pass lengthens the longest episodes to 15,000-19,500 steps,
  so a cell takes 34-46 min as fleet waves (was 21-23).
* **Then the servo speed.** The cable suites declared none, so `go_to_pose` ran at 0.31 m/s
  where the graph's authors had about 0.12. Declaring 0.12 for this tier scored 83/120 (69 %,
  mean reward 0.924) against 60/120 (50 %, 0.876); every declared cell beat every undeclared
  one. See "The servo speed" below.

## What fails, per clause

The goal is `worst_of(route, hand clear and settled)`, and the route is six clauses: seat0-3
(three of the 71 scored rod points in each peg's box, on its required side), the rod in the
port box, and the free end in the port box. The reward is the fraction of the six held.

Every baseline episode runs the whole graph (four crossings, `plug`, `restore`, `verify`) and
ends at `abort` because `verify` sees a seat missing. That abort is the graph's own verdict
and costs nothing; the evaluator scores the settled scene either way. What it scores:

| Baseline cell (graph as shipped plus the L1 fix) | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end |
| --- | --- | --- | --- | --- |
| fleet, 15 x 2 (the coordinator's run) | 2/30 | 0.656 | 20 / 9 / 25 / 30 / 30 / 4 |
| fleet, 15 x 2 (this copy, repeat) | 1/30 | 0.661 | 22 / 11 / 23 / 30 / 30 / 3 |
| single world, 6 MPS workers (the coordinator's run; stopped at 29 of 30) | 2/29 | 0.644 | 23 / 6 / 24 / 29 / 28 / 2 |

Two clauses carry the cell: **seat1** (9-11 of 30) and **the free end** (3-4 of 30). The rod
reaches the port box every time; the end does not stay in it. The rest follows from those two.

## Root cause 1: the crossing's open jaws come down on the NEXT spool

Each crossing grasps the rod with the jaws across it, fully open: 47.5 mm a finger
(`REFERENCE_GRIP_OPEN`), so the two fingertips stand 95 mm apart. Because the rod's start is
welded to the anchor, `choose_grasp` walks the take point away from the anchor until enough
cable lies behind it for the crossing, and on this board that is 34-143 mm past the station.
L2's pegs stand 72-80 mm apart, staggered 30-45 mm in x. So the take for station 1 lands
abreast of peg 2, with the +x fingertip on or beside peg 2's spool (rim radius 12.5 mm,
23.5 mm tall): in one baseline cell the nearer open fingertip came down 3-30 mm from peg 2's
axis on 24 of 30 episodes.

L1 never meets this. Its two pegs stand 240 mm apart; replayed over an L1 cell's 60 recorded
takes, no fingertip comes within 32 mm of a peg.

### Evidence

**The seat follows the jaws, not the station.** For every recorded crossing of three baseline
cells, the take's perceived grasp, tangent and opening were replayed against the perceived
pegs (`tools/replay_jaw2.py`): does either finger's stroke, from fully open to closed, come
within 32 mm of a peg axis?

| Three baseline cells (89 episodes) | Fully open jaws clear every peg | They do not |
| --- | --- | --- |
| station 1 crossings: seat1 held | **13 of 13** | 13 of 76 |
| station 2 crossings: seat2 held | 60 of 63 | 12 of 26 |
| stations 0 and 3 | always clear | -- |

(Fleet 15 x 2 twice and single world, `fast`. Post perception is exact to 0.4 mm against the
frozen instance poses, so the replay's geometry is the scene's.)

**What it does to the arm, recorded.** Two baseline episodes were re-run single-world with
`--record` (privileged state and the `cable` camera every 10 steps):

* 10044: the station-1 take's +x fingertip came down on peg 2 (frames 230-254 of
  `cable.mp4`). From the start of the carry the arm stopped following its own stream:
  wrist joint 5, commanded 0.0 rad throughout, was twisted to -0.66 rad, and the tracking
  error grew to 0.67 rad. The rod did not move (capsule 31 stayed put while the hand drifted
  68 mm off it), the hand opened 137 mm from the planned end of its wrap, and seat1 ended on
  the wrong side. On the same episode stations 0, 2 and 3 tracked within 0.25, 0.20 and 0.03 rad.
* 10003: the fingertip landed 21 mm off peg 2's axis; the arm tracked (0.04 rad) but the
  station-1 carry moved the rod 2 mm: the jaws had closed above it, on air.

The graph's own seat read after each crossing says the same (`seat_station`: 0.000 at
station 1 on 22 of 30 fleet episodes), but it never acts on it (`RETRY_DEEPER = 0`).

**The failed station-1 crossing also costs seat0.** With the jaws fixed, seat0 went from
20-22 of 30 to 28-29 (below); the hooked finger was dragging the run back through peg 0.

## Root cause 2: on Medium the end is let go on top of the port, and springs out

`plug_port` carries the free end to the port and opens. On Easy it lowers the end into the
port's channel (walls 12.5 mm tall, 10 mm apart) and, if the end then lies more than 35 mm
off, makes one measured correction. On Medium (`len(stations) == 4`) it does neither:

```python
if len(stations) == 4:   # release with the cable axis 2 mm above the port's rim
    inset[2] = terminal_release_height(surface, float(port[2]), surface, axis_z)
...
correction = ({'terminal_correction_attempted': False} if len(stations) == 4 else _correct_terminal(...))
```

So on L2 the end is released on top of the walls, nothing retains it, and the ~290 mm tail
behind it (the straight run from seat 3 to the port is ~90 mm) springs it out:

* 10044 (recorded): the end entered the port box during the carry (reward 0.67 -> 0.83) and
  had slid 80 mm, back to seat 3, within 80 steps of the jaws opening.
* 10003 (recorded): in the box at release; out 20 steps later; 100 mm off the port when it
  settled.
* 10004 and 10043 (clause log every 10 steps, jaw fix on): the end was in the box during the
  carry and out of it by the end of the plug stage. 10043 then held all four seats and scored
  0.83.

Across the baseline cells the rod is in the port box 30 of 30 times and the end 3-4.

**Where the Medium branches came from.** They, and the skip in `restore_seats` below, arrived
in one commit of the dev repo (`AUTOLab-GaP-NVIDIA-Collaboration` 7e837ed, 2026-09-11,
"updated .task files"), on the old delivery stack: the 720 mm rod, the suite's 1280x960
`cable` view and the vendored runtime. The 840 mm rod (2026-10-22) and the 1280x720 station
camera (2026-10-18) came later. On the shipped camera the terminal was never detected until
the L1 fix, so these branches never ran in a VABAR measurement before today. No run of that
commit is on disk; the two old-stack runs that are (4cb878bd and ae6de67e, before it) lowered
the end into the channel and corrected on Medium, and kept the end in 7/30 and 10/20.

## The restore pass was switched off on exactly this tier

`restore_seats.py` returns at once on four stations ("Medium uses its first crossing action
without recovery pickups"), so a seat the later crossings or the terminal carry pull out is
never repaired. RoboSimStudio-dev reverted the same line in 48df9c6 (2026-09-12): "disables
the repair pass on exactly the tier that needs it ... It has never executed on any measured
run", with ground truth showing the terminal carry unseats seat 2. VABAR kept the skip.

## A thin port floor

With the L1 fix the port has to mask at least 1,912 px at 720 rows. On L2 at plug time it
masks 1,651-2,055 px (three cells), and 2 of 90 episodes fell under the floor, so the end was
never carried. The spools reach 1,786 px here, so no floor separates the two any more. None
has to: `plug_port` takes the block furthest from the anchored end, which is the port
whenever the port is in the list. The floor is now 1,500 px at 720 rows (2,667 at the
reference focal length, still scaled by `fx * fy`), the value of the dev repo's own branch
for the same symptom (`lgulich/cable-port-floor`). In the three fixed cells the port was
found 90 times in 90, picked out of up to four blocks, and every pick was the port
(y = 0.015-0.026 m, beyond the last peg).

## The fixes, tested head to head

All rows: the full 30-instance cell, `--config fast`, cell protocol, fleet waves of 15 worlds
on 2 workers unless stated. "Stroke" is the jaw rule as shipped: each finger's whole path from
open to closed must clear every perceived peg by 32 mm. Round 1 checked the open fingertip
only, which misses a peg lying between the fingertip and the rod (it then blocks the close);
it let 11-20 station-2 takes through with a finger's stroke within 32 mm of peg 3.

| Run | Jaws at the take | Medium terminal | Restore on Medium | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end | Mean / max steps | Wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (coordinator) | fully open | over the rim, no correction | off | 2/30 | 0.656 | 20 / 9 / 25 / 30 / 30 / 4 | 8,086 / 8,429 | 20.8 min |
| baseline (repeat) | fully open | as shipped | off | 1/30 | 0.661 | 22 / 11 / 23 / 30 / 30 / 3 | 8,073 / 8,241 | 23.1 min |
| narrow only | narrowed to clear (tip check), floor 15 mm | as shipped | off | 4/30 | 0.789 | 28 / 26 / 21 / 29 / 30 / 8 | 8,094 / 8,314 | 27.8 min |
| narrow, else move the take | same, take moved when 15 mm does not clear | as shipped | off | 9/30 | 0.800 | 29 / 25 / 18 / 29 / 30 / 13 | 8,128 / 8,465 | 28.1 min |
| + Easy's terminal | stroke, narrow-else-move | into the channel + correction | off | 11/30 | 0.850 | 27 / 22 / 24 / 30 / 28 / 22 | 8,416 / 8,993 | 28.4 min |
| **+ restore (shipped)** | stroke, narrow-else-move | into the channel + correction | on | **18/30** | **0.878** | 29 / 29 / 28 / 27 / 25 / 20 | 10,353 / 18,500 | 45.8 min |
| RoboSimStudio-dev's jaw rule | 16 mm a finger on every Medium take | into the channel + correction | on | 16/30 | 0.900 | 29 / 28 / 27 / 30 / 30 / 19 | 10,491 / 17,119 | 42.9 min |

(The port-floor change applies from the "+ Easy's terminal" row on.)

* **The jaws fix the seats.** seat1 9-11 -> 25-29 of 30, and seat0 with it (20-22 -> 27-29):
  the hooked finger was dragging the run back through peg 0.
* **Easy's terminal sequence fixes the end**: 13 -> 22 of 30 with the jaws fixed. The
  correction is the part that does it. On 10043 (clause log) the end was in the box during
  the carry, slid out on release even at bench height, and the correction put it back.
* **The restore pass then buys the rest** (11 -> 18 here, 13-18 over three cells of the
  shipped code): it repaired a seat on 13 of 30 episodes, and the repaired seat held 17 times
  in 22 (15 of 16 single repairs). Its cost is time: episodes with several repairs run to
  15,000-19,500 steps, and a wave lasts as long as its longest episode, so the fleet cell takes
  34-46 min instead of 21-28.
* **The two jaw rules are equivalent at this sample size** (18 vs 16 successes, 0.878 vs
  0.900). The geometric rule is shipped because it opens fully wherever that is safe (more
  tolerance for a misread rod), narrows only where a peg is within reach, and leaves L1
  untouched by construction; RoboSimStudio-dev's blanket 16 mm narrows every Medium take.
* Run-to-run spread is large here as on L1 (single episodes flip between 0.5 and 1.0), so
  read totals, not rows.

## The shipped graph, measured again, fleet and single world

The "+ restore" row above ran the shipped logic behind environment switches. With the
switches removed and nothing else changed, the cell was run twice more as 15 x 2 fleet waves
and once single-world (10 MPS workers; L2's default is single-world):

| Run | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end | Mean / max steps | Wall | GPU peak |
| --- | --- | --- | --- | --- | --- | --- |
| fleet 15 x 2 (the "+ restore" row) | 18/30 | 0.878 | 29 / 29 / 28 / 27 / 25 / 20 | 10,353 / 18,500 | 45.8 min | (GPU shared with a diagnostic run) |
| fleet 15 x 2 | 14/30 | 0.856 | 28 / 24 / 30 / 26 / 23 / 23 | 9,855 / 15,038 | 34.1 min | 11.2 GB |
| fleet 15 x 2 | 13/30 | 0.889 | 29 / 27 / 27 / 29 / 26 / 22 | 10,614 / 17,492 | 35.6 min | 12.3 GB |
| **single world, 10 workers** | **15/30** | **0.883** | 27 / 27 / 29 / 29 / 25 / 22 | 10,454 / 19,457 | 57.0 min | 54.9 GB |

Fleet fidelity on L2 had only been validated for L1 ([cable_fleet](cable_fleet.md)). Here the
single-world cell sits inside the fleet cells' spread on every clause, and per instance:

| Pair | Same success (of 30) | Mean abs. reward difference |
| --- | --- | --- |
| fleet vs fleet, three pairs | 12, 17, 19 | 0.111-0.178 |
| fleet vs single world, three pairs | 13, 15, 16 | 0.128-0.172 |

Success per instance is close to a coin toss after the fix: over the four cells, 2 instances
never succeed (10053, 10097), 2 always do, and the other 26 succeed one to three times in four.

### `fast` physics is not what holds this cell back

The fixed graph was also run under `--config default` (the reference
`cable_coupled_960hz@v1`) on the cell's first 12 instances, single world, 12 MPS workers:

| Same 12 instances | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end | Wall |
| --- | --- | --- | --- | --- |
| `default` (`cable_coupled_960hz@v1`) | 6/12 | 0.833 | 11 / 8 / 11 / 11 / 11 / 8 | 56.2 min for 12 |
| `fast`, the four fixed cells above | 5, 7, 6, 5 of 12 | 0.778-0.889 | | |

The one `default` episode with no reward at all (10084) aborted at step 1,985: the planner
refused station 1 ("No feasible grasp relative to the fixed anchor ... 263 mm, against 281
needed"), `plan_crossing` raised, and the workflow ended. That refusal never happened in the
eleven `fast` cells here (1,320 crossings). It is a graph robustness gap, below.

## The servo speed

Upstream documents the problem on `TaskSpec.servo_speed_mps`: a serial servo turn used to span
two or three simulator steps, "so the tool crossed the cell at 0.10-0.13 m/s on the machines
the paper's graphs were authored on", and both runners now advance the law every step, so a
suite has to state the speed it relied on. RoboSimStudio-dev's cable suite states 0.12
(cb96425e: "at the law's nominal cruise the routing legs miss"). No VABAR suite did, so
`robot.go_to_pose` ran at the law's 0.31 m/s. Only `go_to_pose` legs are affected: on this
path the terminal stage's clear, carry, retreat and correction. The crossings stream joint
trajectories.

| `go_to_pose` speed (fixed graph) | Cells (of 30; `s` = single world) | Success | Mean reward | seat0 / seat1 / seat2 / seat3 / port box / end (summed) |
| --- | --- | --- | --- | --- |
| 0.31 m/s, undeclared | 18, 14, 13, s15 | 60/120 (50 %) | 0.876 | 113 / 107 / 114 / 111 / 99 / 87 |
| **0.12 m/s declared in `port_medium.task`, `plug_port`'s `go_to_pose` budgets x 2.5 (cb96425e's)** | 19, 22; at a 14,000-step budget 21, s21 | **83/120 (69 %)** | **0.924** | 117 / 111 / 113 / 118 / 112 / 96 |
| 0.39 x cruise from the graph alone: `plug_port` through `robot.go_to_pose_cartesian(speed_scale=0.39)` | 13, 17 | 30/60 (50 %) | 0.892 | 57 / 57 / 55 / 59 / 55 / 38 |

(The second 0.39 cell merges a wave of 15 with a rerun of its other wave; see "Left open".)

| The two replication cells, fresh copy of the tree, budget 14,000 | Success | Mean reward | Longest episode (longest success) | Truncated | Wall | GPU peak |
| --- | --- | --- | --- | --- | --- | --- |
| fleet 15 x 2 | 21/30 | 0.922 | 14,000 (12,741) | 2 | 32.6 min | 11.6 GB |
| single world, 10 MPS workers | 21/30 | 0.922 | 14,000 (13,783) | 2 | 53.5 min | 53.5 GB |

* **The first two declared cells suggested it and the second two replicated it**: 41/60, then
  42/60. All four (19-22) beat all four undeclared cells (13-18). Per instance (the same 30 in
  every cell), 19 succeed more often declared, 6 less, 5 the same; a sign-flip test by instance
  gives p = 0.010 on success and 0.016 on reward. The two replication cells also ran on a newer
  tree (other suites' merges) and a 14,000-step budget; everything else ran at 24,000.
* **The gain is in the terminal clauses**: port box 99 -> 112, end 87 -> 96 and seat 3, the seat
  nearest the carry, 111 -> 118. Seats 0-2 are unchanged within noise.
* **A 14,000-step budget costs little.** 1 of the 83 declared successes ran past it (4 past
  13,000, 7 past 12,000). The 4 episodes it truncated in the two replication cells scored
  0.33-0.50, cut off mid-repair.
* **The graph-side version still does not reproduce the gain**, unexplained. `go_to_pose_cartesian`
  takes no `max_steps`, so a stalled leg pushes for 1,000 steps instead of 400-600, which may be
  why; it was not tested.

**What the declaration reaches.** Only `cable_route/port_medium`: `port_easy` still loads
`servo_speed_mps = None` and runs at 0.31 m/s. The budgets are in the shared `plug_port`, so they
do reach L1, where they matter only on a leg that stalls: 16/30, mean reward 0.825, longest
episode 12,394, 21.5 min, inside its earlier 16-20/30. L1 at 0.12 m/s has not been run. The line
is in the task document, so it applies under every config, `paper` included, while the frozen
per-instance documents in `benchmark/cable_tasks/dataset/`, which `vabar bench` does not run, do
not carry it.

## The gap to the paper's 46.7 %

There is none left to explain on this cell: the fixed graph scores 50 % undeclared (60 of 120
over four cells; single world 15/30) and 69 % with the servo speed declared (83 of 120; single
world 21/30), against the paper's 46.7 %. Which graph and task the paper's
number came from is not established. The dev repos show at least two candidates that differ
from what VABAR stages: RoboSimStudio-dev's L2 (recessed pegs, 720 mm rod and its own Medium
graph fixes, a1140aa) and the collaboration repo's (840 mm rod, the Medium-only branches).
Before the L1 fix no VABAR measurement could have exercised the terminal stage at all.

## Changes

| File | Change | Why |
| --- | --- | --- |
| `tasks/cable_route/graphs/weave/scripts/plan_crossing.py` | New `PEG_KEEPOUT` (32 mm), `JAW_MIN_REACH` (15 mm), `REGRASP_MARGINS`, and `rod_tangent`, `nearest_tip_to_post`, `jaws_clear_posts`, `jaw_reach_clearing_posts`, `take_opening`. After the plan, the take opens to the widest width (47.5 mm down to 15) whose fingers clear every perceived peg over their whole stroke; when none does, the planner is asked again at larger slack margins, which walk the take along the rod, and the first take that clears is kept. The opening rides in the plan as `take_open_m`, with a `jaw` note for the trace. | Root cause 1. Only the perceived pegs, the perceived rod and its tangent, and the plan's own grasp and opening are read. The planner (`cable.plan_station`), its IK and its choreography are unchanged. |
| `.../scripts/run_legs.py` | The take sets the jaws to `take_open_m` before its approach when that is narrower than the open command. | Applies the opening. The release, carry and close are untouched. |
| `.../scripts/restore_seats.py` | The Medium early return is gone; both `cable.plan_station` calls are wrapped in `try/except (ValueError, RuntimeError)` (the delivery ae6de67e's form); repairs use `take_opening` too. | The repair pass now runs on four stations, as RoboSimStudio-dev's 48df9c6 had it. The `try/except` is the fix [cable_routing_fix](cable_routing_fix.md) left open. |
| `.../scripts/plug_port.py` | The two Medium-only branches are gone, so Medium releases at the grasp height and runs the one measured correction, as Easy does. `BLOCK_MIN_PX` 3,400 -> 2,667 (1,500 px at 720 rows, still scaled by `fx * fy`). The `[plug]` log counts mask pixels as `> 0` (a 0/255 mask was logged 255x). | Root cause 2 and the port floor. The log fix is from `lgulich/cable-port-floor`; on palette masks (0/1) it prints the same numbers. |
| `robot_skills/skills/{computing-routing-waypoints,grasping-curve-point,executing-held-object-motion,releasing-routed-cable,restoring-displaced-seats,terminating-the-run}/scripts/` | The same edits in the exported bundle copies; `restoring-displaced-seats` also carries `plan_crossing.py` now, for `take_opening`. | The bundles are generated from the graph with sibling imports scoped to the package (`tools/sync_robot_skills.py` is not in this tree; regenerated by the same rewrite as before). |
| `tests/test_weave_jaws.py` (new) | The opening rule: full open far from pegs (L1's spacing), narrowed until the stroke clears a neighbouring spool, refused when a peg sits inside the closing stroke. | |

The servo speed adds two more: `(:servo-speed-mps 0.12)` in `port_medium.task`, and `plug_port`'s
four `go_to_pose` budgets x 2.5 (400, 600, 500, 400), in the graph and in its bundle copy.

Not changed: the task's goal and scoring, its physics (`cable_route_fast@v1` under `fast`), the
camera, the planner in `tasks/cable_route/routine.py` and `grasp.py`, and every other node. The
servo line is the one edit to the task document.

## L1 did not regress

Same code, `cable_routing/L1`, `fast`, fleet 15 x 2: **19/30, mean reward 0.850**, seats
28 / 28, port box 23, end 23, 20.3 min. Its four earlier fleet runs scored 16-20/30
([cable_fleet](cable_fleet.md)). All 60 takes opened fully (no peg is within reach at a
240 mm pitch), the port was found 30/30 (picked out of 1-3 blocks under the lower floor), and
no restore call was refused.

## What came from the dev repositories

`VABAR` v1.0 and the public RoboSimStudio are squashed exports, so both dev repos were
searched for cable-routing fixes VABAR lacks, by content (`agent_scratch/l2route/repos/`,
blob-filtered clones):

* **AUTOLab-GaP-NVIDIA-Collaboration.** `vabar-robosimstudio` is VABAR v1.0 byte for byte;
  `main` (Oct 26) differs only in the pre-port import paths. `lgulich/cable-port-floor`
  (138b438, Oct 27) lowers the floor to 1,500 px and fixes the area log: **both taken** (the
  floor kept as a focal-length-scaled 2,667). `cable-tasks` and `yam-station-cameras`
  (Oct 18) are older than VABAR's graph. The history dates the Medium-only branches to
  7e837ed (Sep 11; see root cause 2).
* **RoboSimStudio-dev.** Its `weave_terminated` graph (on `fix/capture-chain-bodies`,
  `fleet/cable-route-per-world`, `feat/network-cable`, `release/oss` and the 2026-10-02 WIP
  snapshot alike) is VABAR's graph plus four Medium fixes VABAR never received:
  * 48df9c6 (Sep 12): **the restore pass runs on four stations** -- **taken**, with its
    rationale (above).
  * 1e699fb8 (Sep 11), the restore `try/except` ("a restore pass that cannot plan one seat
    still repairs the others") -- **taken** (also in the collaboration repo's ae6de67e).
  * a1140aa (Sep 12): every Medium take approaches at 16 mm a finger, "a narrow approach keeps
    open fingers clear of adjacent pegs" -- the same diagnosis as root cause 1. **Measured
    head to head** (16/30 against 18/30 for the geometric rule) and not shipped.
  * a1140aa also moved the Medium grasp 38 mm clear of every peg, widened the Medium wrap to
    38 mm, and rebuilt the Medium terminal around a low drag beside the pegs -- but on a
    *different task*: the same commit recessed the pegs 6 mm (`PORT_BOARD_RECESSED`) and went
    back to the 720 mm rod. Not taken: they were tuned to that board, and the staged L2 is
    the other one.
  * `fix/capture-chain-bodies` (19a43c8, "plan_support takes a grasp off a 3-d centreline"):
    VABAR's graph only passes 2-vectors there, and no L2 log shows the reshape error. Not
    needed. The `fleet/cable-route-per-world` changes are the ones [cable_fleet](cable_fleet.md)
    already made; their `peg_xy_of` fix (33039f90, the build-time pose read in a wave) touches
    only the privileged `sim.get_cable_state` and the oracle/routine paths, since the vision
    graph hands the planner its perceived stations. d9b875de (`sim_step` span tags) is
    capture bookkeeping.
  * cb96425e (Sep 16): the cable suite declares `(:servo-speed-mps 0.12)`, "the tool speed the
    graphs were written and measured at", and the weave's `go_to_pose` budgets rise 2.5x to
    match. VABAR declares no servo speed, so `go_to_pose` legs (the terminal carry, its
    correction, the step-aside before a re-look) run at the OSC law's 0.31 m/s; the crossings
    stream joint trajectories and are unaffected. **Taken for `port_medium`**; see "The servo
    speed" above.

## How to reproduce

```bash
export PYTHONPATH=<robosimstudio>:<graph-as-policy>:<graph-as-policy>/gap-core/src
# the cell as fleet waves (what the tables above ran)
CUDA_VISIBLE_DEVICES=0 python -m vabar bench --cell cable_routing/L2 --solve graph --config fast \
  --set "suites.cable_route/port_medium.num_worlds=15" --out <dir>
# single world, 10 MPS workers
CUDA_VISIBLE_DEVICES=1 python -m vabar bench --cell cable_routing/L2 --solve graph --config fast \
  --set suites.cable_route.workers=10 --out <dir>
```

Scratch scripts and raw runs: `/home/shadeform/speed_ws/agent_scratch/l2route/` --
`runs/` (every cell above, its log, and `NOTES.txt` naming each run's variant), `tools/`
(`tally.py`, `table.py` per-clause tallies; `replay_jaw2.py` the jaw-clearance replay;
`jawpegs.py`, `plans.py` the take geometry; `restoresum.py`, `plugsum.py` the restore and
terminal stages; `timeline.py` with `diag_run.py`, a clause logger for single-world runs;
`sync_bundles.py` the bundle rewrite), and `frames/` (the 10044 crossing).

The variants in the head-to-head table were selected by environment switches in the scratch
copy (removed before shipping); each run's traces keep the exact scripts it ran
(`traces/<episode>/scripts/`).

## Left open

* **The end is still the weakest clause** (about 22 of 30). After release the correction
  looks for the tip; on about two thirds of episodes it is "not unambiguously visible" and no
  correction runs. On L1 that always meant the tip lay hidden in the channel; on L2 it did 15
  times in 20 in the best cell, and the other 5 ends were out of the box. A second look from
  another height, or a correction that does not need the tip, would be the next step; both are
  policy redesign rather than repair.
* **Wall time.** The repair pass is what pays for most of the gain, and it is slow: an
  episode with several repairs runs to 15,000-19,500 steps (budget 24,000), and a fleet wave
  lasts as long as its longest episode, so the cell went from 21-28 min to 34-46 min as
  15 x 2 waves (57 min single-world on 10 workers). The coordinator's L2 table in the README (21.4 min, 2/30) predates this.
* **The restore pass's "seated" bar is two capsules (26 mm); the scorer's is three points.**
  `port_medium` writes the seat threshold as 0.028169014085, which is 2/71 rounded *up*, so two
  scored points fail it and three are needed (L1's 0.039999999999 against 51 points is the same
  three). Of 98 spans the pass left alone in the best cell, 2 then scored two points. A bar
  near 33 mm would catch those and also re-seat some that held; it was not tuned.
* **The wrap's yoke clearance.** The recorded station-0 crossings twisted the wrist 0.24 rad
  and stalled the hand at the start of the arc; the wrap radius (32 mm) clears the closed
  yoke by 1.1 mm by design, which arm lag can eat. Seat 0 still held 27-29 of 30 once the
  jaws were fixed, so it was not pursued. RoboSimStudio-dev widened the Medium wrap to 38 mm
  (a1140aa) on its recessed-peg board; untested here.
* **A planner refusal mid-weave aborts the episode.** `plan_crossing` lets
  `cable.plan_station`'s "No feasible grasp relative to the fixed anchor" escape, so the node
  errors and the workflow ends at `abort` with nothing scored (10084 under `default`, above).
  The workflow already routes `infeasible` to the next crossing; returning that on a refusal
  is the same repair the restore pass got. Not made: on a usable fit it never fired under
  `fast`, so it could not be measured here.
* **A bad opening fit is not caught.** In one wave of 27 run here (`L2f_gspeed039b`, worker 1,
  all 15 worlds) the opening rod fit came back 266 mm long, at x 0.22-0.28 m, where the left
  arm's white base plate stands, and every world aborted at step 85 on a planner refusal. The
  other worker's wave in the same run was normal. That is what a frame without the rod would
  give, but no frames were kept, so the cause is unknown. It is not the known RTX stall (no
  timeout in the log). The rod's rough length is known to the graph's perception only from
  this same fit, so nothing refuses it. The wave was rerun for the table above, and it is in
  no cell of the shipped graph.

## L1 at the declared speed: not adopted

Measured 2026-11-05 with `(:servo-speed-mps 0.12)` added to `port_easy.task` on a copy, `fast`, fleet 15 x 2, two runs. L1 keeps
the law's 0.31 m/s.

| Run | Solved | Mean reward | Wall |
| --- | --- | --- | --- |
| Declared, run b | 14/30 | 0.80 | 20.2 min |
| Declared, run a | 8/30 | 0.44 | 16.6 min |
| Shipped speed, earlier runs | 16-20/30 | 0.83-0.89 | 17-24 min |

Run a's score is not about speed. One worker's whole wave of 15 worlds aborted at step 85, on a rod fit 400-460 mm off
at the graph's first perception, while the other wave scored 8/15. The same whole-wave failure appeared in one other
fleet run today, an L2 cell with the speed set from the graph (`L2f_gspeed039b`), so it does not come from declaring the speed.
Without that wave the declared speed scored 22/45 (49%), no better than the shipped speed's 53-67%.
