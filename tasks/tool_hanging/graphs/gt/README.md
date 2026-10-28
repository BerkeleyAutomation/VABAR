# `tool_hanging/gt` — the ground-truth graph

One workflow serves the three original graph variants and all four level-1
variants: `wrenches`, `scissors`, `screwdrivers`, and the one-object wrench,
scissor, screwdriver, and plier tasks. It is
the suite's first graph, authored here rather than retargeted, and it replaces
nothing — the scripted routines still ship beside it.

The original two-plier task and `hammers` remain out of scope. The former is
genuinely bimanual; the level-1 plier task needs only one arm and is supported.

```bash
uv run python -m vabar run --solve graph \
    --task tool_hanging/wrenches --graph gt --episodes 0:10
```

## Measured

The level-1 compatibility campaign (fixed seed/layout, three independent
rollouts per task) passed **12/12**: 3/3 each for wrench, scissors,
screwdriver, and pliers. The wrench uses the shared 200 g large-tool mass and
passes without wrench-only slow motion or a special shallow insertion.

`--episodes 0:10`, one RTX PRO 6000, one process. Every variant here declares
`(:episodes (:count 1))`, so all ten episodes are the **same layout** — this is
run-to-run variance, not generalisation. The routine column is the shipped
`--solve routine` on the same command and the same machine.

| variant | graph reward (mean / median) | graph success | routine reward | routine success |
| --- | --- | --- | --- | --- |
| `wrenches` | **1.00** / 1.0 | **10/10** | 0.45 / 0.5 | 2/10 |
| `scissors` | **0.90** / 1.0 | 8/10 | — (suite README: 4/5) | 8/10 (prior campaign) |
| `screwdrivers` | 0.80 / 1.0 | 6/10 | 0.70 / 0.5 | 4/10 |

Read the reward, not the rate. `(fraction)` pays half for one tool of two, and
on `wrenches` that is the whole story. The routine's ten rewards are

    0.5  0.5  1.0  0.5  0.0  0.0  0.5  0.5  0.0  1.0

— a lottery whose median outcome is one spanner and whose worst is neither.
The graph's are ten 1.0s.

`screwdrivers` is the honest one: 0.80 against 0.70 and 6/10 against 4/10, which
at n=10 is inside the noise. Call it *level*, and see "What still fails".

## The shape: a per-job cursor loop

```
START ─▶ next_job ──more──▶ perceive ─▶ grasp ─▶ carry ─▶ place ─▶ release ─┐
             ▲                  │          │       │        │              │
             │                  └── unreachable ── lost ── errored ─▶ give_up
             └──────────────────────────────────────────────────────────────┘
             │
             └──done──▶ verify ──▶ done | abort
```

Every variant hangs **two** tools, and the document says which tool goes on
which holder as a pair of `threaded` clauses in its own order. So the loop is
not an optimisation — it is the task. `sim.next_job` reads the goal tree against
the evaluator's live verdict and answers "which clause is still outstanding";
`release` and `give_up` both route *back* to it conditionally, and the runtime
resets the forward-reachable body when a conditional edge lands on a completed
node (`gap/runtime/executor.py`). Nothing is unrolled, so a third tool would
need no edit here.

The cursor is also what makes the graph give up rather than livelock. Each tool
gets two passes; the second re-measures from wherever the first left it, which
is the case that matters — a spanner nudged 30 mm across the table is a new
grasp problem, not the old one retried.

## What is shared and what is bespoke

**Shared across all supported variants:** every node script, every edge, and three
of the four phases — `reach` (approach, descend), `carry` (lift, turn) and
`retreat` (back off, go home). A *phase*, not a stage, is what `move_stage.py`
drives, which is what lets the same node run three legs onto a hook for a
spanner and three quite different ones into a rack for a screwdriver. It is
also what keeps the episode inside its step budget: the runtime floors every
`execute_trajectory` settle at 60 control steps, so one call per phase costs
240 a job where one per leg costs 540.

**Bespoke:** the two branches inside `sim.plan_place`, chosen by *which side of
the goal clause the tool is on*:

| | tool carries | fixture carries | place legs |
| --- | --- | --- | --- |
| `wrenches`, `scissors` | the `loop` | the `rod` | stage in front of the tip, push the loop past it, lower until it rests |
| `screwdrivers` | the `rod` | the `loop` | rise clear of the rack, cross above the hole, drive the tip in along its own axis |

That is one `if`, and it is keyed off the document rather than off the variant's
name. The `place` subgraph's skill is `generic` because this really is bespoke;
All GT subgraphs are labelled `generic`. This is deliberate: their suite tools
read the compiled goal, live simulator poses, and evaluator geometry, so they
are privileged reference-policy operations rather than implementations of the
portable open skill-library contracts. The separate `gap_perception` policy is
where those contracts are used.

## The two things the routine cannot do

**It re-measures the tool in the jaws.** The scripted routine plans every
station — pre-hook, over-hook, lower — from the pose it measured *before* it
closed the fingers, and never looks again. Two things happen in between: the
close shifts the tool by up to a centimetre as the pads bottom out on it, and
the turn to vertical swings whatever the pads did not hold. `sim.plan_place`
runs at the turn station with the tool in hand, reads its live pose, computes
where the landmark rides relative to the tool centre *now*, and puts the
stations where that landmark lands on the holder. Measured slips on the
spanners: 6–10 mm, against a `threaded` budget of 7.71 mm on the 19 and 6.71 mm
on the 17 — the correction is the size of the whole margin.

**It checks whether it is holding anything.** `grasp` reads the jaw gap — the
pads reach 0.7 mm on air and stall at 13–17 mm on a spanner shaft — and `carry`
reads whether the tool actually left the table. Either failure routes to
`give_up`, which frees the budget for a second pass instead of flying an empty
hand through a full hang and reporting a hook nothing landed on. That is the
suite README's one open routine item, exactly: *"the 17's jaws close ~1 mm above
its thinner shaft and the wrench stays on the table."*

## The measurement that mattered most

The grasp was a lottery, and the reason was not the plan. Every tool here lies
flat on a table and the landmark the plan steers by is its *mid-thickness* —
4.25 mm above the surface on the 19 mm spanner. Commanding the tool centre there
drives the fingertips into the table about ten millimetres before they arrive.
Traced on the shipped routine: the descend ends 8.2 mm short and stays there
(asking for 25 mm more changes nothing but the shortfall), and the jaws sit at
92–95 mm through the whole close, only running shut once the *lift* unloads them
— by which time the hand is above the tool.

`FINGER_TABLE_CLEARANCE` in `tools.py` stops the descend where the fingers can
still move. Swept at 1 mm steps on both spanners, measuring whether the tool
leaves the table at all:

| clearance | outcome |
| --- | --- |
| 2 mm | jaws jammed part-open at 26–28 mm; neither lifts |
| **4 mm** | **both lift**; 7–10 mm of in-jaw slip |
| 5 mm | both lift; 6–9 mm of slip |
| 6 mm | the 17 lifts, the 19 does not |
| 7 mm+ | the pads close above the shaft; neither lifts |

The window is four millimetres wide, which is why the workflow does not rely on
hitting it: `carry` checks and the cursor gives the job another pass.

## Read the 17 mm spanner's margin before trusting its number

`wrenches` scores 10/10, and the second half of that is won by tenths of a
millimetre. `threaded` allows `inner_r - rod_r` of offset, which is 7.71 mm on
the 19 and **6.71 mm on the 17**, and a spanner hanging naturally on this hook
sits right at that line: the recorded hung pose in `poses/wrenches.json`,
re-planted and re-settled, reads 6.74 mm — 0.03 mm *outside* its own gate.

What buys the 17 its pass is the derived seat. The routine lowers a flat 12 mm,
which leaves the ring ~10 mm above where it will come to rest, so it *drops* onto
the arm and lands cocked at 6.8–6.9 mm. This graph lowers until the loop rests on
the rod — `loop_above_tip` plus the loop's own declared hanging radius, plus a
2 mm preload — and the ring is already seated when the jaws open. Measured across
ten episodes: 6.61–6.69 mm, clearing by 0.02–0.10 mm every time.

That is a real and repeatable improvement and it is also 20 microns of margin. A
change to the hull sets, the contact solver or the declared `inner_r` would move
it, and the honest reading of `wrenches 10/10` is "the 19 hangs with half a
millimetre to spare and the 17 hangs on the right side of a line it used to land
on top of".

## Roll, and where it actually bit

The routines here use `ArmIK.solve_q_full` exclusively, which is the combination
that broke `cloth_fold/shirt`: its roll-closing step translates the tool centre
on an off-axis hand — 62 mm at 90° on this YAM's 44 mm offset — and its position
polish is hard-capped. This graph does not use it. Every waypoint is solved by
the arm's own full-pose LM objective, warm-started along a Cartesian path, which
closes position and rotation together and reports 0.001–0.01 mm at every
station.

The *shape* of the cloth_fold problem still turned up, in the other axis. The
screwdriver's turn station is 167 mm above its grasp, and at that point the
**upright** wrist solves to 0.001 mm while the **top-down** one parks 43 mm out:
same arm, same point, two wrists. A plan that insisted on holding top-down there
passed the 45 mm gate and then dropped the tool on the way. The fix is
`_holdable_height`, which is `FoldRoutine._holdable_roll`'s argument in the
vertical: ladder each wrist separately at a tolerance far tighter than the plan
gate, and let the *turn* finish the rising rather than requiring a pose the arm
cannot hold.

## Where the geometry comes from

Every target — the hook tips, the rack holes, the rings, the bows, the shafts —
is read off the asset records through the same point/direction transform the
evaluator's feature reader uses, so the plan aims at the geometry the goal
scores. `sim.get_hang_state` computes the `threaded` verdict itself from those
same features, which is why a node can route on "is it hung" before the release
without waiting for a reward that cannot be true until the hand lets go.

Two things are restated in `tools.py` rather than read: where the jaws close on
each tool class, and how far above a hook's rod the loop has to ride to clear
the tip. Both are measurements of the mesh and neither is derivable — a record's
`inner_r` is sized to the *resting hang*, and a scissor bow is a 30 mm oval
whose physical opening is much larger than the circle it comes to rest on.

## What still fails

`screwdrivers` is the weak one -- 6/10 against the routine's 4/10, which is
not a difference at this sample size -- and the cause is neither the grasp nor
the plan. In
every failing episode the tool that misses was **seated while still held** —
`place` reports `threaded: true` at 1.83 mm offset against a 5.2 mm budget — and
then leaves the hole after the jaws open. The cursor gives it a second pass, but
a full re-pick costs about 800 steps against a 2600 budget already 2200 spent,
so the retry runs out rather than recovering. Worth measuring next: whether the
retract lifts before the pads have finished opening, and whether the second
insertion disturbs the first (the two rack holes are 38 mm apart).

`scissors` misses one bow in two of ten, one of them by truncation.

## Could this re-record `poses/*.json`?

Yes, for these three. `tasks/tool_hanging/rigs/record_tool_hanging_poses.py` drives
`PolicyScripted` through the ordinary `rollout` path and writes the settled
object poses; pointing it at `PolicyGraph(env, graph="gt")` is a one-line
change, and the graph reaches reward 1.00 on all three variants, which is the
condition the recorder gates its write on.

It would also let the recorder drop `_mirror_wrench_17` — the branch of
`record()` that *plants* the 17 mm spanner at the 19's hook relation because the
routine cannot hang it. This graph hangs it for real.

`pliers` still needs the routine, so `routine.py` cannot go yet.
