# `tool_sort/gt` — the ground-truth graph

Retargeted from VAB-industrial's `graph/tool_sort_hammer_bddl`. The structure,
the skills, the routing and four of the six node scripts are that workflow's,
unmodified — `compute_align_pose.py`, `compute_drop_pose.py`, `waypoint_move.py`
and `descend_release.py` are byte-identical to the originals, so "only the
backend-specific nodes changed" is a claim you can check with `diff`.

## What changed, and why

**Two nodes.** The reference's `target_obb_mujoco.py` and
`container_obb_mujoco.py` returned *literal* boxes measured once at seed 0,
because no VLM credentials were available on that machine. Here they ask the
simulator:

| Node | Was | Is |
| --- | --- | --- |
| `target_sg.target_obb` | a hardcoded `OrientedBoundingBox` | `sim.get_object_obb` |
| `container_sg.container_obb` | a hardcoded `OrientedBoundingBox` | `sim.get_region_obb` |

Both return a `gap_core.types.OrientedBoundingBox`, so they occupy exactly the
slot a perception pipeline would: a `perceive` node plus
`geometry.filter_and_compute_obb`. Everything downstream — the grasp candidates,
the align pose, the transport — is untouched. That substitutability is the point
of a ground-truth graph: it is a *reference* for a perception graph, not a
separate program.

`sim.*` tools exist only in the in-process executor. Over the remote bridge this
graph cannot run until those two nodes are replaced with real perception, which
is enforced structurally — see `robosimstudio_policy_agent.contract`.

**And one node added, in `grasp_sg`.** `clamp_grasp_depth.py` sits between `yaw`
and `compute_align` and owns how deep the fingertips go. It is one file shared
with `astro` -- `tasks/_shared/scripts/clamp_grasp_depth.py`, resolved from
there because this directory does not ship it (`tests/test_gap_shared_nodes.py`)
-- and the reference workflow has no equivalent because it never grasped
anything that had tipped over. What it is for is below, under "The grasp had no depth rule".

## One scene-specific constant

`move_above` passes `safe_height: 1.10`, overriding `waypoint_move.py`'s own
`_LIFT_Z_M = 0.353`. That default is a table height from the scene the script
was authored for; this workcell's table top is at ~0.80 m with the Panda's base
at 0.912 m, so the default puts the lift leg below the floor and every waypoint
of the lateral leg is unreachable. 1.10 m clears the bin walls by about 28 cm
and solves to ~1 mm along the whole path.

It is passed as a node input rather than by editing the script, so the script
stays byte-identical to the reference.

## The IK budget is not this graph's, but it decides whether it runs

`IKBackendNewton.iterations` is a **wall-clock** budget, not an accuracy knob,
and at its old default of 300 this graph could not complete at all.
`ArmIK.solve_q` spends its first `iterations // 3` in a position-only stage with
no early exit, and every iteration costs eight finite-difference FK round trips,
so a 40-waypoint `plan_linear` is linear in the number: 60.8 s at 300, 20.9 s at
100, 8.5 s at 40 — all to the *same* final tip error of 0.0 mm.
`SimBus`'s watchdog is 30 s and `waypoint_move.py` issues two legs, so at 300
`transport_sg.move_above` always died with `_plan_linear was not serviced within
30 s`, having never been given the chance to fail on its merits.

If a graph in this stack dies in transport, check that number before suspecting
the scene.

## Which tool, which compartment

Neither is named here any more. `target_sg` and `container_sg` still accept
`object_name` / `region_name` node inputs, but they default to empty, and empty
means *ask the task*: `sim.next_goal` returns the first `(object, region)` pair
of this variant's `(:goal)` conjunction that the evaluator has not yet
satisfied.

That tool is contributed by the suite through `TaskSpec.tools`, not by the
generic `sim.*` set — "which goal is outstanding" is a question about a BDDL
problem, not about a scene. It is what turns one workflow into six. Naming
`hammer_0` as a literal is what previously made this graph unrunnable on the
four variants that contain no hammer: they failed in the first node with
`no entity 'hammer_0'`.

## The loop

The reference workflow is a straight line — find, grasp, transport, done — which
solves a one-goal problem. A sorting problem is a *conjunction*, so this one is
a loop:

```
START → check ──found──→ target → container → grasp ──grasped──→ transport ──placed──→ check
          │                                     │                    │
       finished                              failed               blocked
          ↓                                     └──────→ give_up ←───┘
        done                                                │
                                                            └──→ check
```

Three things in that diagram are not in the reference:

* **`check`** is the gate. It is a subgraph of its own rather than a branch
  inside `target_sg` because a subgraph binds its declared outputs by `$ref`
  into its own nodes, so one that sometimes skipped the node producing
  `target_obb` would export an output with nothing behind it.
* **`grasp` verifies.** `robot.close_gripper` returning means the fingers
  finished travelling, not that anything is between them. Without the check the
  graph flies an empty hand to the compartment, opens it, and reports
  `exit_status: success` on an episode that scored zero — every node ran and
  nothing raised. `sim.check_grasp` is the same contact test the evaluators use.
* **`give_up`** abandons one goal for the rest of the episode instead of
  aborting the whole run. Three of these variants ship a deliberately
  ungraspable tool, and a baseline that stops at the first one is scoring the
  goal *ordering* rather than the task. `sim.skip_goal` holds that set
  connector-side, because a subgraph re-entered by a loop edge gets a fresh
  scope. A goal also gets abandoned after `MAX_ATTEMPTS` passes, which is what
  stops a tool that bounces out of its compartment from being retried forever.

Every subgraph also declares `on_error`, so a node that raises routes to
`give_up` rather than killing the workflow.

## Releasing into the compartment, not onto it

`compute_drop` wires the region as `container_interior_obb` as well as
`container_obb`. Without it the script takes its no-interior fallback, treats
the region box's *top face* as the floor and releases `drop_clearance` above
that — 0.90 m over a compartment whose floor is at 0.80. The tool is dropped
10 cm and bounces.

That went unnoticed for as long as the graph only ever did one goal: the
environment terminates the instant the evaluator is satisfied, so a single-goal
episode ends at first containment and never observes the tool bouncing back out.
`sort_all` is what surfaced it — the hammer entered its compartment, the graph
moved on to the drill, and the hammer was no longer there at the end. The
evaluator does not latch, by design.

## Status

Measured 2026-07-29, re-measured 2026-08-07, and re-measured again 2026-08-15
on `e6c7f0c` plus the depth clamp below, `--episodes 0:N --renderer ""`. **Every
row here is a serial measurement on an otherwise idle GPU, and that is part of
the number** — see "Do not measure this arm under contention" below.

| Variant | Result |
| --- | --- |
| `hammer` | **4/4 success**, reward 1.000, 704 steps [702, 706] — 2026-08-15, `e6c7f0c` + the depth clamp, four fresh processes, serial. Reproduces the 6/6 at ~705 recorded 2026-08-07; between those two dates it was **0/4 at ~1305**, see "The grasp had no depth rule" |
| `syringe` | **1/1 success with `--ik pyroki`**, reward 1.0, 1486 steps — posted through the aperture, and released before it scored. Unsolvable on the default IK; see below |
| `sort_new` | runs end to end; grasp fails on the first screwdriver |
| `sort_all` | the loop advances; hammer leg lands marginally and the other tools exceed the gripper span — see the ceilings table |

The `hammer` row was confirmed again on 2026-08-07 against the ported document:
3/3 at 708/705/705 steps, `containment/hammer_0/bin2_hammer_region` true in
every episode. That is the whole of this variant's episode row now — the
documents state one `(In ...)` clause and no `:observe`, so `sorted_count`,
`goal_count`, `held_at_goal` and `in/hammer_0` are all retired. "Released
before it scored" above used to be the `held_at_goal` reading; it is now a
property of the goal rather than a metric, because containment is scored on a
tool at rest in the bin.

The hammer figure was **0 in 6** before the IK budget was corrected, and 1 in 6
on an older tree. It is not a grasp-quality result: what changed was the watchdog
interaction described above, and the grasp was never the marginal part.

The step count dropped from ~2900 to ~680 for the same task when two GaP-facade
bugs were fixed — `stream_joint_trajectory` was calling `apply_joint_action`
(which steps the env) from a graph's worker thread, and `PolicyGraph.get_action`
was returning an idle action that commands the *measured* configuration between
every streamed waypoint. Both amounted to the arm being told to stop pushing on
roughly every other step. Neither showed up as a failure here, only as an
episode three times longer than it needed to be; they were found by a deformable
task, where the same defect breaks the physics outright.

### It went to 0/3 in between, and the cause was in the servo

Between the two measurements the hammer read **0/3, every episode truncated at
2400** — and the accounting says where the steps went: `grasp_sg.rotate_align`
and the lateral leg of `transport_sg.move_above` each ran their full 400-iteration
servo budget, about 1030 control steps apiece, ending with the tool 0.04 mm from
its target and **4.3 degrees** off in approach angle against a 3 degree arrival
test. Nothing raised. The graph simply ran out of episode.

The trigger was a correct change elsewhere: the Panda's `close_axis` was measured
and found a right angle out, so `sim.get_object_obb`'s `grasp_quat` began asking
for the *real* cross-the-handle roll — about 84 degrees off the one a free wrist
holds. `CartesianServo` then bumped into a defect it had always had.
`OscSetpoint`'s lead clamp bounds a **full quaternion** angle, so a goal carrying
a roll the DLS backend is going to discard spends most of that bound on it: the
setpoint slerps only `rot_lead / total` of the way from the measurement, and the
approach axis — the one component the solver controls — is corrected by that
same fraction. The residual is a fixed point at `solver_slop * total / rot_lead`,
which is 1.5° × 1.46 rad ÷ 0.5 ≈ 4.4°, i.e. exactly where it sat.

`CartesianServo._solver_goal` now restates the goal in the backend's own terms
before the law sees it: for a roll-free solver, the measured orientation turned
by the shortest arc onto the goal's approach axis. The roll was already being
dropped by `IKBackendNewton._target` half a step later; what this removes is its
weight in the clamp. `--ik pyroki` honours roll and is handed the goal unchanged.
`tests/test_gap_contract.py::TestServoGoalRestatement` reproduces the stall in
arithmetic, without a GPU.

If a graph in this stack is slow rather than wrong, count the steps per servo
move before suspecting the scene: a move that returns having spent its whole
budget looks identical to one that arrived.

### The syringe needs `--ik pyroki`

The default backend constrains position and the *approach axis* —
`ArmIK.solve_q` aligns the tool's +z and leaves wrist roll free by design,
because constraining it makes a top-down chain stall in a saddle. Standing a
tool on end is a roll about the approach axis, i.e. exactly the component that
solver discards. `post_upright.py` checks the result and raises rather than
releasing a horizontal syringe over a hole it cannot fit through.

`--ik pyroki` swaps in the full-orientation solver
(`robosimstudio.robot.ik.pyroki`, wrapping the backend gap's own
pyproject ships for exactly this seam), and with it the same graph posts the
syringe: reward 1.0, `in/syringe_0` true, released. This mirrors the reference
implementation, whose OSC controller tracks a full 3-DOF rotation error — the
capability was never optional for this goal, our default solver just traded it
away for saddle-free top-down chains. The backend self-calibrates against the
scene at construction and refuses to run if the vendored URDF disagrees with
the imported MJCF (`assets/panda/panda_kinematics.urdf`, parity measured at
≤0.001 mm / 0.0000°).

The scripted routine is a different policy from this graph, and on `syringe` it
currently scores **0/9**: measured 2026-08-17 over episodes 0:9, identically on
`tool_sort/syringe` and on the `sharps_disposal/syringe` fork at the commit
before the split, every episode ending in `policy_request` at 1201 steps.
`tools/make_splash.py` still records 1.00 for it from 2026-08-12 and its tile
claims to show the task done; those two statements cannot both be true, and the
splash is the one that has not been re-measured. (`sharps_disposal/gt` was this
file, copied whole when the syringe was briefly its own suite; the fork merged
back on 2026-09-02 and the copy is gone.)

**Read repeated episodes as repeated attempts at one scene, not as breadth.**
`tool_sort` declares `EpisodeSpec(count=1, seed=0, sampler=fixed)`, so every
episode id builds the identical pile — the export records one settled
arrangement and re-sampling it would discard the only one known to be physically
consistent. Layout variation means re-exporting with another seed.

And the outcome is genuinely a *rate* rather than a function of the episode id:
the graph executes on a worker thread, so the number of control steps that
elapse between one `robot.*` tool returning and the next being issued depends on
scheduling. Do not quote a success rate off a single episode.

### The UR5 runs this graph, and what stops it is that it obeys the wrist roll

Measured 2026-08-15 on `abf9637` + the depth clamp, serial on an idle GPU,
`--embodiment ur5 --allow-undeclared-graph`, `tool_sort/hammer`:

| UR5 configuration | result |
| --- | --- |
| `--ik auto` — honours roll, because `ur5.py` declares `honour_roll=True` | **0/5**, 1504–1515 steps, gripper closes to 0.016–0.019 on nothing |
| `--ik newton` — roll free, *which is exactly what the Panda runs by default* | **2/2 success**, 879, 880 steps |

The pairing was never the problem. It builds, steps, and executes every node of
the workflow with `error=None` — perception, grasp, align, descend, close,
verify, give-up and the goal cursor all work on a 6-DOF arm with a different
hand. What fails is one thing: the grasp closes on air whenever the arm actually
*holds* the orientation the graph asked for.

**Two independent derivations of that orientation were refuted.** The shipped
route is `sim.get_object_obb`'s precomputed `grasp_quat`. The alternative is
`astro`'s: take the box's narrowest horizontal axis and ask `robot.grasp_frame`
to put this hand's measured axes on it. Swapping to the second leaves the Panda
unchanged (706, 707 against 704, 704) and leaves the UR5 exactly as broken (0/2
at 1507, 1504 against 0/5 at 1504–1515) — two derivations failing within three
steps of each other. So it is **not which roll is commanded**. It is that this
arm holds one at all. That refutation is recorded in `apply_grasp_yaw.py` and
the experiment was not kept.

The open hypothesis, untested when this was written: six joints and no
redundancy make a demanded position *plus* a demanded orientation exactly
determined, so `solve_q_full` has nothing to trade and may satisfy the
orientation while giving up millimetres of position. The Panda's seventh joint
would absorb that, which would be a second reason it never noticed. The
measurement that settles it is commanded-versus-achieved at the grasp waypoint,
split into position, approach axis and roll, under both backends
(`scratchpad/g3/ik_probe.py` was written for exactly this and did not get run).

**No `graphs=` claim is written for this pairing, deliberately.**
`check_graph_claims` only tests that a binding names the graph — it knows
nothing about `--ik`. Writing `graphs=("gt",)` would therefore pass its own gate
while a default-flag run still failed at the grasp, which is worse than the
honest "compiles; no policy". The two ways to earn the claim are to make the
grasp survive an honoured roll, or to give `Binding` a way to say "this graph
wants the wrist free here", which does not exist today.

### Two things a third robot will need to know

Both were found porting this graph to the UR5 and neither is a fact about
`tool_sort`. They are here because this is where they were measured.

**Which term of the depth clamp is load-bearing depends on the hand, and on one
of these two it is a fallback nobody measured.** `clamp_grasp_depth.py` takes
`max(fraction, top − housing_rise)`. Those do not merely differ per hand — a
*different one of them governs*:

| Hand | `housing_rise` | box height | fraction | `top − rise` | grips | governed by |
| --- | --- | --- | --- | --- | --- | --- |
| Panda (this graph's native) | **unmeasured**, 0.035 fallback | 80.0 mm | 28.00 | 45.00 | 45.00 mm | housing |
| UR5's Robotiq 2F-85 | 0.0597, declared | 85.0 mm | 29.75 | 25.30 | 29.75 mm | fraction |

So the hammer grasp that works on the Panda works because an *unmeasured*
constant happens to lift it to 45 mm. Declare a real housing rise and that term
relaxes, the fraction takes over, and the grasp drops 15 mm. `grip_height_frac`
is passed as `0.5` in the workflow for this reason — it is the routine's own
rule, grasp at the box centre, and it is a provable no-op on the Panda (the
housing term still binds at 45.0 mm; measured 704, 704 steps unchanged) while
lifting the UR5 to 42.5 mm. Anyone bringing a third hand should work out which
term governs on it before trusting the depth.

**`--ik auto` and `--ik newton` are the same run on the Panda and a different
run on the UR5.** `auto` reads `EmbodimentSpec.honour_roll`; the Panda does not
declare it, so `auto` *is* `newton` there and the two names are interchangeable.
The UR5 declares it, and on this variant that is the difference between 0/5
(1504–1515 steps, closes on nothing) and 2/2 (879, 880 steps, success). A
measurement of this graph that does not state its `--ik` is therefore not
reproducible on any robot declaring `honour_roll`, and every number in this file
is `--ik auto` on the Panda unless it says otherwise.

### Do not measure this arm under contention

Measured 2026-08-15 on this variant, and it is not a small effect. Four fresh
graph processes run **serially** on an idle GPU give 0/4 at 1305 steps
[1299, 1308] — tight enough to read as deterministic. The identical four run
**concurrently** on one GPU give 2/4, at 1179, 2177, 2191 and 2401 steps, the
last of them truncated against the 2400 budget. Same build, same seeds, same
tree.

The mechanism is the paragraph above: the graph's step budget is spent by a
worker thread racing the simulator, and taking the GPU away changes how many
control steps elapse per `robot.*` call. `film_peel` recorded the same hazard
from the other direction — eleven oversubscribed runs returning a zeroed metric
where the serial build was perfect. The scripted arm is immune both times
(4-way concurrent matched a serial control at 786 steps exactly, on this variant
on the same day), which is exactly what makes the trap easy to walk into: the
baseline survives oversubscription and the graph does not, so a campaign that
oversubscribes measures the graph badly and the routine correctly, and then
reports the difference as the graph's.

### The grasp had no depth rule, and a re-settled pile found out

Between 2026-08-07 and 2026-08-15 this variant went from 6/6 to **0/4**, and
nothing in this directory changed. What changed is the scene: `75a953f` replaced
this suite's authored-box colliders with the hulls of the meshes the export
actually draws, which is a straightforward physics correction, and re-settled
every export's start poses with it ("the ten-tool bins move 25-40 mm"). The
hammer came to rest nearly on end — its long axis 97.8% aligned with world z.

That was enough because the grasp had no depth rule to be robust with.
`geometry.top_down_grasp_candidates` places the fingertips a fixed `z_offset`
below the box's **tilt-aware** top, which is `centre + Σ|R[2,i]|·half` and not
the z half-extent. Lying flat the hammer's support is its 40 mm half-extent, so
`-0.03` put the fingertips 10 mm above the box centre, around the handle.
Standing up, the support is the *long* axis instead — 73.2 mm — so the same
`-0.03` put them 43.2 mm above the centre, in the air above the claw. The
fingers closed on nothing, `verify` routed to `failed` both attempts, and the
episode gave up at ~1305 steps having done everything else correctly.

The rule that fixes it is `clamp_grasp_depth.py`, the shared node `astro`
had and this suite did not: aim at a fraction of the object's
own height measured up from its underside, raise it if the hand's housing
cannot reach that low, and clamp into a band. Being anchored to the box centre
rather than to the top face, it does not move when the object tips. On this
hammer it aims at 0.855 m against the routine's 0.850 m and the old rule's
0.893 m, and the variant returns to 4/4 at 704 steps.

**A `bottom_z` clamp alone would not have caught this, and that is worth
stating** because `top_down_grasp_candidates`' own docstring names exactly that
guard and this graph already had it, in `apply_grasp_yaw.py`. A floor clamp only
ever raises the fingertips. The failure here was 43.2 mm *above* the box centre
and comfortably inside a 146 mm-tall box, so the floor guard was never reached
and never fired. Necessary, and not sufficient: what was missing was a ceiling
and a rule that knows how tall the object is. `apply_grasp_yaw.py` no longer
clamps depth at all — one rule, in one node.

### What is still unsolved

* **Containers are not compartments.** `syringe` and the third goal of
  `sort_power` want an object *posted through* a hole, and this graph descends
  and releases as if aiming at an open compartment, so the syringe ends up on
  the sealed box's lid. The reference does this in a separate `post_upright()`
  path; the export already records the box's `rim_z` and `aperture` and no tool
  exposes them yet.
* **Grasp quality, per tool.** The verification node now makes this legible
  rather than silent — `sort_new` reports a failed grasp instead of carrying air
  — but reporting it is not fixing it. The levers not yet explored are the
  gripper's `target_ke`, the closed width (currently driven to zero, which can
  pop a thin handle out from between the pads), and choosing among the planner's
  candidate poses rather than always taking `poses.0`.
* **Some tools cannot be done at all**, by construction, and that is the
  benchmark's intent rather than a defect: the tape measure is 161 mm across
  against an 80 mm span, the adjustable wrench is longer than a compartment's
  interior diagonal, the utility knife and insulating tape are thinner than the
  fingertips' reach above the bin floor. `sort_bench` is largely made of these.
  A full-marks result on those variants is not the target.

### What full marks would even mean, per variant

Worth stating, because a multi-goal variant scored against 1.0 looks like a
total failure when it is at its ceiling. Measured widths against the Panda's
80 mm span:

| Variant | Reachable ceiling | Why |
| --- | --- | --- |
| `hammer` | **1.0** | one tool, graspable by its 36 mm handle |
| `syringe` | 1.0 | one tool, but needs an orientation the IK cannot command |
| `sort_all` | 0.5 of 4 | `drill` 211 mm and `clamp` 147 mm are wider than the span everywhere |
| `sort_power` | 0.33 of 3 | both power tools likewise; only the syringe leg is in reach |
| `sort_new` | 0.75 of 8 | the utility knife is 10.8 mm thick — fingertips hit the bin floor first |
| `sort_bench` | 0.25 of 8 | three of four types out of reach by construction |

On `sort_all` the observed behaviour matches: the hammer is picked and
transported, the other three never leave the pile. The hammer's placement is
still marginal though — it ends about 3 mm outside its compartment in x and high
on a divider, so it registers `in` while carried through the window and not at
rest. Making the release land it squarely is the open item, and it is invisible
on the single-goal variant because the episode terminates the moment the goal
first reads true.

## Running it

```bash
python -m vabar run --solve graph \
    --task tool_sort/hammer --graph gt --episodes 0:1 --renderer tiled
```

Needs the `[gap]` extra. The runtime resolves `geometry.*` from the
open-robot-skills checkout beside the gap one; `--no-privileged` withholds the
`sim.*` tools, which is how you check that a graph has actually been converted
to perception.
