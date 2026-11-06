# cable_routing/L1 as fleet waves

Cell `cable_routing/L1` = `cable_route/port_easy`, 30 instances, graph `gt` (= `graphs/weave`),
`--config fast`, cell protocol (`--no-privileged --perception vision --renderer rtx --bus-timeout 600
--settle-steps 60`, palette segmentation). Measured 2026-11-05 on RTX PRO 6000 Blackwell GPUs 2 and 6,
RoboSimStudio = `robosimstudio-sph` plus the patch below (a private copy, `robosimstudio-fleet`),
graph-as-policy `graph-as-policy-share`. The host was shared (other tenants' jobs on other GPUs,
load average 35-80).

Until now this cell ran as single-world processes under MPS (`num_worlds = 1`, `workers = 6`): 34-46 min
a cell. Its fleet run had failed twice over: every instance keyed into its own model, so a "wave" held one
instance, and the weave graph's carry aborted on `robot.stream_dual: this facade cannot stream several
arms together`.

**Result: yes, a big win, at the same fidelity.** With five fixes (one in the suite, four in RoboSimStudio,
one of them a pre-existing friction bug that broke every fleet world but the first), L1 runs as two waves
of 15 worlds on two MPS workers in **16.9-18.0 min** (three runs) against **46.5 min** for the shipped
6-worker single-world pool on the same stack -- 2.6-2.75x, at 11.7 GB of GPU memory against 32.8 GB.
Outcomes are inside the serial runs' own spread: 20, 16 and 19/30 (mean reward 0.875, 0.833, 0.867)
against 18, 16 and 15/30 (0.817, 0.842, 0.825) serially; per instance, fleet and serial agree as often as
two serial runs do. Every fleet world starts from its own instance's single-world state (rod and fixtures
to 0.0001 mm). `default.toml` now runs L1 as 15-world waves (`[suites.cable_route] num_worlds = 15`;
the 2 workers follow from the 2 waves) and keeps L2 single-world. The floor is set by the cell's longest
episode, not the GPU: a wave advances at 9-12 steps/s, so a cell whose longest episode truncates at
13,000 steps takes about 24 min.

## What had to change

Five changes. One is in the suite, four in RoboSimStudio (the fleet runtime and the rod).

### 1. The anchor and port claim a body per world (`tasks/cable_route/assets/terminals.py`)

`PORT_BOARD` draws all four fixtures per episode: the pegs, the anchor (it follows the rod) and the port
(the slop of its bolt slots). The fleet keys episodes into one model only when every static an episode
moves can be placed per world (`robosimstudio_bench/run/fleet.py`, `fleet_model_key`). The pegs could
(`EntityPegRound` claims a body); `EntityCableTerminal` and `EntityCablePort` drew their parts world-hung
(`body=-1`) and did not, so all 30 instances had 30 distinct fleet keys and every wave held one instance
(`[fleet] rebuilding for 1 drawn bench(es)`).

`EntityCableTerminal` (and through it the port and the cleat) now mixes in `PlacedAtBuild`, sets
`claims_body = True` and calls `_claim_body` in `build`. On a replicated build it draws its parts on a
kinematic body welded where the episode put it, in the body's frame; the body is placed per world at build
and again at reset. A single-world build never claims the body and draws exactly what it drew before
(world-hung, at the episode's pose). After the change the 30 instances share one fleet key.

### 2. `WorldFacade.stream_dual_trajectory` (`robosimstudio_policy_agent/fleet/world_facade.py`, `intents.py`)

`GapEnvFacade.stream_dual_trajectory` semantics, as fleet motions:

* waypoint `i` of every track on the same control step. One `JointTrajectory` per arm, submitted as one
  group: the new `IntentTable.submit_group` puts every member in the table under one lock, so one drain
  installs all of them and they start on the same step;
* a short track holds its last waypoint (padded), and two ids naming one arm keep the later track, as the
  serial frame's later write wins;
* the serial cadence of two control steps a waypoint (`JointTrajectory.steps_per_waypoint`), then each arm
  settles on its own end point (`settle_max_steps=0`, which the cable carry passes, waits for nothing);
* a path longer than the segment driver holds (512) is cut into consecutive pieces rather than resampled.
  The weave's longest carry on this cell is 529 waypoints; resampled, it would have run 34 steps fast.

The group also needs the park to be the group's: a completion unparks its waiter outright
(`ParkTable.unpark`), so a thread waiting on two arms inside one park would be unparked by the first arm to
finish, read as running while still blocked on the second, and hold the barrier until the stall watchdog
retired the world. `_HandleGroup` makes only the group's last completion unpark. The method is also added
to the `EnvFacade` protocol. Tests: `tests/test_fleet_dual_stream.py` (13).

Nothing else in the weave graph or the `cable.*` tools needed a fleet fix. Unlike `usbc_insert`, the
cable tools only *read* `env.state` (through `WorldView`'s per-world slices) and write the gripper through
the facade/bank; no tool writes `joint_q` or steps a solver. `cable.wait_steps` and the `set_gripper` ramp
keep wall-clock deadlines (60 s, 10 s a step) that a wide wave could in principle hit while another world
holds the barrier; none came back short in any run here (checked on every trace).

### 3. Every world's rod gets the rod's friction (`robosimstudio_materials/rod/entity.py`)

**This was the fidelity bug, and it predates this work.** `EntityRod.apply_model_materials` runs on the
*replicated* model in a wave, but it took the rod's own shapes as template indices -- so only world 0's 50
capsules got `mu_cable` (60). Every other world's rod fell through to `mu_base` (2.0): the pad-on-cable
grip, sqrt-combined, was sqrt(2 x 60) = 11 instead of 60. The rod now writes its capsules in every world.
Tests: `tests/test_rod_materials_per_world.py` (5).

How it was found: with changes 1 and 2 (and 4 in one of them) the full-cell fleet runs scored 5/30 (two
configurations, 5/30 each) against 18/30 serially; seats held (30/25 of 30) but the terminal clauses collapsed (port box 11/30 against
21-25/30). A controlled contact probe (`agent_scratch/fleet/tools/port_contact.py`) moves one fixture so
it presses 2 mm into the resting rod, in an otherwise unchanged episode, and steps both builds idle:

| Rod pressed by | single vs single (noise) | single vs fleet world 0 | single vs worlds 1-3, before | after the fix |
| --- | --- | --- | --- | --- |
| port, 2 mm | 0.07-0.12 mm | 0.05-0.06 mm | **0.74-0.76 mm** | 0.03-0.08 mm |
| peg1, 0.75 mm | 0.01-0.02 mm | 0.01-0.02 mm | **0.38-0.45 mm** | 0.01-0.03 mm |

(max over the rod of the position gap after 60 steps.) Worlds 1-3 were identical to one another and wrong
whether or not world 0 touched anything; a diff of every per-shape/body/joint model array between world 0
and world 1 copies (`world_diff.py`) found one physical difference, `shape_material_mu` on the rod's 50
capsules. The same template-index pattern is in `robosimstudio_materials/paper.py`
(`apply_model_materials`, `values[self.pinned] = 0.0`); not touched here.

### 4. The fleet servo turns at the serial servo's cadence (`fleet/stepper.py`, `ServoToPose.steps_per_turn`)

`robot.go_to_pose` (the plug stage's carry) runs as `ServoToPose`. Under the lockstep clock a serial
servo turn is one bus call that steps once and then the rollout's hold step: it measures once, advances the
setpoint law twice from that measurement, and its command reaches the arm after the hold. The fleet
re-aimed every step from a fresh measurement and commanded at once, so the lead clamp bound later and the
arm moved faster. `ServoToPose.steps_per_turn = 2` restores the turn (re-aim and judge once per two steps,
both law ticks from one measurement, command a hold later); 1 is the old behaviour. On the first plug move
of 10062 (same start, same target) serial takes 25 steps in both serial runs, the fleet took 20 before and
22 after; over all matched moves the fleet/serial step ratio went from a median 0.81-0.86 to 0.90-0.91.
The rest is not explained. It changes every suite's fleet servo, so the tool_hanging fleet cell
(29-30/30) should be re-checked once. Tests: `tests/test_fleet_servo_turn.py` (6).

### 5. A failed workflow is reported as failed (`fleet/solution.py`)

The serial cells end 1-4 episodes each "graph failed" (8 of 90: the graph routed to `abort`, e.g. a
planner refusal in `restore` or `mate`, once at step 90 on a misperceived centreline); the fleet runs
reported none (0 of 180). The behaviour was the same -- the fleet rows carry the failure in `graph_error`
on 2-3 episodes a run (16 of 180, the serial rate), e.g. `restore`'s "No feasible grasp relative to the
fixed anchor" on 10022 -- but `SolutionGraphBatched` worded the termination off the graph thread's
exception only, and `gap.execute` puts a workflow failure on its result instead of raising. It now reads both, as the single-world policy does. Reporting only: the close kind, the step at
which the world closes and the score are unchanged. Tests: `tests/test_fleet_termination_reason.py` (3).

## Fidelity

### Initial state, per world

`agent_scratch/fleet/tools/init_state.py` builds the instances as one fleet wave (each world built for its
own instance, as the runner does) and each instance as a single-world `Env`, resets both, and compares
every rod body and every shape of every fixture in world coordinates (body pose composed with the shape's
own transform, so a fitting on its own body and a world-hung one compare like for like), the fixture poses
the evaluator's regions ride on, and the same after 60 idle steps.

Eight L1 instances (10009, 10062, 10084, 10043, 10053, 10010, 10020, 10035) in one 8-world wave
(`fidelity/init8_steps60.json`):

| Per world | At reset | After 60 idle steps |
| --- | --- | --- |
| rod bodies (50), max position gap to the single-world build | 0.0000 mm | 0.0007-0.0036 mm |
| fixture shapes (anchor 6, port 5, pegs 1+1), max gap | 0.00003-0.00006 mm | same |
| fixture pose as the evaluator's context resolves it (regions ride on it) | <= 0.00003 mm | -- |
| 29 reset metrics, max difference | 1.2e-5 | -- |
| distance to the *nearest other* world's instance | 20-53 mm | -- |

Every world matches its own instance and no other. (At reset only, the gripper fingertip bodies differ by
27 mm between builds -- one build's `body_q` for the fingers is stale until the first step, with joint
coordinates identical; after the first step they agree.)

The contact probe in "3." above is the check that found the friction bug; after the fix it shows every
world responding to a fixture as the single-world build does, within that build's own repeatability.

### The full cell

All runs: the whole cell, `--config fast`, cell protocol. "Serial" is the shipped single-world setup
(1 world per process, 6 MPS workers). `SW6` was run on this exact stack (the same RoboSimStudio copy and
VABAR tree, so the terminals change is inert in it); fix 1 and fix 2 are the cells of
[cable_routing_fix](cable_routing_fix.md) (fix 2 is the shipped `plug_port.py`; fix 1 differs only in the
terminal grasp's jaw opening).

| Run | Worlds x workers | Success | Mean reward | seat0 / seat1 / port box / end | Mean steps | Ended |
| --- | --- | --- | --- | --- | --- | --- |
| serial SW6 | 1 x 6 | 18/30 | 0.817 | 28 / 27 / 21 / 22 | 6,444 | 25 graph finished, 3 graph failed, 2 truncated |
| serial fix 2 | 1 x 6 | 16/30 | 0.842 | 27 / 26 / 25 / 23 | 6,339 | 26 finished, 4 failed |
| serial fix 1 | 1 x 6 | 15/30 | 0.825 | 29 / 30 / 21 / 19 | 6,380 | 29 finished, 1 failed |
| **fleet** | 15 x 2 | **20/30** | **0.875** | 29 / 28 / 22 / 26 | 6,113 | 28 finished, 2 failed* |
| **fleet** | 8 x 2 | **17/30** | **0.800** | 26 / 27 / 20 / 23 | 6,071 | 26 finished, 3 failed*, 1 truncated |
| **fleet** | 30 x 1 | **17/30** | **0.792** | 29 / 25 / 20 / 21 | 6,618 | 26 finished, 3 failed*, 1 truncated |
| **fleet**, repeat | 15 x 2 | **16/30** | **0.833** | 27 / 30 / 21 / 22 | 6,029 | 27 finished, 3 failed* |
| **fleet** | 8 x 4 | **16/30** | **0.800** | 29 / 27 / 20 / 20 | 6,345 | 27 finished, 3 failed* |
| **fleet**, third run (the shipped config) | 15 x 2 | **19/30** | **0.867** | 28 / 29 / 22 / 25 | 5,705 | 28 finished, 2 failed* |
| *before fix 3 (rod friction)* | 8 x 2 | 5/30 | 0.675 | 30 / 25 / 11 / 15 | 6,393 | |
| *before fix 3 (rod friction)* | 15 x 2 | 5/30 | 0.692 | 30 / 24 / 11 / 18 | 6,340 | |

\* counted from the row's `graph_error`; these runs predate change 5 and reported them "graph finished".

Per instance, the fleet is no further from serial than serial is from itself
(`tools/agreement.py`; same success on N of 30 instances):

| Pair | Same success | Same reward | Mean abs. reward difference |
| --- | --- | --- | --- |
| serial SW6 vs serial fix 2 (same code) | 20 | 14 | 0.175 |
| serial vs serial, all three pairs | 15-20 (mean 17.3) | 12-14 | 0.175-0.225 |
| fleet vs fleet, all 15 pairs | 16-23 (mean 19.8) | | |
| serial vs fleet, all 18 pairs | 13-23 (mean 18.3) | | |
| serial SW6 vs fleet 15 x 2 | 22 | 18 | 0.158 |

No world ran another instance's board or rod: every row's evaluator scored its own document (rows are
labelled by world and episode), the reset check above matches each world to its own instance only, and
the seat clauses -- which are measured against the drawn pegs of the row's own instance -- held 25-30 of 30
in every fleet run, as serially.

Two smaller differences remain and are reported, not fixed:

* **Servo speed.** After change 4 the fleet's `go_to_pose` still reaches its target in a median 0.90 of
  the serial step count (155 matched moves of the 15 x 2 run against SW6; outcomes agree on 181 of 184).
* **Trajectory settle.** `robot.execute_trajectory` returns with the arm a little less settled in a wave
  (final tracking error median 0.0029 rad against 0.0015 serially): the fleet ends the settle on the first
  step inside the tolerance, while the serial blocking move checks every second step and from its second
  iteration. Both are well inside the 0.01 rad tolerance the graph asks for.

## Speed

Wall time from each run's own `[run] ... min all in` line (start of the pool to its end, builds included);
steps/s from its `[run] ... control steps, N/s end to end` line. GPU memory is the run's peak over the
GPU's idle level, sampled every 5 s with `nvidia-smi` (MPS daemon included); a fleet worker holds 5.0 GB
at 8 worlds, 5.8 GB at 15 and 6.2 GB at 30. A wave's rate is its longest episode's steps over its
seconds (`tools/wave_stats.py`).

| Run | Worlds x workers | Wall, all in | Steps/s end to end | Wave rate (steps/s) | Longest episode | GPU memory peak | Success / mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| serial SW6 | 1 x 6 (MPS) | 46.5 min | 70 | (13.0 per episode, median) | 13,001 | 32.8 GB | 18/30, 0.817 |
| **fleet** | **15 x 2** | **17.0 min** | **186** | 9.3, 9.3 | 9,232 | **11.7 GB** | 20/30, 0.875 |
| fleet, repeat | 15 x 2 | 16.9 min | 185 | 9.3, 9.6 | 9,393 | 11.7 GB | 16/30, 0.833 |
| fleet, third run (the shipped config) | 15 x 2 | 18.0 min | 163 | 10.2, 9.6 | 10,665 | 11.7 GB | 19/30, 0.867 |
| fleet | 8 x 4 | 18.1 min | 180 | 8.9-10.5 | 10,981 | 22.3 GB | 16/30, 0.800 |
| fleet | 30 x 1 | 24.4 min (wall of the run script; 23.5 min of episodes) | 141 | 9.2 | 13,000 (truncated) | 6.2 GB | 17/30, 0.792 |
| fleet | 8 x 2 | 28.7 min | 108 | 11.5-12.1 | 13,000 (truncated) | 11.5 GB | 17/30, 0.800 |

The 8 x 2 run shared its GPU with the contact probes for part of its first wave; its wave rates are clean
(11.5-12.1), its wall time is two waves per worker.

What sets the wall time:

* **A wave advances at 9-12 control steps a second however many worlds it holds** (8 worlds: 11.5-12.1
  on 2 workers; 15: 9.3; 30: 9.2; 8 on 4 workers: 8.9-10.5), against about 13 for one world alone. The GPU
  is not the limit -- 30 worlds hold 6.2 GB. A barrier step waits until every live world's graph thread
  is parked (all of them Python threads under one process's GIL), and any world that is perceiving or
  planning holds the whole wave. The idle probe's 21 steps/s for 8 worlds is the physics alone.
* So the cell takes its **longest episode** at that rate, plus about a minute of builds. L1's longest
  episodes run 9,000-13,000 steps (13,000 is the budget; 1-2 truncate in most runs): 15-24 min. One wave
  per worker is what matters; how the 30 are split hardly does (15 x 2 and 8 x 4 within a minute of each
  other), and a second wave per worker (8 x 2) adds a whole wave.
* **15 x 2 is the setting**: the fastest measured (16.9-18.0 min over three runs), the least GPU memory of
  the multi-worker options, and two processes rather than four. It is 2.6-2.75x the serial pool's speed on
  the same stack (against 46.5 min), delivering 163-186 control steps/s against 70. A cell whose longest
  episode truncates at 13,000 steps will take about 24 min at this setting, still about half the serial
  time.

## Reproduce

```bash
export PYTHONPATH=<worktree>:<robosimstudio-fleet>:<graph-as-policy>:<graph-as-policy>/gap-core/src
export __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1
CUDA_VISIBLE_DEVICES=6 python -m vabar bench --cell cable_routing/L1 --solve graph --config fast --out <dir>
# another combination
CUDA_VISIBLE_DEVICES=6 python -m vabar bench --cell cable_routing/L1 --solve graph --config fast \
  --set suites.cable_route.num_worlds=8 --set workers=2 --out <dir>
```

Raw runs, logs (`<run>.log`), GPU samples (`<run>.gpu`, memory MiB and utilisation every 5 s) and the
probes are in `/home/shadeform/speed_ws/agent_scratch/fleet/` (`runs/`, `fidelity/`, `tools/`); the
patches are in `patches/`.
