# VABAR on Isaac Lab Arena: System Design

## Objective

Connect industrial Isaac Lab Arena environments to GaP without putting benchmark content into the
adapter or integration behavior into the benchmark package. Arena remains the simulation,
recording, and scoring authority.

## Robot I/O boundary for GaP graphs

Every graph is AUTOLab v1.0's, run by RoboSimStudio's own policy agent in AUTOLab's locked
environment (`isaaclab_arena_gap.rss_adapter`, launched by `scripts/run_rss_graph.sh`).
RoboSimStudio's `SolutionGraph`, servo, IK, cuRobo planning and graph-facing tools run unmodified
against `rss_adapter.arena_env.ArenaEnv`, a RoboSimStudio robot-only `Sim` with three things
replaced:

- **Time.** One `step` decodes the action with RoboSimStudio's controllers, sends the arm and
  gripper targets to Arena, and blocks for Arena's next sample: one step is one Arena control
  period.
- **State.** The robot's joints, including the hand's passive linkage, are written from each Arena
  sample, and forward kinematics fills the body poses.
- **Cameras.** Frames, intrinsics and extrinsics are Arena's.

The robot is Arena's own asset carrying AUTOLab's registered hand (`rss_adapter/embodiments.py`);
the scene holds the robot and nothing else, so a tool that reaches for a simulated object finds
none. Task identity and instruction come from AUTOLab's task document; geometry never does,
because AUTOLab's world frame is not Arena's. The only non-robot handles are a hand calibration a
task declares as an entity (USB-C's `rig`) and the static fixtures a task declares
(`gap_scene_statics`: cable's pegs, anchor and port, at Arena's poses), neither of them simulated
object state.

The Arena driver is `bridge_robot_io.ArenaBridgeRobotIO` over the episode bridge
(`cap_franka_real_env` / `cap_bimanual_yam_env`), whose transport and scoring lifecycle stay with
Arena. A sample contains timestamped cameras plus measured, named joint positions and velocities;
one `exchange` accepts an absolute target for every named joint and returns a fresh joint sample.
The driver raises on per-command gains it cannot apply. The FR3 sample includes the measured
passive gripper joints; only arm and gripper entries are actuated. Arena does not calculate or
transmit fingertip geometry to the policy.

Task-pair workspace measurements are validated policy configuration and passed to the adapter
through child-process configuration; they are not robot sensor samples.

## Workspace ownership

`isaaclab_arena_gap/` owns:

- the synchronous msgpack bridge and GaP process lifecycle;
- the RSS adapter and the Arena robot I/O drivers it runs over;
- typed Arena Experiment Definitions and policy configuration;
- pair-specific `rss_*` graph adaptations;
- local setup, run, show, and preflight launchers;
- the exact-source evaluator image and narrow Arena OSMO workflow adapter; and
- tests of the public integration interfaces.

`isaaclab_arena_vabar/` owns:

- policy-independent scenes, tasks, assets, scoring, and definitions;
- FR3/Robotiq and bimanual-YAM embodiments;
- generated registrations and package-owned task adaptations; and
- physical asset and environment acceptance tests.

`tools/` owns independently movable authoring and conversion utilities. Arena owns reusable scene
building, placement, camera, recording, scoring, and typed experiment-runner framework behavior.

The `external/` gitlinks are immutable source pins. `graph-as-policy` owns the graph runtime;
rigid-task tools and skills come from `open-robot-skills`; the RSS version at the root of the VABAR
repository, which encloses this folder, owns the reusable model and motion stack.

## Shared camera rigs

Every task uses the camera rig its AUTOLab suite stages, one rig per suite.
Franka tasks use `gear_mesh_views@v2` (gear), `tool_sort_agentview@v3` (tool
sorting) and `sharps_agentview@v1` (syringe, formerly `agentview@v2`). Their fixed views ride the FR3 base at
their RSS offset from the Panda base, and the tool-sorting and syringe
rigs add a Robotiq wrist camera. YAM tasks use AUTOLab's YAM station, every
camera 1280 x 720 at 30 Hz. Tool hanging uses the shared rig: its
`top_camera` and `side_camera`, each kept at its RSS offset from the
base-pair midpoint, the cable station's `cable_camera` raised to look over the
pegboard, and `left_wrist_camera`/`right_wrist_camera` D405s on link 6.
USB-C uses `yam_station_usbc@v1` at its own poses (`YamUsbcCameraCfg`): its
head and left-shoulder views, each higher than tool hanging's, with the D405s on
the `linear_4310` hand's gripper body.
Cable uses `yam_station_cable@v1` at its own poses (`YamCableCameraCfg`): its
overhead, its oblique `cable_camera`, and the D405s on the `linear_4310` hand's
gripper body. Scenes select the FR3 rig through the
embodiment's `camera_rig` parameter; enabling FR3 cameras without one fails.
Graph aliases keep the RSS camera names.
Embodiments own camera calibration and asset-frame conversions. Task factories
do not modify it. RGB-D output remains the GaP sensor contract.

Every camera scene converges its RTX frames before an episode's first
observation. A frame rendered right after a reset is soft and loses thin detail
for 20 to 40 frames. Isaac Lab's `num_rerenders_on_reset` cannot help headless:
`SimulationContext.render()` syncs poses but never runs Kit's update, and RTX
renders a frame only when a camera reads its data, so the rerenders collapse
into one frame.
The camera contract therefore adds a last reset event, `rtx_camera_warmup`,
that renders 60 frames of the reset scene. It steps no physics, so episode time
and step counts do not change for any policy; each reset costs only render time.

## Embodiment assets, lighting and recording

The industrial FR3 is a generated robot USD of Franka's original `fr3_v1` hardware revision
with the Robotiq 2F-85 (dark covers on links 3 to 5) and keeps its source `fr3_*`
joint and body names; the all-white FR3-v2 closure is not used.
`vabar__bimanual_yam` uses the vendored canonical I2RT YAM closure twice with
independent left/right articulation state. Every tool-hanging, USB-C and cable task
selects it; tool sorting uses FR3/Robotiq. Its USD has a `gripper` variant set:
`crank_4310` (default; crank-shaft hand, 75 mm total opening) and `linear_4310`
(the linear hand, 95 mm, a generated robot USD of AUTOLab's MJCF hand with its CoACD fingertip
colliders).
Cable selects `linear_4310`, which AUTOLab's cable graph is written against, and
the `yam_bimanual_linear` GaP profile; it stands the arms at AUTOLab's cable workcell mounts
(0.62 m apart) and cable-station home, and replaces the fingertip contact material
with the cable's through `embodiments/bimanual_yam/contacts.py`.

Every graph scene uses `vabar__empty_warehouse_dome_light`: Arena's
registered `empty_warehouse_robolab` latitude-longitude HDR with neutral
`(0.75, 0.75, 0.75)` color and intensity `1500`, visible to the primary camera
ray. The scene loads that HDR from Isaac Lab's asset root (`ISAACLAB_NUCLEUS_DIR`)
rather than from Arena's own asset root, which does not host the Isaac Sim 6.1 copy, so camera or
render acceptance needs network access to that URL. Each graph scene also selects
the hidden, collision-free `vabar__hdr_shadow_receiver` (its USD
lives under `vabar__hdr_shadow_receiver/`). Standalone cable routing (Easy
and Medium) builds the shadow receiver directly but no warehouse HDR: it is lit
by the dome and panel lights authored in AUTOLab's RSS room
(`cable_routing_assets/native_appearance/room.usda`). Arena's rollout wrapper records at each
environment's native rate (50 FPS for syringe tasks).

## Processes and lifecycle

```text
20 independent GaP sessions                     One Arena process (Python 3.12)
  private graph/session/episode state              20 Newton environments
  private tool bundles                             NxD policy actions
  scalar ctx.tool() calls                          per-env reset/retirement
```

Managed jobs set `auto_spawn=true`. Scalar jobs retain the established one-process, one-socket
lifecycle. Fleet jobs allocate one independent GaP process, bridge socket, graph state, trace, seed,
session ID, and episode ID per Arena environment. Their request/reply exchanges run concurrently and
produce one ordered `[num_envs, action_dim]` action tensor; graph sessions are never advanced in
lockstep.

Each request and reply carries session, episode, and request identity. Stale replies fail only their
own slot. Partial reset increments that slot's episode identity and replaces only its GaP child.
Unexpected child exit likewise remains local. In one-wave evaluation, a completed slot is retired
instead of respawned so 20 Arena environments produce exactly 20 authoritative episodes.

Each vector environment's GaP session starts its own tool bundles; no tool process is shared
between sessions. The `num_envs=1` path is the same lifecycle with one session.

On the simulator side every environment is its own Newton world, which needs Arena's Newton
default `scene.replicate_physics: true`. Environment YAMLs must not set it to `false`: Newton then
parses the stage into a single world, and `num_envs > 1` fails articulation initialization with
`Invalid default value tensor shape`.

Before a graph marks an arm or gripper command valid, the adapter holds that channel at its observed
state instead of applying GaP startup seeds.

If GaP finishes first, its wrapper publishes a graph-result sidecar before the socket closes. A
declared failure requests immediate non-timeout episode termination. A declared success gives the
Arena task predicate a short settle interval. In every case, Arena computes the independent task
success value and writes the authoritative episode JSONL.

The observation monitor in the bridge arms only after the first RGB-bearing observation and uses
changed wire timestamps as progress. The first timestamp transition has a 180-second startup allowance for
deferred simulator work such as Newton CUDA-graph capture; after that transition, the steady-state
timeout is 60 seconds. `CAP_STARTUP_OBSERVATION_TIMEOUT_S` and
`CAP_STALE_OBSERVATION_TIMEOUT_S` override those values. On expiry the bridge raises
`StaleObservationError` into the adapter's next step, which ends the graph; the adapter publishes
its final graph result and Arena completes its own result path.

## Observation and action interface

Arena sends:

- ordered robot joint positions;
- normalized gripper state in GaP's open-fraction convention;
- one or more pair-configured RGB images and metric `distance_to_image_plane` depth arrays;
- live camera intrinsics;
- live OpenCV-optical camera poses in the robot-base frame.

Validated task-pair workspace measurements are passed separately through
child-process configuration; they are not part of the robot observation.

GaP returns absolute arm joint targets plus a normalized gripper open fraction. The FR3 profile
maps seven arm joints to the FR3/Robotiq action and converts open fraction to binary closedness.
The bimanual YAM profiles (`yam_bimanual`, `yam_bimanual_wide`, `yam_bimanual_linear`, one per
hand) map both six-joint arms and both continuous grippers, with independent per-arm and
per-gripper command-valid flags. A profile the adapter does not bind is rejected. Every action
must have the expected shape, numeric type, and finite values; malformed messages fail closed.

The bridge transport uses a four-byte big-endian length prefix followed by msgpack-numpy payloads.
Frames have an explicit size limit, connections have bounded startup and I/O timeouts, and a peer
close is an episode-ending condition rather than permission to respawn.

Arena episode JSONL remains the scoring authority.

## Industrial pair interface

The reusable integration is an Arena environment paired with a policy definition in one typed
Experiment YAML.
A GaP job may additionally name:

- a `<task>/<graph>` bound in `rss_adapter/runner.py` `BINDINGS`, from AUTOLab's pinned sources
  (`task_sources.autolab_root()`; graphs are never copied here), and optionally the AUTOLab task
  document (`gap_inputs.autolab_task`);
- camera scene keys and names exposed to the graph;
- one bound robot profile and workspace measurements; and
- an optional `rss_*` graph adaptation.

Shared bridge code contains no task-specific conditionals. The benchmark environment owns scene and
task semantics; the adapter owns robot and protocol conversion; the job owns their compatibility
selection.

The maintained managed pairs include `vabar_syringe_disposal__single`,
`__both`, `__designated`, and `__cluttered` with the unchanged
`syringe_packing/gap_perception` graph,
`vabar_tool_sorting__easy_1`, `_2`, and `_3` with `tool_sort/gap_perception_loop`, `vabar_tool_hanging__wrench_easy` with `tool_hanging/gap_perception`, and the
`vabar_gear_insertion__easy`, `__easy_pair`, and `__medium_train`
environments with `gear_mesh/gap_perception`.

Syringe both and cluttered lay their syringes out from AUTOLab's own trials (30 `tray_pair` trials,
7 clutter trials;
`isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/syringe_disposal/tray_layouts.py`). The trial is keyed on the placement seed:
seed `s` starts env `e` from trial `(s + e) mod count`, so a seed always gives the same trial, and the
seeds 42-51 sweep gives ten distinct pair trials and all seven clutter trials. The tray and box
positions within their YAML boxes come from the relation solve for that seed.

The baseline catalog contains 22 one-episode, one-rebuild managed GaP remote jobs. Every managed
GaP job runs through the RSS adapter (`isaaclab_arena_gap.rss_adapter`, launcher
`scripts/run_rss_graph.sh`) and names a graph bound there. The maintained test surface validates configuration and integration
only.

The registered `vabar_gear_insertion__easy` environment reproduces AUTOLab's
16-, 20-, and 24-tooth `gearmesh-easy-single` family: one loose gear, the matching board's fixed
plate and tapered peg, actuated 14-tooth pinion, spring button, and source-relative layout. A
coupled build-time variation selects the gear and board together. `on(table)` solves support Z for
the mat and board, and `on(gear_mat)` solves it for the loose gear. The mat remains fixed; bounded
X/Y and yaw relations reproduce the RSS generator's board drift, board turn, family-specific
gear footprint, 8 mm gear inset, and full-circle gear clocking. Pooled resets sample bounded pose
ranges through per-environment seeded RNGs, preserving reproducibility without collapsing yaw to
the solver midpoint. Its
policy-independent task latches the pinion after the button is depressed, then requires the released gear to remain seated,
upright, and turning at the expected meshing ratio for one second. The easy factory uses the source's
strict source-bounded layout pool, disables best-loss placement fallbacks, and removes the global
inter-object clearance that would otherwise reject the gear through its thin mat support. It also uses AUTOLab's
60 Hz, 16-substep Newton-contact profile in eager Warp mode. It preserves
generated contacts, uses latest-frame contact matching, and allocates the
RSS 32,768-contact ceiling.

The pair and train environments select one complete RSS generated row per rebuild in scalar
runs. That draw, and the easy family's tooth count, is keyed on the run's placement seed rather than
torch's global RNG, which Arena's build-time variations would otherwise use before the builder
applies the seed: the same seed builds the same row in every process. The gear
`*_gap_published_layouts.yaml` experiments go further and pin seeds 42-51: pair and train to fixed
AUTOLab documents, with the graph given the matching `autolab_task`, and easy to fixed tooth counts.
Vector runs accept an ordered `environment.layout_names` list with one distinct RSS row
per slot. Board variants and gear assets share the same clone index; placement and success targets
use that slot's row. Resetting a slot restores its assigned source start, and the episode JSONL records
its `gear_layout` ID. Every reset replays the layout the slot drew on its first reset
(`fixed_layouts.ReplayFirstLayouts`, also used by the syringe tray trials): Arena 0.3.1's pool would
otherwise solve fresh layouts that skip the source-row rewrite. Vector runs require explicit names to prevent shared starts by accident. The pair
pool contains 60 rows over six ordered two-gear tooth combinations. The train pool contains 100
rows over all 27 ordered tooth triples, retaining each row's two alternating chain bends. Their
recorded board and loose-gear X/Y/yaw values remain coupled; `on(table)` and `on(gear_mat)` solve
support height. Arena's conservative generic box-overlap verdict is not a rejection gate for these
already source-validated circular-clearance rows. Their shared task assigns indistinguishable
same-tooth gears to unused matching stations and requires every station to remain seated, turning,
and released after the assemble-then-start latch. Pair and train retain the generated 200-second
and 300-second horizons, respectively.

## Tool frames and motion

The graph-facing Cartesian convention matches AUTOLab's RSS version (BDDL): camera extrinsics, point
clouds, TCP poses, motion targets, and collision geometry share one task/world frame. In vector
Arena jobs this is environment-local world, so the adapter removes `scene.env_origins` rather than
exposing global stage coordinates. Robot motion remains an implementation detail behind that
interface: the adapter mounts each arm at its measured environment-local base pose, so shared-world
targets and collision geometry need no further conversion.

RoboSimStudio owns FK, IK, the Cartesian servo, cuRobo planning and the graph-facing `robot.*`
tools, on the embodiment spec the adapter builds from Arena's asset and AUTOLab's registered hand
and kinematics (`panda_robotiq` for the FR3; `yam_tool_hanging`, or `yam_i2rt` on the linear hand,
for the YAMs). This folder
supplies camera and joint samples and accepts whole-robot absolute joint-position commands. The
bimanual YAM sample keeps its `left` and `right` wire blocks; neither is implicitly mirrored.

Tool hanging supplies each task instruction through `gap_inputs`, which the adapter passes to the
graph in place of the document's. Planned trajectories are consumed as absolute joint targets; each
adapter step is one bridge exchange, which returns after one fresh Arena control step. The bridge
adds no per-target hold or settle cadence for any task.

## Robot-model differences from AUTOLab

AUTOLab authored the FR3 tasks on the Panda with the Franka hand; Arena runs the FR3 with the
Robotiq 2F-85. The differences below come from that robot model, not from a bug in Arena or the
adapter.

- **Syringe disposal, far picks.** Before the carry, the syringe graph turns the wrist to point the
  tip down, and keeps a turn only if it leaves at least 0.15 rad of joint room. The FR3's joint-6
  lower stop is 0.5445 rad; the Panda's is -0.0175 rad. On the FR3, that stop plus the 0.15 rad
  margin leaves too little joint-6 travel for the tip-down turn on picks about 0.69 m or more from
  the arm base, so those picks cannot take the path the RSS run flies. At the same reach the
  Panda keeps more than 0.6 rad of room. The joint-6 stop is the main cause; the 2F-85's TCP,
  52 mm further from the flange than the Franka hand's (0.1558 m against 0.1034 m), costs a
  further 0.07-0.08 rad, about 3 cm of reach.
- **Syringe disposal, fingertips on the tray floor.** The syringe graph targets the TCP at the
  centre of the perceived syringe box, about 6.6 mm above the floor the syringe lies on. The
  Franka hand's fingertips sit a fixed 9.3 mm past its TCP, so they meet the floor on the way
  down and close across the barrel. The 2F-85's fingertips sit 3.1 mm past its TCP when open and
  swing down to 17.8 mm past it as they close to the barrel's width, so they close into the floor
  and push the hand up about 11 mm. With AUTOLab's hand drive (below) they still close onto the
  barrel within GaP's 40-step ramp and 80-step settle.

## FR3 hand parity with AUTOLab's 2F-85

AUTOLab's 2F-85 (RoboSimStudio's `robotiq_2f85_panda` on `2f85.xml`) is the hand AUTOLab authored
gear insertion and tool sorting on, and the hand that the RSS adapter's robot-only shadow closes
on every FR3 task. Arena's
vendored hand is MuJoCo Menagerie's `robotiq_2f85_v4`, and its drive was carried over from DROID. The FR3
embodiment restates AUTOLab's hand drive:

- **Driver gains.** AUTOLab drives each finger's driver joint with 20 N.m/rad and 1 N.m.s/rad.
  Arena drives only the left driver, and the USD makes the right driver its mimic, so the left
  drive carries both fingers at 40 N.m/rad and 2 N.m.s/rad.
- **No effort cap.** AUTOLab states no effort for the drive, so Newton's 1e6 default stands rather
  than the vendored hand's 5 N.m cap.
- **Spring link.** MuJoCo Menagerie holds each spring link with 0.05 N.m/rad toward 2.62 rad. AUTOLab
  raises that tenfold, to 0.5 N.m/rad with 0.02 damping. The `vabar__fr3_robotiq_2f85_on_cart`
  layers restate AUTOLab's values and leave the vendored hand as generated.

With the vendored hand's drive (one 20 N.m/rad driver capped at 5 N.m), fingertips pressed into a floor reached a driver angle of 0.50-0.70 rad by the
end of GaP's close, then finished closing above the syringe barrel during the lift. With AUTOLab's
drive they reach 0.69-0.73 rad on the barrel. The spring holds the open hand at 0.06 rad rather than
0.01 rad, and the hand opens to 0.066 rad within 60 steps of a release: inside the 0.1 rad that
`objects_centered_in_regions_and_released` reads as open. With the spring alone, under the
vendored drive, the open hand rested at 0.125 rad and never read as open.

## Registration and launch

`isaaclab_arena_gap.arena_runner` wraps Arena's `SimulationAppContext`. Once AppLauncher has started, the
context idempotently registers the package-owned `isaaclab_arena_vabar` components.
Arena bundles no application definitions of this repository; conflicting registrations
fail closed. Benchmark FR3 graphs select `vabar__fr3_robotiq_2f85`:
its local robot asset and continuous gripper commands match the GaP static model.
Benchmark camera calibrations and the warehouse lighting preset also live in this repository.
The runner does not boot Isaac Sim before Arena's launcher initializes it.
Task success is declared through Arena's `CompletionCriteria` and evaluated by
its progress tracker; predicate conditions and dwell requirements remain unchanged.

`vabar-arena run` selects an Experiment YAML. `vabar-arena show` loads a graph YAML or registered environment
without GaP, holds the robot still, and disables outcome recording.

## Authored embodiment physics

Embodiment-owned contact, material, gravity-compensation, joint-limit, and drive properties live
in the embodiment's USD closure whenever USD has a native representation for them. Scene
configuration selects the asset and supplies Arena/Isaac Lab runtime configuration; it does not
rewrite those authored properties after spawning. Fixed Gear Mesh, syringe, tool-sorting,
Tool Hanging, and USB-C Newton solver settings are graph-YAML overrides. Python callbacks
retain only fields stock Lab does not expose and runtime-derived scene options.

Newton's MuJoCo-Warp importer requires MuJoCo custom attributes to be registered on each
`ModelBuilder` before USD ingestion. Isaac Lab 3 EA's stock `NewtonMJWarpManager` performs that
registration, so this repository carries no Arena manager or runtime patch. Benchmark assets remain
the source of authored values. Two benchmark-only config subclasses retain fields that stock Lab
does not yet expose: `enable_multiccd` on the MuJoCo-Warp solver and `contact_matching` on Newton's
collision pipeline. Tool Hanging also retains a sparse-Jacobian subclass because stock
`MJWarpSolverCfg` does not expose `jacobian`. Their values live under
`benchmark_physics` in the same shared environment YAML as Arena's native
overrides; callbacks only construct the benchmark subclasses because Arena's
Hydra target allowlist excludes application packages. Cable routing has no
Arena graph: its fixed values and embodiment body paths live in
`cable_env_config.yaml`, while Python assembles the coupled rigid/VBD solver and
derives bend stiffness from each source layout's cable radius and pitch. USB-C
source-contact coefficients and Plier Easy's scoped fingertip friction are
also YAML-owned; runtime callbacks still select the matching Newton shapes.

The canonical I2RT YAM layer contains its BDDL-derived fingertip contacts, native
`mjc:gravcomp`, and standard USD joint drives. Its instanceable layer is an authored render
variant of the same physical asset. The terminated Easy and Medium cable tasks use the
package-owned conversion of AUTOLab's side-qualified I2RT robot and linear grippers, with its source provenance,
passive joint dynamics, contact layers and palette appearance recorded alongside
the asset. Easy keeps the physical composition of the AUTOLab revision its `easy.yaml` records. Medium uses the
`cable-tasks` task and a separate CoACD 1.0.11 fingertip layer. Both use the
YAM station cable rig. Medium's synchronized two-arm stream applies each command
frame atomically. Other YAM tasks use the `vabar__bimanual_yam` embodiment.

Cable routing scores its final state at the step AUTOLab v1.0 does. Its documents set
`terminate-on-success false`, so the goal is read on the step the episode ends, and its cell runs
`--settle-steps 60`: the episode ends 60 control steps (1 s at 60 Hz) after the graph finishes, win
or lose. The RSS adapter steps those 60 itself before the graph process exits, so cable experiments
set `graph_success_settle_s` and `graph_failure_settle_s` to `0.0`.

## Evaluator image

One self-contained image is built per accepted source state of this folder and serves
all maintained GaP/Arena pairs. Its public interface is `vabar-eval manifest`,
`vabar-eval verify`, and `vabar-eval run`; graph and Experiment selection remain
runtime configuration. Arena owns distributed Experiment construction,
scheduling, output collection, publication, and sensitivity analysis.

Clean builds are reproducible from the recorded revision and immutable gitlinks.
An explicit `--allow-dirty` development build copies tracked and non-ignored
untracked changes and records their patch, hashes, Git status, and combined source
fingerprint. External dependency gitlinks remain immutable and clean in both
modes. Package/model credentials are build or OSMO secrets and never image
contents. Every OSMO Experiment output carries the embedded image manifest and
the submitted image digest alongside Arena's authoritative episode JSONL.

The image separates immutable dependency provenance from mutable application
provenance. The expensive environment layer consumes only `external/`, the
bootstrap contracts, and repository-owned `pyproject.toml` dependency
manifests. Repository-owned implementation source is copied later and installed
as an offline, dependency-free editable overlay. Ordinary graph, adapter,
benchmark, asset, documentation, and Docker runtime changes therefore retain
the pinned GaP/skill/Arena environments; changes to an external pin, lock,
bootstrap contract, or declared dependency invalidate the appropriate closure.

## Test contracts

`isaaclab_arena_gap/tests` exposes four durable contracts:

1. Bridge transport, adapter conversion, scalar/fleet lifecycle, partial reset, termination, and
   result-sidecar behavior.
2. Static validation of every shipped Arena Experiment and its referenced plan or graph adaptation.
3. One opt-in fresh-process construction/reset/finite-step smoke per baseline Experiment YAML.
4. Idempotent benchmark registration and public launcher delegation.

Tests assert observable behavior through these interfaces. Historical repository layout, shell
implementation, generated inventories, distributed schedulers, and internal helper structure are
not pytest contracts.

The live job smokes use a hold action. They prove environment construction and finite stepping, not
policy success. Full GaP runs remain integration evidence and must report Arena JSONL separately
from graph completion.

## Deliberate limits

- No lockstep batching of graph sessions and no tool processes shared between sessions.
- No graph-completion claim is treated as benchmark success.
- No simulator ground truth is injected into GaP perception outside opt-in oracle runs.
- No generic distributed campaign scheduler is maintained in `isaaclab_arena_gap`; only
  the exact-source evaluator image and GaP-specific adapter for Arena-owned OSMO
  workflows live here.

## Tool Hanging Level 2

Medium scenes use Arena's `on`, `at_position`, and `random_around_solution`
relations for tool placement. Fixtures stay fixed. Each scene contains two
ordinary distractor objects; Arena samples their placement on reset.

## Spawn Spread

Arena 0.3.1 writes the solved layout as is and ignores `random_around_solution`.
`build_graph_environment` therefore adds a `random_around_solution` reset term to every graph
that declares one (gear easy, tool-hanging medium, USB-C easy and medium). It runs after
`placement_reset`, ahead of any family term such as the USB-C cable reset, and moves each marked
object, anchors included, by one uniform draw within the marker's half-ranges, turning it about
its bounding-box centre so the body turns in place. A draw whose
bounding box would cut into a neighbour's is redrawn, up to 64 times, after which the object keeps
its solved pose. Draws are keyed on `(placement_seed, env_id, that env's reset count)`, so one
placement seed gives one start and env 0 of a vector job matches a `num_envs=1` job.

RSS graphs run from the task repository root with the GaP UV project
selected explicitly, preserving relative CAD paths in evaluator images.

## Tool Hanging Scoring

Every tool-hanging scene scores the RSS version's goal (`tasks/tool_hanging/*_level{1,2}.task`):
each tool's clause -- geometry, contact force of at least 0.2 N on the fixture, speed of at most
0.15 m/s, and the release clause `(> (hands-clear TOOL) R)` -- holds for one second of simulated
time (`hold-for 1.0`), and success ends the episode. `hands-clear` is the distance from the tool to
the nearest tool centre of either YAM (`link_6` plus RoboSimStudio's `yam_linear_arena` grip
offset). `R` follows the RSS rule -- the tool's radius off its collision hulls plus 15 mm, rounded
up to the centimetre -- applied to Arena's own tools and measured from each tool's physical centre,
the centre of its hulls' bounds (`release_point_xyz`): plier 0.10 m, scissors 0.10 m, screwdriver
0.11 m and ring spanner 0.14 m. The RSS documents state 0.11, 0.10, 0.20 and 0.21 m about their own
tools' origins; the screwdriver's sits at one end, and the RSS wrench is a larger tool. A tool held against its fixture, or one that
slides off as it is let go, does not pass. The RSS adapter already steps AUTOLab's 60 settle steps
after the graph ends, so tool-hanging experiments add no `graph_success_settle_s` or
`graph_failure_settle_s` of their own.
