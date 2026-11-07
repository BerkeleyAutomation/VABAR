# Workers and MPS: filling one GPU

Measured 2026-11-05 on RTX PRO 6000 Blackwell GPUs (96 GB), one cell per GPU, with RoboSimStudio
`speed/vabar-runtime` and graph-as-policy `speed/tool-bundle-linger`. Every run uses the lockstep clock and
on-demand RTX observations, and keeps its tool servers and planner warm (`default`'s runtime). The 240-core host
was shared: other cells ran on the other GPUs, and another project's short unit-test jobs sometimes ran on
GPUs 1–4 and 7, so wall times carry a few percent of noise. Raw output is in the session scratchpad
(`default_runs/`). The pool is upstream's (`robosimstudio_bench/run/pool.py`): workers pull episodes from one
queue.

## Why more processes do not change outcomes

Under `clock = "lockstep"`, a worker's simulation advances only when its own graph asks for steps. A second
process on the GPU makes each step take longer in wall time. It does not change how many steps the graph gets
or what they contain. So the worker count is a scheduling choice, and `default` sets one per suite.
Per-instance outcomes across worker counts:

| Cell | 1 worker | 2 workers | 4 workers + MPS |
| --- | --- | --- | --- |
| cable_insertion/L1 (`mate`) | 0/30, median lateral 15.8 mm, depth −0.46 mm | 0/30 | 0/30, median lateral 15.8 mm, depth −0.50 mm |
| syringe_packing/L1 | 1/30; reward 0.5 on 29 | 1/30; 0.5 on 27, 0.0 on 2 | 2/30; 0.5 on 26, 0.0 on 2 |
| gear_meshing/L1 | 24/30 | 27/30 | 28/30 |
| tool_hanging/L1 (`default`) | 30/30 (twice) | 29/30 | -- |

The spread is the policy's own run-to-run variation. Its perception inputs are not bit-identical between runs:
GroundingDINO's scores on the same instance differ in the third decimal even between two 1-worker runs.

## Why MPS

Without MPS, processes on one GPU time-slice it: each CUDA context has the whole GPU in turn. A cell whose
episodes are physics-bound then gains little from a second process. Under NVIDIA MPS the contexts run kernels
concurrently.

| Cell (`fast` physics) | 1 worker | N workers, no MPS | N workers + MPS |
| --- | --- | --- | --- |
| cable_insertion/L1, N = 4 | 43 min | 37.5 min (2 workers: 44 min) | **19.7 min** |
| cable_routing/L1, first 6 instances, N = 3 | ~25 min (249 s/instance) | 23.5 min | **9.1 min** |
| syringe_packing/L1, N = 4 | 48 min | 29.5 min (N = 2) | **18.2 min** |
| gear_meshing/L1, N = 4 | 35 min | 24.4 min (N = 2) | 25.0 min (one 6-minute start, see below) |

With MPS, each cable_routing episode on 3 workers took as long as it did alone (250 s against 249 s). Without
MPS each took 2.7 times as long. Six MPS workers made each episode 1.6 times slower than three, so the full cable
cell took 34 min on 6 against an estimated 45 on 3.

**The gear start.** In the 4-worker MPS gear run, every worker took about 6 minutes to become ready. With 2
workers and no MPS, workers were ready in 25 s. Repeated alone on a clean GPU, four gear builds started together
under MPS took 14 s, the same as without MPS. That run had started together with two other MPS runs (15 worker
processes at once) on a GPU that another job was also using. The cause is not established, and it did not
recur.

## How `vabar bench` uses it

- `runtime.mps = true` (in `default`): when a cell runs more than one worker, `vabar bench` starts a private
  `nvidia-cuda-mps-control -d` for the run and stops it afterwards (`vabar/mps.py`).
  - Its pipe directory is private (`/tmp/vabar-mps-*`). Only the run's own processes, the pool workers and the
    tool servers they spawn, connect to it. Other jobs on the GPU are not routed through it.
  - The daemon serves the first GPU `CUDA_VISIBLE_DEVICES` names, and the run's processes name it by UUID. The
    daemon numbers its one GPU 0, so a client asking for index 3 is refused ("not a valid GPU ID in the MPS
    visible set").
  - Without the binary, or if the daemon will not start, the run continues as plain processes and its
    `vabar_config.json` records why.
- `[suites.<suite>] workers` is a ceiling measured on a 96 GB GPU. `worker_gb` is the GPU memory one of that
  suite's workers holds, with headroom. On a smaller card the count is held to `floor(free GB / worker_gb)`.
  `--set workers=N`, or `--workers N` on `vabar bench`, overrides the suite's count.

| Suite | `workers` | Measured GB per worker | `worker_gb` |
| --- | --- | --- | --- |
| tool_hanging (8-world fleet waves) | 2 | -- | 24 |
| gear_mesh | 4 | 10.4 | 13 |
| syringe_packing | 4 | ~13.5 (its own SAM 3 and DINO servers) | 16 |
| tool_sort | 5 | ~13.3 | 16 |
| usbc_insert | 4 | 5.4–6.4 | 8 |
| cable_route | 6 | 5.4 | 7 |

## tool_sort: the remote VLM decides, MPS does not

Under 5 MPS workers (12k budget), tool_sort scored 1 success in its first 19 instances, against 9/30 on 3
workers without MPS and 6/20 on 1. `level1_9` and `level1_14` had succeeded in the 1- and 3-worker runs at
4,939/4,937 and 5,665/5,659 steps. Under 5 MPS workers both ran out the budget.

The cause is the graph's first `vlm.query` in `pair_sg.select`, which is remote (`GAP_VLM_PROVIDER=vertex`,
`GAP_VLM_MODEL=gemini-2.5-flash`, temperature 0). On the same instance it answers "A" in one run and "B" in
another, and it names the same candidate "painters tape" in one run and "tactical flashlight" in another. When
the pliers are picked first, a later grasp is planned for a 27 mm grip instead of 12-14 mm and closes on
nothing, and the graph retries until the budget runs out. When the flashlight is picked first, the instance
succeeds at about 4.9k steps.

`level1_9` repeated alone, 14k budget (2026-11-05, 08:45-09:35):

| Setting | Successes | Failures |
| --- | --- | --- |
| serial, 1 process, no MPS | 3/4 (4,949; 5,350; 4,954) | 1 at the budget, reward 0.75 |
| 5 at once, MPS | 4/5 (3,126; 4,939; 4,939; 4,950) | 1 at the budget, reward 0.75 |
| 5 at once, no MPS | 2/5 (4,899; 5,347) | 3 at the budget, reward 0.50-0.75 |

So neither MPS nor concurrency is the cause: the instance fails alone on its GPU too. 9 successes in 14 attempts
overall, against 2/2 in the earlier 1- and 3-worker cell runs and 0/1 in the 5-worker MPS cell run.
Read tool_sort numbers as the policy's success rate with a nondeterministic VLM in the loop. One cell run
cannot pin it down (6/20, 9/30 and 1/19 across three runs), and two runs of the same instance disagree about as
often as two configs do.

## Final validation: `fast`, workers and MPS from the config

`vabar bench --cell <cell> --solve graph --config fast`, with no other flags, one L1 cell per GPU and all six at
once. Two runs were made, because the first died when the host's disk filled up. In the second run GPUs 4 and 7
(cable insertion, cable routing) also carried another tenant's processes, so those two times are pessimistic. The
clean-GPU runs above are the better guide.

| Cell | Workers | Wall | Success | Notes |
| --- | --- | --- | --- | --- |
| tool_hanging/L1 | 2 | 5.1 min (7.1 under load) | 29/30 (both runs) | same as `default` |
| gear_meshing/L1 | 4 | 17.9 min | 27/30 | workers ready in 28-68 s |
| syringe_packing/L1 | 4 | 18.4 min | 2/30 | reward 0.5 on 27: one syringe of two posted, as the reference |
| cable_insertion/L1 | 4 | 19.7 min on a clean GPU | 0/30 | `mate`'s easy-tier miss (lateral ~16 mm), as under `paper` |
| tool_sorting/L1 | 5 | 45.5 min | 7/30 | 23 episodes ran out the 14k budget; one success at 13,626 steps |
| cable_routing/L1 | 6 | 34.2 min on a clean GPU (41.5 shared) | 2/30 and 4/30, mean reward 0.60-0.61 | |

**A hung worker.** In the second cable-insertion run, one of the four workers became ready and then never
started its first episode (10010). It sat for 47 minutes with RTX (`OVRTX:Operation`) and TBB threads busy and
793 Python threads. The bus watchdog does not cover the time before a graph starts. Stopping the worker let the
pool finish: it reported "1 episodes never ran: [10010]". Run alone, 10010 completed normally (failure at 3,317
steps, like the other 29). This was 1 such hang in about 130 cable-insertion episodes run on 4 workers, and none
in any other cell. Upstream's pool has no per-episode wall-clock watchdog that would restart such a worker.


## More workers: shared tool servers, and the renderer's caches

Measured 2026-11-05 afternoon, one cell per GPU, `fast`, with RoboSimStudio's cuRobo attachment slots and
graph-as-policy's shared tool servers (both on the branches named at the top).

**Shared tool servers.** Every worker of a perception graph used to start its own SAM 3 (about 6 GB) and
GroundingDINO (about 3 GB). With `share_tool_servers = true` (now in `default`), `vabar bench` gives the run a
private directory (`GAP_TOOL_SHARE_DIR`). The first worker that needs a bundle starts one broker for it
(`gap_core.rpc.broker`), and every worker reaches that broker through a relay (`gap_core.rpc.relay`) in place of a
server of its own. The broker forwards one call at a time. The shared tools take everything they need in the
call, so no call's answer depends on which worker asked. A tool-sort worker went from about 13.5 GB to 4.4 GB,
plus 1 GB for its own cuRobo server.

| Cell (L1) | Before (success) | After (success) |
| --- | --- | --- |
| tool_sorting | 45.5 min, 5 workers (7/30) | 36.5 min on 10 (6/30); 29.1 min on 15 (5/30, 88 GB peak) |
| gear_meshing | 17.9 min, 4 workers (27/30) | 15.1 min on 8 (25/30) |
| cable_insertion (no perception servers) | 21.8 min, 4 workers (30/30) | 15.0 min on 8 (30/30); 16.8 on 15 (30/30) |
| syringe_packing | 18.4 min, 4 workers (2/30) | 13.2 min on 4 with cuRobo attachment slots (0/30, mean reward 0.42); 13.2 on 8 (2/30, 0.52) |

**The driver's shader cache was evicting the warm-up.** Corrected later: pipelines do not differ by scene, and each
process needs a complete cache file of its own (`driver_shader_cache.md`). The measurements here stand. The first
explanation, that each scene adds 130-230 MB, was wrong: those were new files, one per compiling process.

A process's first render builds its ray-tracing pipelines: several minutes of CPU, 19 s once cached. The NVIDIA driver
keeps them in `ovrtx/bin/cache/nv_shadercache`. OVRTX sets `__GL_SHADER_DISK_CACHE_SIZE` to 4,194,304,000 bytes when
the renderer starts, overwriting the environment, and offers no config key for it. Past that cap the driver drops the
oldest files. After warming every scene, the cache held 34 chunks, and 26 of them were written by the next hour's runs. In a
15-worker tool-sorting run, 12 of the first 15 builds compiled again, for 410-500 s each, and a scene warmed
twenty minutes earlier (`level1_11`) compiled again alone in 248 s. RoboSimStudio now sets the driver's
`__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1` before the renderer starts (`keep_driver_shader_cache` in
`solvers/render/rtx.py`), unless the caller set it. OVRTX leaves that variable alone. The cache then grows past
the cap instead of rotating. Delete `nv_shadercache` to reclaim the space.

Checked by re-warming all six L1 cells (96 scenes, `vabar warmup --jobs 12`) with the fix. Every scene built.
91 of them took 29-67 s, and 5 compiled again for 300-360 s. The cache finished at 4,504,621,095 bytes in 68 files, past the
4,194,304,000-byte cap that had held it at 34 chunks before.

**A 30-minute renderer stall.** Four builds out of about 500 today waited exactly the renderer's streaming
timeout. The log line is `[omni.rtx] StreamingStatusManager : Streaming client gpu.resourceManager timed out`,
after about 1,800 s. It comes on the scene's first frame, followed by `Leaking step result outputs` and
the eviction of the scene's texture assets. The episode then ran normally:
- one scene of the 162-scene warm-up, 1,834 s;
- `tool_sort/level1_24` in a 15-worker run: ready after 1,811 s, then succeeded in 75 s. It alone took that cell
  from 28 min to 45;
- `tool_sort/level1_5` in the 96-scene re-warm above, 1,816 s;
- probably the cable-insertion worker above.

A fifth hit the syringe-packing L2 warm-up (`tray_clutter_t25`). It is not tied to a cell, a scene or MPS.
- **The cause.** A scene's first frame waits until every streaming client of OVRTX's `StreamingStatusManager`
  reports idle. Now and then the texture-file loader (`gpu.resourceManager`) goes busy and never reports idle
  again, though everything is loaded.
- **The bound.** It is a Carbonite setting with no `RendererConfig` field. OVRTX reads it from the environment as
  `OVRTX_rtx-transient_gpu_streamingStatusManager_timeoutS`. RoboSimStudio now sets it to 120 s
  (`bound_first_frame_wait` in `solvers/render/rtx.py`), unless the caller set it.
- **The test.** On a stress rig building tool-sorting scenes 12-16 at a time on one GPU, with the bound at 60 s:
  - 9 of 903 builds stalled, and each cost about 60 s instead of 30 min;
  - a stalled build's first frame matched ordinary first frames within render noise;
  - no healthy warm build timed out. The longest healthy busy spell was 25 s.
- **Texture streaming off** (`texture_streaming_mode = DISABLE`) made stalls about 8× rarer but did not remove
  them, and builds under load were 2-8% slower, so it stays at OVRTX's default.
- **Not covered by the bound.** The same rig saw 1 build in 904 hang with no timeout at all. So did the
  single-world cable-routing L2 run, between two episodes: 37 min at 340% CPU, nothing logged.
- **The pool watchdog.** RoboSimStudio's pool now has one, on by default (`robosimstudio_bench/run/pool.py`).
  - Each worker stamps its progress: on taking an episode, before each reset, and after every control step.
  - A worker silent past its budget is SIGKILLed and replaced, and its episode re-runs on the new worker.
  - The budgets: 900 s from taking an episode to its first step, doubled on the retry; 300 s between steps,
    or the graph's bus timeout plus 120 s.
  - After one retry the episode is written as a `close_kind="watchdog"` error row.
  - Every kill is in the run's provenance.
  - It is checked end to end through `vabar bench` under MPS with an injected hang.
  - `ROBOSIMSTUDIO_POOL_WATCHDOG=off` turns it off.
- **Ctrl-C with a hung worker** used to leave the run and its MPS daemon behind. `quit` waited on the hung
  client, and the interpreter waited on the worker. The pool now stops its workers on any early exit, and
  `vabar/mps.py` quits with `quit -t 10` and then SIGTERMs the daemon by its pid file.

## Final L1 validation, every fix in

Measured 2026-11-05, 16:02-16:26. `vabar bench --config fast`, all six L1 cells launched together, one per
RTX PRO 6000. The stack:
- shared tool servers;
- cuRobo attachment slots;
- the shader-cache fix, after a warm re-run;
- cable routing as 15-world fleet waves (`cable_fleet.md`);
- tool sorting on Gemini 3.1 Flash-Lite.

The host's other tenants kept its load at 82-95 on 240 cores. This note read the fleet as host-bound, which
[fleet_profile.md](fleet_profile.md) later showed wrong: GPU physics is its limit. It stepped at 131
steps/s here against 163-186 alone, so these are upper bounds. Wall time is launch to exit. Peak is the GPU's
`memory.used`, sampled every 10 s, minus another tenant's 3.9 GB on the two cards that had one.

| Cell (L1) | Workers | Wall | Success | Mean reward | GPU peak |
| --- | --- | --- | --- | --- | --- |
| tool_hanging | 2 (8-world waves) | 6.8 min | 29/30 | 0.98 | 21.5 GB |
| gear_meshing | 8 | 16.9 min | 26/30 | 0.93 | 41.9 GB |
| syringe_packing | 4 | 15.9 min | 0/30 | 0.45 | 33.3 GB |
| cable_insertion | 8 | 16.5 min | 30/30 | 1.00 | 39.2 GB |
| tool_sorting | 15 | 19.5 min | 8/30 | 0.60 | 87.1 GB |
| cable_routing | 2 (15-world waves) | 23.9 min | 20/30 | 0.89 | 10.8 GB |

No build stalled, and none recompiled a shader. Tool sorting went from 45.4 min to 19.5 because neither
happened. Its successes (8/30) are inside the 5-7/30 of earlier runs. Tool hanging is the other cell that runs
fleet waves (4 tasks of 7-8 instances). It scored the same 29/30 after the fleet servo change as before it
(`cable_fleet.md`, fix 4). The other four cells' instances are each a variant of their own, so they run
single-world episodes.

**Sizing workers on a smaller card.** `worker_gb` had been measured with every worker holding its own SAM 3 and
GroundingDINO. With the servers shared, a worker costs 4-7 GB, and the servers about 9 GB once per run (6.5 GB
for a graph that calls only SAM 3). A run now fits `floor((free GB - shared_gb) / worker_gb)` workers, at most the
suite's ceiling. `worker_gb` is the peak above, less `shared_gb`, per worker, plus about 10%.

| Suite | `worker_gb` (was) | `shared_gb` | Workers on 47 GB free | 23 GB free |
| --- | --- | --- | --- | --- |
| tool_hanging | 7.0 (24) | 9.0 | 2 (was 1) | 2 |
| gear_mesh | 5.0 (10) | 6.5 | 8 (was 4) | 3 |
| syringe_packing | 7.5 (16) | 6.5 | 4 (was 2) | 2 |
| tool_sort | 5.5 (6) | 9.0 | 6 (was 7, which would not fit) | 2 |
| usbc_insert | 4.5 (8) | 6.5 | 8 (was 5) | 3 |
| cable_route | 6.0 (7) | 6.5 | 6 (L1: 2 waves) | 2 |

## Gear meshing: SAM 3 asked for 48 masks, not 200

Measured 2026-11-05, 20:08-20:19: gear_meshing/L1, `fast`, 8 MPS workers, other agents' runs on the host.
- **The fix.** The gear graph's `sight.py` called `sam3.segment_text` with `max_results=0`, so every call
  returned all 200 masks: 246 MB of full-frame masks through one shared SAM 3 server that serves a call at a
  time. In the validation run above (400 calls), no call had more than 14 masks over the 0.10 floor
  `segment()` keeps. The tool sorts by score before capping, so a cap of 48 drops nothing the graph uses.
  The fix is RoboSimStudio-dev f4bd50b6 (2026-09-10), which never reached VABAR.
- **The result.** The cell went from 16.9 min (26/30, mean reward 0.93) to 10.7 min (27/30, 0.95). Mean
  episode wall time fell from 182 s to 129 s.
- **The other suites.** None makes the same call. Cable routing and cable insertion also ask for every mask, but
  their scenes return at most 12 and 2.

## Gear meshing: what the suite's Sept 10 snapshot had missed

VABAR's `tasks/gear_mesh` is RoboSimStudio-dev's suite as of 2026-09-10, reached through the AUTOLab repo. Later
dev commits never arrived. Three are now in. All runs: `fast`, reference graph, single-world, 8 MPS workers.

| Cell | Before | SAM 3 cap | + declared servo speed | + bent chains (L2) |
| --- | --- | --- | --- | --- |
| L1 | 16.9 min, 26/30 | 10.7 min, 27/30 | about 14 min, 29/30 (0.98) | n/a |
| L2 | 22.5 min, 22/30 | | 26/30 and 28/30, on straight boards | 13.7 min, 23/30; 15.0 min, 21/30 |

- **Declared servo speed** (dev cb96425e). `(:servo-speed-mps 0.12)` in `gear_mesh.suite` states the speed the
  graph was tuned against, and `carry.py` and `press.py` get the matching contact budgets. Without it the tool
  servoes at the OSC law's 0.31 m/s.
- **Bent chains** (dev 74e0cfd8). All 100 L2 documents say the chain turns at each station, and every board was
  built straight, because their poses files lacked `bends`. With the bends alone, L2 scored 16/30 against 22/30
  on the straight boards, so earlier L2 numbers were on an easier scene. L2's 21-23/30 compares with the paper's
  90%, which dev's commit says was measured on bent boards.
- **Fleet waves, measured and not taken.** Dev's `fleet/gear-mesh-easy-per-world` (a per-world board, `(:fleet-base)` on
  91 documents, per-execution graph memory) runs L1 as 6 waves in 8.9 min, 26/30 (0.90). That saves 2 min
  against the 10.7 measured at the same stage, and takes 21% more sim steps per episode (4,616 against 3,812),
  which would bias cycle time. Each world's start state matched its single-world build to 6e-5 mm.
- **The hang again, this time caught.** In the 20.2-min L1 run, worker w3 hung while building `pair-15`. The
  watchdog killed it at 900 s, and the fresh worker built the scene in 29 s and succeeded. No renderer timeout was
  logged, so this is the unbounded hang, not the streaming stall. It is the third sighting today.
