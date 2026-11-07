# One full pass from a fresh install

Measured 2026-11-06 at `1b22b3f`, from a new `git clone` and `uv sync` with the README's commands, with no shader cache.
This pass predates the fix in `driver_shader_cache.md`, which has the same pass repeated after it.
Each run had one RTX PRO 6000 Blackwell. Seven ran at once on a shared 240-core host, at a load of 60-150.

## Every cell under `fast`

One run of each. Time includes the first-run warmup, which a second run on the same install skips. The percentages for
tool sorting, tool hanging and gear meshing are mean reward, which tracks the paper's per-object and per-gear measure.

| Cell | Solved | Paper's measure | Paper | Time | Warmup | Build hangs |
| --- | --- | --- | --- | --- | --- | --- |
| Tool hanging L1 | 30/30 | 100% | 86.7% | 7.9 min | 1.3 min | 0 |
| Tool hanging L2 | 25/30 | 92% | 81.7% | 12.9 min | 0.5 min | 0 |
| Gear meshing L1 | 29/30 | 98% | 100% | 26.4 min | 9.1 min | 1 |
| Gear meshing L2 | 23/30 | 90% | 90% | 27.6 min | 7.0 min | 1 |
| Syringe packing L1 | 26/30 | 87% | 87% | 13.2 min | none | 0 |
| Syringe packing L2 | 17/30 | 57% | 63% | 33.7 min | none | 0 |
| Cable insertion L1 | 30/30 | 100% | 86.7% | 22.9 min | 7.1 min | 0 |
| Cable insertion L2 | 24/30 | 80% | 30% | 37.9 min | 8.2 min | 1 |
| Tool sorting L1 | 7/30 | 74% | 47.1% | 19.2 min | 5.6 min | 0 |
| Tool sorting L2 | 1/30 | 58% | 33.7% | 52.1 min | 16.6 min | 0 |
| Cable routing L1 | 16/30 | 53% | 46.7% | 16.8 min | 0.9 min | 0 |
| Cable routing L2 | 12/30 | 40% | 46.7% | 39.9 min | 8.2 min | 0 |

There were no crashes, failed warmups or physics overflows, and the checkout stayed clean. All three hangs recovered
without losing a result.

**Syringe packing under the other configs.** `default`: L1 30/30 in 21.1 min, L2 17/30 (reward 0.786) in 59.5 min, the
first runs on the larger buffers at 500 Hz, with no overflow. `paper`: three L1 episodes, all solved, at about 7 min each.

**Cable routing L2 alone.** Rerun with nothing else of ours on the machine, warm: 20/30, reward 0.939, 20.8 min. So the
time above is mostly warmup and load. The 12/30 is its lowest yet, against 14-21 before. One run each cannot separate
load from noise (p = 0.07), and the lockstep clock should keep load from changing outcomes.

## The build limit stays at 15 minutes

The pool's watchdog kills a worker silent for `build_s` while building, and re-runs its unit with twice the budget. A
shorter limit would cut what a real hang costs, so each limit ran on the two most exposed cells, warm and all at once
(`ROBOSIMSTUDIO_POOL_WATCHDOG=build=N`).

| Cell | Limit | Workers killed | Solved | Reward | Time | Slowest build |
| --- | --- | --- | --- | --- | --- | --- |
| Tool sorting L2 | 15 min | 0 | 2/30 | 0.575 | 36.6 min | 685 s |
| | 10 min | 7 | 1/30 | 0.578 | 35.8 min | 144 s, after the kills |
| | 5 min | 6 | 0/30 | 0.528 | 38.2 min | 482 s, after the kills |
| Cable insertion L2 | 15 min | 0 | 20/30 | 0.759 | 25.0 min | 670 s |
| | 10 min | 2 | 22/30 | 0.809 | 26.8 min | 164 s, after the kills |
| | 5 min | 3, one episode lost | 17/30 | 0.660 | 29.2 min | 462 s, after the kills |

Neither shorter limit saved time, and no real hang happened in these runs, so every kill looks like a healthy slow build:
the 15-minute tool sorting arm had 7 builds over 600 s, and the 10-minute arm killed 7. At 5 minutes one cable insertion
episode was killed twice and written as a watchdog error.

## What makes those builds slow

Of 642 builds across these runs and the syringe cells, 285 were a worker's first and 357 were later ones. Later builds
took a median of 7 s and at most 133 s. First builds took a median of 33 s, with 32 over 5 minutes, and all three hangs
were first builds.

The slow builds match one renderer line, `Streaming client gpu.foundation.TextureStreaming timed out`:

| Run | Builds over 300 s | Lines | | Run | Builds over 300 s | Lines |
| --- | --- | --- | --- | --- | --- | --- |
| Tool sorting L2 | 9 | 9 | | Tool sorting L2, 15 min | 7 | 7 |
| Tool sorting L1 | 1 | 1 | | Tool sorting L2, 5 min | 6 | 6 |
| Gear meshing L1 | 4 | 4 | | Cable insertion L2, 15 min | 1 | 1 |
| Cable insertion L2 | 4 | 5 | | Cable insertion L2, 5 min | 2 | 2 |
| Cable insertion L1 | 1 | 1 | | Cable routing L2 | 0 | 2 |

RoboSimStudio's `rtx.py` records that this client stays busy through a pipeline compile and times out when the compile
ends. So these workers look to be compiling shaders after warmup has compiled them. In the tool sorting and cable
insertion arms, the slow builds fall on the last workers to start (`w8`-`w14` of 15, `w7` of 8).

**Later measured: they were.** The driver's cache needs a complete file per process, and warmup left fewer files than a
cell has workers. With that fixed, the same pass had no build over 300 s and no stream timeouts, and tool sorting L2
took 35.9 min. See `driver_shader_cache.md`. The kills in the limit test above were these compiles, so the case for
keeping a long limit is weaker than it was, though no shorter limit has been measured since the fix.
