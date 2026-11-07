# Why warmed workers compiled again

Measured 2026-11-06 on NVIDIA driver 580.126.09 and RTX PRO 6000 Blackwell GPUs, from new installs with the README's
commands, on a shared 240-core host. The raw output was not kept.

## Answer

**The driver's shader cache is per process, not per scene.** The pipelines compile once per install, and each worker
then needs its own complete copy of the file that holds them. `vabar warmup` used to warm every scene, 8 at a time,
which took most of a cell's warmup and still left 8 files for a cell with 15 workers.

`warmup.py` now compiles once, alone, and copies the result for the workers to come. A cell's first run warms only its
first scene, for its Warp kernels. Across all the cells, time before the first episode fell from 68 to 27 minutes, no
first build took over 300 s (19 before), and the renderer's stream timeouts went from 22 to none.

## How the driver's cache works

The files are under `ovrtx/bin/cache/nv_shadercache/vulkan/GLCache/<driver hash>/<app hash>/`, as `<name>.bin` and
`<name>.toc`.

- **Names follow a fixed sequence**, counting up by one.
- **One process, one file.** A process takes a POSIX write lock (visible in `/proc/locks`) on the first name nobody
  holds, and uses only that file. If the file is new, the process compiles everything and fills it.
- **Every file holds the same pipelines, whichever scene wrote it.** An install that had run all the RTX cells held 69
  files, 9 GB, each with the same 673 entries (557 distinct keys) and about 134 MB.
- **So a run needs a complete file per process.** `warmup --jobs 8` left 8 files, so a 15-worker cell compiled the other
  7. The limit test in `complete_test.md` started 69 workers in all, and its cache ended with exactly 69 files.
- **A copy under the next name is accepted**, since a file does not record its own name. Copies on names outside the
  sequence are ignored. Hard links cannot help, because a lock belongs to the inode.
- **`__GL_SHADER_DISK_CACHE_READ_ONLY=1` is honoured:** no locks, one file shared, nothing saved.

This also explains the slow first builds in `complete_test.md`. Each was a worker compiling, and the renderer's
`gpu.foundation.TextureStreaming` client timed out as the compile ended.

## The test

Tool sorting L2, 15 workers, 15 instances:

| Files at the start | First builds | New files | Other |
| --- | --- | --- | --- |
| 69, written by compiling, machine otherwise idle | 15-35 s | 0 | |
| 8, written by compiling | 8 at 114-153 s, then 7 at 456-546 s | 7 | 7 stream timeouts |
| 1, read-only | 117-162 s | 0 | 1 hang, file unchanged |
| 15, copies of one | 120-161 s | 0 | no stream timeouts |

The last three ran at the same time, so their builds are slower than the first row's. Seven new files is what the
mechanism predicts for 8 files and 15 workers.

**Scenes do not add pipelines.** One read-only file served every scene of every cell under `fast`, `default` and
`paper`: 396 RTX builds. The slowest took 228 s, none compiled, and the file did not change.

**A new user's first cell.** Tool sorting L2, all 30 instances, from a new install under heavy load:

| Install | Total | Warmup | Pool | Recompiles | Driver files |
| --- | --- | --- | --- | --- | --- |
| Before | 57.2 min | about 20 min | 37.2 min | 7 | 15 files, 1.9 GB |
| Read-only file, every scene warmed | 55.8 min | 20.7 min | 35.0 min | 0 | 128 MB |
| Read-only file, first scene warmed | 47.6 min | 9.5 min | 37.1 min | 0 | 128 MB |
| **Copies, first scene warmed** | **40.9 min** | **6.6 min** | **33.5 min** | **0** | 72 files, 9.0 GB |

## Why copies, and not the read-only switch

Both stop the recompiles, and they build equally fast: 71-88 s and 81-88 s with 15 workers side by side. The choice
rests on what happens when something changes:

- **A scene that needs a new pipeline.** Under copies, the worker compiles it once and keeps it. Read-only keeps
  nothing, so it would compile again on every run.
- **A driver that names its files differently.** A copy would be ignored, and workers would compile their own files as
  they do without this.

Copies cost more disk: 134 MB per worker, 16 files per GPU up to 72, so about 2 GB on one GPU and 9.6 GB at most. Deleting
`nv_shadercache` reclaims them, and the next run compiles once more.

## One full pass

The same 15 runs as `complete_test.md`, in the same order, on 7 GPUs at once from a new install. Time includes everything
from launch to exit. *Before first episode* is what warmup costs; the "after" cells at about 6 minutes were among the 4
that started together, so 3 of them waited for the one compile (5.2 min). A one-GPU user running cells in turn does not
wait.

| Cell (`fast`) | Solved | Reward | Time (min) | Before first episode (min) |
| --- | --- | --- | --- | --- |
| Tool hanging L1 | 30 → 29 | 1.000 → 0.978 | 7.8 → 5.4 | 2.1 → 0.5 |
| Tool hanging L2 | 25 → 21 | 0.917 → 0.833 | 12.9 → 12.1 | 0.8 → 0.5 |
| Gear meshing L1 | 29 → 30 | 0.983 → 1.000 | 26.4 → 9.8 | 9.3 → 0.4 |
| Gear meshing L2 | 23 → 23 | 0.900 → 0.922 | 27.6 → 14.2 | 7.6 → 0.4 |
| Cable insertion L1 | 30 → 30 | 1.000 → 1.000 | 22.9 → 19.8 | 7.4 → 5.9 |
| Cable insertion L2 | 24 → 23 | 0.877 → 0.807 | 37.9 → 34.6 | 8.6 → 6.2 |
| Tool sorting L1 | 7 → 12 | 0.738 → 0.808 | 19.2 → 13.3 | 5.9 → 0.4 |
| Tool sorting L2 | 1 → 2 | 0.582 → 0.578 | 52.1 → 35.9 | 16.8 → 5.4 |
| Cable routing L1 | 16 → 17 | 0.817 → 0.850 | 16.8 → 18.4 | 1.1 → 1.1 |
| Cable routing L2 | 12 → 14 | 0.817 → 0.856 | 39.9 → 29.9 | 8.7 → 6.4 |
| **The 10 RTX cells** | **197 → 201** | | **263.5 → 193.4** | **68.3 → 27.3** |

**The syringe cells are the control.** They render with the tiled renderer, so the fix cannot touch them, and their five
runs went from 149.3 to 152.7 min in all. The host was about as busy for both passes.

Across all 272 builds of the pass:

| | Before | After |
| --- | --- | --- |
| First builds over 300 s | 19 | 0 |
| Slowest finished build | 456 s | 166 s |
| Stream timeouts | 22 | 0 |
| Hangs | 3 | 1 |

All 72 tables were still byte-identical at the end, so no cell needed a pipeline outside the first compile.

## One GPU with nothing else running

Tool sorting L2, all 30 instances on one GPU, warm, with none of our other runs on the machine: **28.3 min**, 0/30, reward
0.617. The 30 first builds took 10-37 s (median 14), with no stream timeouts or hangs, and the driver files did not
change. The same cell took 32-37 min warm before this fix, and 52.1 min in the first pass. A new install adds the one
compile, 5-8 min, on its first RTX cell.

## Does it change outcomes

Tool hanging L2's 25 → 21 was the largest drop in the pass, so that cell ran 16 times, on two installs with identical
packages. The fix does not change the code a run executes after warmup. Its files are the one other difference: two
compiles of the same pipelines are not byte-identical (only about 15% of entries match), so workers used to read several
compiles and now all read copies of one. So the driver files were swapped between the installs too, with warming off.

| Code | Driver files | Solved /30 | Mean reward |
| --- | --- | --- | --- |
| Before the fix | Compiled by each process | 24, 26, 25, 26, 24 | 0.903 |
| Fixed | Copies of one | 23, 23, 26, 21, 23 | 0.877 |
| Before the fix | Copies of one | 24, 25, 26 | 0.917 |
| Fixed | Compiled by each process | 26, 22, 25 | 0.900 |

- **The gap does not follow the files:** 0.902 for files compiled by each process against 0.892 for copies (exact
  permutation test, p = 0.51). The code from before the fix did best of all on copies.
- **Nor does it clearly follow the code:** 0.908 before the fix against 0.885 after (p = 0.10), and the code a run
  executes is the same.
- **A difference of about 2 points cannot be ruled out** from 16 runs. The cell's spread is 21-26/30 and 83-93%.
- **Frames differ on every run anyway.** The first frame of one episode differed by about 0.3-0.5 of 255 grey levels
  between any two runs: 0.39 on average between two compiles, and 0.35 between two runs of one compile. So a compile
  adds little, if anything, to the noise that makes episodes branch, and that noise does not separate the installs: one
  episode reached the same 5,488 steps on both.

The other cells moved in both directions, and the 10 RTX cells solved 197 before and 201 after.

## What this does not fix

- **The rare build hang.** A process stops in its first build with 789 threads (normal is about 2,080) and about 4.9 GB
  of memory, using 3-8 cores, with no timeout line. It happens with copies and with read-only: 1 in 621 and 2 in 632 side
  by side. A new process on the same scene builds normally. The cause is unknown, and the pool's watchdog still recovers
  it, at a cost of 15 minutes for one worker.
- **Load.** Builds slow down with the machine's load: 15 workers take 10-37 s on an idle host and 100-230 s at a load of
  about 85. The cause is not established.
