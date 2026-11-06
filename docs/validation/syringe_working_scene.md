# `syringe_packing`: the working scene

`speed/default-fast` gave this suite the multi-syringe graph, but kept the scene the v1.0 export shipped. Its notes
named the colliders as the likely reason the cells stayed below the paper, untested. This change ports the scene the
graph was written on, and runs it on the hand and renderer it was measured with.

**Source.** RoboSimStudio-dev `vabar/tasks/sharps_disposal` at `1cd9f76a`, chiefly `18ab35f6` and `5c561ad9`. A second,
independent port of the same commits reached identical layouts: all 302 syringe and tray poses, and every document's
objects and budget, match.

## What changed

| Area | Before | Now |
| --- | --- | --- |
| Syringe collider | One authored box, 23.6 mm across its 150 mm, on a 10.7 mm barrel | 61 CoACD hulls of the mesh it draws |
| Tray collision | An invisible floor slab, 6 mm high and 4 mm thin, and no walls | Four walls on the drawn walls, and a corrected 20 mm floor |
| Layouts | Syringes crossing walls that did not exist (`tray_clutter_t13` holds 7) | 22 out-of-tray syringes removed, documents and budgets re-derived (`t13` holds 5) |
| Robot | Panda with the Franka hand | Panda with a Robotiq 2F-85 in every config; the host declares `gap_perception` |
| Renderer and look | RTX, house look | Tiled and the bare look in every config |
| Physics buffers | `vab_500hz@v1`: `njmax` and `nconmax` 1024 | `vab_sharps_union@v1`: 8192 and 4096, otherwise the same physics |
| `fast` | 400 Hz, 8 workers | 200 Hz (`vab_syringe_200hz@v1`), 16 workers |
| Variants | 94, including the dropped `tray_pair_tNN` sweep | 64 |

Unchanged: the graphs and the variant names the cells use. `paper` keeps its protocol apart from the hand and renderer,
which is its one departure from v1.0's commands. The frozen copies under `benchmark/syringe_packing/` are now the
documents that run, byte for byte.

`vab_syringe_fast@v1` stays registered, but nothing uses it: it was measured on the old scene.

## Results

Full cells of 30 at `044706c` under `fast`, with the renderer, rate and workers below, all with the bare look and the
Robotiq. One run per score unless two are given. A cell passes a tray only if every syringe in it is posted. The paper's
L1 and L2 are 87.0% (26/30) and 63.0% (19/30).

| Setup | L1 solved | L1 reward | L1 wall | L2 solved | L2 reward | L2 posted | L2 wall | GPU peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX, 500 Hz, 8 workers, 1024 buffers | 29/30 | 0.983 | 29.6 min, 8 of it warmup | 15/30 | 0.783 | 138/176 | 72 min, with warmup | 55-56 GB |
| Tiled, 500 Hz, 8 workers, 1024 buffers | 30/30 | 1.000 | 22.8 min | 15/30 | 0.808 | 137/176 | 68 min | 24 GB |
| **Tiled, 200 Hz, 16 workers (what `fast` ships)** | 27/30 | 0.950 | 13.8 min | 19/30, 14/30 | 0.806, 0.739 | 139, 126/176 | 34.5, 31 min | 40 GB |
| The same, servo speed unset | 28/30 | 0.967 | 12.9 min | 14/30 | 0.737 | 129/176 | 31.5 min | 40 GB |

- **Rate:** 200 Hz halves L2's wall time and solves no fewer trays (33/60 against 15/30, Fisher p = 0.66). L1 solves 2-3
  fewer than at 500 Hz and stays above the paper.
- **Renderer:** tiled scores the same as RTX, needs no shader warmup, and holds less than half the GPU memory.
- **Servo speed:** the suite's `(:servo-speed-mps 0.12)` made no measurable difference, so it stays as dev wrote it.
- **L2 against the paper:** one of these five runs reached 19/30. All 73 failures were graph aborts, none ran out of
  budget. `tray_clutter_t15` and `t18` were never solved.

### Workers, rate and budget

Measured on the second port (same layouts, tiled, bare look, 8192 and 4096 buffers, servo speed unset, an earlier
RoboSimStudio). One L2 run of 30 per row, on shared tool servers with the documents' budgets unless stated.

| Workers | Rate | Other | Wall | Solved | Reward | Posted | GPU peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 400 Hz | | 57 min | 21/30 | 0.843 | 148/176 | 24.2 GB |
| 12 | 400 Hz | | 46 min | 14/30 | 0.776 | 136/176 | 32.4 GB |
| 16 | 400 Hz | | 45 min | 20/30 | 0.840 | 147/176 | 40.1 GB |
| 24 | 400 Hz | | 44 min | 21/30 | 0.861 | 149/176 | 56.2 GB |
| 10 | 400 Hz | a tool server per worker | 47 min | 18/30 | 0.819 | 143/176 | 90.8 GB |
| 16 | 400 Hz | budget capped at 35,000 | 42 min | 13/30 | 0.793 | 135/176 | 40.3 GB |
| 10 | 300 Hz | a tool server per worker | 39 min | 18/30 | 0.810 | 140/176 | 91.6 GB |
| 16 | 300 Hz | | 36 min | 19/30 | 0.833 | 145/176 | 40.3 GB |
| 16 | 200 Hz | | 30.5 min | 18/30 | 0.804 | 139/176 | 40.4 GB |
| 24 | 200 Hz | | 34.6 min | 23/30 | 0.891 | 156/176 | 56.2 GB |

- **Workers:** more than 16 saves little, because the shared SAM 3 saturates. A call takes 0.66 s alone, 2.1 s at 8
  workers, 3.5 s at 12 and 12.1 s at 24.
- **Rate** is what moves the time: 400 to 200 Hz at 16 workers took a third off.
- **Budget:** a cap that saves time cuts into successes. A tool server per worker costs 90 GB and saves nothing.
- **Memory:** peak is about 8.6 GB + 1.98 GB per worker, so the config states 8.6 and 2.2.

### Buffers

At `njmax` and `nconmax` 1024, one full L2 cell printed 14,517 `broadphase overflow` lines. MuJoCo-Warp drops the
collision pairs and carries on. The burst ended when `tray_clutter_t00` did, which still solved, so one episode looks
responsible and it cost no score that time. The other three full cells at 1024 printed none, so it is rare, but the
physics was silently wrong while it lasted. Raising the buffers cost no measurable step rate (122 against 126 steps/s,
about 32 MiB per scene).

## Not done

- **Overlapping starts.** Seven L2 layouts start with a pair of hulls overlapping by up to 2 mm. Four of them kick a
  syringe at 0.12-0.47 m/s at reset, and in `t12` `syringe_0` ends 57 mm past the wall. Three of the seven were solved in
  only 1 of 5 runs, though so were several other trays, so what this costs is not known.
- **A white spare's grasp station** is still about 22 mm off. The Robotiq tolerates it, but it remains a graph bug.
- **L2 scored higher on the second port** (mean 18.5/30 over ten runs at mixed settings, against 15.4 over these five).
  The layouts are identical, while settings and runtime versions differ. This is a hint, not a finding, and it is
  unexplained.
- `gt` was not touched. RoboSimStudio-dev has later fixes for it (`114329fe`), and it does not post here yet.
