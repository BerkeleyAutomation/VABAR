# tool_sorting/L2: the heap tier, ported and measured

Measured 2026-11-05 and 06 on RTX PRO 6000 GPUs 4 and 6 of a shared 240-core host, one cell per GPU, with other jobs
running. Everything ran from a copy of VABAR under `--config fast`, with the overrides each row names. Raw output and
the counting scripts are under `/home/shadeform/speed_ws/agent_scratch/sortl2/`.

**What ships is in "The graph's two fixes" below: 57-59% of objects in 32 min.** The "Result" table measures the graph
as it was found, before those fixes, so its "as patched" rows are no longer what `--config fast` runs.

## Result

Thirty documents of 8 to 12 goal objects, 273 in all. The paper counts this cell per object, so **objects** is the
number to compare with its 33.7%. *Solved* is the whole instance.

| Run | Budget | Objects | Solved | Mean reward | Steps (mean) | Wall | GPU peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`final_1`, `--config fast` as patched** | **2,400 per goal** | **145/273 (53.1%)** | **0/30** | **0.525** | **21,653** | **32.9 min** | **87.9 GB** |
| **`final_2`, the same** | **2,400 per goal** | **130/273 (47.6%)** | **0/30** | **0.467** | **21,840** | **33.1 min** | **87.8 GB** |
| `fastdoc_1`, the same budgets as a separate config | 2,400 per goal | 130/273 (47.6%) | 0/30 | 0.467 | 21,653 | 32.0 min | 88.0 GB |
| `fastdoc_2`, the same | 2,400 per goal | 125/273 (45.8%) | 0/30 | 0.452 | 21,840 | 35.9 min | 88.0 GB |
| `fast14k_1`, `fast` as found | 14,000 flat | 98/273 (35.9%) | 0/30 | 0.357 | 14,000 | 20.6 min | 88.2 GB |
| `fast20k_1` | 20,000 flat | 117/273 (42.9%) | 0/30 | 0.432 | 20,000 | 28.2 min | 88.5 GB |
| `fast40k_1` | 40,000 flat | 136/273 (49.8%) | 1/30 | 0.494 | 38,924 | 51.9 min | 88.4 GB |
| `doc500_1`, the reference's 500 Hz | 2,400 per goal | 131/273 (48.0%) | 0/30 | 0.477 | 21,840 | 43.2 min | 88.4 GB |

- **The graph sorts about half the objects and almost never a whole instance.** The four runs at the chosen budget
  pool to 530/1,092, **48.5%** (95% interval 41.9-55.2%, resampling instances). One instance in 240 episodes was solved:
  `level2_28`, at 19,895 steps, in the 40,000-step run.
- **Above the paper's 33.7%, on a different protocol.** See "Against the paper".
- **No worker hung.** No run logged a `[pool] watchdog:` line, and all 240 episodes finished.
- **The port is sound.** The graph read all four labels exactly in 30 of 30 scenes, in all eight runs.

## The graph's two fixes

The failure analysis below found two problems, both also on L1, so both were fixed in `gap_perception_loop` and
measured on both cells. Raw output is under `/home/shadeform/speed_ws/agent_scratch/sortl2/cap/`.

1. **The attempt log reaches `select_pair`.** The graph was written when the runtime ignored a subgraph node's own
   `inputs` and fed it the latest output of the same name, which is how the log came round the loop. graph-as-policy
   `a5bb898` (2026-09-05) made call-site inputs take precedence, so `pair`'s `$ref layout.attempted_json`, the empty
   seed, won on every visit from then on. Removing that one binding restores the feedback. `select_pair`'s `retry` exit
   also has to return the log, or a retry wipes it. A scan of the other VABAR graphs found no other subgraph node whose
   call site shadows its own output on a cycle.
2. **A blocked descent stops after 1 s without progress.** `descend` uses `robot.move_cartesian_until_contact` with
   `stall_steps = 60`, not `robot.go_to_pose`. Two L2 runs, before the change:

   | Descent | Share of descents | Share of an episode's steps | Ends holding an object |
   | --- | --- | --- | --- |
   | Times out at 1,000 steps | 60-66% | 42-45% | 6-7% |
   | Reaches its target | 34-40% | 3-4% | about 88% |

   A flat, lower timeout was rejected, because descents that reach their target can take 300-970 steps and those almost
   always hold.

| Cell | Graph | Objects | Solved | Steps (mean) | Wall |
| --- | --- | --- | --- | --- | --- |
| L2 | as found, four runs | 45.8-53.1% | 0/30 | about 21,750 | 32.0-35.9 min |
| L2 | log only | 51.6% | 1/30 | 21,587 | 36.5 min |
| L2 | log, 2 s stall | 52.7%, 56.4% | 0/30, 0/30 | 19,741, 20,112 | 33.3 min, 41.6 min (a build hang) |
| L2 | log, 1 s stall | 60.4%, 57.9% | 0/30, 0/30 | 19,273, 20,174 | 37.1 min, 33.0 min |
| **L2** | **log, 1 s stall, 20,000-step cap (ships)** | **59.3%, 57.1%** | **1/30, 2/30** | **18,516, 18,051** | **31.9 min, 32.0 min** |
| L1 | as found | 64-75% | 7-13/30 | about 11,000-12,000 | 14-16 min |
| L1 | log only | 79.3% | 9/30 | 11,013 | 19.2 min (slow builds) |
| L1 | log, 2 s stall | 81.4%, 80.0% | 10/30, 10/30 | 8,277, 8,572 | 11.7 min, 12.5 min |
| **L1** | **log, 1 s stall (ships)** | **78.6%, 75.7%** | **9/30, 9/30** | **8,718, 8,718** | **28.6 min, 22.1 min (each lost 15 min to a build hang)** |

- **1 s or 2 s:** 1 s averages 58.7% on L2 and 77.1% on L1; 2 s averages 54.6% and 80.7%. Neither gap is clearly outside
  the noise. 1 s ships, with more L2 runs behind it.
- **The cost:** empty grips rise (85 against about 33 on L2), because a descent stopped short sometimes passes
  `verify_reach`. Some episodes also end early with the graph reporting itself finished once every spot it can see is
  spent, with objects still unsorted: 12-21 of 30 on L2 and 19-21 of 30 on L1. Scores still rise, so the retries given
  up were worth less than the steps they cost.
- **The cap now costs almost nothing:** a flat 20,000 scored about 6 points below the documents' budgets before these
  fixes, and is level with them after.
- **The real cause is still open:** more than half of descents still stop short. Why the fingers land on the target is
  not known (see "What fails").

## What was ported

| What | Where | Source |
| --- | --- | --- |
| 30 documents | `tasks/tool_sort/level2_N.task` | RoboSimStudio-dev `tool_perception` 067645c6, plus one clause (below). Byte-identical to AUTOLab `tool-sorting` 69c8fcb on all 30 |
| 90 exported scene files (3.0 MB) | `assets/vab_tool_sort/level2/level2_N/` | The same commits. Byte-identical across `tool_perception`, AUTOLab and 7ffc4017 |
| 15 label plates (94 KB) | `assets/vab_tool_sort/assets/bin2_*.png` | `tool_perception`. **AUTOLab's tree lacks four of them** (`bin2_0_spur_gear_20t`, `bin2_2_spur_gear_20t`, `bin2_1_e27_bulb_socket`, `bin2_3_e27_bulb_socket`), used by 5 scenes |
| Frozen copies | `benchmark/tool_sort/level2/tasks/` | Equal to the documents; a new test holds them equal |
| Variants, cell | `tasks/tool_sort/tool_sort.suite`, `vabar/benchmark.py`, `benchmark/vabar_v1.json` | L1's shape |

No mesh is new. Of the 178 files the scenes name, 163 were already in VABAR, each byte-identical to dev's.

**Choices.**
- **`(:entity tool_sort_workcell)` is added to every document.** VABAR's L1 documents carry it (ac12747) and dev's L2
  documents predate it. It paints the bins and the printed labels. Without it L2 would render differently from L1, and
  the labels are what the VLM reads.
- **`tool_perception`'s documents, not `release/oss`'s.** Theirs name `robosuite_bin`, a `:fleet-base` that needs a
  merged export, and a reorganised asset tree. `tool_perception`'s match how VABAR's L1 is staged. Apart from comments,
  the bin type, the fleet base and provenance, all 30 are the same in both.
- **No fleet base.** Each document runs as its own one-world job, as L1 does.

**Dev's page gets one thing wrong.** `docs/tool_sort_level2.md` says "Distractors: none". Every document has one
(`distractor_industrial_*`, 303 objects against 273 goals), of a type no compartment names. It is not scored.

**`--verify` passes but checks nothing here,** as on L1: these instances record no transforms to compare. In its place,
offline: all 303 sidecar poses start inside `bin1_region`, every document's budget is 2,400 x its goals, and each
compartment's region, sidecar entry and label plate agree.

## The graph

VABAR's `gap_perception_loop` is what dev used. Against `tool_perception` 067645c6 ("Fixing the graph, add level2") it
differs only by the removal of one debug PNG write in `select_pair.py`, and the same against AUTOLab 69c8fcb. Against
dev's main, `select_pair.py` is identical and `workflow.json` differs in two skill labels. All 30 documents declare it.

## `fast` on L2

| Key | L2 under `fast` | How checked |
| --- | --- | --- |
| Physics | `vab_tool_sort_fast@v1`, 6 substeps. The suite's mapping applies because L2 stages the same `vab_industrial_robotiq@v1` | `run_meta.json` of every episode: 6 substeps, 10 in the 500 Hz run |
| VLM | `gemini-3.1-flash-lite` | The run's `[config]` line |
| Workers | 15, under MPS, one shared SAM 3 and GroundingDINO | Peak 88.5 GB, inside `default.toml`'s 9 + 15 x 5.5 |
| Budget | **Changed**: each document's own | See below |

### Budget: each document's own, 2,400 per goal

`fast` as found gives every tool_sort variant 14,000 steps, a number measured on L1's 4 or 5 objects. All 30 L2 episodes
ran it out. Dev's documents state 2,400 per goal (19,200 to 28,800, mean 21,840). The cell's own flag says 20,000.

| Budget | Objects | 8-goal documents | 12-goal documents | Picks at a spot already picked | Wall |
| --- | --- | --- | --- | --- | --- |
| 14,000 | 35.9% | 44/120 | 25/60 | 34% | 20.6 min |
| 20,000 | 42.9% | 51/120 | 22/60 | 50% | 28.2 min |
| **2,400 per goal** (four runs) | **45.8-53.1%** | **49-66/120** | **31-40/60** | **46-49%** | **32.0-35.9 min** |
| 40,000 | 49.8% | 65/120 | 37/60 | 68% | 51.9 min |

**Chosen: the documents' budgets.** They are dev's number rather than a new one, and they give the 12-goal documents the
room a flat budget takes from them. Past them, about 55% more wall time bought nothing outside the four runs' own range.
The cost against a flat 20,000 is 4-8 minutes; that one run sits below all four.

The 14,000, 20,000 and 40,000 rows are one run each, and one run's interval is wide: resampling instances, 35.9% has
28.4-43.4%. Instances also move between runs: the same document differed by 3.3 objects on average across the four
runs, and by 8 at most. The ordering is firmer than any one gap between neighbours.

`fast.toml` states the 30 budgets one line each, because its suite-wide `max_steps` would otherwise win over them. L1
still gets 14,000. The cell's `--max-steps 20000`, which `default` and `paper` read, is unchanged and was not measured.

### Physics: 300 Hz holds

| Probe | 500 Hz (reference) | 300 Hz (`fast`) |
| --- | --- | --- |
| Full cell, same budgets | 131/273 (48.0%), 43.2 min | 125-145/273 (45.8-53.1%), 32.0-35.9 min |
| Peak contacts / constraint rows, idle | 37-67 / 118-199 | 37-65 / 121-196 |
| Peak contacts / constraint rows, routine | 47-69 / 142-205 | 49-73 / 142-215 |
| Worst object in 29 s idle, `level2_2 / 3 / 5 / 6 / 7 / 21` | 8 / 52 / 14 / 16 / 46 / 13 mm | 9 / 47 / 74 / 15 / 27 / 12 mm |
| Routine, objects placed, `level2_1 / 3 / 12` | 2/8, 0/12, 1/8 | 5/8, 0/12, 3/8 |

No new arrangement was needed. Contacts stay far below `nconmax` and `njmax` of 1,024. The 500 Hz cell is one run, and it
ran beside `fastdoc_2`.

**A heap is not at rest on either physics.** Dev's sidecars record it as settled, but those settled in MuJoCo with box
colliders, and this suite's `drawn_hull` tools then slump: 8-52 mm at 500 Hz. `level2_5` moved 74 mm at 300 Hz against
14 mm at 500, the one large gap; the other five were within 20 mm, in both directions. The graph looks again before each
pick, so a slump is not a stale pose.

## What fails

Over the four runs at the chosen budget: 1,796 descents, of which 1,153 timed out.

The first item is fixed and the second is shortened (see "The graph's two fixes"). The analysis stands as measured
before those fixes.

1. **The retry cap never acts, so the graph goes back to the same spot.** `pair`'s `attempted_json` input is wired to
   `layout.attempted_json`, the empty seed, not to the log `select_pair.py` returns. On all 3,946 `select` visits of the
   eight runs the log arrived empty, so `max_attempts_per_instance = 4` and `max_total_attempts = 40` never fire. 46-49%
   of picks go to a spot already picked in that episode, one spot up to 15 times. At 40,000 steps it is 68%, and up to 27
   times, which is why more budget stops helping.
   - **This is not L2's alone.** The same wiring is in dev's `tool_perception` and main, and in AUTOLab's `tool-sorting`
     and `lgulich/tool-sort-loop-failure-fixes`. An L1 run shows the same: 0 of 289 visits got a non-empty log. So the
     "retries per position" in `dev_history_merge.md` and the "exclusion by pixel position" in `tool_sort_vlm.md` describe
     code that does not run.
2. **More than half the descents stop short, and each costs 1,000 steps.** `verify_reach` rejects 54-59% of attempts.
   `go_to_pose` runs to its 1,000-step timeout, against a median of 49 steps for a clear descent. Descents are 46-49% of
   all steps and 34-37% of wall time (28.3 s blocked, 4.1 s clear).
3. **That is not the pile.** It was the first explanation, and L1 rules it out: the separated tier does the same.

   | | L1, nothing rests on anything (2 runs) | L2, heap (4 runs) |
   | --- | --- | --- |
   | Attempts rejected | 49-55% | 54-59% |
   | First attempt of an episode rejected | 8-9 of 30 | 9-11 of 30 |
   | Rejected: hand short of the pose, median | 24-27 mm | 20-24 mm |
   | Rejected: above the commanded height, median | 19-21 mm | 15-17 mm |
   | Rejected: sideways, median (reached: about 3 mm) | 16-17 mm | 13-15 mm |
   | Rejected: tool point above the target's perceived top, median | 2-4 mm | 5-7 mm |
   | Objects sorted per attempt | 33-42% | 27-32% |

   The hand comes down about level with the top of the object it is after, pushed sideways as it stops. That reads as the
   fingers landing on the target rather than straddling it. **Neither that nor why was established:** it needs the
   contacts or the wrist views, and these runs kept neither. The commanded height does not explain it. Every grasp is commanded at `bin_floor_z +
   0.014` (all 1,873 are z = 0.836), but rejected and reached targets sit at the same perceived height on L2, a median
   17-19 mm above the floor.
4. **It gets worse as the bin empties.** Rejections rise from 43% of the first four attempts to 70% after twelve. What
   remains is what the graph keeps failing on, and item 1 keeps it there.
5. **The budget buys too few attempts.** About 15 attempts at about 30% each, for 8-12 objects, gives about half. Whole
   instances are out of reach at that yield.
6. **Types.** Worst, over the four runs: `sanding_block_large` 3/24, `wire_cutter_wall_mount` 5/28, `esp32_breadboard`
   3/16, `breadboard` 28/96. Best: `d_battery` 35/44, `spur_gear_20t` 29/40, `tactical_flashlight` 25/36,
   `cutting_pliers` 11/16.
7. **The graph can stop early.** Six episodes of 240 ended on the graph's own "finished" with goal objects left, for
   example `level2_15` at 4/8 and `level2_28` at 6/8. On the three checked, its detector returned only bin-sized boxes,
   which `select_pair.py` filters out, and it took the bin for empty.
8. **A drop is not always a sorted object.** 624 drops gave 530 sorted objects. The trace does not record where an object
   landed, so wrong compartments, misses and dropped distractors cannot be told apart here.

## Against the paper

| | Objects sorted | Median cycle (simulated) |
| --- | --- | --- |
| Paper, GaP Human-Newton | 33.7% | 913.8 s |
| This port, `fast` (four runs) | 45.8-53.1% | 408 s |
| This port, 40,000 steps | 49.8% | 800 s |

The rate is above the paper's, but this is not a reproduction, and the difference should not be read as an improvement
to the graph:
- **Clock.** `fast` holds the simulator while the graph thinks. The paper's free-running clock spends steps during
  perception, so step budgets and simulated seconds do not compare.
- **Physics and VLM.** 300 Hz and Gemini 3.1 Flash-Lite. The `paper` config has 500 Hz, and this repository's tool-sort
  numbers before `fast` used Gemini 2.5 Flash. 500 Hz alone did not move this cell.
- **Budget decides the number.** At 14,000 steps this cell scores 35.9%, close to the paper. That match is the budget's,
  not the graph's.
- **Cycle time here is the budget.** Nearly every episode is truncated, so the median is not how long sorting takes.

## Wall time

**The target of about 30 minutes is missed by 2 to 6 minutes:** 32.0, 32.9, 33.1 and 35.9 min, with another cell on the
next GPU each time. Wall time follows the budget, because nearly every episode runs it out: 20.6 min at 14,000, 28.2 at
20,000, 51.9 at 40,000.

Where an episode's time went, summed over every visit (`final_1`, 815 s mean):

| Node | Share | Each |
| --- | --- | --- |
| `grasp_sg.descend` | 34% | 18.1 s |
| `pair_sg.select` (DINO, VLM tournament, SAM 3) | 21% | 11.0 s |
| `transport_sg.transport_move` | 16% | 22.7 s |
| `pair_sg.move_aside` | 6% | 3.1 s |
| Other nodes | 23% | |

Build is 4-8%. The GPU read 100% utilisation when sampled with 15 workers, and at `default.toml`'s 5.5 GB a worker a 16th
does not fit. So the cell is bound by the budget and by what the graph does with it. The VLM is not the bound here: `select` is a fifth of
the time.

`tools/profile_instance.py` under-reports this graph: it reads each node once, so a looping episode shows over 90%
"between nodes". The table above comes from the trace's events instead.

A flat 20,000 meets the target, at about 6 points of objects in its one run. A descent that gives up sooner would meet it without that
loss, but that is a graph change and was not measured.

## Warm-up

`vabar warmup --cell tool_sorting/L2 --config fast`: 30 of 30 scenes in about 4 minutes on a host already warm for L1,
19-49 s to build each.

## Not done

- Find out what the fingers land on, with the wrist views kept (`--set trace_assets=true`). Not measured.
- `default` and `paper` on L2: not run. Their budget is the cell's 20,000.
- `gt` on L2, which the documents declare: not run.
- Dev's merged `level2_base` export and fleet waves: not ported.
