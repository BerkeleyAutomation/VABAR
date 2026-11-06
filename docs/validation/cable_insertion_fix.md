# Cable insertion L1 (`usbc_insert/insert_easy`, graph `mate`): why it scored 0/30, and the fix

`cable_insertion/L1` scored 0/30 under the shipped protocol: palette segmentation,
`--no-privileged --perception vision --renderer rtx`. Every instance released the plug at
the mouth about 15 mm off the bore (`cable.md`; today's `fast` cell: median lateral 15.8 mm,
depth −0.5 mm). The cause is a camera-rig mismatch inside the graph's perception, not physics,
segmentation or graph constants for the stroke. One change in `ground_vision.py`
makes the graph work as designed: **the full L1 cell under `fast` goes 0/30 → 30/30**
(depth median 12.55 mm, lateral median 0.20 mm, 21.8 min on 4 workers). Goal, tolerances,
success checks, the camera rig and what the vision graph may read are all unchanged.

## Root cause

**The `mate` graph's perception was written and measured against `usbc_perception@v1`.
The documents stage `yam_station_usbc@v1`.**

| | `usbc_perception@v1` (what the graph's numbers describe) | `yam_station_usbc@v1` (what `insert_easy` stages) |
| --- | --- | --- |
| `topdown` | 960×720, straight down from 0.40 m over the bench, 0.35 mm/px | 1280×720 head camera, eye (0.2525, 0, 1.69), 0.95 m from its look-at point, 33° off vertical, ~1.3 mm/px at the parts |
| `wrist` | 320×240, fov 58° | 1280×720, fov 58°, same mount |
| `side` | none | 1280×720 shoulder view |

The evidence that the graph targets the first rig:

* `staging.py`'s module docstring says "Both documents stage" `usbc_perception@v1`. `bulkhead.task` still does.
* `graphs/mate/README.md` says the wrists are "320×240 RGB-D".
* The perception docstrings quote the straight-down topdown's 0.35–0.39 mm/px and its 0.40 m standoff.

The YAM station rig is appended at the end of `staging.py`, and the four `insert_*` documents
(and the frozen dataset files) stage it.

What that breaks is one gate. The chain:

1. **`ground_vision` measures the fixed port from its whole silhouette** (the `shape is None`
   branch: `_principal_xy(rec_pts)` for the heading, `median(rec_pts @ perp)` for the
   lateral). Seen straight down, a block's silhouette is its roof. From the oblique head
   camera, the 150 mm-tall, 40 mm-wide port also shows its camera-facing (−x) wall: on
   10009 that is about 1,500 of the 3,606 mask points (2,096 are roof), all at the wall's x. The read comes out
   biased toward the camera: on 10009 the mouth is (429.8, 11.9) mm against the true
   (439.2, 11.1), with heading −6.7° against +0.6°. Over nine layouts the bias is lateral
   +7.7 to +9.7 mm and heading −6.5° to −10.6° (offline replay, table below).
2. **`inspect_vision` gates the wrist read against it.** The gates are `MAX_MOVE_MM = 6`
   horizontal and `AXIS_MAX_DEG = 8`. Their docstring states the premise: "the bench read
   is good to about a millimetre in x and y". On this rig it is not. The wrist read
   (`bore_cv.recess_frame`) is excellent: replayed offline on the very frame the live run
   rejected, it reads 10009's mouth to (+0.04, −0.01, −0.16) mm. It is 9.4 mm from the
   bench read, so it is discarded.
3. **The fallback `bore_cv.find_bore` cannot run at this resolution.** Its `SEARCH_PX = 34`
   is a 320×240 constant. At 1280×720 the slot is about 65 px from the face centroid, so it
   reports `no slot-shaped recess`. The graph then "keeps the earlier camera read".
4. **For a bolted port nothing re-measures the mouth after that.** `relook_vision` keeps a
   fixed socket's earlier read (`[earlier camera frame of bolted fixture]`). The stroke is
   therefore aimed at the bench read: 9.4 mm off laterally, plus the bench read's
   deliberately crude height (the plug's, 12.4 mm under the bore). That is √(9.4² + 12.4²)
   = 15.6 mm, which is exactly the ~15–17 mm lateral error every L1 instance ended at.
5. `reseat: no supported 'orange plug' read` is a symptom, not a cause. At the reseat station
   the wrist lens is against the port and sees only blue (trace frame), and this happens on
   *successful* runs too, including on the original rig. The first push already seats the
   plug (12.5 mm), so the reseat routes to `release` without consequence.

L2 (`insert_medium`) was not affected because a held bulkhead is re-measured at the relook
(`fit_receptacle` against its catalogue mesh), so it never relies on the gated inspect read.

### Stage by stage, instance 10009 (`fast`, palette; printed node outputs)

| Stage | Oracle body (`--perception oracle`) | Vision body, shipped | Vision body, fixed |
| --- | --- | --- | --- |
| ground: mouth [mm] | (439.2, 11.1, 823.4) | (429.8, 11.9, 811.0) | (438.5, 12.2, 811.0) |
| ground: bore axis | (+0.01, +1.00) | (−0.12, +0.99) | (+0.00, +1.00) |
| ground: plug [mm] | (441.5, −67.5, 811.0) | (439.0, −67.7, 811.0) | (439.0, −67.7, 811.0) |
| inspect | exact by construction | `no slot-shaped recess; keeping the earlier camera read` | (439.2, 11.1, 823.4) from 534 px of aperture, 1.35 mm from the bench read |
| relook mouth | (439.2, 11.1, 823.4) | (429.8, 11.9, 811.0) | (439.2, 11.1, 823.4) |
| relook plug in jaws [mm] | (439.6, −24.2, 943.4) | (430.3, −23.0, 930.9) | (439.3, −23.9, 943.4) |
| stroke | 35.0 mm along (0.01, 1.0, 0) | 34.5 mm along (−0.116, 0.993, 0) | 34.7 mm along (0.01, 1.0, 0) |
| result | **success**, depth 11.54, lat 0.21 mm | fail, depth −0.26, lat 16.82, mate gap 13.26 mm | **success**, depth 12.51, lat 0.23 mm |

The oracle run takes the same schedule and succeeds, so the stroke, grasp, cable and
contact are sound. Only the frame is wrong.

### The rig, tested directly

The **unmodified** graph, run on `usbc_perception@v1`
(`--set 'suites.usbc_insert.cameras="usbc_perception@v1"'`, everything else the cell's):

| Instance | Shipped rig | `usbc_perception@v1` |
| --- | --- | --- |
| 10009 | fail, lat 16.8 mm | **success**, depth 12.48, lat 0.20 mm |
| 10024 | fail (`fast_usbc_insert.md`: lat 14.5 mm) | **success**, depth 12.48, lat 0.17 mm |

On that rig the ground read is (439.8, 11.9) mm with heading −1.1° (truth (439.2, 11.1),
+0.6°), and the inspect accepts
its wrist read: "52 px of aperture at 318 mm (1.06 mm from the bench read)". This is the
graph working as designed. The rig is part of the task, so it is not the fix. The graph is
made to hold its premise on the rig the task stages.

### The segmentation hypothesis (SAM3)

`cable.md` guessed that the paper used SAM3. The test is the same instance with
`ROBOSIMSTUDIO_SEGMENTATION_BACKEND=auto`, open-robot-skills resolvable, and SAM3 serving
`sam3.segment_text`. It **fails identically**: lat 17.97 mm, depth −0.05 mm. SAM3's
silhouette includes the same wall, giving a ground mouth of (426.8, 13.4) mm, and
`inspect_vision` does not use the segmenter at all (HSV threshold plus depth). Segmentation
is not the cause. After the fix, SAM3 runs: see Results.

## The change

| File | What | Why |
| --- | --- | --- |
| `tasks/usbc_insert/graphs/mate/scripts/ground_vision.py` | New `ROOF_BAND = 0.004` and `_roof(points)`. For a **fixed** receptacle (no catalogue shape), the heading (`_principal_xy`) and the lateral (`median(· @ perp)`) are taken over the roof: points within 4 mm of the part's 98th-percentile height. The `front`, the height (still the plug's) and the whole bulkhead path (`shape` present) are unchanged. | The roof is what the straight-down view this body was written for saw. On an oblique camera, the visible side wall dragged the read 8–10 mm and 7–11° off. That broke `inspect_vision`'s premise and made it reject a 0.2 mm wrist read. |
| `tasks/usbc_insert/graphs/mate/scripts/inspect_vision.py` | Docstring only: `MAX_MOVE_MM` says where its premise is now kept. | Points the next reader at the fix and away from loosening the gate. |
| `robot_skills/skills/perceiving-connector-pair/scripts/ground_vision.py`, `robot_skills/skills/inspecting-receptacle-mouth/scripts/inspect_vision.py` | The same two edits. | The README requires the exported skill bundles to stay in sync. They differed from the graph copies only in their import lines before and after. |

Not changed, and why:

* **The inspect gates.** Loosening `MAX_MOVE_MM` would also pass the misdetections it was
  written for (37–40 mm off).
* **`bore_cv.find_bore`'s pixel constants.** Deriving them from the focal length
  (×3.0 from 320×240 to 1280×720) was tried offline on nine frames. In 8 of 9 it locks onto
  the wrong recess, the step between bench and port, about 29 mm high. That read could
  slip through the 6 mm / 45 mm gates. At shipped values the fallback fails safe ("keeping
  the earlier camera read"), and with a correct bench read it is not needed.
* **The plug read.** It carries the same kind of wall bias (~2.5 mm on 10009), but the jaws
  centre the plug and the relook re-measures it in the jaws (the fixed run's in-jaw read on
  10009 is within 0.4 mm of the oracle run's), so it does not reach the mate.
* No goal, tolerance, check, step budget, camera rig, prompt or privilege boundary.

### Offline check, nine layouts (old vs new ground, then `inspect_vision` replayed on the traced wrist frame)

Seeds 10000–10007 are not in either cell. They were used to design the fix, with 10009 as
the reference. Ground truth is computed from the drawn port pose and the cavity payload,
and agrees with the oracle and inspect reads to 0.1 mm.

| Seed | Old ground lat / heading | Old final mouth error | New ground lat / heading | Inspect | New final mouth error |
| --- | --- | --- | --- | --- | --- |
| 10009 | +9.4 mm / −7.2° | 15.6 mm (kept) | +0.7 mm / −0.7° | wrist | 0.16 mm |
| 10000 | +7.7 / −10.6 | 14.7 | +1.1 / −0.1 | wrist | 0.02 |
| 10001 | +9.0 / −7.5 | 15.3 | +0.9 / −0.3 | wrist | 0.17 |
| 10002 | +7.7 / −10.2 | 14.7 | +1.1 / −0.1 | wrist | 0.02 |
| 10003 | +9.5 / −6.5 | 15.6 | +1.2 / +0.2 | wrist | 0.20 |
| 10004 | +9.1 / −7.4 | 15.4 | +0.9 / −0.1 | wrist | 0.14 |
| 10005 | +9.7 / −7.2 | 15.8 | +1.3 / +0.5 | wrist | 0.15 |
| 10006 | +9.2 / −7.9 | 15.5 | +1.1 / +0.1 | wrist | 0.19 |
| 10007 | +8.0 / −10.5 | 14.8 | +1.3 / +0.2 | wrist | 0.07 |

## Results

All runs use `--config fast`, the cell's flags, and the one GPU per process (RTX PRO 6000).
Depth and lateral are the goal's `insertion` and `insertion:lat`; the gates are ≥ 10.4 mm
and ≤ 8.79 mm.

### Full cell: `vabar bench --cell cable_insertion/L1 --solve graph --config fast`

| | Before (today's cell, from the brief) | After |
| --- | --- | --- |
| Success | **0/30** | **30/30** (95% CI 89–100%) |
| Depth [mm] | median −0.5 | 12.41–12.64, median 12.55 |
| Lateral [mm] | median 15.8 | 0.17–0.23, median 0.20 |
| Tilt [rad] | -- | ≤ 0.051 |
| `inspect_vision` | bench read kept on every instance | fresh wrist read on 30/30, 1.22–2.38 mm from the bench read |
| Steps | -- | 3,309–3,841, mean 3,482 |

Wall time: **21.8 min all in** (bench 1,323 s; 21.0 min of episodes). That is 4 workers
sharing GPU 0 through the config's private MPS daemon, 83 steps/s aggregate end to end,
with every step captured (`cuda graph: 4 captured, 0 eager`). The host load average was
95–150 from other tenants throughout. `fast_usbc_insert.md` estimated ~21 min for this cell.

### Single instances

| Instance | Before (shipped graph) | After (fixed graph) |
| --- | --- | --- |
| L1 10009, palette | fail: depth −0.26, lat 16.82, mate gap 13.26 mm | **success**: 12.51 / 0.23 (alone); 12.55 / 0.23 (cell) |
| L1 10009, SAM3 (`auto`) | fail: depth −0.05, lat 17.97 mm | **success**: 12.53 / 0.22 |
| L1 10024, SAM3 (`auto`) | -- (palette: fail, lat 14.45, `fast_usbc_insert.md`) | **success**: 12.46 / 0.18 |
| design seed 10000, palette | bench read kept (offline), 14.7 mm off | **success**: 12.62 / 0.22 |
| design seed 10001, palette | bench read kept (offline), 15.3 mm off | **success**: 12.58 / 0.17 |
| design seed 10002, palette | bench read kept (offline), 14.7 mm off | **success**: 12.59 / 0.21 |

With the fix, the vision graph lands where the privileged runs do: the oracle body on 10009
reached 11.54 / 0.21, and `gt` on 10009 reached 12.6 / 0.18 (`fast_usbc_insert.md`).

### L2 (`insert_medium`), regression check

The change cannot reach L2. A held bulkhead has a catalogue shape, so `footprint = rec_pts`
and its ground read is the shipped one (10032's printed ground line is identical to the
baseline graph's to 0.1 mm). `inspect_vision` changed only in a docstring. Measured anyway,
first four cell instances:

| Instance | Fixed graph | Baseline graph |
| --- | --- | --- |
| 10085 | **success**, 12.81 / 0.55 | -- (`fast_usbc_insert.md`: success) |
| 10010 | **success**, 13.21 / 0.10 | -- (`fast_usbc_insert.md`: success) |
| 10032 | fail: `relook: no supported fresh socket surface fit`, abort | 1/2: **success** 13.07 / 0.43, and fail with the **same** relook abort |
| 10024 | fail: depth 3.14, tilt 0.30 rad (stopped cocked) | not run |

10032's abort happens under the baseline graph too, so it is L2's own run-to-run variance in
the bulkhead relook, not this change. The full L2 cell was not run.

## Left open

* **The paper's configuration is not established.** The frozen dataset's `.task` files
  stage `yam_station_usbc@v1`, as the documents do. The evidence above (the graph's
  docstrings and README, and 2/2 on `usbc_perception@v1` with the unmodified graph) points
  to the graph's numbers having been taken on `usbc_perception@v1` before the documents
  switched rigs, but nothing in the tree records when.
* **The reseat never reads the plug** on L1, on either rig, fixed or not: the wrist is
  against the port at that station and the `side` fallback finds no qualifying mask. It is
  harmless here (the first push seats 12.5 mm), but the "one reseat" the workflow describes
  does not run on the fixed port.
* `bore_cv.find_bore` is effectively disabled at 1280×720 (see above). It is a 320×240
  detector, kept as a fail-safe.
* `graphs/mate/README.md` still describes 320×240 wrists. Left as is, because the
  bulkhead variant stages that rig.
* **L2 10032 aborts at the bulkhead relook** (`no supported fresh socket surface fit`) in
  2 of 3 runs today, baseline graph included. Earlier records have it seating 3/3
  (`fast_usbc_insert.md`, `fast_resolution.md`). Not investigated: it is outside this
  change's path.
* `benchmark/vabar_v1.json`'s note on both cable-insertion cells ("the release does not
  reproduce the paper's cable rates") is generated by `vabar/benchmark.py` and was left
  for the owners. For L1 it no longer holds.

## Reproduce

```bash
export PYTHONPATH=<robosimstudio>:<graph-as-policy>:<graph-as-policy>/gap-core/src
# one instance
ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette python -m vabar run --config fast \
  --task usbc_insert/insert_easy --episodes 10009 --solve graph --graph mate \
  --no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps 600
# the cell
python -m vabar bench --cell cable_insertion/L1 --solve graph --config fast --out <dir>
# the rig test (unmodified graph): add
#   --set 'suites.usbc_insert.cameras="usbc_perception@v1"'
```

Scratch runs (not part of the release): `agent_scratch/ins/runs/` — `base` (before, 10009:
palette, oracle, SAM3), `probe` (design seeds, traced to the inspect station), `rig`,
`fix1`, `cellL1`, `l2`, `sam3fix`.
