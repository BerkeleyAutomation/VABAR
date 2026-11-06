# cable_routing/L1: why the reference graph rarely succeeded, and the fix

Cell `cable_routing/L1` = `cable_route/port_easy`, graph `gt` (= `graphs/weave`),
`--no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps 60`,
`ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`. Same stack as [cable](cable.md) and
[fast_cable_route](fast_cable_route.md).

## Result

| Run (`--config fast`, 30 instances, 6 MPS workers, one RTX PRO 6000) | Terminal detected | Success | Mean reward | seat0 / seat1 / port box / end | Wall |
| --- | --- | --- | --- | --- | --- |
| As shipped (benchmark rig, 1280x720) | 0/30 | **4/30** | 0.600 | 29 / 30 / 4 / 9 | 40.2 min |
| Floor scaled by the frame's `fx * fy` (fix 1) | 30/30 | **15/30** (50 %, CI 33-67 %) | 0.825 | 29 / 30 / 21 / 19 | 46.2 min |
| Fix 1 + narrow jaw opening at the port (fix 2, shipped) | 30/30 | **16/30** (53 %, CI 36-70 %) | 0.842 | 27 / 26 / 25 / 23 | 42.8 min |
| *A/B:* as shipped, but on the suite's own 1280x960 `cable` view | 30/30 | 15/30 | 0.758 | 27 / 27 / 18 / 19 | 45.8 min |

* **The fault was the camera, not the graph's logic or the physics.** The unmodified graph,
  given the camera its floor was measured on, finds the terminal every time and scores
  15/30. On the benchmark's own camera, fix 1 scores the same 15/30. The paper reports
  46.7 % on L1.
* The successes come from the clauses the terminal stage exists for: the port box went
  4 -> 21 and the end 9 -> 19. Seats are unchanged (29-30 of 30).
* Wall time rises because the insertion and the restore pass now run (mean 6,380 steps
  per episode, against about 5,000). The walls are indicative only. The host was shared
  (load average 45-110), and single diagnostic runs shared the GPU for part of three
  cells: the tail of the shipped one, the start of the A/B and the first minutes of
  fix 1. Fix 2 (42.8 min) ran alone.
* Run-to-run variance per instance is large (`10009`: 1.00 as shipped, 0.50 with fix 1,
  1.00 on the A/B), so read the totals, not single rows.
* **Fix 2 is a real mechanism fix whose cell-level gain is inside the noise.** It changes
  the run only where a fingertip would land on the port. On those 14 episodes, success
  went 4 -> 6 and mean reward 0.732 -> 0.821. Four of the five episodes that stalled under
  fix 1 now succeed (10003, 10021, 10035, 10046). On the other 16 episodes the code path
  is identical, and the two cells still differed by 11 -> 10 successes and 0.906 -> 0.859.
  That is this cell's run-to-run noise, and 16/30 vs 15/30 is inside it. Fix 2 moves
  the terminal clauses (port box 21 -> 25, end 19 -> 23). It also has a cost: 5 of the
  7 seats lost under fix 2 (10017, 10026, 10077, 10084, 10090) are on narrowed episodes,
  where fix 1 lost none. Once the carry actually picks the cable up, it drags the seats
  more than a carry that holds nothing, and `restore` does not always recover them. On
  those episodes the net is still positive (reward +0.09).

### `fast` physics is not what holds this cell back

Fix 1 was also run under `--config default` (the reference `cable_coupled_960hz@v1`) on
the cell's first 12 instances, same graph, 6 MPS workers:

| Same 12 instances | Success | Mean reward | seat0 / seat1 / port box / end | Steps/s end to end | Wall |
| --- | --- | --- | --- | --- | --- |
| `fast` (`cable_route_fast@v1`), from the fix-1 cell | 6/12 | 0.792 | 12 / 12 / 6 / 8 | 71 | (46.2 min for 30) |
| `default` (`cable_coupled_960hz@v1`) | 5/12 | 0.812 | 11 / 12 / 8 / 8 | 28 | 43.5 min for 12 |

Per instance the rewards differ on 5 of 12, in both directions, which is the cell's
run-to-run spread. There is no sign that `fast` costs seats or terminal clauses
(`agent_scratch/rte/default_vs_fast.md`).

## Root cause: the terminal floor is a pixel count of a camera the benchmark no longer uses

`plug_port.py` finds the port as an "orange block" mask of at least
`BLOCK_MIN_PX = 3400` pixels. Its docstring measured that floor on `port3`'s opening
frame: port 4,766 px, spools 2,479 and 2,779 px. That frame came from the suite's own
`cable` camera, `_weave_rig()` in `tasks/cable_route/staging.py`: **1280x960**,
`fov_y` 50 deg, 0.60 m from (0.46, -0.06, bench) at 40 deg elevation. So
fx = fy = 480 / tan 25 deg = **1029.4 px**.

VABAR's `port_easy.task` and `port_medium.task` stage `yam_station_cable@v1`
instead. Its `cable` camera has the **same eye, look-at and `fov_y`**, at
**1280x720**, so fx = fy = 360 / tan 25 deg = **772.0 px**. The old source trees
(`autolab_cable_tasks`, both `AUTOLab-GaP-NVIDIA-Collaboration` copies) stage
`cable_port_easy@v1`, the 960-row view.

At a fixed pose, a mask's pixel count scales with fx * fy. Every mask is therefore
(720/960)^2 = **0.5625** of what the floor was set against, and the port drops under it.

### Evidence

**Same scene, rendered both ways.** `cable_route/port_easy` was built with both `cable`
cameras in one scene (benchmark rig and suite rig, same pose) and reset at six seeds.
Palette "orange block" components were back-projected through the frame's own `K` and
pose and matched to the port fixture's true position (diagnostic only):

| Seed | 720 rows: port / spools / anchor [px] | 960 rows: port / spools / anchor [px] | port 720/960 |
| --- | --- | --- | --- |
| 10009 | 2,137 / 1,539, 1,516 / 1,048 | 3,965 / 2,689, 2,568 / 1,791 | 0.539 |
| 10062 | 2,170 / 1,606, 1,513 / 634 | 3,744 / 2,776, 2,593 / 1,122 | 0.580 |
| 10084 | 2,218 / 1,480, 1,438 / 889 | 3,794 / 2,654, 2,529 / 980 | 0.585 |
| 10043 | 2,169 / 1,682, 1,649 / 567 | 3,826 / 2,957, 2,926 / 992 | 0.567 |
| 10053 | 2,098 / 1,703, 1,617 / 988 | 3,758 / 2,917, 2,889 / 1,750 | 0.558 |
| 10010 | 2,173 / 1,707, 1,661 / 655 | 3,941 / 2,968, 2,907 / 1,162 | 0.551 |

* On the 960-row view the port is 3,744-3,965 px, above 3,400 every time. On the
  720-row view it is 2,098-2,218 px, under 3,400 every time. The mean ratio, 0.563,
  is the predicted 0.5625.
* **The camera mapping is not at fault.** The port's mask back-projects through
  `CameraSpec.intrinsics` and `pose_on_body` to within 1.5-3.1 mm of the fixture's true
  position on both views. The RTX render honours the spec's `K` and pose.
* At plug time in the baseline full cell (720 rows), the port read 2,028-2,207 px and
  the spools at most 1,660 px. The floor rejected the port in every episode
  ("no terminal detected"), so the endpoint insertion never ran.

**The delivery this graph came from found the port every time on its 960-row view.**
Two old-stack runs of this cell's 30 instances are in
`/home/shadeform/autolab_cable_tasks/outputs/`. Both used `--segmentation palette` and the
1280x960 `cable` view (their saved `initial-cable.png` frames are 1280x960):

| Run | Crossing template | Port mask at plug | Terminal detected | Success | Mean reward |
| --- | --- | --- | --- | --- | --- |
| `repro` (delivery 4cb878bd) | pre-wrap | 3,416-4,415 px | 30/30 | 20/30 | 0.883 |
| `repro_ae6de67e` | wrap (as VABAR's graph) | 4,256-4,523 px | 30/30 | 8/30 | 0.717 |

The "old runtime 2,513-2,562 px" in [cable](cable.md) fits a 720-row view on the old
runtime: 4,400 x 0.5625 = 2,475. That this is what that A/B rendered is inferred; its
logs are not on disk.

**SAM3 is not the missing configuration.** In-process `facebook/sam3` was run on the same
frames ("orange block", both resolutions). It scores every candidate 0.007-0.06,
against `BLOCK_SCORE = 0.50`, so it grounds nothing. Both old runs that found the port
used palette.

### A second, smaller effect: this runtime renders the port's shaded faces out of the palette

At 960 rows the port masks 3,744-3,965 px here, against about 4,380 px on the old
runtime at the same view (13 % less). The spools are unchanged. Upstream's RTX renders
the same `ACCENT_PROP` (0.85, 0.35, 0.12) darker and redder, on a dark bench. The
old runtime drew a pale orange on a light bench (`agent_scratch/rte/probe/port_old_vs_new_960.png`).
The port's two shaded faces fall under the palette's hue floor of 8. That squeezes the
floor's margins from the docstring's 22 % below / 40 % above to about 10 % each side. It
does not by itself move the port under the floor; the resolution does.

## Second defect, exposed once the terminal is found: the terminal grasp lands a finger on the port

With the terminal detected, `plug_port` runs its full sequence: clear the hand, re-observe
the tail, grasp it 15 mm from the tip, carry it around the stations, release, then make
one measured correction. In the first fixed cell the main carry often left the end
where it was. The tail endpoint before the carry and after release agreed to 0.0-1.5 mm
on 10003, 10021, 10033, 10035 and 10046. The frames refresh between those reads
(the hand moves in them), so these are not stale images; the cable truly did not move.

An instrumented copy (`agent_scratch/rte/variants/diag_instrumented_plug_port.py`, 10003 and 10009) recorded the
support hand's **measured** finger width (`gripper_fraction` x 94 mm span) after each step:

| Step | 10003 | 10009 |
| --- | --- | --- |
| jaws closed (24-step ramp + 24-step wait) | **69.7 mm** | **69.8 mm** |
| after the 70 mm lift | 7.1 mm (closed on air; the cable is still on the bench) | 7.3 mm (on air, likewise) |
| correction grasp closed | 69.7 mm | 69.9 mm |

On air the same hands close within the same 48 steps (`grip_speed.py`, both arms alike:
94 mm open, 11.0 mm after 24 steps, 3.4 mm after 48 on a 3 mm command; the tool's own
close command is about 6 mm, which is the 5.9-7.3 mm read above). So these jaws are
**blocked, not slow**. The cable-camera frame at the close shows why: one
fully opened finger has come down inside the port's channel
(`agent_scratch/rte/probe/diag_10003_close_zoom.png`). The hand also sits 10.6 mm above
its planned grasp height, on the 12.5 mm walls.

Geometry over all 30 episodes of the fix-1 cell. Jaws close across the cable, so the
open fingertips sit 47.5 mm (`grip_open_m`) either side of the grasp, normal to the
tail's tangent. Distance from the nearer fingertip to the perceived port centre:

| Main carry outcome | Episodes | Nearer open fingertip to port centre |
| --- | --- | --- |
| end did not move (< 2 mm) | 10003, 10021, 10033, 10035, 10046 | 7.0, 14.1, 27.2, 29.6, 31.9 mm |
| end moved >= 44 mm, or was delivered with no correction needed | 21 others | 33.8-161 mm, except 10017 (25.4) and 10026 (26.1) |

The boundary is soft, around 32-34 mm; the jaws' exact footprint and the tangent estimate
blur it.

**It is not a camera effect.** The unmodified graph on the 960-row view (the A/B below)
stalls the same way: 10002, 10009, 10021, 10022 and 10024 did not move the end (0.1-3.0 mm),
with the nearer open fingertip 7.8-33.4 mm from the port. The carried ends there had
24.9-55.8 mm.

The grasp point is chosen as the first reachable tail candidate (15, 25, 40, 60 mm from the
tip, then centreline points out to 150 mm) with no check that the open jaws clear the
fitting the tip is lying next to. The one-shot correction re-reads the same tail and picks
the same point, so it usually stalls the same way. When it succeeds, the cable has
caught as the fingers slide off the wall.

### Two fixes for the stall, tested head to head

Both keep a fingertip `PORT_KEEPOUT` = 40 mm off the perceived port centre. That is
above every stall measured, 33.4 mm at most over both cells, and it is a little
conservative: some deliveries had a fingertip at 34-39 mm.

1. **Move the grasp inboard** (`agent_scratch/rte/variants/fix2_keepout_rejected_plug_port.py`). Candidates whose fully
   open jaws reach the keep-out go to the back of the list. **Rejected.** It moved the
   grasp to 40-80 mm from the tip on 9 of 12 episodes. It fixed the stalled 10003 and
   10021, but broke 10002, 10010, 10020 and 10026, which the 15 mm grasp had solved.
   Result: 5/12 against 7/12 for the same seeds in the threshold-only cell, stopped at 12.
   A tip overhanging the hand by 60 mm is a different thing to place.
2. **Keep the grasp and open the jaws only as far as clears the port** (kept). Each
   candidate is opened to the widest opening per finger, from 47.5 mm down to 15 mm
   (`JAW_MIN_REACH`), whose fingertips clear the keep-out. A candidate moves back only if
   even 15 mm does not clear. Replayed on the threshold-only cell's recorded candidates,
   it changes only the stalled episodes: 10003 opens to 33 mm, 10017 to 24, 10033 to 20,
   10035 to 38, 10046 to 31 and 10084 to 36; 10021 takes its 40 mm point at 19 mm. Every
   other episode keeps the full opening and the 15 mm grasp. The same opening is used
   for the one-shot correction.

On 10033, alone and traced, the narrow opening (21.5 mm) moved the end 70.5 mm where
the full opening had moved it 1.5 mm. The episode closed at 1.00, against 0.50 in the
threshold-only cell. In the full fix-2 cell 10033 closed at 0.50 again: after the
release its free end was not unambiguously visible, so no correction ran, and the end was
not in the box.

## Seats

Seat losses were not what kept this cell down. Over the four complete `fast` cells
(shipped, fix 1, A/B, fix 2; 120 episodes) seat0 held 29, 29, 27 and 27 times and seat1
30, 30, 27 and 26 times. When the terminal
stage runs, its carry does disturb seats. `restore` then re-seats them: in the fix-1 cell
it repaired at least one station on 12 of 30 episodes, and both seats still held in 29.
The one lost seat there (10025, seat0) was read by `restore` as seated (33 mm visible,
against its 26 mm bar), and station 1 was repaired after it. So the station-1 repair
either dragged it or the reading was generous; the trace cannot tell which.

The two seat failures that remain are known and minor:

* **`restore_seats.py` lets a planner refusal escape.** `cable.plan_station` raises
  `ValueError` ("No feasible grasp relative to the fixed anchor ...") when a repair needs
  more cable than the anchor leaves. The node then errors before checking the next
  station, routes to `verify`, and the run aborts. The ae6de67e delivery wrapped both
  `plan_station` calls in `try/except (ValueError, RuntimeError)` ("planner refused;
  check the next seat"); VABAR's copy does not. It fired once in 120 episodes (shipped
  run, 10022, reward 0.50 either way). Left as is; restoring the `try/except` is the fix.
  *Since restored, with the restore pass enabled on L2: [cable_routing_l2](cable_routing_l2.md).*
* The A/B's 27/27 seats (vs 29/30 at 720 rows) is within run-to-run spread. One cell per
  arm cannot separate it.

Under `default` physics on 12 instances the seats held 11 and 12 times. Seat retention
is no different from `fast` here.

## Changes

| File | Change | Why |
| --- | --- | --- |
| `tasks/cable_route/graphs/weave/scripts/plug_port.py` | `BLOCK_MIN_PX` (3,400) is now read through `block_min_px(K)` = `BLOCK_MIN_PX * fx * fy / BLOCK_REF_FOCAL_PX^2`. `BLOCK_REF_FOCAL_PX` = 1029.4 px, the focal length of the 1280x960 / 50 deg view the floor was measured on. The `[plug]` log line prints the floor actually used. | The floor is a solid angle, so it has to scale with the camera's focal length. At the benchmark's 1280x720 view it is 1,912 px; on the 960-row view it is unchanged at 3,400. The value, the score floor, the query, the segmentation backend, the block choice ("furthest from the anchored end") and every task clause are untouched. The graph reads only the frame's own `intrinsics`, which every observation already carries. |
| same file | New `PORT_KEEPOUT` (40 mm), `JAW_MIN_REACH` (15 mm), `jaws_clear_port`, `jaw_reach_clearing_port`, `clear_of_port`. `_regrasp_free_end` and `_correct_terminal` open the grasping hand to each candidate's port-clearing width (`set_gripper(width=jaw)`) instead of full open (`-1.0`) just before the take. They keep the candidate order unless no width down to 15 mm clears. They take `reach=grip_open_m`. Diagnostics: `terminal_jaw_open_m`, `candidates_jaws_on_port`, `candidates_opened_narrower` and their `terminal_correction_*` twins. | Fully opened jaws put a fingertip in the port's channel and the pinch stalled at 70 mm, so the carry moved nothing (above). The geometry uses only the perceived port centroid, the perceived tail and tangent, and the node's own `grip_open_m`. The set-gripper call now also targets the hand the plan picked, which matters when the plan falls back to the work arm. |
| `robot_skills/skills/terminating-the-run/scripts/plug_port.py` | the same edits | The exported bundle copy of the same script (`SKILL.md`: generated from the source graph). Only its sibling import differs, as before. `tools/sync_robot_skills.py` is not in this tree, so the copy was regenerated by hand. |
| `docs/validation/cable.md`, `docs/validation/fast_resolution.md` | pointers to this note | Both said the terminal is never detected. |

Not changed: the camera rig. Going back to `cable_port_easy@v1` (1280x960) would also
restore detection; the A/B below measures exactly that. But it changes the benchmark's
observation spec, and it would leave the graph tied to one resolution.

## How to reproduce

```bash
export PYTHONPATH=<robosimstudio>:<graph-as-policy>:<graph-as-policy>/gap-core/src
# the cell (6 MPS workers on the first visible GPU)
CUDA_VISIBLE_DEVICES=2 python -m vabar bench --cell cable_routing/L1 --solve graph --config fast --out <dir>
# the A/B: the graph as it was, on the suite's own 1280x960 cable view
CUDA_VISIBLE_DEVICES=3 python -m vabar bench --cell cable_routing/L1 --solve graph --config fast \
  --set 'suites.cable_route.cameras="cable_port_easy@v1"' --out <dir>
```

Scratch scripts and raw runs: `/home/shadeform/speed_ws/agent_scratch/rte/`.
`runs/` holds the bench outputs and `probe/` the frames above. `tools/` holds:

* `probe_block.py`: both `cable` views in one scene, and the palette/SAM3 components
  measured against the port.
* `tally_new.py`, `compare.py`: per-clause tallies.
* `plugsum.py`, `jawclear.py`, `check_keepout.py`: the plug-stage geometry.
* `grip_speed.py`: on-air jaw closing.
* `make_diag.py`: the instrumentation applied to a copy of fix 1.

`variants/` holds each `plug_port.py` that was run: as shipped, fix 1, the rejected
keep-out, fix 2 (shipped) and the instrumented copy.

## Left open

* **"Carried, then lost."** This is now the main terminal failure. In 11 of 30 fix-1
  episodes the main carry moved the end 44-225 mm and then lost it. The end landed
  47-160 mm from the port, often near the carry's `across` waypoint (port y + 60 mm).
  The pinch is by design shallow: a 3 mm command per finger, about 6 mm of jaw, on a 7 mm
  cable, with no latch. The likely mechanism is the detour pulling the short post-weave
  tail taut until it slips out. It was not instrumented further. The one-shot correction
  recovers some of these; a second correction or a shorter detour would be the next
  thing to try, and both are policy redesign rather than repair.
* **The floor's margin in this runtime is thin.** *(On L2 it was too thin: 2 of 90 ports fell under it. The floor is now 1,500 px at 720 rows and the selection rule alone picks the port; see [cable_routing_l2](cable_routing_l2.md).)* At plug time the port read
  2,034-2,232 px against the scaled 1,912 px, about 6 % headroom, and the spools
  <= 1,660 px. On the old runtime the same floor had about 25 % above it. The difference
  is upstream's RTX look (above), not geometry. If the render changes again, a gate that
  does not depend on colour fill would be sturdier: height, since the port walls are
  12.5 mm and the spool flanges 23.5 mm. The selection rule ("furthest from the anchored
  end") already keeps a spool from being chosen whenever the port itself is found.
* **SAM3.** In-process `facebook/sam3` does not ground "orange block" on these frames
  (scores <= 0.06). The open-robot-skills bundle's SAM3 was not tried. The cell's protocol
  is palette, and both old-stack runs that found the port used palette.
* **The paper's 46.7 %.** Fix 1 (15/30) and the A/B (15/30) are consistent with it. Which
  delivery commit and seeds the paper ran was not established. The two old-stack runs on
  disk scored 20/30 (pre-wrap crossing) and 8/30 (wrap crossing, older `plug_port`).
