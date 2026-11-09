# gap_perception: the gear board from the overhead camera

The suite's non-privileged graph, in the shape the other suites' vision graphs
take: `--perception vision --no-privileged --renderer tiled`, every read a
camera or the robot's own proprioception, the segmenter the shared `sam3`
tool bundle (booted by the workflow's `requires`), no `sim.*` anywhere.

| Question | Answer | Where |
| --- | --- | --- |
| Where are the gears, which class is each | `sam3.segment_text(rgb, "gear")` masks back-projected through the frame's intrinsics; tip radius reads the tooth count at module 2.5 mm, the hue (cobalt 16, teal 20, plum 24) is asked to agree | `scripts/sight.py` `read`, `teeth_from_radius`, `hue_class` |
| Which is the pinion | the orange one of those | `sight._is_orange` |
| Where is the button | `"red circle"` (else `"orange disc"`), colour-gated | `sight.read` |
| Where are the pegs | `"pin"` masks 30-50 mm above the bench, plus whatever the depth says stands that proud inside the plate; every station records `peg_source` (`sam3`, `depth`, `both`) | `sight.read`, `View.high_blobs` |
| Which class each station wants | peg spacing: `module * (z_a + z_b) / 2`, seeded by the pinion's radius | `sight.geometry` |
| Which way the chain runs | the pinion-to-last-peg line; the jaws close across it | `sight.geometry`, `gearboard.base_quat` via `robot.grasp_frame` |
| Is a station seated / perched / empty / hidden | the height of an annulus around the peg against the plate's, read in a ring around the button | `sight.station_state`, `_plate_height` |
| Is something in the jaws | `robot.get_gripper` stopped part-way | `gearboard.holding` |
| Where the fingertips are below the TCP | `robot.get_gripper`'s live `tip_reach_m`, read at the opening in question -- on the 2F-85 the pads move 10 mm along the approach axis over the stroke | `gearboard.reach_now` |
| How far the held gear's centre sits under the TCP | measured as the jaws close: the TCP's height over the gear the camera placed on the mat | `gearboard.take_hold` |

The choreography (pick, carry, a turning descent released where it stops, clear,
press, watch) is the paper table's `gear_mesh_vision` graph's, with every
height taken relative to the bench the camera measured and every orientation
built on the hand's own grasp frame turned to the measured chain. Only the
cursor (`next_station_sg`) publishes `index`.

Measured on `outputs/snapshots/gear_after2` before the graph was written: SAM3
names every gear and the pinion at 0.95 with `gear`, the button at 0.55-0.86
with `red circle` / `orange disc`, and the peg tips at 0.11-0.44 with `pin` --
which is why the pegs also come from the depth band, and why the report says
how often each source found them.
