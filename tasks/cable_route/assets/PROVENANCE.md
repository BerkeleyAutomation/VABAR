# `cable_fixtures/` — ManipulationNet's F1 round peg

**Source:** `ManipulationNet/mnet_client` @ `2745ccc6` (branch `ros_2`),
`assets/cable_management/cad_files/round_peg.stl`. A byte-identical copy. The
task it belongs to is
[Cable Routing](https://manipulation-net.org/tasks/cable_routing.html).

One file of the reference's twelve, vendored because one is what is used. The
other eleven — the four NIST ATB M2 board segments, the C/U/Y clips, the
base connector and its harness — belong to the staged board tiers, which this
corpus does not ship.

**Units: centimetres.** The mesh measures 2.5 × 2.5 × 2.35 and the reference
calls it a 25 mm fixture, so a loader applies ×0.01. Stated here because it is
the one fact about this file that a reader cannot get from the file.

## What it is, and why it is not a cylinder

**F1 is a spool, not a post.** Measured off the mesh:

| | | measured at |
| --- | --- | --- |
| Flange radius | 12.5 mm, top and bottom | z = 0 and z = 22.5 mm |
| Shaft radius | 4.75 mm | z = 2.5 … 19.68 mm |
| Winding groove | 17.2 mm tall | between those two |
| Overall height | 23.5 mm | |
| Axial bore | 4.5 mm diameter, for the board's 10 mm hole grid | |

The reference's own note calls the groove 18 mm; it measures 17.18. The measured
figure is the one used, here and in the loader, because the mesh is the thing
that gets built.

That profile is the whole reason to prefer it over the 13 mm procedural cylinder
`cable_peg` builds. A cable laid against a post is held there by nothing: it
rides up the post under tension and lifts off the top, which is the failure mode
a weave spends its retention budget on. A cable seated in a groove is captured
between two flanges and cannot.

It also moves a number the weave's goal depends on. A seated cable rests against
the **shaft**, not the flange, so it sits `4.75 + 5 = 9.75 mm` from the axis
rather than the 18 mm a 13 mm post would put it at. The seat regions in
`tasks/cable_route/*.task` are stated against that.

**Nothing was modified.**
