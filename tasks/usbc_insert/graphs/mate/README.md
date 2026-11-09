# `usbc_insert/mate` — the insertion perception graph

The Easy and Medium tasks share this workflow and its registered GaP skill
bundles. Easy expects a fixed receptacle and one working arm. Medium expects a
free bulkhead braced by the second arm and a plug with a trailing cable.

The graph locates the connector pair, braces a free receptacle when needed,
positions the wrist for mouth inspection, grasps the plug, and reobserves the
held pair. It then aligns and pushes the connector, performs an axial reseating
pass, releases, and verifies insertion.

`workflow.json` and its `scripts/` directory contain the execution policy.
The perception helpers estimate the mouth and bore axis from camera data;
they do not resize the receptacle or generate collision geometry.

Both wrists use tool hanging's standard D405 optical pose relative to the
calibrated TCP, with a 58-degree vertical field of view and 320×240 RGB-D.
The empty-hand inspection round trip keeps the mate heading to avoid a wrist
joint-limit violation when the plug starts at a different angle.

The planner grasps 32 mm behind the shoulder and tilts the wrist 20 degrees
away from the port. Full fingertip and gripper housing collisions remain
enabled on both hands. Entry, preload, and axial retry share a full-hand
clearance bound of 3 mm from the measured mouth plane; release stays outside
that plane. Perception and servo errors can still affect actual clearance.

For Medium, the support hand lifts the barrel 30 mm clear of the bench. Fresh
RGB-D surface fits estimate the held socket's mouth and 3D axis, and the plug's
shoulder and attitude, using known local meshes without scene poses. Both
hands use the evaluation sliding-friction material of 8, finite contact-patch
torsional friction of 2 mm (`condim=4`), and a friction/normal impedance ratio
of 10. Retention comes from contact; no grasp weld or socket detent is added.

## Use the graph

See [the fixed cable files](../../../../benchmark/cable_tasks/README.md) for
fixed-file execution, [`docs/running_graphs.md`](../../../../docs/running_graphs.md)
for the evaluation protocol, and the [skill registry instructions](../../../../robot_skills/README.md)
for graph selection and skill discovery.

Vision and oracle bodies share perception interfaces. Select
`--perception vision --no-privileged` for RGB-D and robot proprioception with
privileged simulator tools withheld. Fixed sockets are localized from RGB-D.
Oracle bodies are available for diagnostics.

Grasping uses contact friction. The graph's insertion verdict is separate from
the evaluator's physical goal after release and settling. Use 600 settling steps
at the task's 60 Hz control rate for the insertion evaluation protocol.

The workflow declares skill owners, inputs, exits, and tool capabilities. Keep
its scripts and contracts synchronized with the corresponding exported bundles
in [robot_skills](../../../../robot_skills).
