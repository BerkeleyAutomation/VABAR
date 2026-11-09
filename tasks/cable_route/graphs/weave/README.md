# `cable_route/weave` — the routing perception graph

The Easy and Medium tasks share this workflow and its registered GaP skill
bundles. The simulator environment supplies these task configurations:

| Case | Pegs | Cable | Starting end | Finishing end |
| --- | --- | --- | --- | --- |
| `cable_route/port_easy` | 2 | One, 600 mm | Physically fixed at the anchor | Free endpoint into the port |
| `cable_route/port_medium` | 4 | One, 720 mm | Physically fixed at the anchor | Free endpoint into the port |

The graph observes the fixtures and cable, sequences the crossings, computes
waypoints, grasps and moves the cable, and checks seating at each station.
It includes support, terminal insertion, displaced-seat recovery, release, and
visual verification stages. The crossing loop follows the observed stations.
`anchor_sg` recognizes the fixed start and skips the initial support-hand grasp
for these two tasks.

`workflow.json` and its `scripts/` directory contain the execution policy and
its perception and motion helpers.

## Use the graph

See [the fixed cable files](../../../../benchmark/cable_tasks/README.md) for
fixed-file execution, [`docs/running_graphs.md`](../../../../docs/running_graphs.md)
for the evaluation protocol, and the [skill registry instructions](../../../../robot_skills/README.md)
for graph selection and skill discovery.

Select `--perception vision --no-privileged` for RGB-D and robot proprioception
with privileged simulator tools withheld. Grasping uses contact friction.
The free endpoint's containment in the port and the routed cable's seat regions
are part of the task's physical goal. Graph exits and visual verification are
separate from that goal after release and settling. Use 60 settling steps at
the task's 60 Hz control rate for the routing evaluation protocol.

The workflow declares skill owners, inputs, exits, and tool capabilities. Keep
its scripts and contracts synchronized with the corresponding exported bundles
in [robot_skills](../../../../robot_skills).
