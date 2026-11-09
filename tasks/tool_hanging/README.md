# `tool_hanging` — five pegboard documents

Ported from the tool-hanging development branch (3e20fc7), where the suite was
a Python `TaskSpec` file whose evaluator returned failure unconditionally — a
scaffold. The port kept what was measured (the cell, every holder transform,
the tools' spawn poses, the CoACD hull sets and their manifests, the four
IK-table routines) and replaced what was missing:

- **Goals.** Each document states what "hung" means — `threaded` over declared
  loop/rod features for the wrenches, scissors and screwdrivers; a support-tip
  region for the pliers, which have no closed loop; borne load and rest
  everywhere (`contact-force` is what makes *let go* part of the task: a tool
  held around its hook by the gripper reads ~0 N). The features live in
  `tasks/tool_hanging/assets/records.py`, measured off the meshes, and the
  routines steer by the same constants.
- **Arrangements.** The branch's per-variant physics blocks became the two
  registered bundles in `tasks/tool_hanging/staging.py` — the CoACD scenes
  use MuJoCo for integration and Newton for contact collision; the hammers run
  the plain rates. Every tool and every holder in the four scripted variants
  uses a convex compound collider. The camera rig frames the one wall every
  variant shares.
- **Compound colliders.** The branch's `AssetSpec.collision_path(s)` mechanism
  landed in the toolkit: the authored mesh stays visible and carries
  mass/inertia, the hull set carries collision, invisibly. Scissor 1 keeps its
  A/B knob (`SCISSOR_1_COLLISION_METHOD`) between the automatic CoACD set and
  the hand-authored 30-piece compound proxy.

`hammers` ships no routine. The other four routines are merged into one
`routine.py`; the wrench policy uses the current meshes' measured ring centres
and corrects its hook waypoints at the ring itself.

Measured 2026-08-10, `--solve routine --episodes 0:5` on an RTX PRO 6000
(the runtime's thread timing is not pinned, so pooled rates, never one batch):

| variant | scripted | note |
| --- | --- | --- |
| `pliers` | **5/5** | mean 1889 steps |
| `scissors` | 4/5 | mean 1613 steps |
| `screwdrivers` | 3/5 | mean 1699 steps |
| `wrenches` | 1/5 | the 19 hangs reliably; the 17's jaws close ~1 mm above its thinner shaft and the wrench stays on the table — the suite's one open routine item. The goal is proven satisfiable: the 17 planted at the 19's hook relation settles hung and scores. |
| `hammers` | — | no routine, deliberately; the holder's slots are side mouths under a solid top, so the engagement itself is the open item. |

## The graph

`wrenches`, `scissors` and `screwdrivers` ship a ground-truth graph —
`graphs/gt`, one workflow for all three, declared in each document's
`(:artifacts)` as `(:graph gt "graphs/gt")` over
`tasks/tool_hanging/tools.py`. It is a per-job cursor loop: the goal's
`threaded` clauses *are* the job list, and `release` routes back to the cursor,
so nothing about the two tools is unrolled. Read `graphs/gt/README.md` for the
shape and for the measurements behind its two constants.

Measured 2026-08-13, `--episodes 0:10` on one RTX PRO 6000, same layout every
episode (`EpisodeSpec(count=1)`), so this is run-to-run variance and not
generalisation:

| variant | graph `gt` | reward mean/median | routine, same command | note |
| --- | --- | --- | --- | --- |
| `wrenches` | **10/10** | **1.00** / 1.0 | 2/10, 0.45 / 0.5 | both spanners, every episode. The 17 clears its `threaded` gate by 0.02–0.10 mm — read the graph README before trusting that margin. |
| `scissors` | 8/10 | 0.90 / 1.0 | 8/10 (prior campaign) | level; the two misses are one bow of two. |
| `screwdrivers` | 6/10 | 0.80 / 1.0 | 4/10, 0.70 / 0.5 | level at this sample size. The failures are one hole of two, seated while held and out again after the release. |

The graph beats the routine on the two things a routine cannot do: it re-plans
the hang from the tool's *measured* pose in the jaws rather than from the pose
it saw before closing them, and it checks whether it is holding anything before
carrying an empty hand to a hook. It also stops driving the fingertips into the
table at the grasp, which is what made the spanners a lottery.

`pliers` and `hammers` have no graph. The plier routine is genuinely bimanual —
it streams both arms at once, which needs a generic synchronised-execute the
toolkit does not have — and the hammers ship no routine to port.

Still open, carried honestly from the branch: mesh licences are pending
confirmation (`tasks/tool_hanging/assets/PROVENANCE.md`), the hammer
engagement above, and the goal thresholds are authored floors calibrated
against the shipped routines — tighten them from larger campaigns, not by eye.
The 17 mm spanner's `threaded` budget is 6.71 mm against a natural hang that
sits at 6.6–6.9, which is a threshold worth revisiting rather than a policy
problem.

```bash
uv run python tools/bench_smoke.py --task tool_hanging/wrenches
uv run python -m vabar run --task tool_hanging/wrenches --solve routine --episodes 0:1
uv run python -m vabar run --task tool_hanging/wrenches --solve graph --graph gt --episodes 0:10
```
