# Arena Experiment Definitions

These typed YAML files are the runnable evaluation interface for both local
execution and OSMO. They are grouped by task family without an additional
`industrial` directory.

Names follow `<environment>_<policy>_<kind>.yaml`:

- `*_gap_baseline.yaml` evaluates the standard GaP pairing.
- `*_gap_published_layouts.yaml` evaluates the baseline over a published set of
  layout seeds, one Run per seed.

Every Experiment runs AUTOLab v1.0's graph, robot-only (`--no-privileged`),
through RoboSimStudio's own policy agent (`isaaclab_arena_gap.rss_adapter`):

- `gap_launcher: isaaclab_arena_gap/scripts/run_rss_graph.sh` (also the default
  when a config omits it; there is no other GaP runtime);
- `gap_graph` is a graph bound in `rss_adapter/runner.py` `BINDINGS`, and
  `robot_profile` is that binding's profile;
- `gap_inputs.autolab_task` names the AUTOLab document where the level or
  variant differs from the binding's default. It must be a v1.0 benchmark
  instance of the graph's suite, or a recorded exception in
  `tests/test_rss_adapter.py` (`NON_V1_DOCUMENTS`). The document also resolves
  the graph: an L2 tool-hanging document's `gap_perception` is AUTOLab's
  `graphs/gap_perception_level2`;
- `gap_adaptation`, where present, is an `rss_*` patch for a scene difference
  Arena keeps on purpose (`graph_adaptations/README.md`).

Each baseline file contains exactly one named Run.

Keep one canonical baseline per environment/policy pairing. Set evaluation
budgets and distinct, explicit seeds through the shared Arena Experiment/OSMO
execution interface. Separate committed experiments describe meaningful policy
or task variations, rather than repeated evaluations of the same baseline.
Arena's per-run episode JSONL is authoritative for benchmark scores.
