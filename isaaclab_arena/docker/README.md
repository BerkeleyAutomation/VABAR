# Arena/GaP evaluator image

This directory builds one self-contained evaluator image for all maintained
GaP/Arena pairs at a selected source state of this folder, plus a small simulator-free
collector image. Arena still owns Experiment fan-out, scheduling, collection,
publication, and benchmark scoring.

The runtime interface is intentionally small:

```text
vabar-eval manifest
vabar-eval verify [--quick] [--require-immutable-base]
vabar-eval run --experiment EXPERIMENT.yaml --output-dir /outputs
```

OSMO invokes the same baked runtime through `isaaclab_arena_gap.container_runner`, which
adds `image-manifest.json` and `run-provenance.json` to every Arena Experiment
output. Arena's episode JSONL remains the authoritative benchmark result.

## Build

Use the simulator base selected by the pinned Isaac Lab Docker support, resolved
to an immutable registry digest for any published image:

```bash
./docker/build.sh \
  --base-image nvcr.io/nvidia/isaac-sim:6.1.0@sha256:<digest> \
  <registry>/arena-gap-evaluator:<tag>
```

For repeated builds, publish a separate BuildKit cache manifest. Stable pinned
environments, model weights, benchmark content, and frequently edited adapter
source are separate layers:

```bash
./docker/build.sh \
  --base-image nvcr.io/nvidia/isaac-sim:6.1.0@sha256:<digest> \
  --cache-image <registry>/arena-gap-evaluator:buildcache \
  --push \
  <registry>/arena-gap-evaluator:<tag>
```

The build uses the pinned Isaac Lab lock to construct the Python 3.12 Arena
environment, AUTOLab v1.0's locked environment for the RSS adapter, and the
independent frozen GaP/skill locks for Python 3.11. It
reads `GH_TOKEN` (or `gh auth token`) for the pinned private SAM3 source and
`HF_TOKEN` (or `hf auth token`) for gated model weights. Both are passed only as
BuildKit secrets. Model downloads are restricted to the files and revisions in
`runtime_dependencies.json` and verified by SHA-256; no token is stored in the
image.

For a local iteration, dirty source can be included explicitly:

```bash
./docker/build.sh \
  --allow-dirty \
  --base-image nvcr.io/nvidia/isaac-sim:6.1.0@sha256:<digest> \
  arena-gap-evaluator:dirty
```

The final runtime source tree includes tracked changes and non-ignored untracked
files under `isaaclab_arena_gap/`, `isaaclab_arena_vabar/`, and the root `pyproject.toml`.
`/opt/cap/provenance/dirty.patch` records those tracked changes, while
`image-manifest.json` records their Git status and hashes, the source
fingerprint, exact gitlinks, dependency-lock hashes, model hashes, and base
reference. Other repository files, ignored outputs, and local caches are not
image inputs. `.dockerignore` and the immutable external gitlinks must be clean.

The expensive environment layer is keyed only by `external/`, the bootstrap
contracts, and the two repository-owned dependency manifests. Other local
source is copied afterward and installed as an offline editable overlay, so
normal graph, adapter, benchmark, asset, documentation, or runtime-source
changes do not reconstruct the pinned CUDA/Python environments. Dependency or
lock changes still fail closed and invalidate the relevant layer.

A mutable base tag is available only for an explicitly local experiment with
`--allow-mutable-base`; `vabar-eval verify --require-immutable-base` rejects the
result.

Build and publish the collector independently. It copies only Arena's result
and HTML-report closure and contains no Isaac Sim, Isaac Lab, Torch, CUDA, model,
or policy payload:

```bash
./docker/build_collector.sh \
  --base-image python:3.12-slim@sha256:<digest> \
  --cache-image <registry>/arena-gap-collector:buildcache \
  --push \
  <registry>/arena-gap-collector:<tag>
```

## Verify and publish

```bash
docker run --rm --gpus all --network none \
  arena-gap-evaluator:dirty verify
docker push <registry>/arena-gap-evaluator:<tag>
docker inspect --format '{{index .RepoDigests 0}}' \
  <registry>/arena-gap-evaluator:<tag>
```

Submit only the resulting `image@sha256:...` reference. Credentials are injected
by OSMO and must not appear in the image, Experiment YAML, or source manifest.

## OSMO smoke

Render and inspect a one-episode workflow before submission:

```bash
uv run python -m isaaclab_arena_gap.osmo.submit_gap_experiment --dry_run \
  --experiment_cfg isaaclab_arena_gap/experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml \
  --collector_image <registry>/arena-gap-collector@sha256:<digest> \
  experiment_runner.image=<registry>/arena-gap-evaluator@sha256:<digest> \
  osmo.workflow_name=tool-sort-smoke \
  experiment_cfg.runs.vabar_tool_sorting_easy_1_gap.rollout_limit.num_episodes=1
```

Remove `--dry_run` only after checking the rendered workflow. Scalar jobs retain
`num_envs=1`. Fleet jobs use one independent GaP graph process per Arena slot
and may share policy-owned SAM3, Grounding DINO, and cuRobo owners, which serve one
heavy-tool call at a time.

By default, Arena fans Runs out to separate evaluator tasks. Add `--serial_runs`
to execute every Run in the Experiment in one evaluator task and pay the image pull
only once. The evaluator still starts each Run in its own Isaac process
(`isaaclab_arena_gap.isolated_runs`): a second Run in the same process can crash in
Warp's `wp_free_device_async`, so a process never
carries one Run's simulator state into the next, and a Run that dies is recorded as
failed while later Runs still run. This is best for small sensitivity sweeps when
reduced parallelism is a good trade. The collector understands both parallel
and serial runner outputs.

Enable persistent cache reuse with:

```bash
--cache_mode auto \
--cache_artifact_root_url "$CAP_CACHE_ARTIFACT_ROOT_URL" \
--cache_driver_branch 580
```

`CAP_CACHE_ARTIFACT_ROOT_URL` stands for your team's writable OSMO object-store prefix,
`swift://<host>/<account>/<container>/<path>`; the cache objects live below it.

`auto` checks the immutable manifest and archive objects with anonymous HTTPS
HEAD requests. A complete hit adds the persistent URL directly to every
evaluator input and emits no GPU cache-builder group. A miss prepends one
builder task using the first Run's representative scene, variations, cameras,
and an eight-step hold policy. The builder publishes through OSMO's persistent
Swift output transport; evaluators in that workflow consume its task output.
Use `--cache_mode rebuild` to refresh deliberately, or `off` for a cold control.

The task-neutral key contains manifest schema, configurable recipe epoch, Kit
build, OSMO resource platform, and the explicitly supplied GPU-driver branch.
The submitter never guesses the driver. At runtime, both builders and evaluators
compare `nvidia-smi`'s actual major branch with the configured value. Evaluators
also validate a completion marker, compatibility fields, archive size and
SHA-256 before extracting it locally. Missing, corrupt, incompatible, or
unreadable artifacts fail before Isaac starts rather than falling back cold.

The archive includes Kit, OV/XDG, CUDA, and GL shader caches. OSMO transfers one
archive and manifest; Isaac reads the extracted files from task-local `/cache`,
not from object storage or a remote small-file tree. The complete evaluator
image digest remains manifest provenance but is excluded from compatibility, so
source-only final-layer changes can reuse a cache. Bump `--cache_recipe_epoch`
when the warmup recipe or renderer compatibility changes. The final collector
continues to depend only on evaluator outputs.

Run at most one miss/rebuild builder for a compatibility key at a time. OSMO's
multi-object upload is not transactional, so concurrent builders can overwrite
the same remote prefix. Publishing the manifest last locally plus runtime size
and checksum validation prevents unsafe reuse, but a reader racing an upload can
fail and should be retried after the builder completes. Cache logs report
`archive_bytes` and `elapsed_seconds` separately from subsequent Kit startup.

The GaP observation monitor allows 180 seconds for the first timestamp to
advance because the first Arena step may perform deferred Newton CUDA-graph and
RTX initialization. It returns to the 60-second stale-stream timeout after that
first transition. Override these independently with
`CAP_STARTUP_OBSERVATION_TIMEOUT_S` and `CAP_STALE_OBSERVATION_TIMEOUT_S` in a
rendered evaluator task when collecting controlled diagnostics.


Root package installation uses a temporary UV cache outside `/cache`; derived
image builds must do the same. The final image build clears cache children and
checks write access as `cap`, matching restoration's non-root cleanup contract.
OSMO evaluator tasks publish `evaluator-full.log` after the runner exits so full
logs survive OSMO log-rate limits without making Arena's output directory nonempty
before startup.

## Startup performance

The collector image, persistent archive caches, `--serial_runs`, layered
asset/source images, registry BuildKit caches and payload hardlink
deduplication reduce evaluator startup. In a matched
fresh-pod syringe comparison (same evaluator digest, GPU type and driver branch),
a cache hit reduced AppLauncher initialization from 891.4 s to 19.9 s and the
whole workflow from 1,351.76 s to 325.10 s. Deduplication saved about 9.6 GB
unpacked and 10.8% of transferred evaluator bytes.
