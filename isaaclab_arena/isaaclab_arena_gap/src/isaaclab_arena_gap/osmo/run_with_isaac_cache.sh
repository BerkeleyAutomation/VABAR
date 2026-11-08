#!/usr/bin/env bash
# Restore or publish one verified Isaac cache artifact using task-local storage.
set -euo pipefail

cache_mode="${CAP_ISAAC_CACHE_MODE:-off}"
if [[ "$cache_mode" == off ]]; then
  exec "$@"
fi
if [[ "$cache_mode" != reuse && "$cache_mode" != builder ]]; then
  printf '[cap-cache] invalid runtime mode: %s\n' "$cache_mode" >&2
  exit 2
fi

: "${CAP_ISAAC_CACHE_ARTIFACT_DIR:?CAP_ISAAC_CACHE_ARTIFACT_DIR is required}"
: "${CAP_ISAAC_CACHE_RECIPE_EPOCH:?CAP_ISAAC_CACHE_RECIPE_EPOCH is required}"
: "${CAP_KIT_BUILD_ID:?CAP_KIT_BUILD_ID is required}"
: "${CAP_OSMO_PLATFORM:?CAP_OSMO_PLATFORM is required}"
: "${CAP_EXPECTED_DRIVER_BRANCH:?CAP_EXPECTED_DRIVER_BRANCH is required}"
: "${CAP_PYTHON:=/workspace/isaaclab_arena/.venv/bin/python}"
: "${CAP_IMAGE_DIGEST:=unknown}"

cache_args=(
  --artifact-directory "$CAP_ISAAC_CACHE_ARTIFACT_DIR"
  --local-cache /cache
  --recipe-epoch "$CAP_ISAAC_CACHE_RECIPE_EPOCH"
  --kit-build "$CAP_KIT_BUILD_ID"
  --platform "$CAP_OSMO_PLATFORM"
  --driver-branch "$CAP_EXPECTED_DRIVER_BRANCH"
)

mkdir -p /cache/{kit,data,logs,documents,xdg,ov} \
  /cache/nvidia/{ComputeCache,GLCache}

if [[ "$cache_mode" == reuse ]]; then
  "$CAP_PYTHON" -m isaaclab_arena_gap.osmo_cache restore "${cache_args[@]}"
  printf '[cap-cache] verified and restored artifact from %s\n' \
    "$CAP_ISAAC_CACHE_ARTIFACT_DIR"
  exec "$@"
fi

if "$@"; then
  command_status=0
else
  command_status=$?
  printf '[cap-cache] warmup failed; artifact will not be published\n' >&2
  exit "$command_status"
fi
"$CAP_PYTHON" -m isaaclab_arena_gap.osmo_cache publish "${cache_args[@]}" \
  --image-digest "$CAP_IMAGE_DIGEST"
printf '[cap-cache] verified artifact published to %s\n' \
  "$CAP_ISAAC_CACHE_ARTIFACT_DIR"
