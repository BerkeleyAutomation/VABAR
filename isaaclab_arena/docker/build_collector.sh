#!/usr/bin/env bash
set -euo pipefail

CAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_GAP_ROOT="$CAP_ROOT/isaaclab_arena_gap"
COLLECTOR_BASE_IMAGE="${CAP_COLLECTOR_BASE_IMAGE:-}"
CACHE_IMAGE=""
PUSH=0

usage() {
  cat <<'EOF'
Usage: docker/build_collector.sh [OPTIONS] IMAGE [BUILDX_ARG ...]

Options:
  --base-image REF   Immutable Python 3.12 slim base image.
  --cache-image REF  Import and export a registry-backed BuildKit cache.
  --push             Push the built image instead of loading it locally.
EOF
}

while (($#)); do
  case "$1" in
    --base-image) COLLECTOR_BASE_IMAGE="$2"; shift 2 ;;
    --cache-image) CACHE_IMAGE="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    *) break ;;
  esac
done

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
IMAGE="$1"
shift
if [[ ! "$COLLECTOR_BASE_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  printf 'Collector base image must be immutable: %s\n' "$COLLECTOR_BASE_IMAGE" >&2
  exit 2
fi

collector_source_paths=(
  .dockerignore
  docker/Dockerfile.collector
  docker/build_collector.sh
  isaaclab_arena_gap/src/isaaclab_arena_gap/osmo/build_experiment_output.py
)
collector_status="$(
  git -C "$CAP_ROOT" status --porcelain=v1 --untracked-files=all -- \
    "${collector_source_paths[@]}" external/IsaacLab-Arena
)"
if [[ -n "$collector_status" ]]; then
  printf 'Collector build inputs must be clean:\n%s\n' "$collector_status" >&2
  exit 1
fi

expected_arena="$(git -C "$CAP_ROOT" rev-parse HEAD:external/IsaacLab-Arena)"
actual_arena="$(git -C "$CAP_ROOT/external/IsaacLab-Arena" rev-parse HEAD)"
[[ "$actual_arena" == "$expected_arena" ]] || {
  printf 'Arena gitlink is at %s, expected %s\n' "$actual_arena" "$expected_arena" >&2
  exit 1
}
[[ -z "$(git -C "$CAP_ROOT/external/IsaacLab-Arena" status --porcelain --untracked-files=all)" ]] || {
  printf 'Arena gitlink must be clean.\n' >&2
  exit 1
}

collector_source_fingerprint="$(
  {
    printf 'arena=%s\n' "$actual_arena"
    for relative in "${collector_source_paths[@]}"; do
      printf '%s=' "$relative"
      git -C "$CAP_ROOT" hash-object -- "$relative"
    done
  } | sha256sum | awk '{print $1}'
)"

build_args=(
  --progress=plain
  --file "$CAP_ROOT/docker/Dockerfile.collector"
  --tag "$IMAGE"
  --build-arg "COLLECTOR_BASE_IMAGE=$COLLECTOR_BASE_IMAGE"
  --build-arg "CAP_SOURCE_REVISION=$(git -C "$CAP_ROOT" rev-parse HEAD)"
  --build-arg "CAP_SOURCE_FINGERPRINT=$collector_source_fingerprint"
  --build-arg "ARENA_SOURCE_REVISION=$actual_arena"
)
if [[ -n "$CACHE_IMAGE" ]]; then
  build_args+=(
    --cache-from "type=registry,ref=$CACHE_IMAGE"
    --cache-to "type=registry,ref=$CACHE_IMAGE,mode=max"
  )
fi
if ((PUSH)); then build_args+=(--push); else build_args+=(--load); fi
docker buildx build "${build_args[@]}" "$@" "$CAP_ROOT"
