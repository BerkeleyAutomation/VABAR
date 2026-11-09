#!/usr/bin/env bash
set -euo pipefail

CAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_GAP_ROOT="$CAP_ROOT/isaaclab_arena_gap"
BASE_IMAGE="${CAP_BASE_IMAGE:-}"
ALLOW_DIRTY=0
ALLOW_MUTABLE_BASE=0
CACHE_IMAGE=""
PUSH=0

usage() {
  cat <<'EOF'
Usage: docker/build.sh [OPTIONS] IMAGE [DOCKER_BUILD_ARG ...]

Build the exact-pin Arena/GaP evaluator image. Dirty superproject changes are
accepted only with --allow-dirty and are copied into the image with an
auditable patch and per-file hashes. Pinned external gitlinks must stay clean.

Options:
  --base-image REF       Simulator-supported base image. Publication builds
                         must use NAME@sha256:<64 hex> (or set CAP_BASE_IMAGE).
  --allow-dirty          Include tracked and untracked local source changes.
  --allow-mutable-base   Permit a mutable base tag for local development only.
  --cache-image REF      Import/export a registry BuildKit cache (mode=max).
  --push                 Push the result; default is to load it locally.
  -h, --help             Show this help.

Credentials:
  GH_TOKEN               GitHub token that can read the pinned SAM3 source.
                         Defaults to `gh auth token` when available.
  HF_TOKEN               Hugging Face token that can read gated SAM3 weights.
                         Defaults to `hf auth token` when available.
EOF
}

while (($#)); do
  case "$1" in
    --base-image)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      BASE_IMAGE="$2"
      shift 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --allow-mutable-base)
      ALLOW_MUTABLE_BASE=1
      shift
      ;;
    --cache-image)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      CACHE_IMAGE="$2"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
IMAGE="$1"
shift
if [[ -z "$BASE_IMAGE" ]]; then
  printf 'Pass --base-image NAME@sha256:<digest> or set CAP_BASE_IMAGE.\n' >&2
  exit 2
fi
if [[ ! "$BASE_IMAGE" =~ @sha256:[0-9a-f]{64}$ && "$ALLOW_MUTABLE_BASE" != 1 ]]; then
  printf 'Base image must be immutable: %s\n' "$BASE_IMAGE" >&2
  printf 'Use --allow-mutable-base only for an explicitly local development build.\n' >&2
  exit 2
fi
if [[ -n "$(git -C "$CAP_ROOT" status --porcelain=v1 --untracked-files=all -- .dockerignore)" ]]; then
  printf '.dockerignore controls the evaluator build context and must be clean.\n' >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]] && command -v hf >/dev/null 2>&1; then
  HF_TOKEN="$(hf auth token 2>/dev/null || true)"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  printf 'Log in with `hf auth login` or export HF_TOKEN for gated SAM3.\n' >&2
  printf 'BuildKit receives it as a secret; it is not stored in the image.\n' >&2
  exit 1
fi
export HF_TOKEN

if [[ -z "${GH_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
  GH_TOKEN="$(gh auth token 2>/dev/null || true)"
fi
if [[ -z "${GH_TOKEN:-}" ]]; then
  printf 'Run `gh auth login` or export GH_TOKEN with read access to the pinned SAM3 source.\n' >&2
  printf 'BuildKit receives it as a secret; it is not stored in the image.\n' >&2
  exit 1
fi
export GH_TOKEN

"$ARENA_GAP_ROOT/scripts/bootstrap.sh" --source-only

CAP_BUILD_METADATA_DIR="$(mktemp -d)"
CAP_AUTOLAB_CONTEXT="$(mktemp -d)"
cleanup() {
  rm -rf -- "$CAP_BUILD_METADATA_DIR" "$CAP_AUTOLAB_CONTEXT"
}
trap cleanup EXIT

# AUTOLab's sources enter the image as the named `autolab` build context, exported from their
# committed revision: the external/gap/autolab-vabar gitlink's checkout (whose cleanliness and
# pin source_manifest.py verifies), or, when this tree is a folder of the VABAR repository, that
# repository's HEAD less this folder.
# shellcheck source=isaaclab_arena_gap/scripts/autolab_layout.sh
. "$ARENA_GAP_ROOT/scripts/autolab_layout.sh"
resolve_autolab_layout "$CAP_ROOT"
autolab_top="$(git -C "$AUTOLAB_VABAR_ROOT" rev-parse --show-toplevel)"
if [[ "$(cd "$autolab_top" && pwd -P)" != "$(cd "$AUTOLAB_VABAR_ROOT" && pwd -P)" ]]; then
  printf 'AUTOLAB_VABAR_ROOT must be the root of AUTOLab'"'"'s repository: %s\n' \
    "$AUTOLAB_VABAR_ROOT" >&2
  exit 1
fi
autolab_pathspec=(.)
if [[ "$(git -C "$CAP_ROOT" rev-parse --show-toplevel)" == "$autolab_top" ]]; then
  cap_prefix="$(git -C "$CAP_ROOT" rev-parse --show-prefix)"
  autolab_pathspec+=(":(exclude)${cap_prefix%/}")
  if [[ -n "$(git -C "$autolab_top" status --porcelain -- "${autolab_pathspec[@]}")" ]]; then
    printf 'Note: uncommitted AUTOLab changes are not built; the image takes its HEAD.\n' >&2
  fi
fi
git -C "$autolab_top" archive --format=tar HEAD -- "${autolab_pathspec[@]}" \
  | tar -x -C "$CAP_AUTOLAB_CONTEXT"

python3 "$CAP_ROOT/docker/source_manifest.py" create \
  --source-root "$CAP_ROOT" \
  --output-dir "$CAP_BUILD_METADATA_DIR" \
  --runtime-path isaaclab_arena_gap \
  --runtime-path isaaclab_arena_vabar \
  --runtime-path pyproject.toml \
  --runtime-path external

readarray -t source_fields < <(python3 - "$CAP_BUILD_METADATA_DIR/source-manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if manifest["dirty"] else "0")
print(manifest["revision"])
print(manifest["source_fingerprint"])
PY
)
if [[ "${source_fields[0]}" == 1 && "$ALLOW_DIRTY" != 1 ]]; then
  printf 'Source tree is dirty; rerun with --allow-dirty to include and record it.\n' >&2
  git -C "$CAP_ROOT" status --short >&2
  exit 1
fi

build_args=(
  --progress=plain
  --file "$CAP_ROOT/docker/Dockerfile"
  --tag "$IMAGE"
  --build-context "source-metadata=$CAP_BUILD_METADATA_DIR"
  --build-context "autolab=$CAP_AUTOLAB_CONTEXT"
  --secret id=hf_token,env=HF_TOKEN
  --secret id=github_token,env=GH_TOKEN
  --build-arg "BASE_IMAGE=$BASE_IMAGE"
  --build-arg "CAP_SOURCE_REVISION=${source_fields[1]}"
  --build-arg "CAP_SOURCE_FINGERPRINT=${source_fields[2]}"
  --build-arg "CAP_SOURCE_DIRTY=${source_fields[0]}"
)
if [[ -n "$CACHE_IMAGE" ]]; then
  build_args+=(
    --cache-from "type=registry,ref=$CACHE_IMAGE"
    --cache-to "type=registry,ref=$CACHE_IMAGE,mode=max"
  )
fi
if ((PUSH)); then build_args+=(--push); else build_args+=(--load); fi

docker buildx build "${build_args[@]}" "$@" "$CAP_ROOT"
