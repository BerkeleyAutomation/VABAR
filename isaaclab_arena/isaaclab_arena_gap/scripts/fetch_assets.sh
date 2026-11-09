#!/usr/bin/env bash
# Fetch the Arena scene assets -- the SimReady USD closures and collision hulls under
# isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/vabar__*/ -- from Hugging Face.
# They are not tracked in git. setup.sh calls this, and so does docker/build.sh through
# `bootstrap.sh --source-only`. Re-running it downloads only what is missing or changed.
set -euo pipefail

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"

ASSETS_REPO="keplerccc/VABAR-arena-assets"
ASSETS_REVISION="7ee8d59e2a809be6617b3a3133faf825fcbccbf5"
ASSETS_DIR="$CAP_ROOT/isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets"
HF_HUB_VERSION="2.3.0"

printf 'Fetching the Arena scene assets from huggingface.co/%s@%s (about 5 GB the first time)...\n' \
  "$ASSETS_REPO" "${ASSETS_REVISION:0:7}"
if [[ -n "${UV_OFFLINE:-}" ]]; then
  # Offline setup: no download, but every pinned file must already be here.
  export HF_HUB_OFFLINE=1
fi
uvx --from "huggingface_hub==$HF_HUB_VERSION" hf download "$ASSETS_REPO" \
  --revision "$ASSETS_REVISION" --include "vabar__*" --local-dir "$ASSETS_DIR" >/dev/null
printf 'Arena scene assets ready in %s\n' "$ASSETS_DIR"
