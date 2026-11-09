#!/usr/bin/env bash
set -euo pipefail

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"
ARENA_SOURCE_ROOT="${ARENA_SOURCE_ROOT:-$CAP_ROOT/external/IsaacLab-Arena}"
ARENA_ENV_ROOT="${ARENA_ENV_ROOT:-$CAP_ROOT}"
ARENA_PYTHON="${ARENA_PYTHON:-$ARENA_ENV_ROOT/.venv/bin/python}"
ARENA_EXPERIMENT_CONFIG="${ARENA_EXPERIMENT_CONFIG:-$ARENA_GAP_ROOT/experiment_configs/tool_hanging/vabar_tool_hanging_wrench_easy_gap_baseline.yaml}"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
ARENA_OUTPUT_DIR="${ARENA_OUTPUT_DIR:-$CAP_ROOT/outputs/arena/$RUN_ID}"

for path in "$ARENA_SOURCE_ROOT" "$ARENA_EXPERIMENT_CONFIG" "$ARENA_PYTHON"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 1
  fi
done

ARENA_SOURCE_ROOT="$(realpath "$ARENA_SOURCE_ROOT")"
ARENA_EXPERIMENT_CONFIG="$(realpath "$ARENA_EXPERIMENT_CONFIG")"
ARENA_PYTHON_DIR="$(cd "$(dirname "$ARENA_PYTHON")" && pwd -P)"
ARENA_PYTHON="$ARENA_PYTHON_DIR/$(basename "$ARENA_PYTHON")"
unset ARENA_PYTHON_DIR

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ARENA_SOURCE_ROOT:$ARENA_GAP_ROOT/src:$CAP_ROOT/isaaclab_arena_vabar/src${PYTHONPATH:+:$PYTHONPATH}"

runtime_import_check='import importlib.metadata; '
runtime_import_check+='import isaaclab_arena_vabar.registration; '
runtime_import_check+='import isaaclab_arena_gap.gap_remote_policy; '
runtime_import_check+='importlib.metadata.version("isaaclab-arena-vabar"); '
runtime_import_check+='importlib.metadata.version("isaaclab-arena-gap")'
if ! import_error="$("$ARENA_PYTHON" -c "$runtime_import_check" 2>&1)"; then
  printf 'Arena Python cannot import the isaaclab_arena_gap and isaaclab_arena_vabar runtime packages:\n%s\n' \
    "$import_error" >&2
  printf 'Install them after Arena uv sync:\n' >&2
  printf '  uv pip install --python %q --editable %q --editable %q\n' \
    "$ARENA_PYTHON" "$CAP_ROOT/isaaclab_arena_vabar" "$ARENA_GAP_ROOT" >&2
  exit 1
fi

mkdir -p "$ARENA_OUTPUT_DIR"
ARENA_OUTPUT_DIR="$(cd "$ARENA_OUTPUT_DIR" && pwd)"
export ARENA_OUTPUT_DIR
printf 'Arena source: %s\n' "$ARENA_SOURCE_ROOT"
printf 'Arena Python: %s\n' "$ARENA_PYTHON"
printf 'Experiment:   %s\n' "$ARENA_EXPERIMENT_CONFIG"
printf 'Output:       %s\n' "$ARENA_OUTPUT_DIR"
printf 'Managed jobs (auto_spawn=true) start one GaP process automatically.\n'

cd "$CAP_ROOT"
runner_args=(
  -m isaaclab_arena_gap.arena_runner
  --experiment_config "$ARENA_EXPERIMENT_CONFIG"
  --output_base_dir "$ARENA_OUTPUT_DIR"
)
if [[ "${ARENA_RECORD_CAMERA_VIDEO:-1}" == "1" ]]; then
  runner_args+=(--record_camera_video)
fi
exec "$ARENA_PYTHON" "${runner_args[@]}" "$@"
