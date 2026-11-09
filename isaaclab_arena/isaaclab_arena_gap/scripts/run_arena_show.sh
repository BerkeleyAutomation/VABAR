#!/usr/bin/env bash
# Open a scene in the Isaac Sim viewer with no policy attached.
#
# Unlike run_arena_eval.sh this runs no GaP process, records no episode, and
# applies no episode budget: it builds the scene, steps it with zero actions and
# holds it open so the environment can be inspected interactively.
set -euo pipefail

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"
ARENA_SOURCE_ROOT="${ARENA_SOURCE_ROOT:-$CAP_ROOT/external/IsaacLab-Arena}"
ARENA_ENV_ROOT="${ARENA_ENV_ROOT:-$CAP_ROOT}"
ARENA_PYTHON="${ARENA_PYTHON:-$ARENA_ENV_ROOT/.venv/bin/python}"
ARENA_SHOW_SCENE="${ARENA_SHOW_SCENE:-$CAP_ROOT/isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/tool_hanging/wrench_easy.yaml}"

# A registered environment is selected by name, not by scene file, so skip the
# scene entirely when the caller passes --env.
use_env=0
for arg in "$@"; do
  if [[ "$arg" == "--env" || "$arg" == --env=* ]]; then
    use_env=1
    break
  fi
done

required_paths=("$ARENA_SOURCE_ROOT" "$ARENA_PYTHON")
if ((!use_env)); then
  required_paths+=("$ARENA_SHOW_SCENE")
fi

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 1
  fi
done

ARENA_SOURCE_ROOT="$(realpath "$ARENA_SOURCE_ROOT")"
if ((!use_env)); then
  ARENA_SHOW_SCENE="$(realpath "$ARENA_SHOW_SCENE")"
fi
ARENA_PYTHON_DIR="$(cd "$(dirname "$ARENA_PYTHON")" && pwd -P)"
ARENA_PYTHON="$ARENA_PYTHON_DIR/$(basename "$ARENA_PYTHON")"
unset ARENA_PYTHON_DIR

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ARENA_SOURCE_ROOT:$ARENA_GAP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

printf 'Arena source: %s\n' "$ARENA_SOURCE_ROOT"
if ((use_env)); then
  printf 'Scene:        (registered environment selected with --env)\n'
else
  printf 'Scene:        %s\n' "$ARENA_SHOW_SCENE"
fi
printf 'Policy:       none (zero actions, runs until interrupted)\n'

cd "$CAP_ROOT"
show_args=(-m isaaclab_arena_gap.arena_show)
if ((!use_env)); then
  show_args+=(--scene "$ARENA_SHOW_SCENE")
fi
if [[ "${ARENA_HEADLESS:-0}" != "0" ]]; then
  show_args+=(--headless)
fi
exec "$ARENA_PYTHON" "${show_args[@]}" "$@"
