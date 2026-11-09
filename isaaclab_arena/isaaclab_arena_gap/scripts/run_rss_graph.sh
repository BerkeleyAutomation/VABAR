#!/usr/bin/env bash
# GapRemotePolicy's gap_launcher for AUTOLab graphs run through RoboSimStudio's
# own policy agent (isaaclab_arena_gap.rss_adapter). Runs in AUTOLab v1.0's
# locked environment; Arena sets GAP_PORT, GAP_GRAPH and the CAP_GAP_* variables.
set -euo pipefail

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"
# shellcheck source=isaaclab_arena_gap/scripts/autolab_layout.sh
. "$ARENA_GAP_ROOT/scripts/autolab_layout.sh"
resolve_autolab_layout "$CAP_ROOT"
SKILLS_ROOT="${SKILLS_ROOT:-$CAP_ROOT/external/gap/open-robot-skills}"
PYTHON="$AUTOLAB_VABAR_VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'AUTOLab v1.0 environment missing: %s (run ./isaaclab_arena_gap/scripts/bootstrap.sh)\n' "$PYTHON" >&2
  exit 1
fi
if [[ -z "${GAP_TRACE_DIR:-}" ]]; then
  if [[ -n "${ARENA_OUTPUT_DIR:-}" ]]; then
    GAP_TRACE_DIR="$ARENA_OUTPUT_DIR/gap-trace/port-${GAP_PORT:-standalone}-pid-$$"
  else
    GAP_TRACE_DIR="$CAP_ROOT/outputs/gap/run_$(date +%Y-%m-%d_%H-%M-%S)"
  fi
fi
export GAP_TRACE_DIR
GAP_VLM_ENV="${GAP_VLM_ENV:-$HOME/.config/gap/vlm.env}"
. "$ARENA_GAP_ROOT/scripts/vlm_env.sh"
# The repository's VLM bundle override (provider-qualified model IDs) first, then AUTOLab's
# own skill bundles, then the generic registry, as AUTOLab's pyproject orders those two.
export GAP_SKILLS_PATH="$ARENA_GAP_ROOT/skill_overrides:$AUTOLAB_VABAR_ROOT/robot_skills:$SKILLS_ROOT"
export PYTHONPATH="$ARENA_GAP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'RSS adapter: %s via %s, trace %s\n' "${GAP_GRAPH:-}" "$PYTHON" "$GAP_TRACE_DIR"

cd "$AUTOLAB_VABAR_ROOT"
exec "$PYTHON" -u -m isaaclab_arena_gap.rss_adapter.runner "$@"
