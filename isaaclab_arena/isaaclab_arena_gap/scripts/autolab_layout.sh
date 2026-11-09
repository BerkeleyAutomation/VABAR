# AUTOLab v1.0 source and environment resolution shared by setup, preflight and the RSS launcher.
# shellcheck shell=bash

# Set AUTOLAB_VABAR_ROOT (AUTOLab's tasks, graphs and lock) and AUTOLAB_VABAR_VENV (the
# environment the RSS adapter runs in) unless the caller already set them. Two layouts resolve
# by default:
#
# - AUTOLab v1.0 pinned as this tree's external/gap/autolab-vabar gitlink: the sources are that
#   checkout and the environment is its own .venv;
# - this tree as a folder of the VABAR repository: the sources are the enclosing directory and
#   the environment is .venv-autolab here, which leaves the repository root's own .venv (with its
#   default dependency groups) alone.
#
# An explicit AUTOLAB_VABAR_ROOT elsewhere keeps its own .venv, as before.
resolve_autolab_layout() {
  local cap_root="$1" parent
  parent="$(cd "$cap_root/.." && pwd -P)"
  if [[ -z "${AUTOLAB_VABAR_ROOT:-}" ]]; then
    if [[ -e "$cap_root/external/gap/autolab-vabar" ]]; then
      AUTOLAB_VABAR_ROOT="$cap_root/external/gap/autolab-vabar"
    else
      AUTOLAB_VABAR_ROOT="$parent"
    fi
  fi
  if [[ -z "${AUTOLAB_VABAR_VENV:-}" ]]; then
    if [[ "$(cd "$AUTOLAB_VABAR_ROOT" 2>/dev/null && pwd -P)" == "$parent" ]]; then
      AUTOLAB_VABAR_VENV="$cap_root/.venv-autolab"
    else
      AUTOLAB_VABAR_VENV="$AUTOLAB_VABAR_ROOT/.venv"
    fi
  fi
}

# True when AUTOLAB_VABAR_ROOT holds the files setup and the launcher read.
autolab_sources_present() {
  [[ -f "$AUTOLAB_VABAR_ROOT/pyproject.toml" && -f "$AUTOLAB_VABAR_ROOT/uv.lock" \
    && -d "$AUTOLAB_VABAR_ROOT/tasks" && -d "$AUTOLAB_VABAR_ROOT/robot_skills" ]]
}
