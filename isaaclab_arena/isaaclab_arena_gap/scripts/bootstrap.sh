#!/usr/bin/env bash
set -euo pipefail

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"
GAP_ROOT="${GAP_ROOT:-$CAP_ROOT/external/gap/graph-as-policy}"
SKILLS_ROOT="${SKILLS_ROOT:-$CAP_ROOT/external/gap/open-robot-skills}"
ARENA_SOURCE_ROOT="${ARENA_SOURCE_ROOT:-$CAP_ROOT/external/IsaacLab-Arena}"
ARENA_ENV_ROOT="${ARENA_ENV_ROOT:-$CAP_ROOT}"
if [[ "$ARENA_ENV_ROOT" != "$CAP_ROOT" ]]; then
  printf 'ARENA_ENV_ROOT must be the workspace root: %s\n' "$CAP_ROOT" >&2
  printf 'The simulator environment is always materialized at %s/.venv.\n' \
    "$CAP_ROOT" >&2
  exit 2
fi
# shellcheck source=isaaclab_arena_gap/scripts/skill_bundles.sh
. "$ARENA_GAP_ROOT/scripts/skill_bundles.sh"
# shellcheck source=isaaclab_arena_gap/scripts/arena_environment_consistency.sh
. "$ARENA_GAP_ROOT/scripts/arena_environment_consistency.sh"
# shellcheck source=isaaclab_arena_gap/scripts/gitlinks.sh
. "$ARENA_GAP_ROOT/scripts/gitlinks.sh"
# shellcheck source=isaaclab_arena_gap/scripts/autolab_layout.sh
. "$ARENA_GAP_ROOT/scripts/autolab_layout.sh"
resolve_autolab_layout "$CAP_ROOT"

skip_skills=0
skip_preflight=0
offline=0
source_only=0
materialized_source=0
external_only=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [OPTIONS]

Initialize the pinned sources of this folder, create the root simulation environment,
and create the separate GaP environments.

Options:
  --offline         Require uv to use already-cached packages.
  --materialized-source
                    Trust an exported source tree without Git metadata. Verify
                    required manifests instead of initializing submodules.
  --external-only   Build only the pinned external GaP, skill, and Isaac Lab
                    environments. Skip repository-owned editable overlays and
                    preflight. Intended for the stable container dependency
                    layer.
  --source-only     Initialize and verify source pins, then stop.
  --skip-skills     Skip the five per-bundle GaP environments.
  --skip-preflight  Skip the final runtime/credential/GPU preflight.
  -h, --help        Show this help.

Environment:
  UV_HTTP_TIMEOUT   HTTP read timeout in seconds (default: 1800 for the
                    multi-gigabyte Isaac Sim extension wheels).
EOF
}

while (($#)); do
  case "$1" in
    --offline) offline=1 ;;
    --materialized-source) materialized_source=1 ;;
    --external-only) external_only=1; skip_preflight=1 ;;
    --source-only) source_only=1 ;;
    --skip-skills) skip_skills=1 ;;
    --skip-preflight) skip_preflight=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

required_commands=(uv)
if ((!materialized_source)); then
  required_commands+=(git)
fi
for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$command" >&2
    exit 1
  fi
done

if ! uv_series_supported; then
  printf 'uv %s.x is required; found %s at %s.\n' \
    "$ARENA_UV_SERIES" "${ARENA_UV_VERSION:-an unknown version}" "$(command -v uv)" >&2
  printf 'Setup checks the exact `uv pip check` output of uv %s, which later releases change.\n' \
    "$ARENA_UV_SERIES" >&2
  printf 'Install it with `uv self update 0.9.25`, or `uv tool install uv==0.9.25` and put\n' >&2
  printf 'the tool bin directory first on PATH.\n' >&2
  exit 1
fi

if ((offline)); then
  export UV_OFFLINE=1
fi

# Isaac Sim includes an extension-cache wheel larger than 5 GiB. uv's normal
# HTTP timeout is too short for a cold install on otherwise healthy links.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"

init_pinned_submodule() {
  local parent="$1" path="$2" name="$3" expected actual status
  expected="$(gitlink_pin "$parent" "$path")"
  if [[ -z "$expected" ]]; then
    printf 'Missing gitlink for %s (%s in %s)\n' "$name" "$path" "$parent" >&2
    exit 1
  fi

  # `git -C empty/child rev-parse` walks upward into the parent repository,
  # so only an exact .git file/directory proves the submodule is initialized.
  if [[ -e "$parent/$path/.git" ]]; then
    status="$(git -C "$parent/$path" status --porcelain)"
    if [[ -n "$status" ]]; then
      printf 'Refusing to update dirty %s checkout: %s\n' "$name" "$parent/$path" >&2
      exit 1
    fi
    actual="$(git -C "$parent/$path" rev-parse HEAD)"
    if [[ "$actual" != "$expected" ]]; then
      printf '%s is initialized at %s, expected %s.\n' "$name" "$actual" "$expected" >&2
      printf 'Move local work elsewhere, then run: git -C %q submodule update --checkout -- %q\n' \
        "$parent" "$path" >&2
      exit 1
    fi
  else
    git -C "$parent" submodule sync -- "$path"
    # Pinned upstream .gitmodules (Arena's Isaac Lab) use SSH URLs for public
    # repositories; fetch them over HTTPS so setup needs no GitHub SSH key.
    git -C "$parent" -c url.https://github.com/.insteadOf=git@github.com: \
      submodule update --init --checkout -- "$path"
  fi

  actual="$(git -C "$parent/$path" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Failed to initialize %s at %s (found %s)\n' \
      "$name" "$expected" "${actual:-nothing}" >&2
    exit 1
  fi
  printf 'Pinned %-34s %s\n' "$name" "$expected"
}

verify_materialized_source() {
  local required_paths=(
    "$ARENA_SOURCE_ROOT/pyproject.toml"
    "$ARENA_SOURCE_ROOT/submodules/IsaacLab/pyproject.toml"
    "$ARENA_SOURCE_ROOT/submodules/IsaacLab/uv.lock"
    "$GAP_ROOT/pyproject.toml"
    "$GAP_ROOT/uv.lock"
    "$SKILLS_ROOT/pyproject.toml"
    "$AUTOLAB_VABAR_ROOT/pyproject.toml"
    "$AUTOLAB_VABAR_ROOT/uv.lock"
    "$AUTOLAB_VABAR_ROOT/robot_skills/pyproject.toml"
    "$AUTOLAB_VABAR_ROOT/tasks/syringe_packing/graphs/gap_perception/workflow.json"
    "$AUTOLAB_VABAR_ROOT/tasks/cable_route/graphs/weave/workflow.json"
    "$GAP_ROOT/third_party/Variational-Automation-Benchmark/setup.py"
    "$GAP_ROOT/third_party/robosuite/pyproject.toml"
  )
  local bundle required_path
  for bundle in "${CAP_SKILL_BUNDLES[@]}"; do
    required_paths+=(
      "$SKILLS_ROOT/tools/$bundle/pyproject.toml"
      "$SKILLS_ROOT/tools/$bundle/uv.lock"
    )
  done
  for required_path in "${required_paths[@]}"; do
    if [[ ! -f "$required_path" ]]; then
      printf 'Materialized source is incomplete: %s\n' "$required_path" >&2
      exit 1
    fi
  done
  printf 'Materialized source tree verified (%d manifests).\n' "${#required_paths[@]}"
}

if ((materialized_source)); then
  verify_materialized_source
else
  init_pinned_submodule "$CAP_ROOT" "external/IsaacLab-Arena" "IsaacLab-Arena"
  init_pinned_submodule "$ARENA_SOURCE_ROOT" "submodules/IsaacLab" "Isaac Lab"
  init_pinned_submodule "$CAP_ROOT" "external/gap/graph-as-policy" "graph-as-policy"
  init_pinned_submodule "$CAP_ROOT" "external/gap/open-robot-skills" "open-robot-skills"
  if [[ -n "$(gitlink_pin "$CAP_ROOT" external/gap/autolab-vabar)" ]]; then
    init_pinned_submodule "$CAP_ROOT" "external/gap/autolab-vabar" "autolab-vabar"
  elif autolab_sources_present; then
    printf 'Using %-35s %s\n' "AUTOLab v1.0 sources" "$AUTOLAB_VABAR_ROOT"
  else
    printf 'AUTOLab v1.0 sources are missing at %s: pin them as the external/gap/autolab-vabar\n' \
      "$AUTOLAB_VABAR_ROOT" >&2
    printf 'gitlink, place this tree in a folder of AUTOLab'"'"'s repository, or set AUTOLAB_VABAR_ROOT.\n' >&2
    exit 1
  fi
  init_pinned_submodule "$GAP_ROOT" \
    "third_party/Variational-Automation-Benchmark" "GaP VAB"
  init_pinned_submodule "$GAP_ROOT" "third_party/robosuite" "GaP robosuite"
  # The scene assets are not in git; a materialized source tree already carries them.
  "$ARENA_GAP_ROOT/scripts/fetch_assets.sh"
fi

if ((source_only)); then
  printf '\nVABAR on Isaac Lab Arena source bootstrap complete.\n'
  exit 0
fi

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
fi

printf '\nSyncing the pinned GaP environment...\n'
# The pinned ORS checkout is an editable sibling path in GaP's lock. The
# source-only changes at the current ORS pin add no dependencies, so its
# metadata may differ without changing the resolved environment. Keep GaP's
# lock authoritative; a future ORS dependency change requires a compatible
# upstream GaP lock/pin.
(cd "$GAP_ROOT" && uv sync --frozen)

# AUTOLab v1.0's own environment (RoboSimStudio, its GaP pin, cuRobo) for
# isaaclab_arena_gap.rss_adapter, from AUTOLab's lock. Its RTX renderer and dev
# groups are left out: the adapter's robot model is never rendered.
printf '\nSyncing the pinned AUTOLab v1.0 environment...\n'
(cd "$AUTOLAB_VABAR_ROOT" \
  && UV_PROJECT_ENVIRONMENT="$AUTOLAB_VABAR_VENV" uv sync --frozen --no-default-groups)
# The adapter loads Arena's robot assets, which are USD; AUTOLab's own robots are
# MJCF, so its environment carries no USD importer. usd-core is the version
# AUTOLab's lock already resolves.
uv pip install --python "$AUTOLAB_VABAR_VENV/bin/python" \
  "usd-core==26.11" "newton-usd-schemas==0.6.0"

if ((!skip_skills)); then
  gap_python="$GAP_ROOT/.venv/bin/python"
  printf '\nSyncing the GaP skill environments with GaP Python...\n'
  UV_LOCKED=1 UV_PYTHON="$gap_python" \
    "$GAP_ROOT/.venv/bin/gap" skills install \
      --skills "$SKILLS_ROOT" \
      "${CAP_SKILL_BUNDLES[@]}"

  gap_python_version="$("$gap_python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  for bundle in "${CAP_SKILL_BUNDLES[@]}"; do
    bundle_python="$SKILLS_ROOT/tools/$bundle/.venv/bin/python"
    bundle_python_version="$("$bundle_python" -c \
      'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$bundle_python_version" != "$gap_python_version" ]]; then
      printf '%s uses Python %s; expected GaP Python %s\n' \
        "$bundle" "$bundle_python_version" "$gap_python_version" >&2
      exit 1
    fi
  done

  # The repository-owned VLM registry override reuses the exact pinned VLM
  # environment; only its tools.py compatibility seam differs.
  vlm_override_dir="$ARENA_GAP_ROOT/skill_overrides/tools/vlm"
  vlm_override_env="$vlm_override_dir/.venv"
  if [[ -d "$vlm_override_dir" ]]; then
    if [[ -L "$vlm_override_env" ]]; then
      ln -sfn "$SKILLS_ROOT/tools/vlm/.venv" "$vlm_override_env"
    elif [[ -e "$vlm_override_env" ]]; then
      printf 'Refusing to replace non-symlink VLM override environment: %s\n' \
        "$vlm_override_env" >&2
      exit 1
    else
      ln -s "$SKILLS_ROOT/tools/vlm/.venv" "$vlm_override_env"
    fi
  fi
fi

printf '\nSyncing the pinned Arena/Isaac Lab/Isaac Sim environment at %s/.venv...\n' \
  "$ARENA_ENV_ROOT"
(
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
  export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
  cd "$ARENA_SOURCE_ROOT"
  UV_PROJECT_ENVIRONMENT="$ARENA_ENV_ROOT/.venv" \
    uv sync --frozen
  if ((external_only)); then
    exit 0
  fi
  # Keep Arena's frozen lock authoritative for the simulator stack, then
  # install the framework and repository overlays with their own runtime dependencies.
  # This pip-style install does not activate Arena's old simulator groups.
  uv pip install --python "$ARENA_ENV_ROOT/.venv/bin/python" \
    "idna==3.10" \
    "Pillow>=10" \
    "pytest>=8" \
    --editable "$ARENA_SOURCE_ROOT" \
    --editable "$CAP_ROOT/isaaclab_arena_vabar" \
    --editable "$ARENA_GAP_ROOT"
  if check_arena_environment_consistency "$ARENA_ENV_ROOT/.venv/bin/python"; then
    :
  elif [[ "$?" == "$ARENA_PIP_CHECK_KNOWN_CONFLICTS" ]]; then
    printf 'Arena environment carries only the 5 pinned Arena source conflicts.\n'
  else
    printf 'Arena environment consistency check failed:\n%s\n' \
      "$ARENA_PIP_CHECK_OUTPUT" >&2
    exit 1
  fi
)

if ((!skip_preflight)); then
  printf '\nRunning VABAR on Isaac Lab Arena preflight...\n'
  GAP_ROOT="$GAP_ROOT" \
  SKILLS_ROOT="$SKILLS_ROOT" \
  AUTOLAB_VABAR_ROOT="$AUTOLAB_VABAR_ROOT" \
  AUTOLAB_VABAR_VENV="$AUTOLAB_VABAR_VENV" \
  ARENA_SOURCE_ROOT="$ARENA_SOURCE_ROOT" \
  ARENA_ENV_ROOT="$ARENA_ENV_ROOT" \
    "$ARENA_GAP_ROOT/scripts/check_setup.sh"
fi

printf '\nVABAR on Isaac Lab Arena bootstrap complete.\n'
printf 'GaP environment:   %s/.venv\n' "$GAP_ROOT"
printf 'Simulation environment: %s/.venv\n' "$ARENA_ENV_ROOT"
