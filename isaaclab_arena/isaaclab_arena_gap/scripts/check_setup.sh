#!/usr/bin/env bash
# Validate a prepared checkout without modifying it.
set -u

ARENA_GAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAP_ROOT="$(cd "$ARENA_GAP_ROOT/.." && pwd)"
GAP_ROOT="${GAP_ROOT:-$CAP_ROOT/external/gap/graph-as-policy}"
SKILLS_ROOT="${SKILLS_ROOT:-$CAP_ROOT/external/gap/open-robot-skills}"
ARENA_SOURCE_ROOT="${ARENA_SOURCE_ROOT:-$CAP_ROOT/external/IsaacLab-Arena}"
ARENA_ENV_ROOT="${ARENA_ENV_ROOT:-$CAP_ROOT}"
ARENA_PYTHON="${ARENA_PYTHON:-$ARENA_ENV_ROOT/.venv/bin/python}"
GAP_VLM_ENV="${GAP_VLM_ENV:-$HOME/.config/gap/vlm.env}"
# shellcheck source=isaaclab_arena_gap/scripts/skill_bundles.sh
. "$ARENA_GAP_ROOT/scripts/skill_bundles.sh"
# shellcheck source=isaaclab_arena_gap/scripts/arena_environment_consistency.sh
. "$ARENA_GAP_ROOT/scripts/arena_environment_consistency.sh"
# shellcheck source=isaaclab_arena_gap/scripts/gitlinks.sh
. "$ARENA_GAP_ROOT/scripts/gitlinks.sh"
# shellcheck source=isaaclab_arena_gap/scripts/autolab_layout.sh
. "$ARENA_GAP_ROOT/scripts/autolab_layout.sh"
resolve_autolab_layout "$CAP_ROOT"

# The legacy-fork SHA describes lineage; executable child revisions come from
# the committed or staged gitlinks.
ARENA_LEGACY_CAP_PIN="1c80330225f7ae932275838f4c49c0692f92bf83"

GAP_PIN="$(gitlink_pin "$CAP_ROOT" external/gap/graph-as-policy)"
SKILLS_PIN="$(gitlink_pin "$CAP_ROOT" external/gap/open-robot-skills)"
AUTOLAB_VABAR_PIN="$(gitlink_pin "$CAP_ROOT" external/gap/autolab-vabar)"
ARENA_PIN="$(gitlink_pin "$CAP_ROOT" external/IsaacLab-Arena)"

errors=0
warnings=0

ok() { printf 'OK:   %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL: %s\n' "$*"; errors=$((errors + 1)); }

warn_if_dirty() {
  local name="$1" path="$2"
  if [[ -n "$(git -C "$path" status --porcelain -- . 2>/dev/null)" ]]; then
    warn "$name worktree is dirty: $path"
  fi
}

check_repo() {
  local name="$1" path="$2" pin="$3" actual
  if [[ ! -d "$path/.git" && ! -f "$path/.git" ]]; then
    fail "$name checkout is missing: $path"
    return
  fi
  actual="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$pin" ]]; then
    fail "$name has no recorded gitlink pin"
  elif [[ "$actual" == "$pin" ]]; then
    ok "$name is at pinned commit $pin"
  else
    fail "$name is not at pinned commit $pin (HEAD: ${actual:-unknown})"
  fi
  warn_if_dirty "$name" "$path"
}

check_nested_submodule() {
  local parent="$1" path="$2" name="$3" expected actual
  expected="$(gitlink_pin "$parent" "$path")"
  if [[ -e "$parent/$path/.git" ]]; then
    actual="$(git -C "$parent/$path" rev-parse HEAD 2>/dev/null || true)"
  else
    actual=""
  fi
  if [[ -z "$expected" ]]; then
    fail "$name has no gitlink pin in $parent"
  elif [[ "$actual" == "$expected" ]]; then
    ok "$name is at pinned commit $expected"
  elif [[ -z "$actual" ]]; then
    fail "$name is not initialized: $parent/$path"
  else
    fail "$name is not at pinned commit $expected (HEAD: $actual)"
  fi
}

warn_if_dirty "isaaclab_arena" "$CAP_ROOT"
if uv_series_supported; then
  ok "uv $ARENA_UV_VERSION is in the supported $ARENA_UV_SERIES series"
else
  fail "uv $ARENA_UV_SERIES.x is required; found ${ARENA_UV_VERSION:-none} (see docs/SETUP.md section 2)"
fi
check_repo "graph-as-policy" "$GAP_ROOT" "$GAP_PIN"
check_repo "open-robot-skills" "$SKILLS_ROOT" "$SKILLS_PIN"
if [[ -n "$AUTOLAB_VABAR_PIN" ]]; then
  check_repo "autolab-vabar" "$AUTOLAB_VABAR_ROOT" "$AUTOLAB_VABAR_PIN"
elif autolab_sources_present; then
  # This tree is a folder of the VABAR repository, whose checked-out revision is the pin.
  ok "AUTOLab v1.0 sources are present: $AUTOLAB_VABAR_ROOT"
else
  fail "AUTOLab v1.0 sources are missing: $AUTOLAB_VABAR_ROOT"
fi
check_repo "Arena framework checkout" "$ARENA_SOURCE_ROOT" "$ARENA_PIN"
check_nested_submodule "$ARENA_SOURCE_ROOT" "submodules/IsaacLab" "Isaac Lab"
check_nested_submodule "$GAP_ROOT" "third_party/Variational-Automation-Benchmark" \
  "GaP Variational-Automation-Benchmark"
check_nested_submodule "$GAP_ROOT" "third_party/robosuite" "GaP robosuite"
if git -C "$ARENA_SOURCE_ROOT" merge-base --is-ancestor "$ARENA_LEGACY_CAP_PIN" HEAD 2>/dev/null; then
  fail "Arena framework checkout contains the legacy pre-release Arena fork $ARENA_LEGACY_CAP_PIN"
else
  ok "Arena framework checkout excludes the legacy pre-release Arena fork"
fi
if [[ -x "$GAP_ROOT/.venv/bin/python" ]]; then
  ok "GaP Python exists: $GAP_ROOT/.venv/bin/python"
  gap_python_version="$("$GAP_ROOT/.venv/bin/python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  # ORS is an editable sibling path whose exact source is enforced above by
  # the gitlink check. The source-only changes at the current ORS pin add no
  # dependencies, so GaP's immutable lock can retain older ORS metadata.
  if (cd "$GAP_ROOT" && uv sync --frozen --dry-run >/dev/null 2>&1); then
    ok "GaP environment is installable from the frozen lock"
  else
    fail "GaP environment cannot be installed from the frozen lock"
  fi
else
  fail "GaP environment is missing; run uv sync in $GAP_ROOT"
fi

# The RSS adapter runs every graph in AUTOLab v1.0's own environment.
autolab_python="$AUTOLAB_VABAR_VENV/bin/python"
if [[ -x "$autolab_python" ]]; then
  if import_error="$(PYTHONPATH="$ARENA_GAP_ROOT/src" "$autolab_python" -c \
      'import robosimstudio, robosimstudio_policy_agent.graph_solution, pxr, isaaclab_arena_gap.rss_adapter' 2>&1)"; then
    ok "AUTOLab v1.0 environment imports RoboSimStudio, its policy agent and the RSS adapter"
  else
    fail "AUTOLab v1.0 environment cannot run the RSS adapter: $import_error"
  fi
else
  fail "AUTOLab v1.0 environment is missing: $autolab_python (run ./setup.sh)"
fi

for bundle in "${CAP_SKILL_BUNDLES[@]}"; do
  bundle_python="$SKILLS_ROOT/tools/$bundle/.venv/bin/python"
  if [[ -x "$bundle_python" ]]; then
    bundle_python_version="$("$bundle_python" -c \
      'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ -n "${gap_python_version:-}" && "$bundle_python_version" != "$gap_python_version" ]]; then
      fail "GaP skill environment $bundle uses Python $bundle_python_version; expected $gap_python_version"
    else
      ok "GaP skill environment exists with Python $bundle_python_version: $bundle"
    fi
  else
    fail "GaP skill environment is missing: $bundle (run ./setup.sh)"
  fi
done

vlm_override="$ARENA_GAP_ROOT/skill_overrides/tools/vlm"
if [[ -f "$vlm_override/tools.py" ]]; then
  ok "repository VLM compatibility override exists"
else
  fail "repository VLM compatibility override is missing: $vlm_override/tools.py"
fi
if [[ -x "$vlm_override/.venv/bin/python" ]]; then
  ok "VLM compatibility override reuses the pinned bundle environment"
else
  fail "VLM compatibility override environment is missing (run ./setup.sh)"
fi

if [[ -x "$ARENA_PYTHON" ]]; then
  ok "Root simulation Python exists: $ARENA_PYTHON"
  simulation_python_version="$("$ARENA_PYTHON" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$simulation_python_version" == "3.12" ]]; then
    ok "Root simulation environment uses Python $simulation_python_version"
  else
    fail "Root simulation environment uses Python $simulation_python_version; expected 3.12"
  fi
  if [[ -x "$ARENA_ENV_ROOT/.venv/bin/vabar-arena" ]]; then
    ok "vabar-arena entrypoint is installed in the root simulation environment"
  else
    fail "vabar-arena entrypoint is missing from the root simulation environment"
  fi
  # Verify the exact frozen Arena-lock sync used by bootstrap without changing
  # the shared Arena environment.
  if (cd "$ARENA_SOURCE_ROOT" \
      && UV_PROJECT_ENVIRONMENT="$ARENA_ENV_ROOT/.venv" \
        uv sync --frozen --dry-run \
        >/dev/null 2>&1); then
    ok "Isaac Lab source environment is installable from the frozen lock"
  else
    fail "Isaac Lab source environment cannot be installed from the frozen lock"
  fi
  if runtime_versions="$("$ARENA_PYTHON" - "$ARENA_SOURCE_ROOT/uv.lock" 2>&1 <<'PYTHON'
import importlib.metadata as metadata
import sys
import tomllib

import newton

with open(sys.argv[1], "rb") as stream:
    packages = tomllib.load(stream)["package"]
names = ("isaaclab", "newton", "isaacsim", "mujoco-warp")
expected = {}
for name in names:
    versions = {package["version"] for package in packages if package["name"] == name}
    if len(versions) != 1:
        raise SystemExit(f"Arena lock does not pin exactly one {name}: {sorted(versions)}")
    expected[name] = versions.pop()
actual = {name: metadata.version(name) for name in names}
if actual != expected:
    raise SystemExit(f"runtime versions do not match the Arena lock: {actual!r} != {expected!r}")
print(", ".join(f"{name} {version}" for name, version in actual.items()))
PYTHON
)"; then
    ok "Root simulation Python matches the Arena lock: $runtime_versions"
  else
    fail "Root simulation Python has unexpected simulator versions: $runtime_versions"
  fi
  if import_error="$(PYTHONPATH="$ARENA_SOURCE_ROOT:$ARENA_GAP_ROOT/src:$CAP_ROOT/isaaclab_arena_vabar/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$ARENA_PYTHON" -c \
      'import importlib.metadata; import isaaclab_arena_vabar.registration; import isaaclab_arena_gap.gap_remote_policy; import msgpack_numpy; importlib.metadata.version("isaaclab-arena-gap")' \
      2>&1)"; then
    ok "Root simulation Python imports Arena, the GaP adapter, and industrial benchmark dependencies"
  else
    fail "Root simulation Python cannot import the GaP Arena policy: $import_error"
    warn "install after Arena sync: uv pip install --python $ARENA_PYTHON --editable $CAP_ROOT/isaaclab_arena_vabar --editable $ARENA_GAP_ROOT"
  fi
  # Arena deliberately overrides five stale or over-strict dependency declarations.
  # Accept that exact frozen
  # conflict profile and continue to fail on a missing or additional line.
  if check_arena_environment_consistency "$ARENA_PYTHON"; then
    ok "Arena environment passes uv pip check"
  elif [[ "$?" == "$ARENA_PIP_CHECK_KNOWN_CONFLICTS" ]]; then
    warn "Arena environment carries only the 5 pinned Arena source conflicts"
  else
    fail "Arena environment consistency check failed: $ARENA_PIP_CHECK_OUTPUT"
  fi
else
  fail "Root simulation environment is missing; run ./setup.sh"
fi

if [[ -f "$GAP_VLM_ENV" ]]; then
  ok "VLM env file exists: $GAP_VLM_ENV"
  if bash -u -c '
    set -a
    # shellcheck disable=SC1090
    . "$1" >/dev/null 2>&1
    [[ -n "${GAP_VLM_API_KEY:-}" || -n "${OPENROUTER_API_KEY:-}" ]]
  ' _ "$GAP_VLM_ENV"; then
    ok "VLM API credential variable is present (value not printed)"
  else
    warn "sourcing $GAP_VLM_ENV did not provide GAP_VLM_API_KEY or OPENROUTER_API_KEY"
  fi
  if grep -Eq '^(export[[:space:]]+)?HF_TOKEN=' "$GAP_VLM_ENV" || [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is configured (value not printed)"
  else
    warn "HF_TOKEN was not found; gated SAM3 download may fail"
  fi
elif [[ -n "${GAP_VLM_API_KEY:-}" ]]; then
  # Matches vlm_env.sh, which accepts an exported key without the file.
  ok "VLM API credential is exported as GAP_VLM_API_KEY (value not printed)"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is configured (value not printed)"
  else
    warn "HF_TOKEN was not found; gated SAM3 download may fail"
  fi
else
  fail "VLM env file is missing and GAP_VLM_API_KEY is not exported: $GAP_VLM_ENV"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ok "NVIDIA GPU is visible"
else
  fail "nvidia-smi cannot see an NVIDIA GPU"
fi

if [[ "${CAP_CHECK_EXTERNAL_PORT:-0}" == "1" ]] && command -v ss >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | grep -Eq '[:.]9000[[:space:]]'; then
    warn "TCP port 9000 already has a listener; stop the stale process or intentionally reuse it"
  else
    ok "TCP port 9000 is available for external-server mode"
  fi
else
  ok "managed mode uses a per-episode ephemeral port (fixed port check skipped)"
fi

printf '\nPreflight complete: %d error(s), %d warning(s).\n' "$errors" "$warnings"
if (( errors > 0 )); then
  exit 1
fi
