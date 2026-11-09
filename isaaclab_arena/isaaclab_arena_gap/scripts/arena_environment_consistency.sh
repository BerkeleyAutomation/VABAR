#!/usr/bin/env bash

# Shared fail-closed interpretation of Arena's pinned `uv pip check` result.
# Return 0 for a compatible environment, 10 for exactly the five deliberate
# dependency overrides in Arena's frozen source environment, and 1 for every other nonzero
# result. The complete captured output remains available to callers in
# ARENA_PIP_CHECK_OUTPUT.

ARENA_PIP_CHECK_OUTPUT=""
ARENA_PIP_CHECK_KNOWN_CONFLICTS=10

check_arena_environment_consistency() {
  local python="$1" status=0 line
  local using_python_lines=0 checked_lines=0 found_lines=0
  local known_conflict_lines=0 seen_conflicts=$'\n' conflict_token

  ARENA_PIP_CHECK_OUTPUT="$(uv pip check --python "$python" 2>&1)" || status=$?
  if ((status == 0)); then
    return 0
  fi

  while IFS= read -r line; do
    case "$line" in
      'The package `openpi-client` requires `numpy>=1.22.4,<2.0.0`, but `2.3.1` is installed'|\
      'The package `isaacsim-core` requires `newton[sim]==1.5.0`, but `1.5.2` is installed'|\
      'The package `isaacsim-kernel` requires `websockets>=12.0,<15`, but `16.1.1` is installed'|\
      'The package `isaacsim-robot` requires `onnxruntime-gpu==1.26.0`, but it'\''s not installed'|\
      'The package `ovphysx` requires `packaging>=20,<24`, but `26.0` is installed')
        conflict_token=$'\n'"$line"$'\n'
        if [[ "$seen_conflicts" == *"$conflict_token"* ]]; then
          return 1
        fi
        seen_conflicts+="$line"$'\n'
        known_conflict_lines=$((known_conflict_lines + 1))
        ;;
      "Found 5 incompatibilities")
        found_lines=$((found_lines + 1))
        ;;
      *)
        if [[ "$line" =~ ^Checked\ [0-9]+\ packages\ in\ [^[:space:]]+$ ]]; then
          checked_lines=$((checked_lines + 1))
        elif [[ "$line" =~ ^Using\ Python\ [0-9]+\.[0-9]+\.[0-9]+\ environment\ at:\ .+$ ]]; then
          using_python_lines=$((using_python_lines + 1))
        else
          return 1
        fi
        ;;
    esac
  done <<< "$ARENA_PIP_CHECK_OUTPUT"

  if ((checked_lines == 1 \
      && found_lines == 1 \
      && using_python_lines <= 1 \
      && known_conflict_lines == 5)); then
    return "$ARENA_PIP_CHECK_KNOWN_CONFLICTS"
  fi
  return 1
}

# The output above is uv 0.9's (the image pins 0.9.25). Later releases add diagnostics to
# `uv pip check`, such as uv 0.12's pre-release-mode deprecation warning, which the exact
# profile rejects, so setup and preflight require a uv 0.9.x on PATH.
ARENA_UV_SERIES="0.9"
ARENA_UV_VERSION=""

uv_series_supported() {
  ARENA_UV_VERSION="$(uv --version 2>/dev/null | awk '{ print $2 }')"
  [[ "$ARENA_UV_VERSION" == "$ARENA_UV_SERIES".* ]]
}
