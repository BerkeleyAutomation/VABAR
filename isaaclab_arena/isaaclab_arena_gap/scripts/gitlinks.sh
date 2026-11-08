# Gitlink lookup shared by bootstrap and preflight.
# shellcheck shell=bash

# Print the commit recorded for a submodule path in a parent repository.
# Prefer the exact index entry so a staged gitlink bump can be bootstrapped and
# preflighted before its parent commit is created.
gitlink_pin() {
  local parent="$1" path="$2" pin
  pin="$(git -C "$parent" ls-files --stage "$path" 2>/dev/null \
    | awk '$1 == "160000" { print $2; exit }')"
  if [[ -z "$pin" ]]; then
    pin="$(git -C "$parent" rev-parse --verify "HEAD:$path" 2>/dev/null || true)"
  fi
  printf '%s' "$pin"
}
