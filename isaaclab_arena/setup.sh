#!/usr/bin/env bash
set -euo pipefail

CAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CAP_ROOT/isaaclab_arena_gap/scripts/bootstrap.sh" "$@"
