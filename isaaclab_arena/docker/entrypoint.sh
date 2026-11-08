#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  set -- vabar-eval --help
elif [[ "$1" == run || "$1" == verify || "$1" == manifest ]]; then
  set -- vabar-eval "$@"
fi
exec "$@"
