# Sourced by the GaP launchers: load the VLM credentials and the repository's VLM
# route. Expects GAP_VLM_ENV to name the credential file.
# shellcheck shell=bash

# OSMO injects GAP_VLM_API_KEY into the baked image, so an exported key can
# replace the local credential file; CAP_BAKED_RUNTIME alone carries no key.
if [[ -f "$GAP_VLM_ENV" ]]; then
  if [[ "${GAP_DRY_RUN:-0}" != "1" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$GAP_VLM_ENV"
    set +a
  fi
elif [[ -n "${GAP_VLM_API_KEY:-}" ]]; then
  GAP_VLM_ENV="(process environment)"
elif [[ "${GAP_DRY_RUN:-0}" == "1" ]]; then
  printf 'WARN: missing VLM env file: %s (dry run, continuing)\n' "$GAP_VLM_ENV" >&2
else
  printf 'Missing VLM env file: %s\n' "$GAP_VLM_ENV" >&2
  exit 1
fi

# Repository-owned defaults keep local and OSMO runs on the supported VLM
# route. Process- or credential-file values remain authoritative overrides.
export GAP_VLM_PROVIDER="${GAP_VLM_PROVIDER:-openrouter}"
export GAP_VLM_BASE_URL="${GAP_VLM_BASE_URL:-https://inference-api.nvidia.com/v1}"
export GAP_VLM_MODEL="${GAP_VLM_MODEL:-openai/openai/gpt-5.6-sol}"

# The container carries immutable model snapshots. Keep native installs'
# download flow unchanged, but never pass a Hugging Face credential at runtime
# from the combined VLM environment into the baked image.
if [[ "${CAP_BAKED_RUNTIME:-0}" == "1" ]]; then
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
fi
