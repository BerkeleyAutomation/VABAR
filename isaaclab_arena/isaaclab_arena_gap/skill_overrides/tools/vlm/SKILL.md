---
name: vlm
description: Repository compatibility override for the pinned Open Robot Skills VLM bundle.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, vlm, api]}
gap:
  requires: {env_any: [OPENROUTER_API_KEY, GAP_VLM_API_KEY, GAP_VLM_BASE_URL, GAP_VLM_PROJECT_ID, GOOGLE_CLOUD_PROJECT, GAP_LLM_PROVIDER]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "vlm"]
    protocol: stdio-msgpack
    env: {UV_NO_SYNC: "1"}
  tools:
    - vlm.query: Free-form visual question answering via a hosted VLM.
    - vlm.query_yes_no: Yes/no visual question answering; coerces the model reply to a bool.
---

# VLM compatibility override

This bundle executes the immutable pinned Open Robot Skills VLM implementation with one
fail-closed source adaptation: GPT-5 detection examines the final component of provider-qualified
model IDs such as `openai/openai/gpt-5.6-sol`. It reuses the pinned VLM bundle environment.
