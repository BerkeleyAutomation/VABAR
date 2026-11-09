"""Compatibility helpers for the pinned Open Robot Skills VLM bundle."""

from __future__ import annotations


_UPSTREAM_GPT5_CHECK = 'if model.lower().startswith("gpt-5"):'
_QUALIFIED_GPT5_CHECK = "if is_gpt5_model(model):"
_UPSTREAM_RETRIES = "_MAX_RETRIES = 3\n"
# Each sweep runner calls the VLM on one shared key, so a wide sweep draws 429s that three
# tries (~3 s of backoff) don't outlast. Seven tries back off 1+2+4+8+16+32 s, about a minute.
_SWEEP_RETRIES = "_MAX_RETRIES = 7\n"


def is_gpt5_model(model: str) -> bool:
    """Return whether the final component of a routed model ID is GPT-5."""
    return model.rsplit("/", maxsplit=1)[-1].lower().startswith("gpt-5")


def patch_upstream_vlm_source(source: str) -> str:
    """Adapt the pinned bundle for provider-qualified model IDs and sweep-scale rate limits."""
    for seam, name in ((_UPSTREAM_GPT5_CHECK, "GPT-5 detection"), (_UPSTREAM_RETRIES, "retry count")):
        count = source.count(seam)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one {name} seam in the pinned VLM bundle, found {count}"
            )
    return source.replace(_UPSTREAM_GPT5_CHECK, _QUALIFIED_GPT5_CHECK).replace(
        _UPSTREAM_RETRIES, _SWEEP_RETRIES
    )
