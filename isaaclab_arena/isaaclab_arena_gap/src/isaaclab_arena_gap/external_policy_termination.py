"""Bridge policy-owned episode completion into Isaac Lab termination terms."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_requested: list[bool] = []


def configure_external_policy_termination(num_envs: int) -> None:
    """Create fresh per-environment state for one Arena environment."""
    if isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    global _requested
    _requested = [False] * num_envs


def request_external_policy_termination(mask: Iterable[bool]) -> None:
    """Replace the current policy termination request mask."""
    values = [bool(value) for value in mask]
    global _requested
    if not _requested:
        _requested = [False] * len(values)
    if len(values) != len(_requested):
        raise ValueError(
            f"termination mask must have shape ({len(_requested)},), got ({len(values)},)"
        )
    _requested[:] = values


def clear_external_policy_termination(env_ids: Any = None) -> None:
    """Clear all requests or only the selected environment indexes."""
    if not _requested:
        return
    if env_ids is None:
        env_ids = range(len(_requested))
    elif hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().tolist()
    elif isinstance(env_ids, int):
        env_ids = [env_ids]
    for env_id in env_ids:
        _requested[int(env_id)] = False


def external_policy_termination(env: Any):
    """Isaac Lab termination term for policy-requested episode completion."""
    import torch

    if len(_requested) != env.num_envs:
        raise RuntimeError(
            "external policy termination state was not configured for this environment"
        )
    return torch.tensor(_requested, dtype=torch.bool, device=env.device)


def requested_external_policy_terminations() -> tuple[bool, ...]:
    """Return an immutable snapshot for diagnostics and unit tests."""
    return tuple(_requested)
