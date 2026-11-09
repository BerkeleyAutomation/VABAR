"""Small numeric helpers shared by benchmark tasks and validators."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def is_finite_real(value: Any) -> bool:
    """Return whether ``value`` is a finite real number and not a boolean."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def as_torch(value: Any) -> torch.Tensor:
    """View an Isaac Lab data buffer (tensor, ProxyArray, or Warp array) as torch."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp

    return wp.to_torch(value)
