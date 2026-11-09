# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred registration for the industrial benchmark Arena components."""

from importlib import import_module


def register_arena_components() -> None:
    """Register the benchmark after Arena has started SimulationApp."""
    benchmark = import_module("isaaclab_arena_vabar.registration")
    # Also spreads placement seeds (isaaclab_arena_vabar.placement_seed_compat).
    benchmark.register_components()
