"""Filesystem ownership boundaries for the VABAR on Isaac Lab Arena workspace."""

from pathlib import Path


ARENA_GAP_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ARENA_GAP_ROOT.parent
