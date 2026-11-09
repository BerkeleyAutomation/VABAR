"""Work around Isaac Lab 3.0 EA's passive-tendon adapter bug on Newton 1.5.

Isaac Lab's Newton 1.5 compatibility layer assumes that every model carrying
fixed tendons also carries MuJoCo actuator rows. The FR3/Robotiq asset has
passive coupling tendons but no ``mujoco:actuator`` rows, so the adapter reads
an attribute Newton correctly did not allocate. Newton 1.6 removes this
compatibility layer; keep this patch strictly version-gated to Newton 1.5.
"""

from __future__ import annotations

from importlib.metadata import version

_installed = False


def install_newton_15_passive_tendon_compat() -> None:
    """Return no tendon controller when a Newton 1.5 model has no actuators."""
    global _installed
    if _installed or not version("newton").startswith("1.5."):
        return

    from isaaclab_newton.physics.mjwarp_tendon_control import MjWarpTendonControl

    original = MjWarpTendonControl.create.__func__
    if getattr(original, "_isaac_cap_passive_tendon_compat", False):
        _installed = True
        return

    def create(cls, articulation, model):
        counts = getattr(model, "custom_frequency_counts", {}) or {}
        if int(counts.get("mujoco:actuator", 0)) == 0:
            return None
        return original(cls, articulation, model)

    create._isaac_cap_passive_tendon_compat = True
    MjWarpTendonControl.create = classmethod(create)
    _installed = True
    print(
        "[vabar] installed Newton 1.5 passive-tendon compatibility fix",
        flush=True,
    )
