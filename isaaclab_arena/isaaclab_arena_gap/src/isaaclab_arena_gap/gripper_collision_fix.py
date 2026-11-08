"""Corrected collision geometry for the DROID Robotiq-2F-85 gripper.

The Arena gripper asset ``franka_robotiq_2f_85_flattened.usd`` gives the
Robotiq linkage bars ``convexHull`` collision meshes. Those hulls
over-approximate the thin bars into a converging V-channel between the
fingers, which cages a grasped object pressed into it: the object pops up past
the pads and stays wedged there even after the gripper opens. The trap is the
gap *between* separate links, so no per-mesh refinement removes it, and it is
geometric rather than frictional. Disabling collision on the two inner
knuckles -- the bars forming the cavity walls -- frees the object; every other
link keeps its collision, so contact with the environment is unchanged and the
inner-finger pads still do the gripping.

Newton imports collision geometry when the articulation is built, so the
correction must land *before* the gripper USD is parsed. Rather than forking
the 14 MB asset, the robot spawn is redirected to a generated override layer
that references the original and flips the two ``collisionEnabled`` flags via
USD composition.

This is the repository's gripper: installed from ``isaaclab_arena_gap/__init__`` so
every robot spawn gets it. ``CAP_FIX_GRIPPER_LINKAGE_COLLISION=0`` opts back
out for an A/B against the stock asset; no other value is accepted, so a typo
cannot silently select either behaviour. Every failure below raises rather
than falling back, because a correction that silently does not apply leaves a
scored run physically unfixed while its evidence reports it as fixed. The
long-term home for this correction is the upstream Arena gripper USD; until
then this module applies it.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import tempfile

#: The robot USD whose spawn we redirect (arm+gripper flattened).
_GRIPPER_BASENAME = "franka_robotiq_2f_85_flattened.usd"

#: The spawn helper every from-USD spawn funnels through.
_SPAWN_MODULE = "isaaclab.sim.spawners.from_files.from_files"
_SPAWN_FUNC = "_spawn_from_usd_file"

#: Collision prims (relative to the /panda default prim) of the two inner
#: knuckles that form the caging cavity. Every other link -- palm, outer
#: knuckles, outer fingers, inner-finger pads -- keeps its collision.
_LINKAGE_COLLISION_SUBPATHS = (
    "Gripper/Robotiq_2F_85/left_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
    "Gripper/Robotiq_2F_85/right_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
)

_ENABLE_VAR = "CAP_FIX_GRIPPER_LINKAGE_COLLISION"

_installed = False
_patched = False
_override_cache: dict[str, str] = {}
_redirects: list[tuple[str, str]] = []


def _log(message: str) -> None:
    # stderr, not stdout: this module is imported by CLIs whose stdout is
    # machine-readable bridge status.
    print(f"[gripper_collision_fix] {message}", file=sys.stderr)


class GripperCollisionFixError(RuntimeError):
    """The gripper correction could not be applied.

    Always fatal: continuing would score a run whose actual geometry
    contradicts its configuration.
    """


def linkage_fix_enabled() -> bool:
    """Resolve the gate. On by default; only "0" and "1" are accepted, since
    guessing at an unrecognised value would silently choose the robot's
    physics for a scored run."""
    value = os.environ.get(_ENABLE_VAR, "1")
    if value not in ("0", "1"):
        raise GripperCollisionFixError(
            f"{_ENABLE_VAR}={value!r} is not valid; use '1' (default, corrected "
            "gripper) or '0' (stock gripper with the caging linkage hulls)"
        )
    return value == "1"


def fix_status() -> dict[str, object]:
    """Applied state, for logging and for stamping into run evidence.

    ``spawn_redirects`` is the ground truth: a non-empty list is the only proof
    that Newton imported the corrected layer rather than the original.
    """
    enabled = linkage_fix_enabled()
    return {
        "enabled": enabled,
        "disabled_collision_prims": list(_LINKAGE_COLLISION_SUBPATHS) if enabled else [],
        "spawn_patched": _patched,
        "spawn_redirects": [
            {"original_usd": original, "override_usd": override}
            for original, override in _redirects
        ],
    }


def _override_layer_path(original_usd_path: str) -> str:
    # Keyed by pid and source asset: concurrent eval processes must not race on
    # one file, and two source assets in one process must not overwrite each
    # other's layer.
    digest = hashlib.sha256(original_usd_path.encode()).hexdigest()[:16]
    return os.path.join(
        tempfile.gettempdir(),
        f"cap_gripper_linkage_fix_{os.getpid()}_{digest}.usda",
    )


def _remove_override_layers() -> None:
    # Best effort: Isaac Sim tears the process down without running atexit
    # handlers, so under Kit the layer outlives the run. Harmless -- every run
    # recreates its own before use.
    for path in _override_cache.values():
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_remove_override_layers)


def _build_override_layer(original_usd_path: str) -> str:
    """Author an override USD that references ``original_usd_path`` and disables
    collision on the inner knuckles. Cached per original path."""
    if original_usd_path in _override_cache:
        return _override_cache[original_usd_path]

    from pxr import Sdf, Usd

    out_path = _override_layer_path(original_usd_path)
    # A leftover file from a recycled pid must never become this run's geometry.
    if os.path.exists(out_path):
        os.unlink(out_path)
    layer = Sdf.Layer.CreateNew(out_path)
    if layer is None:
        raise GripperCollisionFixError(f"could not create override layer at {out_path}")
    stage = Usd.Stage.Open(layer)
    panda = stage.DefinePrim("/panda")
    panda.GetReferences().AddReference(assetPath=original_usd_path)
    stage.SetDefaultPrim(panda)
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        prim = stage.OverridePrim(f"/panda/{subpath}")
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    stage.GetRootLayer().Save()

    # An ``over`` on a renamed prim composes to nothing, and an instanced or
    # restructured prim can resolve while ignoring the opinion -- both leave the
    # correction inert, so require the composed value to be exactly False.
    unresolved = []
    for subpath in _LINKAGE_COLLISION_SUBPATHS:
        prim = stage.GetPrimAtPath(f"/panda/{subpath}")
        if not prim.IsDefined():
            unresolved.append(f"{subpath} (prim not found)")
            continue
        attribute = prim.GetAttribute("physics:collisionEnabled")
        if not attribute or attribute.Get() is not False:
            unresolved.append(f"{subpath} (collisionEnabled did not compose to False)")
    if unresolved:
        raise GripperCollisionFixError(
            f"the gripper collision override does not apply to {original_usd_path}: "
            f"{len(unresolved)}/{len(_LINKAGE_COLLISION_SUBPATHS)} targets unresolved "
            "-- the asset's linkage prim paths may have changed. Refusing to run with "
            f"unfixed gripper geometry (set {_ENABLE_VAR}=0 to accept it). "
            f"Unresolved: {unresolved}"
        )

    _override_cache[original_usd_path] = out_path
    _log(
        f"wrote override layer {out_path} (references {original_usd_path}, "
        f"{len(_LINKAGE_COLLISION_SUBPATHS)} inner-knuckle collisions off)"
    )
    return out_path


def _redirect_usd_path(usd_path: str) -> str:
    """Swap the gripper spawn path for the override; pass everything else through."""
    if not usd_path or not linkage_fix_enabled():
        return usd_path
    if not usd_path.replace("\\", "/").endswith(_GRIPPER_BASENAME):
        return usd_path
    override = _build_override_layer(usd_path)
    _redirects.append((usd_path, override))
    _log(f"redirecting robot spawn -> {override}")
    return override


def _patch_spawn_module(module) -> None:
    global _patched
    orig = getattr(module, _SPAWN_FUNC, None)
    if orig is None:
        raise GripperCollisionFixError(
            f"{_SPAWN_MODULE}.{_SPAWN_FUNC} does not exist -- Isaac Lab's private "
            "spawn helper has been renamed or removed, so the gripper correction "
            f"cannot be applied. Refusing to run with unfixed geometry (set "
            f"{_ENABLE_VAR}=0 to accept it)."
        )
    if getattr(orig, "_cap_gripper_patched", False):
        _patched = True
        return

    def wrapped(prim_path, usd_path, cfg, *args, **kwargs):
        return orig(prim_path, _redirect_usd_path(usd_path), cfg, *args, **kwargs)

    wrapped._cap_gripper_patched = True
    setattr(module, _SPAWN_FUNC, wrapped)
    _patched = True
    _log(f"patched {_SPAWN_MODULE}.{_SPAWN_FUNC}")


def install_gripper_fix() -> None:
    """Install the spawn redirect, patching Isaac Lab's spawn module if it is
    already imported and otherwise registering an import hook that patches it
    the moment it loads. Safe to call before SimulationApp starts."""
    global _installed
    if _installed or not linkage_fix_enabled():
        return
    _installed = True

    if _SPAWN_MODULE in sys.modules:
        _patch_spawn_module(sys.modules[_SPAWN_MODULE])
        return

    import importlib.abc
    import importlib.util

    class _SpawnPatchFinder(importlib.abc.MetaPathFinder):
        # Stays installed for the process lifetime: a speculative find_spec
        # probe that never imports the module would otherwise disarm the
        # correction permanently and silently. Patching is idempotent.
        def __init__(self) -> None:
            self._resolving = False

        def find_spec(self, fullname, path, target=None):
            if fullname != _SPAWN_MODULE or self._resolving:
                return None
            self._resolving = True  # our own lookup re-enters sys.meta_path
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._resolving = False
            if spec is None or spec.loader is None:
                raise GripperCollisionFixError(
                    f"{_SPAWN_MODULE} could not be resolved to a loadable module, so "
                    "the gripper correction cannot be applied."
                )
            real_exec = spec.loader.exec_module

            def exec_module(mod):
                real_exec(mod)
                _patch_spawn_module(mod)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _SpawnPatchFinder())
