"""The slim screwdriver simulates its steel/handle inertia at the delivered mass and centre of mass."""

import numpy as np
import pytest

pytest.importorskip("isaaclab_newton")
pytest.importorskip("newton")
pytest.importorskip("pxr")

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Usd

from ib_support import ASSETS_ROOT

_SLIM_USD = ASSETS_ROOT / "vabar__screwdriver_slim" / "vabar__screwdriver_slim.usda"


def test_newton_simulates_the_slim_screwdriver_two_density_inertia() -> None:
    builder = ModelBuilder()
    NewtonMJWarpManager._register_builder_attributes(builder)
    builder.add_usd(
        Usd.Stage.Open(str(_SLIM_USD)),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )
    (body,) = [index for index in range(builder.body_count) if builder.body_mass[index] > 0]
    assert builder.body_mass[body] == pytest.approx(0.07839327, rel=1e-6)
    # The delivered centre of mass: a steel shaft and tip against a plastic handle.
    assert tuple(builder.body_com[body]) == pytest.approx((0.0695, 0.0, 0.0), abs=1e-4)
    principal = np.linalg.eigvalsh(np.asarray(builder.body_inertia[body]).reshape(3, 3))
    # Steel parts 6.8x as dense as the handle. The delivered 1.6e-6/6.6e-4 transverse was
    # 0.8/4.6x this and above the largest inertia any mass distribution in the shape can have.
    assert principal == pytest.approx((1.9462e-6, 1.46032e-4, 1.46079e-4), rel=1e-3)
