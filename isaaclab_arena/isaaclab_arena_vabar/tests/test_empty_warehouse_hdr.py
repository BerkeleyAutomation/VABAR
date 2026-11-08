"""The benchmark dome light loads its HDR from a bucket that still hosts it."""

import pytest

pytest.importorskip("isaaclab_arena")


def test_hdr_comes_from_the_production_bucket():
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    from isaaclab_arena_vabar.assets.empty_warehouse_hdr import (
        EMPTY_WAREHOUSE_HDR_TEXTURE_FILE,
        empty_warehouse_dome_light_cfg,
    )

    # Arena's staging asset root does not host Isaac Sim 6.1's copy of this HDR.
    assert "omniverse-content-staging" not in EMPTY_WAREHOUSE_HDR_TEXTURE_FILE
    assert EMPTY_WAREHOUSE_HDR_TEXTURE_FILE.startswith(ISAACLAB_NUCLEUS_DIR)
    assert EMPTY_WAREHOUSE_HDR_TEXTURE_FILE.endswith(
        "/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/empty_warehouse.hdr"
    )
    assert empty_warehouse_dome_light_cfg().texture_file == EMPTY_WAREHOUSE_HDR_TEXTURE_FILE
