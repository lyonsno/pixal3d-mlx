"""Verify moge-mlx package is importable and re-exports work.

The full MoGe model test suite lives in the standalone moge-mlx package.
This file only checks that pixal3d-mlx can reach MoGe through moge_mlx.
"""

import pytest


def test_moge_mlx_import():
    """moge_mlx package should be importable with expected symbols."""
    from moge_mlx import MoGeModel, load_moge_weights, estimate_camera_params_mlx

    assert callable(MoGeModel)
    assert callable(load_moge_weights)
    assert callable(estimate_camera_params_mlx)


def test_moge_camera_reexports():
    """trellmlx.moge_camera should re-export from moge_mlx.camera."""
    from trellmlx.moge_camera import (
        estimate_camera_params,
        estimate_camera_params_mlx,
        _compute_distance_from_fov,
    )

    assert callable(estimate_camera_params)
    assert callable(estimate_camera_params_mlx)
    assert callable(_compute_distance_from_fov)


def test_moge_camera_reexport_identity():
    """Re-exports should be the exact same objects, not copies."""
    from trellmlx.moge_camera import estimate_camera_params_mlx as from_trellmlx
    from moge_mlx.camera import estimate_camera_params_mlx as from_moge

    assert from_trellmlx is from_moge
