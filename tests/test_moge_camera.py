"""Tests for MoGe-2 camera estimation module."""

import math
import sys

import numpy as np
import pytest


def test_module_docstring_matches_cli_backend_contract():
    """The public backend labels should match generate_pixal3d.py flags."""
    import trellmlx.moge_camera as moge_camera

    doc = moge_camera.__doc__

    assert "MLX backend only" in doc
    assert "--no-moge" in doc
    assert "--mlx-moge" not in doc
    assert "--pytorch-moge" not in doc
    assert "PyTorch/MPS (default)" not in doc


def test_mlx_function_docstring_matches_default_backend_contract():
    """The default MLX estimator must not tell callers to use PyTorch instead."""
    from trellmlx.moge_camera import estimate_camera_params_mlx

    doc = estimate_camera_params_mlx.__doc__

    assert "pure MLX MoGe-2 port" in doc
    assert "EXPERIMENTAL" not in doc
    assert "Use estimate_camera_params() (PyTorch/MPS) for production." not in doc


def test_generate_pixal3d_falls_back_when_moge_estimator_raises(monkeypatch, tmp_path, capsys):
    """MoGe inference failures should not crash the generation entrypoint."""
    from PIL import Image
    import trellmlx.moge_camera as moge_camera
    import generate_pixal3d

    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(image_path)
    output_path = tmp_path / "mesh.glb"

    def fail_estimator(_image):
        raise ValueError("bad fov metadata")

    monkeypatch.setattr(moge_camera, "estimate_camera_params_mlx", fail_estimator)
    monkeypatch.setattr(sys, "argv", [
        "generate_pixal3d.py",
        "--image", str(image_path),
        "--output", str(output_path),
    ])

    real_import = __import__

    def stop_after_camera(name, *args, **kwargs):
        if name == "trellmlx.weight_loader":
            raise RuntimeError("stop after camera fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", stop_after_camera)

    with pytest.raises(RuntimeError, match="stop after camera fallback"):
        generate_pixal3d.main()

    stdout = capsys.readouterr().out
    assert "MoGe unavailable (bad fov metadata), falling back to default FOV." in stdout
    assert "Default FOV:" in stdout



class TestComputeDistanceFromFov:
    """Test the pure-math distance computation (no PyTorch needed)."""

    def test_default_fov_matches_expected(self):
        """Distance for default 49.1° FOV matches precomputed reference value."""
        from trellmlx.moge_camera import _compute_distance_from_fov

        default_fov = 0.8575560450553894  # ~49.1 degrees
        distance = _compute_distance_from_fov(default_fov)

        # Precomputed from upstream Pixal3D distance_from_fov() with
        # grid_point=[-1,0,0], mesh_scale=1.0, image_resolution=512
        expected = 1.0937500142266299
        assert abs(distance - expected) < 1e-8, (
            f"Distance {distance} != expected {expected}"
        )

    def test_wider_fov_gives_shorter_distance(self):
        """Wider FOV should result in shorter camera distance."""
        from trellmlx.moge_camera import _compute_distance_from_fov

        narrow_fov = 0.6  # ~34 degrees
        wide_fov = 1.2    # ~69 degrees

        d_narrow = _compute_distance_from_fov(narrow_fov)
        d_wide = _compute_distance_from_fov(wide_fov)

        assert d_narrow > d_wide, (
            f"Narrow FOV distance ({d_narrow}) should be > wide ({d_wide})"
        )

    def test_mesh_scale_affects_distance(self):
        from trellmlx.moge_camera import _compute_distance_from_fov

        d1 = _compute_distance_from_fov(0.8, mesh_scale=1.0)
        d2 = _compute_distance_from_fov(0.8, mesh_scale=2.0)

        # Different mesh scales should produce different distances
        assert d1 != d2

    def test_reasonable_fov_range(self):
        """Distances should be positive for reasonable FOV values."""
        from trellmlx.moge_camera import _compute_distance_from_fov

        for fov_deg in [30, 45, 60, 75, 90]:
            fov_rad = math.radians(fov_deg)
            d = _compute_distance_from_fov(fov_rad)
            assert d > 0, f"Distance should be positive for {fov_deg}° FOV, got {d}"

    def test_extreme_fov_values(self):
        """Very narrow and very wide FOVs should still produce finite results."""
        from trellmlx.moge_camera import _compute_distance_from_fov

        d_narrow = _compute_distance_from_fov(math.radians(10))
        d_wide = _compute_distance_from_fov(math.radians(120))

        assert math.isfinite(d_narrow)
        assert math.isfinite(d_wide)


def test_pytorch_estimate_not_exported():
    """PyTorch estimate_camera_params should not be re-exported (pure-MLX pipeline)."""
    import trellmlx.moge_camera as moge_camera

    assert not hasattr(moge_camera, "estimate_camera_params"), (
        "estimate_camera_params (PyTorch) should not be exported — "
        "pixal3d-mlx is a pure-MLX pipeline"
    )
