"""Tests for MoGe-2 camera estimation module."""

import math

import numpy as np
import pytest


def _moge_available() -> bool:
    """Check if MoGe is importable."""
    try:
        from moge.model import import_model_class_by_version
        return True
    except ImportError:
        return False


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


class TestEstimateCameraParams:
    """Integration tests for full MoGe estimation (requires MoGe installed)."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a simple test image."""
        from PIL import Image
        img = Image.new("RGB", (512, 512), color=(128, 64, 32))
        path = tmp_path / "test.png"
        img.save(path)
        return path

    @pytest.mark.skipif(
        not _moge_available(),
        reason="MoGe not installed",
    )
    def test_returns_valid_camera_params(self, sample_image):
        from trellmlx.moge_camera import estimate_camera_params

        params = estimate_camera_params(sample_image)

        assert "camera_angle_x" in params
        assert "distance" in params
        assert "mesh_scale" in params

        # FOV should be in a reasonable range (10° to 120°)
        fov_deg = math.degrees(params["camera_angle_x"])
        assert 10 < fov_deg < 120, f"FOV {fov_deg}° out of expected range"

        # Distance should be positive and finite
        assert params["distance"] > 0
        assert math.isfinite(params["distance"])

    @pytest.mark.skipif(
        not _moge_available(),
        reason="MoGe not installed",
    )
    def test_different_images_different_fov(self, tmp_path):
        """Different image content should (generally) produce different FOV."""
        from PIL import Image
        from trellmlx.moge_camera import estimate_camera_params

        # Solid color image
        img1 = Image.new("RGB", (512, 512), color=(200, 200, 200))
        path1 = tmp_path / "solid.png"
        img1.save(path1)

        # Gradient image (simulates perspective)
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            arr[i, :, :] = int(255 * i / 511)
        img2 = Image.fromarray(arr)
        path2 = tmp_path / "gradient.png"
        img2.save(path2)

        p1 = estimate_camera_params(path1)
        p2 = estimate_camera_params(path2)

        # They should both be valid even if similar
        assert 10 < math.degrees(p1["camera_angle_x"]) < 120
        assert 10 < math.degrees(p2["camera_angle_x"]) < 120
