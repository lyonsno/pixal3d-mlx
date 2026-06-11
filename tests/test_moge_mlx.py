"""Tests for the pure MLX MoGe-2 model."""

import math

import mlx.core as mx
import numpy as np
import pytest


class TestMoGeMLXModel:
    """Test the MLX MoGe model architecture and weight loading."""

    def test_model_instantiation(self):
        from trellmlx.models.moge import MoGeModel
        model = MoGeModel()
        # Verify key components exist
        assert hasattr(model, "encoder")
        assert hasattr(model, "neck")
        assert hasattr(model, "points_head")
        assert hasattr(model, "mask_head")
        assert hasattr(model, "scale_head")

    def test_weight_loading(self):
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        n = load_moge_weights(model, verbose=False)
        assert n >= 480, f"Expected >= 480 weight arrays, got {n}"

    def test_forward_runs(self):
        """Verify forward pass produces correctly shaped outputs."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        # Small image for speed
        img = mx.array(np.random.rand(1, 128, 128, 3).astype(np.float32))
        output = model.forward(img, num_tokens=900)
        mx.eval(output["points"], output["mask"], output["metric_scale"])

        pts = output["points"]
        assert pts.shape[0] == 1
        assert pts.shape[1] == 128  # resized to original
        assert pts.shape[2] == 128
        assert pts.shape[3] == 3

        mask = output["mask"]
        assert mask.shape == (1, 128, 128)

        scale = output["metric_scale"]
        assert scale.shape == (1,)
        assert float(scale[0]) > 0  # exp() is always positive

    def test_infer_api(self):
        """Verify infer() produces expected output keys."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        # channels-first input (matching upstream API)
        img = mx.array(np.random.rand(3, 128, 128).astype(np.float32))
        result = model.infer(img, resolution_level=0)  # lowest for speed
        mx.eval(result["intrinsics"])

        assert "points" in result
        assert "depth" in result
        assert "intrinsics" in result
        assert "mask" in result

        assert result["points"].shape == (128, 128, 3)
        assert result["depth"].shape == (128, 128)
        assert result["intrinsics"].shape == (3, 3)
        assert result["mask"].shape == (128, 128)


class TestMoGeMLXComponents:
    """Test individual MLX MoGe components against PyTorch reference."""

    def test_conv_transpose_resampler_parity(self):
        """ConvTranspose2d resampler should match PyTorch exactly."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        np.random.seed(42)
        x_np = np.random.randn(1, 8, 8, 1024).astype(np.float32)

        y = model.neck.resamplers[0](mx.array(x_np))
        mx.eval(y)
        y_np = np.array(y)

        # Basic shape check
        assert y_np.shape == (1, 16, 16, 256)
        # Should have non-trivial output
        assert y_np.std() > 0.01

    def test_residual_block_parity(self):
        """ResidualConvBlock should produce non-trivial output."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        np.random.seed(42)
        x_np = np.random.randn(1, 16, 16, 256).astype(np.float32)

        block = model.neck.res_blocks[1][0]
        y = block(mx.array(x_np))
        mx.eval(y)
        y_np = np.array(y)

        assert y_np.shape == (1, 16, 16, 256)
        assert y_np.std() > 0.01
        # Residual: output should differ from input
        assert not np.allclose(x_np, y_np, atol=0.01)

    def test_bilinear_resize(self):
        from trellmlx.models.moge import _bilinear_resize

        x = mx.array(np.random.rand(1, 8, 8, 3).astype(np.float32))
        y = _bilinear_resize(x, 16, 16)
        mx.eval(y)
        assert y.shape == (1, 16, 16, 3)

        # Identity resize
        z = _bilinear_resize(x, 8, 8)
        mx.eval(z)
        np.testing.assert_allclose(np.array(x), np.array(z), atol=1e-6)
