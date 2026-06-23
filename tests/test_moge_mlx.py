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

        # Spot-check key parameter shapes (F4)
        pe = model.encoder.backbone.patch_embed.weight
        assert pe.shape == (1024, 14, 14, 3), f"patch_embed shape: {pe.shape}"
        cls = model.encoder.backbone.cls_token
        assert cls.shape == (1, 1, 1024), f"cls_token shape: {cls.shape}"
        pos = model.encoder.backbone.pos_embed
        assert pos.shape == (1, 1370, 1024), f"pos_embed shape: {pos.shape}"

    def test_forward_runs(self):
        """Verify forward pass produces correctly shaped, non-trivial outputs."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        img = mx.array(np.random.rand(1, 128, 128, 3).astype(np.float32))
        output = model.forward(img, num_tokens=900)
        mx.eval(output["points"], output["mask"], output["metric_scale"])

        pts = output["points"]
        assert pts.shape == (1, 128, 128, 3)

        # F1: verify output is non-trivial (not all zeros / dead model)
        pts_np = np.array(pts)
        assert pts_np.std() > 0.01, f"Points appear trivial: std={pts_np.std()}"
        assert not np.all(pts_np == 0), "Points are all zeros"

        mask = output["mask"]
        assert mask.shape == (1, 128, 128)

        scale = output["metric_scale"]
        assert scale.shape == (1,)

    def test_infer_api_nonsquare(self):
        """Verify infer() handles non-square channels-first input correctly."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        # F3: non-square input catches channels-first/last transpose errors
        img = mx.array(np.random.rand(3, 100, 140).astype(np.float32))
        result = model.infer(img, resolution_level=0)
        mx.eval(result["intrinsics"])

        assert "points" in result
        assert "depth" in result
        assert "intrinsics" in result
        assert "mask" in result

        assert result["points"].shape == (100, 140, 3), (
            f"Expected (100, 140, 3), got {result['points'].shape}"
        )
        assert result["depth"].shape == (100, 140)
        assert result["intrinsics"].shape == (3, 3)
        assert result["mask"].shape == (100, 140)


class TestMoGeMLXComponentSmoke:
    """Smoke tests for individual MLX MoGe components.

    These verify that loaded components produce correct shapes and non-trivial
    output. They are NOT cross-backend parity tests against PyTorch — parity
    was verified interactively during development (see topos for evidence).
    """

    def test_conv_transpose_resampler(self):
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        np.random.seed(42)
        x_np = np.random.randn(1, 8, 8, 1024).astype(np.float32)

        y = model.neck.resamplers[0](mx.array(x_np))
        mx.eval(y)
        y_np = np.array(y)

        assert y_np.shape == (1, 16, 16, 256)
        assert y_np.std() > 0.01

    def test_residual_block(self):
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

    def test_missing_weights_error(self):
        """Verify helpful error when weights are not found."""
        from trellmlx.models.moge_loader import _find_hf_weights
        with pytest.raises(FileNotFoundError, match="huggingface-cli download"):
            _find_hf_weights("nonexistent/model-xyz")


class TestMoGeNormalHead:
    """Tests for the normal estimation head (moge-2-vitl-normal variant)."""

    def test_model_has_no_normal_head_by_default(self):
        """Base model should not have a normal_head attribute."""
        from trellmlx.models.moge import MoGeModel
        model = MoGeModel()
        assert not hasattr(model, "normal_head") or model.normal_head is None

    def test_model_with_normal_head(self):
        """Model instantiated with normal_head=True should have the head."""
        from trellmlx.models.moge import MoGeModel
        model = MoGeModel(normal_head=True)
        assert model.normal_head is not None

    def test_normal_head_weight_loading(self):
        """Load weights from moge-2-vitl-normal checkpoint; normal_head must be populated."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel(normal_head=True)
        n = load_moge_weights(model, weights_path=None,
                              model_name="Ruicheng/moge-2-vitl-normal",
                              verbose=False)
        # Base model has ~480 arrays; normal head adds 38 more
        assert n >= 510, f"Expected >= 510 weight arrays with normal head, got {n}"

        # Spot-check: output_blocks.4 should have shape [3, 1, 1, 32] (MLX conv format)
        ob4 = model.normal_head.output_blocks[4]
        assert ob4 is not None
        assert ob4.weight.shape == (3, 1, 1, 32), f"Got shape {ob4.weight.shape}"

    def test_forward_produces_normals(self):
        """Forward pass with normal_head should include 'normal' in output."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel(normal_head=True)
        load_moge_weights(model, model_name="Ruicheng/moge-2-vitl-normal",
                          verbose=False)

        img = mx.array(np.random.rand(1, 128, 128, 3).astype(np.float32))
        output = model.forward(img, num_tokens=900)
        mx.eval(output["points"], output["normal"], output["mask"])

        assert "normal" in output, "forward() must return 'normal' when normal_head is present"
        normals = output["normal"]
        assert normals.shape == (1, 128, 128, 3), f"Expected (1,128,128,3), got {normals.shape}"

        # Normals should be L2-normalized (unit vectors)
        normals_np = np.array(normals)
        norms = np.linalg.norm(normals_np, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01,
                                   err_msg="Normals should be approximately unit vectors")

    def test_forward_without_normal_head_still_works(self):
        """Base model (no normal_head) should still work, returning no 'normal' key."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel()
        load_moge_weights(model, verbose=False)

        img = mx.array(np.random.rand(1, 128, 128, 3).astype(np.float32))
        output = model.forward(img, num_tokens=900)
        mx.eval(output["points"], output["mask"])

        assert "normal" not in output, "Base model should not produce normals"
        assert output["points"].shape == (1, 128, 128, 3)

    def test_infer_returns_normals(self):
        """infer() with normal_head should include normals in result."""
        from trellmlx.models.moge import MoGeModel
        from trellmlx.models.moge_loader import load_moge_weights
        model = MoGeModel(normal_head=True)
        load_moge_weights(model, model_name="Ruicheng/moge-2-vitl-normal",
                          verbose=False)

        img = mx.array(np.random.rand(3, 100, 140).astype(np.float32))
        result = model.infer(img, resolution_level=0)
        mx.eval(result["points"])

        assert "normal" in result, "infer() must return 'normal' when normal_head is present"
        assert result["normal"].shape == (100, 140, 3), \
            f"Expected (100,140,3), got {result['normal'].shape}"
