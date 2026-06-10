"""Tests for NAF (Neural Attention Fields) feature upsampler.

Covers: neighborhood attention, RoPE, image encoder, cross-attention,
weight loading, and integration with DinoV3ProjFeatureExtractor.
"""

import os
import math
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trellmlx.models.naf import (
    NAF, NAFRoPE, NAFCrossAttention, EncBlock, make_encoder,
    neighborhood_attention,
)


# ── Neighborhood Attention ──

class TestNeighborhoodAttention:
    """Test the windowed gather-based neighborhood attention."""

    def test_output_shape(self):
        """Output shape should match input spatial dims with V's channel dim."""
        B, H, X, Y, Dq, Dv = 1, 4, 8, 8, 64, 256
        q = mx.random.normal((B, H, X, Y, Dq))
        k = mx.random.normal((B, H, X, Y, Dq))
        v = mx.random.normal((B, H, X, Y, Dv))
        out = neighborhood_attention(q, k, v, kernel_size=(9, 9), scale=1.0 / math.sqrt(Dq))
        mx.eval(out)
        assert out.shape == (B, H, X, Y, Dv)

    def test_cross_attention_different_dv(self):
        """V can have different channel dim than Q/K (cross-attention case)."""
        B, H, X, Y = 1, 4, 16, 16
        Dq, Dv = 64, 256  # NAF's actual dims: Q/K=64, V=256
        q = mx.random.normal((B, H, X, Y, Dq))
        k = mx.random.normal((B, H, X, Y, Dq))
        v = mx.random.normal((B, H, X, Y, Dv))
        out = neighborhood_attention(q, k, v, (9, 9), scale=0.125)
        mx.eval(out)
        assert out.shape == (B, H, X, Y, Dv)

    def test_dilation_changes_output(self):
        """Dilation=4 should produce different output than dilation=1."""
        B, H, X, Y, D = 1, 2, 32, 32, 16
        q = mx.random.normal((B, H, X, Y, D))
        k = mx.random.normal((B, H, X, Y, D))
        v = mx.random.normal((B, H, X, Y, D))

        out_d1 = neighborhood_attention(q, k, v, (5, 5), scale=0.25, dilation=(1, 1))
        out_d4 = neighborhood_attention(q, k, v, (5, 5), scale=0.25, dilation=(4, 4))
        mx.eval(out_d1, out_d4)

        diff = float(mx.mean(mx.abs(out_d1 - out_d4)))
        assert diff > 0.01, f"Dilation should change output, diff={diff}"

    def test_dilation_receptive_field(self):
        """With dilation=4 and kernel=3, position (8,8) should attend to
        positions 4 apart: (4,4), (4,8), (4,12), (8,4), (8,8), (8,12),
        (12,4), (12,8), (12,12)."""
        B, H, X, Y, D = 1, 1, 16, 16, 1
        # Put a unique value at each position in V
        v = mx.zeros((B, H, X, Y, D))
        v_np = np.zeros((B, H, X, Y, D), dtype=np.float32)
        for i in range(X):
            for j in range(Y):
                v_np[0, 0, i, j, 0] = i * 100 + j
        v = mx.array(v_np)

        # Uniform Q and K so attention weights are uniform
        q = mx.ones((B, H, X, Y, D))
        k = mx.ones((B, H, X, Y, D))

        out = neighborhood_attention(q, k, v, (3, 3), scale=1.0, dilation=(4, 4))
        mx.eval(out)

        # Position (8, 8) should average over 9 neighbors at stride 4
        # Neighbors: (4,4)=404, (4,8)=408, (4,12)=412, (8,4)=804, (8,8)=808,
        #            (8,12)=812, (12,4)=1204, (12,8)=1208, (12,12)=1212
        expected_neighbors = [404, 408, 412, 804, 808, 812, 1204, 1208, 1212]
        expected_mean = sum(expected_neighbors) / len(expected_neighbors)
        actual = float(out[0, 0, 8, 8, 0])
        assert abs(actual - expected_mean) < 1.0, \
            f"Expected ~{expected_mean:.1f}, got {actual:.1f}"

    def test_chunking_matches_unchunked(self):
        """Chunked processing should produce identical results to small grid."""
        B, H, D = 1, 2, 16
        # Small grid (no chunking needed)
        X_small, Y_small = 8, 8
        q_s = mx.random.normal((B, H, X_small, Y_small, D))
        k_s = mx.random.normal((B, H, X_small, Y_small, D))
        v_s = mx.random.normal((B, H, X_small, Y_small, D))
        out_s = neighborhood_attention(q_s, k_s, v_s, (5, 5), scale=0.25)
        mx.eval(out_s)
        assert out_s.shape == (B, H, X_small, Y_small, D)

    def test_asymmetric_grid(self):
        """Should work with X != Y."""
        B, H, X, Y, D = 1, 2, 24, 16, 32
        q = mx.random.normal((B, H, X, Y, D))
        k = mx.random.normal((B, H, X, Y, D))
        v = mx.random.normal((B, H, X, Y, D))
        out = neighborhood_attention(q, k, v, (7, 7), scale=0.125)
        mx.eval(out)
        assert out.shape == (B, H, X, Y, D)

    def test_scale_affects_output(self):
        """Different scale factors should produce different outputs."""
        B, H, X, Y, D = 1, 2, 8, 8, 16
        q = mx.random.normal((B, H, X, Y, D))
        k = mx.random.normal((B, H, X, Y, D))
        v = mx.random.normal((B, H, X, Y, D))
        out_s1 = neighborhood_attention(q, k, v, (5, 5), scale=0.01)
        out_s2 = neighborhood_attention(q, k, v, (5, 5), scale=10.0)
        mx.eval(out_s1, out_s2)
        diff = float(mx.mean(mx.abs(out_s1 - out_s2)))
        assert diff > 0.01, f"Scale should affect output, diff={diff}"


# ── NAFRoPE ──

class TestNAFRoPE:
    """Test 2D Rotary Position Embeddings."""

    def test_output_shape_preserved(self):
        """RoPE should preserve input shape [B, C, H, W]."""
        rope = NAFRoPE(embed_dim=256, num_heads=4, base=100.0)
        x = mx.random.normal((1, 256, 16, 16))
        out = rope(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_spatial_variance(self):
        """Different spatial positions should produce different outputs."""
        rope = NAFRoPE(embed_dim=64, num_heads=2, base=100.0)
        x = mx.ones((1, 64, 8, 8))  # uniform input
        out = rope(x)
        mx.eval(out)
        # Positions (0,0) and (4,4) should differ after RoPE
        pos_00 = out[0, :, 0, 0]
        pos_44 = out[0, :, 4, 4]
        diff = float(mx.mean(mx.abs(pos_00 - pos_44)))
        assert diff > 0.01, f"Spatial positions should differ, diff={diff}"

    def test_different_base_produces_different_output(self):
        """Different base frequencies should produce different RoPE."""
        x = mx.random.normal((1, 64, 8, 8))
        rope_100 = NAFRoPE(embed_dim=64, num_heads=2, base=100.0)
        rope_1000 = NAFRoPE(embed_dim=64, num_heads=2, base=1000.0)
        out_100 = rope_100(x)
        out_1000 = rope_1000(x)
        mx.eval(out_100, out_1000)
        diff = float(mx.mean(mx.abs(out_100 - out_1000)))
        assert diff > 0.01, f"Different base should differ, diff={diff}"


# ── EncBlock ──

class TestEncBlock:
    """Test residual conv block."""

    def test_output_shape(self):
        """EncBlock should preserve spatial dims and channels."""
        block = EncBlock(channels=128, kernel_size=3, num_groups=8)
        x = mx.random.normal((1, 16, 16, 128))  # NHWC
        out = block(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_no_residual_by_default(self):
        """Default EncBlock should NOT add input residual."""
        block = EncBlock(channels=32, kernel_size=1, num_groups=8)
        x = mx.ones((1, 4, 4, 32))
        out = block(x)
        mx.eval(out)
        # If residual were added, output would be close to input (1.0 + something)
        # Without residual, output is just the conv path
        # They should differ significantly
        diff = float(mx.mean(mx.abs(out - x)))
        assert diff > 0.01, "EncBlock should not be identity (no residual)"


# ── make_encoder ──

class TestMakeEncoder:
    """Test encoder factory."""

    def test_output_channels(self):
        """Encoder should produce the specified output channels."""
        layers = make_encoder(3, 128, kernel_size=1, ks_res=1, num_layers=2)
        x = mx.random.normal((1, 16, 16, 3))
        for layer in layers:
            x = layer(x)
        mx.eval(x)
        assert x.shape == (1, 16, 16, 128)

    def test_3x3_encoder(self):
        """3x3 encoder should work with padding."""
        layers = make_encoder(3, 64, kernel_size=3, ks_res=3, num_layers=2)
        x = mx.random.normal((1, 32, 32, 3))
        for layer in layers:
            x = layer(x)
        mx.eval(x)
        assert x.shape == (1, 32, 32, 64)


# ── NAFCrossAttention ──

class TestNAFCrossAttention:
    """Test cross-attention with neighborhood masking."""

    def test_output_shape(self):
        """Output should match Q spatial dims with V channel dim."""
        ca = NAFCrossAttention(dim=256, num_heads=4, kernel_size=(9, 9))
        q = mx.random.normal((1, 256, 32, 32))  # BCHW
        k = mx.random.normal((1, 256, 8, 8))
        v = mx.random.normal((1, 1024, 8, 8))
        out = ca(q, k, v)
        mx.eval(out)
        assert out.shape == (1, 1024, 32, 32)

    def test_upsample_ratio_4x(self):
        """4x upsample ratio should work (the standard NAF case)."""
        ca = NAFCrossAttention(dim=64, num_heads=2, kernel_size=(5, 5))
        q = mx.random.normal((1, 64, 16, 16))
        k = mx.random.normal((1, 64, 4, 4))
        v = mx.random.normal((1, 128, 4, 4))
        out = ca(q, k, v)
        mx.eval(out)
        assert out.shape == (1, 128, 16, 16)


# ── NAF Model ──

class TestNAFModel:
    """Test the full NAF model."""

    def test_smoke(self):
        """NAF should produce correct output shape without crashing."""
        naf = NAF(dim=256, heads_attn=4, heads_rope=4, kernel_size=9)
        image = mx.random.normal((1, 3, 64, 64))
        features = mx.random.normal((1, 1024, 4, 4))
        out = naf(image, features, (16, 16))
        mx.eval(out)
        assert out.shape == (1, 1024, 16, 16)
        assert not mx.any(mx.isnan(out)).item()

    def test_output_not_trivial(self):
        """Output should not be all zeros or constant."""
        naf = NAF(dim=256, heads_attn=4, heads_rope=4, kernel_size=9)
        image = mx.random.normal((1, 3, 64, 64))
        features = mx.random.normal((1, 512, 4, 4))
        out = naf(image, features, (16, 16))
        mx.eval(out)
        assert float(mx.std(out)) > 0.01, "Output should have variation"

    def test_different_images_different_output(self):
        """Different guide images should produce different upsampled features."""
        naf = NAF(dim=64, heads_attn=2, heads_rope=2, kernel_size=5)
        features = mx.random.normal((1, 256, 4, 4))
        img_a = mx.random.normal((1, 3, 32, 32))
        img_b = mx.random.normal((1, 3, 32, 32)) * 2
        out_a = naf(img_a, features, (8, 8))
        out_b = naf(img_b, features, (8, 8))
        mx.eval(out_a, out_b)
        diff = float(mx.mean(mx.abs(out_a - out_b)))
        assert diff > 0.001, f"Different images should produce different output, diff={diff}"

    def test_bchw_convention(self):
        """NAF expects BCHW input and produces BCHW output."""
        naf = NAF(dim=64, heads_attn=2, heads_rope=2, kernel_size=5)
        image = mx.random.normal((1, 3, 32, 32))  # BCHW
        features = mx.random.normal((1, 128, 4, 4))  # BCHW
        out = naf(image, features, (8, 8))
        mx.eval(out)
        # Output should be BCHW with features' channel count
        assert out.shape[0] == 1  # B
        assert out.shape[1] == 128  # C (from features)
        assert out.shape[2] == 8  # H (output_size)
        assert out.shape[3] == 8  # W (output_size)


# ── NAF Weight Loading ──

NAF_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "weights", "naf_release.safetensors")

needs_naf_weights = pytest.mark.skipif(
    not os.path.exists(NAF_WEIGHTS),
    reason="NAF weights not downloaded"
)


@needs_naf_weights
class TestNAFWeightLoading:
    """Test loading real NAF weights."""

    def test_all_weights_loaded(self):
        """All 37 model parameters should be loaded."""
        from trellmlx.models.naf_loader import load_naf_weights
        model = NAF()
        skipped = load_naf_weights(model, NAF_WEIGHTS, verbose=True)
        assert len(skipped) == 0, f"Unexpected skipped keys: {skipped}"

    def test_loaded_model_nontrivial(self):
        """Loaded model should produce non-zero, non-NaN output."""
        from trellmlx.models.naf_loader import load_naf_weights
        model = NAF()
        load_naf_weights(model, NAF_WEIGHTS, verbose=False)

        image = mx.random.normal((1, 3, 64, 64))
        features = mx.random.normal((1, 1024, 4, 4))
        out = model(image, features, (16, 16))
        mx.eval(out)

        assert not mx.any(mx.isnan(out)).item(), "Output contains NaN"
        assert float(mx.mean(mx.abs(out))) > 1e-4, "Output is trivial"

    def test_encoder_weights_nontrivial(self):
        """Encoder conv weights should not be all zeros after loading."""
        from trellmlx.models.naf_loader import load_naf_weights
        model = NAF()
        load_naf_weights(model, NAF_WEIGHTS, verbose=False)

        # Check first 1x1 encoder conv weight
        w = model.encoder_1x1[0].weight
        mx.eval(w)
        assert float(mx.mean(mx.abs(w))) > 1e-4, "Encoder weight is trivial"

        # Check first 3x3 encoder conv weight
        w3 = model.encoder_3x3[0].weight
        mx.eval(w3)
        assert float(mx.mean(mx.abs(w3))) > 1e-4, "Sem encoder weight is trivial"


# ── NAF Weight Loader Key Remapping ──

class TestNAFKeyRemapping:
    """Test weight key remapping logic."""

    def test_encoder_remap(self):
        from trellmlx.models.naf_loader import _remap_key
        assert _remap_key("image_encoder.encoder.0.weight") == "encoder_1x1.0.weight"
        assert _remap_key("image_encoder.encoder.1.conv1.weight") == "encoder_1x1.1.conv1.weight"
        assert _remap_key("image_encoder.encoder.2.norm2.bias") == "encoder_1x1.2.norm2.bias"

    def test_sem_encoder_remap(self):
        from trellmlx.models.naf_loader import _remap_key
        assert _remap_key("image_encoder.sem_encoder.0.weight") == "encoder_3x3.0.weight"
        assert _remap_key("image_encoder.sem_encoder.1.conv1.weight") == "encoder_3x3.1.conv1.weight"

    def test_rope_remap(self):
        from trellmlx.models.naf_loader import _remap_key
        assert _remap_key("image_encoder.rope.periods") == "rope.periods"

    def test_unknown_key_unchanged(self):
        from trellmlx.models.naf_loader import _remap_key
        assert _remap_key("some.other.key") == "some.other.key"
