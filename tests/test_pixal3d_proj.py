"""Tests for Pixal3D projection conditioning modules.

Tests the core Pixal3D innovation: pixel-aligned back-projection.
Covers ProjGrid, ProjectAttention, and DinoV3ProjFeatureExtractor.
"""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trellmlx.modules.proj_grid import ProjGrid, project_points_to_image, _grid_sample_bilinear
from trellmlx.modules.proj_attention import ProjectAttention
from trellmlx.models.sparse_structure_flow import MultiHeadAttention
from trellmlx.models.pixal3d_flow import (
    ProjModulatedBlock, Pixal3DSparseStructureFlowModel, Pixal3DSLatFlowModel,
)


# ── grid_sample bilinear ──

class TestGridSampleBilinear:
    """Test bilinear sampling from feature maps."""

    def test_center_sample_returns_center_value(self):
        """Sampling at NDC (0, 0) should return the center of the feature map."""
        B, C, H, W = 1, 4, 8, 8
        fmap = mx.zeros((B, C, H, W))
        # Put a known value at center
        center_val = mx.array([1.0, 2.0, 3.0, 4.0])
        fmap_np = np.zeros((B, C, H, W), dtype=np.float32)
        # Center pixels at (3,3), (3,4), (4,3), (4,4) for 8x8
        for i in range(C):
            fmap_np[0, i, 3, 3] = center_val[i].item()
            fmap_np[0, i, 3, 4] = center_val[i].item()
            fmap_np[0, i, 4, 3] = center_val[i].item()
            fmap_np[0, i, 4, 4] = center_val[i].item()
        fmap = mx.array(fmap_np)

        queries = mx.array([[[0.0, 0.0]]])  # NDC center
        result = _grid_sample_bilinear(fmap, queries)
        mx.eval(result)

        assert result.shape == (1, 4, 1), f"Expected (1, 4, 1), got {result.shape}"
        # Should be close to center_val (exact depends on align_corners convention)
        for i in range(C):
            assert abs(float(result[0, i, 0]) - float(center_val[i])) < 0.5, \
                f"Channel {i}: expected ~{float(center_val[i])}, got {float(result[0, i, 0])}"

    def test_output_shape(self):
        """Grid sample output shape should be [B, C, K]."""
        B, C, H, W, K = 2, 16, 32, 32, 100
        fmap = mx.ones((B, C, H, W))
        queries = mx.zeros((B, K, 2))
        result = _grid_sample_bilinear(fmap, queries)
        mx.eval(result)
        assert result.shape == (B, C, K)

    def test_border_padding(self):
        """Out-of-range queries should clamp to border values (not crash)."""
        fmap = mx.ones((1, 2, 4, 4))
        # Way outside [-1, 1]
        queries = mx.array([[[5.0, 5.0], [-5.0, -5.0]]])
        result = _grid_sample_bilinear(fmap, queries)
        mx.eval(result)
        assert result.shape == (1, 2, 2)
        # Should still be 1.0 (border clamped)
        assert float(mx.mean(result)) == pytest.approx(1.0, abs=0.01)


# ── project_points_to_image ──

class TestProjectPoints:
    """Test 3D-to-2D perspective projection."""

    def test_front_center_projects_to_image_center(self):
        """A point at origin should project near image center with front-facing camera."""
        B = 1
        points = mx.array([[[0.0, 0.0, 0.0]]])  # Origin
        # Front-facing camera at distance 2 along -Y
        transform = mx.array([[[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]]])
        fov = mx.array([0.8])  # ~45 degrees
        resolution = 512

        pts_2d, depth, valid = project_points_to_image(
            points, transform[0], fov, resolution
        )
        mx.eval(pts_2d, depth, valid)

        # Should be near center (256, 256)
        x, y = float(pts_2d[0, 0, 0]), float(pts_2d[0, 0, 1])
        assert abs(x - 256) < 10, f"X should be near 256, got {x}"
        assert abs(y - 256) < 10, f"Y should be near 256, got {y}"
        assert bool(valid[0, 0]), "Center point should be valid"
        assert float(depth[0, 0]) > 0, "Depth should be positive"

    def test_batch_projection_shape(self):
        """Output shapes should match input batch and point count."""
        B, N = 2, 100
        points = mx.zeros((B, N, 3))
        transform = mx.broadcast_to(mx.eye(4)[None], (B, 4, 4))
        fov = mx.array([0.8, 0.8])

        pts_2d, depth, valid = project_points_to_image(points, transform, fov, 512)
        mx.eval(pts_2d, depth, valid)

        assert pts_2d.shape == (B, N, 2)
        assert depth.shape == (B, N)
        assert valid.shape == (B, N)


# ── ProjGrid ──

class TestProjGrid:
    """Test the 3D grid projection module."""

    def test_output_shape(self):
        """ProjGrid should output [B, R^3, C] features."""
        R = 8
        grid = ProjGrid(grid_resolution=R, image_resolution=512)
        B, C, H, W = 1, 64, 32, 32
        fmap = mx.ones((B, H, W, C))  # BHWC
        fov = mx.array([0.8])
        dist = mx.array([2.0])
        scale = mx.array([1.0])

        result = grid(fmap, fov, dist, scale)
        mx.eval(result)

        expected_tokens = R ** 3
        assert result.shape == (B, expected_tokens, C), \
            f"Expected (1, {expected_tokens}, {C}), got {result.shape}"

    def test_different_grid_resolutions(self):
        """Different grid resolutions should produce different token counts."""
        for R in [4, 8, 16]:
            grid = ProjGrid(grid_resolution=R, image_resolution=512)
            fmap = mx.ones((1, 16, 16, 32))
            result = grid(fmap, mx.array([0.8]), mx.array([2.0]), mx.array([1.0]))
            mx.eval(result)
            assert result.shape[1] == R ** 3

    def test_bchw_input(self):
        """ProjGrid should handle BCHW input when BHWC=False."""
        R = 4
        grid = ProjGrid(grid_resolution=R, image_resolution=512)
        B, C, H, W = 1, 32, 16, 16
        fmap = mx.ones((B, C, H, W))

        result = grid(fmap, mx.array([0.8]), mx.array([2.0]), mx.array([1.0]), BHWC=False)
        mx.eval(result)
        assert result.shape == (B, R ** 3, C)


# ── ProjectAttention ──

class TestProjectAttention:
    """Test the projection attention wrapper."""

    def test_output_shape_matches_input(self):
        """ProjectAttention output should match input spatial dimensions."""
        channels = 64
        ctx_channels = 32
        proj_in = 48
        num_heads = 4
        T = 100  # tokens

        cross_attn = MultiHeadAttention(channels, num_heads, ctx_channels)
        proj_attn = ProjectAttention(cross_attn, channels, proj_in)

        x = mx.ones((T, channels))
        context = {
            'global': mx.ones((1, 5, ctx_channels)),  # B=1, 5 global tokens
            'proj': mx.ones((1, T, proj_in)),  # B=1, T projected tokens
        }

        result = proj_attn(x, context)
        mx.eval(result)
        assert result.shape == (T, channels), f"Expected ({T}, {channels}), got {result.shape}"

    def test_proj_linear_affects_output(self):
        """Changing proj features should change the output."""
        channels = 32
        cross_attn = MultiHeadAttention(channels, 4, 16)
        proj_attn = ProjectAttention(cross_attn, channels, 16)

        x = mx.ones((10, channels))
        ctx_a = {'global': mx.ones((1, 5, 16)), 'proj': mx.ones((1, 10, 16))}
        ctx_b = {'global': mx.ones((1, 5, 16)), 'proj': mx.ones((1, 10, 16)) * 5.0}

        out_a = proj_attn(x, ctx_a)
        out_b = proj_attn(x, ctx_b)
        mx.eval(out_a, out_b)

        diff = float(mx.mean(mx.abs(out_a - out_b)))
        assert diff > 0.01, f"Proj features should affect output, diff={diff}"


# ── ProjModulatedBlock ──

class TestProjModulatedBlock:
    """Test the projection-conditioned DiT block."""

    def test_forward_shape(self):
        """Block should preserve token count and channel dimension."""
        channels = 64
        num_heads = 4
        ctx_channels = 32
        mlp_hidden = 128
        proj_in = 48
        T = 50

        block = ProjModulatedBlock(channels, num_heads, ctx_channels, mlp_hidden, proj_in)

        x = mx.ones((T, channels))
        mod = mx.zeros((6 * channels,))
        context = {
            'global': mx.ones((1, 5, ctx_channels)),
            'proj': mx.ones((1, T, proj_in)),
        }

        out = block(x, mod, context)
        mx.eval(out)
        assert out.shape == (T, channels), f"Expected ({T}, {channels}), got {out.shape}"

    def test_block_is_deterministic(self):
        """Same inputs should produce same outputs."""
        block = ProjModulatedBlock(32, 4, 16, 64, 16)
        x = mx.ones((10, 32))
        mod = mx.zeros((192,))
        ctx = {'global': mx.ones((1, 5, 16)), 'proj': mx.ones((1, 10, 16))}

        out1 = block(x, mod, ctx)
        out2 = block(x, mod, ctx)
        mx.eval(out1, out2)

        diff = float(mx.max(mx.abs(out1 - out2)))
        assert diff < 1e-5, f"Block should be deterministic, max diff={diff}"


# ── Pixal3DSparseStructureFlowModel ──

class TestPixal3DSSFlowModel:
    """Test the full Pixal3D sparse structure flow model."""

    def test_forward_shape(self):
        """Model should produce [B, out_C, R, R, R] from [B, in_C, R, R, R]."""
        R = 4  # Small for testing
        model = Pixal3DSparseStructureFlowModel(
            in_channels=8, out_channels=8,
            model_channels=64, num_heads=4,
            num_blocks=2, mlp_hidden=128,
            context_channels=32, proj_in_channels=48,
            resolution=R,
        )

        x = mx.ones((1, 8, R, R, R))
        t = mx.array([0.5])
        cond = {
            'global': mx.ones((1, 5, 32)),
            'proj': mx.ones((1, R ** 3, 48)),
        }

        out = model(x, t, cond)
        mx.eval(out)
        assert out.shape == (1, 8, R, R, R), f"Expected (1, 8, {R}, {R}, {R}), got {out.shape}"

    def test_different_timesteps_produce_different_outputs(self):
        """Different timesteps should produce different noise predictions."""
        R = 4
        model = Pixal3DSparseStructureFlowModel(
            in_channels=4, out_channels=4,
            model_channels=32, num_heads=4,
            num_blocks=2, mlp_hidden=64,
            context_channels=16, proj_in_channels=16,
            resolution=R,
        )

        x = mx.ones((1, 4, R, R, R))
        cond = {'global': mx.ones((1, 5, 16)), 'proj': mx.ones((1, R**3, 16))}

        out_t0 = model(x, mx.array([0.1]), cond)
        out_t1 = model(x, mx.array([0.9]), cond)
        mx.eval(out_t0, out_t1)

        diff = float(mx.mean(mx.abs(out_t0 - out_t1)))
        assert diff > 1e-4, f"Different timesteps should give different outputs, diff={diff}"


# ── Pixal3DSLatFlowModel ──

class TestPixal3DSLatFlowModel:
    """Test the Pixal3D structured latent flow model with projection."""

    def test_forward_shape(self):
        """SLat model should preserve token count and output channels."""
        N = 50  # sparse tokens
        model = Pixal3DSLatFlowModel(
            in_channels=16, out_channels=16,
            model_channels=64, num_heads=4,
            num_blocks=2, mlp_hidden=128,
            context_channels=32, proj_in_channels=48,
        )

        x = mx.ones((N, 16))
        t = mx.array([0.5])
        # For SLat, proj features are already sparse: [N, proj_in]
        cond = {
            'global': mx.ones((1, 5, 32)),
            'proj': mx.ones((1, N, 48)),  # pre-indexed
        }

        out = model(x, t, cond)
        mx.eval(out)
        assert out.shape == (N, 16), f"Expected ({N}, 16), got {out.shape}"

    def test_with_coords_rope(self):
        """SLat model should accept coordinates for RoPE."""
        N = 30
        model = Pixal3DSLatFlowModel(
            in_channels=8, out_channels=8,
            model_channels=32, num_heads=4,
            num_blocks=2, mlp_hidden=64,
            context_channels=16, proj_in_channels=16,
        )

        x = mx.ones((N, 8))
        t = mx.array([0.3])
        cond = {
            'global': mx.ones((1, 5, 16)),
            'proj': mx.ones((1, N, 16)),
        }
        coords = mx.array(np.random.randint(0, 64, size=(N, 3)).astype(np.int32))

        out = model(x, t, cond, coords=coords)
        mx.eval(out)
        assert out.shape == (N, 8)

    def test_with_concat_cond(self):
        """SLat model should accept concat_cond (e.g. shape features for texture)."""
        N = 20
        concat_dim = 8
        model = Pixal3DSLatFlowModel(
            in_channels=16 + concat_dim, out_channels=16,
            model_channels=32, num_heads=4,
            num_blocks=2, mlp_hidden=64,
            context_channels=16, proj_in_channels=16,
        )

        x = mx.ones((N, 16))
        concat = mx.ones((N, concat_dim))
        t = mx.array([0.5])
        cond = {
            'global': mx.ones((1, 5, 16)),
            'proj': mx.ones((1, N, 16)),
        }

        out = model(x, t, cond, concat_cond=concat)
        mx.eval(out)
        assert out.shape == (N, 16)
