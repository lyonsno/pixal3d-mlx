"""Tests for loading real Pixal3D checkpoints into MLX models.

These tests require the actual Pixal3D checkpoints from HuggingFace.
They verify that weight names map correctly and that the loaded model
can produce non-trivial output (not zeros/NaN).
"""

import os
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trellmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel
from trellmlx.weight_loader import load_weights


SS_FLOW_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
    "snapshots/0b31f9160aa400719af409098bff7936a932f726/"
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
)

needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(SS_FLOW_CKPT),
    reason="Pixal3D SS flow checkpoint not downloaded"
)


@needs_checkpoint
class TestSSFlowWeightLoading:
    """Test loading the real Pixal3D SS flow model weights."""

    def test_weight_loading_no_missing_keys(self):
        """All model parameters should be filled from the checkpoint.

        Checks both directions: no checkpoint keys skipped, and no model
        params left unfilled.
        """
        import mlx.utils
        model = Pixal3DSparseStructureFlowModel(
            in_channels=8, out_channels=8,
            model_channels=1536, num_heads=12,
            num_blocks=30, mlp_hidden=8192,
            context_channels=1024, proj_in_channels=1024,
            resolution=16,
        )

        unloaded = load_weights(model, SS_FLOW_CKPT, verbose=True)

        # Check no checkpoint keys were skipped (except metadata and buffers)
        # rope_phases is a precomputed buffer, not a learned parameter —
        # our model computes RoPE on the fly from input resolution
        allowed_skips = {"rope_phases", "__metadata__"}
        unexpected_skips = [k for k in unloaded if k not in allowed_skips]
        assert len(unexpected_skips) == 0, \
            f"Unexpected skipped checkpoint keys: {unexpected_skips[:10]}"

    def test_loaded_model_produces_nonzero_output(self):
        """Loaded model should produce non-trivial output, not zeros or NaN."""
        model = Pixal3DSparseStructureFlowModel(
            in_channels=8, out_channels=8,
            model_channels=1536, num_heads=12,
            num_blocks=30, mlp_hidden=8192,
            context_channels=1024, proj_in_channels=1024,
            resolution=16,
        )
        load_weights(model, SS_FLOW_CKPT, verbose=False)

        R = 16
        x = mx.random.normal((1, 8, R, R, R))
        t = mx.array([0.5])
        cond = {
            'global': mx.random.normal((1, 5, 1024)),
            'proj': mx.random.normal((1, R ** 3, 1024)),
        }

        out = model(x, t, cond)
        mx.eval(out)

        assert out.shape == (1, 8, R, R, R)
        assert not mx.any(mx.isnan(out)).item(), "Output contains NaN"
        assert float(mx.mean(mx.abs(out))) > 1e-6, "Output is all zeros"

    def test_proj_linear_weights_are_nontrivial(self):
        """The proj_linear weights should be loaded (not zero-initialized)."""
        model = Pixal3DSparseStructureFlowModel(
            in_channels=8, out_channels=8,
            model_channels=1536, num_heads=12,
            num_blocks=30, mlp_hidden=8192,
            context_channels=1024, proj_in_channels=1024,
            resolution=16,
        )
        load_weights(model, SS_FLOW_CKPT, verbose=False)

        # Check first block's proj_linear is not all zeros
        proj_w = model.blocks[0].cross_attn.proj_linear.weight
        mx.eval(proj_w)
        w_mag = float(mx.mean(mx.abs(proj_w)))
        assert w_mag > 1e-6, f"proj_linear weight is trivial (mean abs = {w_mag})"
