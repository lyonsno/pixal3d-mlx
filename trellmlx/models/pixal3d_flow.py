"""Pixal3D Flow Models — projection-conditioned variants of TRELLIS.2 flow models.

These models are architecturally identical to the TRELLIS.2 flow models
(SparseStructureFlowModel, SLatFlowModel) but replace standard cross-attention
with ProjectAttention: cross_attn(global) + proj_linear(projected).

The weight mapping is:
    blocks.N.cross_attn.cross_attn_block.* → inner cross-attention weights
    blocks.N.cross_attn.proj_linear.*      → per-block projection linear
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..modules.norm import LayerNorm32
from ..modules.attention import scaled_dot_product_attention, MultiHeadRMSNorm
from ..modules.rope import apply_rope, build_rope_phases
from ..modules.proj_attention import ProjectAttention
from .sparse_structure_flow import (
    TimestepEmbedder,
    MultiHeadAttention,
    FeedForward,
    _layernorm_noaffine,
)


class ProjModulatedBlock(nn.Module):
    """DiT block with ProjectAttention instead of plain cross-attention.

    Weight name mapping to Pixal3D checkpoints:
        self_attn.*                          → blocks.N.self_attn.*
        cross_attn.cross_attn_block.*        → blocks.N.cross_attn.cross_attn_block.*
        cross_attn.proj_linear.*             → blocks.N.cross_attn.proj_linear.*
        norm2.*                              → blocks.N.norm2.*
        mlp.*                                → blocks.N.mlp.*
        modulation                           → blocks.N.modulation
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_channels: int,
        mlp_hidden: int,
        proj_in_channels: int,
    ):
        super().__init__()
        self.channels = channels

        # Self-attention
        self.self_attn = MultiHeadAttention(channels, num_heads)

        # Cross-attention with projection wrapping
        self.norm2 = LayerNorm32(channels, affine=True)
        inner_cross_attn = MultiHeadAttention(channels, num_heads, context_channels)
        self.cross_attn = ProjectAttention(inner_cross_attn, channels, proj_in_channels)

        # FFN
        self.mlp = FeedForward(channels, mlp_hidden)

        # Per-block learned modulation bias
        self.modulation = mx.zeros((6 * channels,))

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context,  # dict with 'global' and 'proj'
        rope_phases: mx.array = None,
    ) -> mx.array:
        mod = mod + self.modulation

        C = self.channels
        shift_msa = mod[0*C:1*C]
        scale_msa = mod[1*C:2*C]
        gate_msa  = mod[2*C:3*C]
        shift_mlp = mod[3*C:4*C]
        scale_mlp = mod[4*C:5*C]
        gate_mlp  = mod[5*C:6*C]

        # Self-attention with adaLN-Zero + RoPE
        h = _layernorm_noaffine(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, rope_phases=rope_phases)
        h = h * gate_msa
        x = x + h

        # Cross-attention with projection
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        # FFN with adaLN-Zero
        h = _layernorm_noaffine(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x


class Pixal3DSparseStructureFlowModel(nn.Module):
    """Pixal3D Sparse Structure Flow Model with projection conditioning.

    Same architecture as SparseStructureFlowModel but uses ProjModulatedBlock
    and accepts context as a dict {'global': ..., 'proj': ...}.

    Config (from Pixal3D, 1.3B):
        in_channels: 8
        out_channels: 8
        model_channels: 1536
        num_heads: 12
        num_blocks: 30
        mlp_hidden: 8192
        context_channels: 1024 (DINOv3 global features)
        proj_in_channels: 1024 (DINOv3 projected features)
        resolution: 16
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        model_channels: int = 1536,
        num_heads: int = 12,
        num_blocks: int = 30,
        mlp_hidden: int = 8192,
        context_channels: int = 1024,
        proj_in_channels: int = 1024,
        resolution: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.resolution = resolution

        self.t_embedder = TimestepEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )

        self.blocks = [
            ProjModulatedBlock(
                model_channels, num_heads, context_channels,
                mlp_hidden, proj_in_channels,
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        x: mx.array,        # [B, in_channels, R, R, R]
        t: mx.array,        # [B] timestep
        cond,                # dict: {'global': [B, L, C], 'proj': [B, R^3, proj_C]}
    ) -> mx.array:
        B = x.shape[0]
        R = x.shape[2]

        t_emb = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb)

        x = x.reshape(B, self.in_channels, -1)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * R * R * R, self.in_channels)
        x = self.input_layer(x)

        head_dim = self.model_channels // self.num_heads
        rope_phases = build_rope_phases(R, head_dim)

        assert B == 1, f"Only B=1 supported for inference, got B={B}"
        for i, block in enumerate(self.blocks):
            x = block(x, mod[0], cond, rope_phases=rope_phases)
            if (i + 1) % 6 == 0:
                mx.eval(x)

        x = _layernorm_noaffine(x)
        x = self.out_layer(x)

        x = x.reshape(B, R, R, R, self.out_channels)
        x = x.transpose(0, 4, 1, 2, 3)
        return x


class Pixal3DSLatFlowModel(nn.Module):
    """Pixal3D Structured Latent Flow Model with projection conditioning.

    Same architecture as SLatFlowModel but uses ProjModulatedBlock
    and accepts context as a dict {'global': ..., 'proj': ...}.

    For SLat stages, projection features are pre-indexed by sparse
    coordinates in the pipeline — each token gets its own projection
    feature at its voxel position.

    Config (from Pixal3D, 1.3B):
        in_channels: 32
        out_channels: 32
        model_channels: 1536
        num_heads: 12
        num_blocks: 30
        mlp_hidden: 8192
        context_channels: 1024
        proj_in_channels: 1024
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        model_channels: int = 1536,
        num_heads: int = 12,
        num_blocks: int = 30,
        mlp_hidden: int = 8192,
        context_channels: int = 1024,
        proj_in_channels: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads

        self.t_embedder = TimestepEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )

        self.blocks = [
            ProjModulatedBlock(
                model_channels, num_heads, context_channels,
                mlp_hidden, proj_in_channels,
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        x: mx.array,        # [N, in_channels] sparse token features
        t: mx.array,        # [B] timestep
        cond,                # dict: {'global': [B, L, C], 'proj': [N, proj_C]}
        coords: mx.array = None,  # [N, 3] for RoPE
        concat_cond: mx.array = None,
    ) -> mx.array:
        N = x.shape[0]

        if concat_cond is not None:
            x = mx.concatenate([x, concat_cond], axis=-1)

        t_emb = self.t_embedder(t)
        mod = self.adaLN_modulation(t_emb)

        x = self.input_layer(x)

        rope_phases = None
        if coords is not None:
            rope_phases = self._coords_to_rope_phases(coords)

        for i, block in enumerate(self.blocks):
            x = block(x, mod[0], cond, rope_phases=rope_phases)
            if (i + 1) % 6 == 0:
                mx.eval(x)

        x = _layernorm_noaffine(x)
        x = self.out_layer(x)
        return x

    def _coords_to_rope_phases(self, coords: mx.array) -> mx.array:
        """Compute RoPE phases from sparse voxel coordinates."""
        import math

        freq_dim = self.head_dim // 2 // 3
        freqs = np.arange(freq_dim, dtype=np.float32) / freq_dim
        freqs = 1.0 / (10000.0 ** freqs)
        freqs = mx.array(freqs)

        coords_f = coords.astype(mx.float32)

        all_angles = []
        for d in range(3):
            angles = coords_f[:, d:d+1] * freqs[None, :]
            all_angles.append(angles)
        angles = mx.concatenate(all_angles, axis=-1)

        target = self.head_dim // 2
        if angles.shape[-1] < target:
            pad = mx.zeros((angles.shape[0], target - angles.shape[-1]))
            angles = mx.concatenate([angles, pad], axis=-1)

        cos_p = mx.cos(angles)
        sin_p = mx.sin(angles)
        return mx.stack([cos_p, sin_p], axis=-1)
