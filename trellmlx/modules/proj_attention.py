"""Projection-based attention modules for Pixal3D MLX port.

Implements view-aligned projection conditioning: combines standard
cross-attention on global image features with a per-block linear projection
of spatially-projected DINOv3 features.

The key Pixal3D innovation — pixel-aligned back-projection — lives here.
"""

import mlx.core as mx
import mlx.nn as nn


class ProjectAttention(nn.Module):
    """Projection-based attention: cross-attn(global) + proj_linear(projected).

    Wraps a standard cross-attention module. Each transformer block owns
    its own proj_linear that projects DINOv3 features from proj_in_channels
    (e.g. 1024) to model_channels (e.g. 1536).

    Context is a dict:
        'global': [B, M, ctx_channels] — global image features (CLS + registers)
        'proj':   [B, N, proj_in_channels] — view-aligned projected features

    Weight mapping to Pixal3D checkpoints:
        cross_attn.cross_attn_block.* → the inner cross-attention weights
        cross_attn.proj_linear.*      → the per-block projection linear
    """

    def __init__(self, cross_attn_module, channels: int, proj_in_channels: int):
        """
        Args:
            cross_attn_module: An existing MultiHeadAttention for cross-attention.
            channels: Model channels (output dimension).
            proj_in_channels: Dimension of projected features (DINOv3 embed_dim).
        """
        super().__init__()
        self.cross_attn_block = cross_attn_module
        self.proj_linear = nn.Linear(proj_in_channels, channels)

    def __call__(self, x: mx.array, context) -> mx.array:
        """
        Args:
            x: Input from self-attention, [T, C] or [B, T, C].
            context: Dict with 'global' and 'proj' keys, or tuple.
        """
        if isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
        else:
            global_context, proj_context = context

        global_out = self.cross_attn_block(x, global_context)
        proj_out = self.proj_linear(proj_context)

        # For dense (SparseStructureFlow): x is [B*R^3, C], proj is [B, R^3, proj_in]
        # Need to reshape proj_out to match. proj_out after linear is [B, R^3, C].
        # When B=1, squeeze to [R^3, C] to match global_out.
        if proj_out.ndim == 3 and global_out.ndim == 2:
            # B=1 case: [1, N, C] -> [N, C]
            proj_out = proj_out.reshape(-1, proj_out.shape[-1])

        return proj_out + global_out
