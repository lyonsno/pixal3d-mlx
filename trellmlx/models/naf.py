"""NAF (Neural Attention Fields) feature upsampler — MLX port.

Upsamples VFM (DINOv3) patch features to higher resolution using
cross-attention with the input image as guidance.

Original: valeoai/NAF (PyTorch + natten CUDA)
This port replaces neighborhood attention with masked full attention.
For grids up to 64×64 (4096 positions), this is fast enough.

Architecture:
    ImageEncoder: dual Conv2d paths (1×1 + 3×3) → pool → RoPE
    CrossAttention: upsample K/V → neighborhood attention → output
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── RoPE ──

class NAFRoPE(nn.Module):
    """2D Rotary Position Embeddings for NAF.

    Matches valeoai/NAF src/layers/rope.py with base=100, rescale=2.0,
    normalize_coords="separate".
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        self.num_heads = num_heads
        D_head = embed_dim // num_heads
        self.D_head = D_head
        # Periods: base^(2*i / (D//2)) for i in 0..D//4-1
        periods = base ** (2 * np.arange(D_head // 4, dtype=np.float32) / (D_head // 2))
        self.periods = mx.array(periods)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 2D RoPE to spatial features.

        Args:
            x: [B, C, H, W] (channels-first)

        Returns:
            [B, C, H, W] with RoPE applied
        """
        B, C, H, W = x.shape
        n = self.num_heads
        d = C // n

        # Reshape to [B, n, H*W, d]
        x = x.reshape(B, n, d, H * W).transpose(0, 1, 3, 2)  # [B, n, HW, d]

        # Create coordinates in [-1, 1]
        coords_h = (mx.arange(H).astype(mx.float32) + 0.5) / H * 2 - 1
        coords_w = (mx.arange(W).astype(mx.float32) + 0.5) / W * 2 - 1
        grid_h, grid_w = mx.meshgrid(coords_h, coords_w, indexing='ij')
        coords = mx.stack([grid_h.reshape(-1), grid_w.reshape(-1)], axis=-1)  # [HW, 2]

        # Compute angles: [HW, 2, D//4] → [HW, D//2] → [HW, D]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.reshape(coords.shape[0], -1)  # [HW, D//2]
        angles = mx.concatenate([angles, angles], axis=-1)  # [HW, D]

        cos = mx.cos(angles)
        sin = mx.sin(angles)

        # Apply RoPE: x * cos + rotate_half(x) * sin
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        x = x * cos + rotated * sin

        # Reshape back to [B, C, H, W]
        x = x.transpose(0, 1, 3, 2).reshape(B, C, H, W)
        return x


# ── Convolution blocks ──

def _reflect_pad(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad spatial dims of NHWC tensor.

    MLX doesn't have mx.flip or mode='reflect', so we use reverse slicing.
    For pad=1 on a [B, H, W, C] tensor with H=4:
        top = x[:, 1:2, :, :]  (row 1, reversed = row 1)
        bottom = x[:, -2:-1, :, :]  (row H-2, reversed = row H-2)
    """
    if pad == 0:
        return x
    # Reflect-pad H: mirror rows [1..pad] at top, [H-pad-1..H-2] at bottom
    top = x[:, pad:0:-1, :, :]        # rows pad, pad-1, ..., 1 (reversed)
    bottom = x[:, -2:-(pad+2):-1, :, :]  # rows H-2, H-3, ..., H-pad-1
    x = mx.concatenate([top, x, bottom], axis=1)
    # Reflect-pad W: same for columns
    left = x[:, :, pad:0:-1, :]
    right = x[:, :, -2:-(pad+2):-1, :]
    x = mx.concatenate([left, x, right], axis=2)
    return x


class EncBlock(nn.Module):
    """Residual conv block with GroupNorm + SiLU + reflect padding."""

    def __init__(self, channels: int, kernel_size: int = 3, num_groups: int = 8):
        super().__init__()
        self.pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=0)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = self.conv1(_reflect_pad(h, self.pad))
        h = nn.silu(self.norm2(h))
        h = self.conv2(_reflect_pad(h, self.pad))
        return h


class ReflectConv2d(nn.Module):
    """Conv2d with reflect padding (matching upstream padding_mode='reflect')."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def __call__(self, x):
        return self.conv(_reflect_pad(x, self.pad))


def make_encoder(in_channels: int, out_channels: int, kernel_size: int = 1,
                 ks_res: int = 1, num_layers: int = 2):
    """Build encoder: ReflectConv2d → EncBlock × num_layers."""
    layers = [ReflectConv2d(in_channels, out_channels, kernel_size)]
    for _ in range(num_layers):
        layers.append(EncBlock(out_channels, kernel_size=ks_res))
    return layers


# ── Neighborhood attention as masked full attention ──

def neighborhood_attention(q, k, v, kernel_size, scale, dilation=(1, 1)):
    """Neighborhood attention via gather-based windowed attention.

    Each query attends only to its K×K spatial neighbors, with optional
    dilation. With dilation=d, neighbors are spaced d apart, giving a
    receptive field of (K*d - d + 1) × (K*d - d + 1).

    O(N × K²) instead of O(N²). Works for any grid size.

    Args:
        q: [B, H, X, Y, Dq] queries
        k: [B, H, X, Y, Dq] keys
        v: [B, H, X, Y, Dv] values (may differ from Dq)
        kernel_size: (Kh, Kw) or int neighborhood size
        scale: attention scale factor
        dilation: (dh, dw) dilation factor for the neighborhood window

    Returns:
        [B, H, X, Y, Dv] output
    """
    B, nH, X, Y, Dq = q.shape
    Dv = v.shape[-1]
    Kh, Kw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    half_kh, half_kw = Kh // 2, Kw // 2

    # With dilation, the effective padding needed is half_k * dilation
    pad_h = half_kh * dh
    pad_w = half_kw * dw

    # Pad K and V so boundary positions have full neighborhoods
    k_padded = mx.pad(k, [(0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)],
                       mode="edge")
    v_padded = mx.pad(v, [(0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)],
                       mode="edge")

    # Process in row chunks to control peak memory.
    # Each chunk allocates [B, H, chunk, Y, K², D] window tensors.
    # At chunk=64, Y=512, K²=81, D=1024, H=4: ~27GB per chunk.
    # At chunk=8: ~3.4GB. At chunk=4: ~1.7GB.
    CHUNK_ROWS = max(1, min(X, 16))

    out_chunks = []
    for row_start in range(0, X, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, X)
        chunk_x = row_end - row_start

        # Extract neighbor windows with dilation
        # Neighbor offsets: di * dh for i in range(Kh), dj * dw for j in range(Kw)
        k_windows = []
        v_windows = []
        for di in range(Kh):
            for dj in range(Kw):
                offset_h = di * dh
                offset_w = dj * dw
                k_windows.append(k_padded[:, :, row_start + offset_h:row_start + offset_h + chunk_x, offset_w:offset_w + Y, :])
                v_windows.append(v_padded[:, :, row_start + offset_h:row_start + offset_h + chunk_x, offset_w:offset_w + Y, :])

        k_nbr = mx.stack(k_windows, axis=4)
        v_nbr = mx.stack(v_windows, axis=4)

        q_chunk = q[:, :, row_start:row_end, :, :]
        q_exp = q_chunk[:, :, :, :, None, :]

        attn = (q_exp @ k_nbr.transpose(0, 1, 2, 3, 5, 4)) * scale
        attn = mx.softmax(attn, axis=-1)
        chunk_out = (attn @ v_nbr).squeeze(4)
        out_chunks.append(chunk_out)
        mx.eval(chunk_out)

    return mx.concatenate(out_chunks, axis=2)


# ── Cross Attention ──

class NAFCrossAttention(nn.Module):
    """Cross-attention with neighborhood attention.

    Q comes from upsampled image features, K/V from DINOv3 patch features.
    K and V are nearest-upsampled to Q resolution, then neighborhood
    attention is applied.
    """

    def __init__(self, dim: int, num_heads: int, kernel_size=(9, 9)):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = (dim // num_heads) ** -0.5

    def __call__(self, q, k, v, image=None):
        """
        Args:
            q: [B, C, Hq, Wq] image-encoded features (query resolution)
            k: [B, C, Hk, Wk] image-encoded features (key resolution, ≤ Hq)
            v: [B, Cv, Hk, Wk] DINOv3 features (value, same res as k)

        Returns:
            [B, Cv, Hq, Wq] upsampled features
        """
        B, C, Hq, Wq = q.shape
        Hk, Wk = k.shape[2], k.shape[3]
        n = self.num_heads

        # Reshape Q to [B, n, Hq, Wq, d]
        dq = C // n
        q = q.reshape(B, n, dq, Hq, Wq).transpose(0, 1, 3, 4, 2)  # [B, n, Hq, Wq, dq]

        # Upsample K to query resolution via nearest
        # K: [B, C, Hk, Wk] → [B, C, Hq, Wq]
        k_up = mx.repeat(mx.repeat(k, Hq // Hk, axis=2), Wq // Wk, axis=3)
        k_up = k_up.reshape(B, n, dq, Hq, Wq).transpose(0, 1, 3, 4, 2)

        # Upsample V to query resolution via nearest
        # V may have different channel dim than Q/K (e.g. 1024 vs 256)
        Cv = v.shape[1]
        v_up = mx.repeat(mx.repeat(v, Hq // Hk, axis=2), Wq // Wk, axis=3)
        # For neighborhood attention, V needs same head structure as Q
        # Split V into same number of heads, each with Cv//n channels
        dv = Cv // n
        v_up = v_up.reshape(B, n, dv, Hq, Wq).transpose(0, 1, 3, 4, 2)

        # Dilation from resolution ratio — each query's K×K window is spaced
        # dilation apart, covering the full original-resolution neighborhood
        dilation = (Hq // Hk, Wq // Wk)

        # Neighborhood attention with dilation
        out = neighborhood_attention(q, k_up, v_up, self.kernel_size, self.scale, dilation=dilation)

        # Reshape back to [B, Cv, Hq, Wq]
        out = out.transpose(0, 1, 4, 2, 3).reshape(B, Cv, Hq, Wq)
        return out


# ── NAF Model ──

class NAF(nn.Module):
    """Neural Attention Fields feature upsampler.

    Takes a guide image and low-res VFM features, produces high-res features
    via cross-attention with neighborhood masking.

    Config (from pretrained):
        dim: 256
        heads_attn: 4
        heads_rope: 4
        kernel_size: 9
        img_layers: 2
        rope_base: 100.0
        rope_rescale: 2.0
    """

    def __init__(
        self,
        dim: int = 256,
        heads_attn: int = 4,
        heads_rope: int = 4,
        kernel_size: int = 9,
        img_layers: int = 2,
        rope_base: float = 100.0,
    ):
        super().__init__()
        self.dim = dim

        # Image encoder: dual path (1×1 + 3×3) → concat → pool → RoPE
        self.encoder_1x1 = make_encoder(3, dim // 2, kernel_size=1, ks_res=1, num_layers=img_layers)
        self.encoder_3x3 = make_encoder(3, dim // 2, kernel_size=3, ks_res=3, num_layers=img_layers)
        self.rope = NAFRoPE(dim, num_heads=heads_rope, base=rope_base)

        # Cross-attention upsampler
        self.upsampler = NAFCrossAttention(dim=dim, num_heads=heads_attn, kernel_size=(kernel_size, kernel_size))

    def _run_encoder(self, layers, x):
        for layer in layers:
            x = layer(x)
        return x

    def _encode_image(self, image, output_size):
        """Encode guide image to feature map at output_size resolution.

        Args:
            image: [B, H, W, 3] in [0, 1] (MLX channels-last)
            output_size: (H', W') target spatial size

        Returns:
            [B, dim, H', W'] encoded features with RoPE
        """
        # Pre-downsample if image is >4x target (matches upstream guard)
        oh, ow = output_size
        if image.shape[1] > 4 * oh or image.shape[2] > 4 * ow:
            target_h = min(image.shape[1], 4 * oh, 4 * ow)
            target_w = min(image.shape[2], 4 * ow, 4 * oh)
            image = nn.Upsample(scale_factor=target_h / image.shape[1],
                                mode="linear", align_corners=False)(image)

        # Run encoders in NHWC
        enc_1x1 = self._run_encoder(self.encoder_1x1, image)  # [B, H, W, dim//2]
        enc_3x3 = self._run_encoder(self.encoder_3x3, image)  # [B, H, W, dim//2]
        x = mx.concatenate([enc_1x1, enc_3x3], axis=-1)  # [B, H, W, dim]

        # Adaptive avg pool: convert to BCHW, pool, convert back
        x_bchw = x.transpose(0, 3, 1, 2)  # [B, dim, H, W]
        # Simple pooling via reshape+mean for exact output_size
        B, C, H, W = x_bchw.shape
        oh, ow = output_size
        x_pooled = x_bchw.reshape(B, C, oh, H // oh, ow, W // ow).mean(axis=(3, 5))

        # Apply RoPE (operates on BCHW)
        x_rope = self.rope(x_pooled)  # [B, dim, oh, ow]

        return x_rope

    def __call__(self, image, features, output_size):
        """Upsample VFM features using image guidance.

        Args:
            image: [B, 3, H, W] guide image in [0, 1] (BCHW for compat with upstream)
            features: [B, C, h, w] low-res VFM features (BCHW)
            output_size: (H', W') target resolution

        Returns:
            [B, C, H', W'] upsampled features (BCHW)
        """
        # Convert image from BCHW to NHWC for MLX Conv2d
        image_nhwc = image.transpose(0, 2, 3, 1)  # [B, H, W, 3]

        # Encode image at output resolution
        x = self._encode_image(image_nhwc, output_size)  # [B, dim, H', W'] BCHW

        # Q = encoded image features at output resolution
        queries = x

        # K = encoded image features pooled to feature resolution
        # adaptive_avg_pool2d to features spatial size
        Hf, Wf = features.shape[2], features.shape[3]
        Hq, Wq = output_size
        B, C = x.shape[0], x.shape[1]
        keys = x.reshape(B, C, Hf, Hq // Hf, Wf, Wq // Wf).mean(axis=(3, 5))

        # V = input features (DINOv3 patch features)
        values = features

        # Cross-attention with neighborhood masking
        out = self.upsampler(queries, keys, values)

        return out
