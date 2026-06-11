"""MoGe-2 model in pure MLX.

Full port of microsoft/MoGe v2 (DINOv2-L backbone + ConvStack decoder heads)
for monocular geometry estimation on Apple Silicon without PyTorch.

Architecture:
  - DINOv2-L encoder: 24 blocks, 1024 dim, 16 heads, 14×14 patches
  - 4× Conv2d 1×1 output projections (sum intermediate features at layers 5,11,17,23)
  - ConvStack neck: 5-level conv decoder with ConvTranspose2d resamplers
  - Points head (ConvStack): 3-channel point map
  - Mask head (ConvStack): 1-channel mask
  - Scale head (MLP): metric scale from CLS token

Total: 326M params, ~1.3 GB float32.
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ============================================================================
# DINOv2-L Backbone
# ============================================================================

class DINOv2Attention(nn.Module):
    """Multi-head attention with fused QKV (DINOv2 style)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # [B, N, H, D] -> [B, H, N, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class DINOv2MLP(nn.Module):
    """GELU MLP (DINOv2 uses standard GELU, not SwiGLU)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class DINOv2Block(nn.Module):
    """DINOv2 transformer block with LayerNorm + LayerScale."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DINOv2Attention(dim, num_heads)
        self.ls1 = mx.zeros((dim,))  # layer scale gamma

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv2MLP(dim, int(dim * mlp_ratio))
        self.ls2 = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-L/14 backbone with learned position embeddings.

    Supports get_intermediate_layers() for multi-scale feature extraction.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_blocks: int = 24,
        patch_size: int = 14,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Embeddings
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, 1370, embed_dim))  # 1 CLS + 37*37 patches (518/14)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # Transformer blocks
        self.blocks = [DINOv2Block(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)]

        # Final norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _interpolate_pos_embed(self, x: mx.array, h: int, w: int) -> mx.array:
        """Interpolate position embeddings for arbitrary resolution.

        Args:
            x: [B, 1+H*W, D] token sequence (CLS prepended)
            h, w: number of patch rows/cols
        """
        num_patches = h * w
        num_pos = self.pos_embed.shape[1] - 1  # exclude CLS

        if num_patches == num_pos:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]  # [1, 1, D]
        patch_pos = self.pos_embed[:, 1:]  # [1, N, D]

        # Original grid size (assumes square training resolution)
        orig_size = int(math.sqrt(num_pos))

        # Reshape to 2D grid: [1, orig_size, orig_size, D] -> [1, D, orig_size, orig_size]
        patch_pos_2d = patch_pos.reshape(1, orig_size, orig_size, -1)
        # MLX Upsample expects channels-last: [B, H, W, C]
        # Use bilinear interpolation
        patch_pos_2d = _bilinear_resize(patch_pos_2d, h, w)
        patch_pos = patch_pos_2d.reshape(1, h * w, -1)

        return mx.concatenate([cls_pos, patch_pos], axis=1)

    def __call__(
        self,
        pixel_values: mx.array,
        token_h: int,
        token_w: int,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass returning final features + CLS token.

        Args:
            pixel_values: [B, H, W, 3] normalized image (channels-last)
            token_h, token_w: target patch grid size

        Returns:
            features: [B, D, token_h, token_w] spatial features
            cls_token: [B, D] CLS token
        """
        B = pixel_values.shape[0]

        # Resize to match target token count
        target_h = token_h * self.patch_size
        target_w = token_w * self.patch_size
        if pixel_values.shape[1] != target_h or pixel_values.shape[2] != target_w:
            pixel_values = _bilinear_resize(pixel_values, target_h, target_w)

        # Patch embed: [B, H, W, 3] -> [B, H/P, W/P, D]
        x = self.patch_embed(pixel_values)
        h_patches, w_patches = x.shape[1], x.shape[2]
        x = x.reshape(B, -1, self.embed_dim)  # [B, N, D]

        # Prepend CLS
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)

        # Add interpolated position embeddings
        pos = self._interpolate_pos_embed(x, h_patches, w_patches)
        x = x + pos

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # [B, D]
        patch_out = x[:, 1:]  # [B, N, D]

        return patch_out, cls_out

    def get_intermediate_layers(
        self,
        pixel_values: mx.array,
        token_h: int,
        token_w: int,
        layers: List[int],
    ) -> Tuple[List[mx.array], mx.array]:
        """Extract features at specific intermediate layers.

        Args:
            pixel_values: [B, H, W, 3]
            token_h, token_w: target patch grid
            layers: list of block indices to extract features from

        Returns:
            features: list of [B, D, token_h, token_w] features at each layer
            cls_token: [B, D] CLS token from final layer
        """
        B = pixel_values.shape[0]

        x = self.patch_embed(pixel_values)
        h_patches, w_patches = x.shape[1], x.shape[2]
        x = x.reshape(B, -1, self.embed_dim)

        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)
        pos = self._interpolate_pos_embed(x, h_patches, w_patches)
        x = x + pos

        intermediates = []
        layer_set = set(layers)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_set:
                # DINOv2 applies norm to each intermediate layer output
                normed = self.norm(x)
                patch = normed[:, 1:]
                spatial = patch.reshape(B, h_patches, w_patches, self.embed_dim)
                intermediates.append(spatial.transpose(0, 3, 1, 2))

        # CLS from final normed output
        x = self.norm(x)
        cls_out = x[:, 0]

        return intermediates, cls_out


# ============================================================================
# DINOv2 Encoder (backbone + output projections)
# ============================================================================

class DINOv2Encoder(nn.Module):
    """DINOv2 encoder with multi-scale output projections.

    Extracts features at intermediate layers [5, 11, 17, 23] and projects
    each through a 1×1 Conv2d, then sums them.
    """

    def __init__(self, embed_dim: int = 1024, intermediate_layers: List[int] = None):
        super().__init__()
        if intermediate_layers is None:
            intermediate_layers = [5, 11, 17, 23]
        self.intermediate_layers = intermediate_layers

        self.backbone = DINOv2Backbone(embed_dim=embed_dim)
        self.output_projections = [
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=True)
            for _ in range(len(intermediate_layers))
        ]

        # ImageNet normalization constants
        self.image_mean = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.image_std = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

    def __call__(
        self,
        image: mx.array,
        token_h: int,
        token_w: int,
    ) -> Tuple[mx.array, mx.array]:
        """Encode image to features.

        Args:
            image: [B, H, W, 3] in [0, 1] range (channels-last)
            token_h, token_w: target patch grid

        Returns:
            features: [B, D, token_h, token_w] summed multi-scale features
            cls_token: [B, D]
        """
        # Resize image to match token grid before normalization.
        # Use PIL LANCZOS for the initial resize — it closely matches
        # PyTorch's F.interpolate(antialias=True) which the model was trained
        # with. MLX's bilinear lacks antialiasing and produces measurably
        # different results that compound through the 24 transformer blocks.
        target_h = token_h * self.backbone.patch_size
        target_w = token_w * self.backbone.patch_size
        if image.shape[1] != target_h or image.shape[2] != target_w:
            image = _pil_resize(image, target_h, target_w)

        # Normalize
        image = (image - self.image_mean) / self.image_std

        # Get intermediate features — pass pre-resized image
        intermediates, cls_token = self.backbone.get_intermediate_layers(
            image, token_h, token_w, self.intermediate_layers,
        )

        # Project and sum: each is [B, D, H, W] channels-first
        # Conv2d in MLX expects [B, H, W, C], so transpose
        features = None
        for proj, feat in zip(self.output_projections, intermediates):
            # [B, D, H, W] -> [B, H, W, D]
            feat_hw = feat.transpose(0, 2, 3, 1)
            projected = proj(feat_hw)
            # [B, H, W, D] -> [B, D, H, W]
            projected = projected.transpose(0, 3, 1, 2)
            if features is None:
                features = projected
            else:
                features = features + projected

        return features, cls_token


# ============================================================================
# Conv modules (ConvStack, Resampler, ResidualConvBlock)
# ============================================================================

def _replicate_pad2d(x: mx.array, padding: int) -> mx.array:
    """Replicate (edge) padding for 2D input.

    Args:
        x: [B, H, W, C] channels-last
        padding: number of pixels to pad on each side
    """
    if padding == 0:
        return x
    # MLX mx.pad with edge mode
    return mx.pad(x, [(0, 0), (padding, padding), (padding, padding), (0, 0)], mode="edge")


class ReplicateConv2d(nn.Module):
    """Conv2d with replicate padding (MLX doesn't support padding_mode)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if self.pad > 0:
            x = _replicate_pad2d(x, self.pad)
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """Residual conv block: ReLU + Conv + ReLU + Conv with skip connection.

    Upstream uses Identity norms (in_norm='none', hidden_norm='none' for this
    model), so the layers are: Identity, ReLU, Conv3x3, Identity, ReLU, Conv3x3.
    Weight keys are layers.2 and layers.5 (indexed through the Sequential).
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        # layers.2 and layers.5 in upstream (layers.0,1,3,4 are Identity/ReLU)
        self.layers_2 = ReplicateConv2d(in_channels, hidden_channels, 3, padding=1)
        self.layers_5 = ReplicateConv2d(hidden_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        skip = self.skip(x) if self.skip is not None else x

        h = nn.relu(x)
        h = self.layers_2(h)
        h = nn.relu(h)
        h = self.layers_5(h)

        return h + skip


class ConvTransposeResampler(nn.Module):
    """Upsample via ConvTranspose2d + Conv2d (matching upstream conv_transpose resampler)."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=scale_factor, stride=scale_factor, bias=True,
        )
        self.conv = ReplicateConv2d(out_channels, out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_t(x)
        x = self.conv(x)
        return x


class BilinearResampler(nn.Module):
    """Upsample via bilinear interpolation + Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ReplicateConv2d(in_channels, out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, H, W, C] channels-last in MLX
        B, H, W, C = x.shape
        x = _bilinear_resize(x, H * self.scale_factor, W * self.scale_factor)
        x = self.conv(x)
        return x


class ConvStack(nn.Module):
    """Multi-scale conv decoder stack.

    5 levels with input conv, residual blocks, resamplers, and output conv.
    """

    def __init__(
        self,
        dims_in: List[Optional[int]],
        dims_res: List[int],
        dims_out: List[Optional[int]],
        resampler_types: List[str],
        num_res_blocks: List[int],
    ):
        super().__init__()
        num_levels = len(dims_res)

        # Input 1×1 convs
        self.input_blocks = []
        for i in range(num_levels):
            if dims_in[i] is not None:
                self.input_blocks.append(nn.Conv2d(dims_in[i], dims_res[i], 1, bias=True))
            else:
                self.input_blocks.append(None)

        # Residual blocks (may be empty Sequential for 0 res blocks)
        self.res_blocks = []
        for i in range(num_levels):
            n = num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks
            if n > 0:
                blocks = [ResidualConvBlock(dims_res[i], dims_res[i], dims_res[i]) for _ in range(n)]
                self.res_blocks.append(blocks)
            else:
                self.res_blocks.append([])

        # Resamplers (between levels)
        self.resamplers = []
        for i in range(num_levels - 1):
            rtype = resampler_types[i] if isinstance(resampler_types, list) else resampler_types
            if rtype == "conv_transpose":
                self.resamplers.append(ConvTransposeResampler(dims_res[i], dims_res[i + 1]))
            elif rtype == "bilinear":
                self.resamplers.append(BilinearResampler(dims_res[i], dims_res[i + 1]))
            else:
                raise ValueError(f"Unsupported resampler: {rtype}")

        # Output 1×1 convs
        self.output_blocks = []
        for i in range(num_levels):
            if dims_out[i] is not None:
                self.output_blocks.append(nn.Conv2d(dims_res[i], dims_out[i], 1, bias=True))
            else:
                self.output_blocks.append(None)

    def __call__(self, in_features: List[mx.array]) -> List[mx.array]:
        """Forward pass through conv stack.

        Args:
            in_features: list of [B, H_i, W_i, C_i] features per level (channels-last)

        Returns:
            list of output features per level
        """
        out_features = []
        x = None

        for i in range(len(self.res_blocks)):
            # Input projection
            if self.input_blocks[i] is not None:
                feat = self.input_blocks[i](in_features[i])
            else:
                feat = in_features[i]

            # Accumulate
            if x is None:
                x = feat
            else:
                x = x + feat

            # Residual blocks
            for block in self.res_blocks[i]:
                x = block(x)

            # Output projection
            if self.output_blocks[i] is not None:
                out_features.append(self.output_blocks[i](x))
            else:
                out_features.append(x)

            # Resample to next level
            if i < len(self.resamplers):
                x = self.resamplers[i](x)

        return out_features


# ============================================================================
# Scale Head (MLP)
# ============================================================================

class ScaleHead(nn.Module):
    """MLP head for metric scale prediction from CLS token."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# Full MoGe-2 Model
# ============================================================================

class MoGeModel(nn.Module):
    """MoGe-2 monocular geometry estimation model in MLX.

    Full architecture:
    - DINOv2-L encoder with multi-scale projection
    - ConvStack neck (shared features)
    - Points head (3-ch point map)
    - Mask head (1-ch mask)
    - Scale head (metric scale from CLS)
    """

    def __init__(self):
        super().__init__()
        D = 1024  # DINOv2-L embed dim

        self.encoder = DINOv2Encoder(embed_dim=D)

        # Neck: 5 levels, actual dims from weights: [1024, 256, 128, 64, 32]
        # Input at level 0 is D+2 (features + UV), levels 1-4 are 2 (UV only)
        neck_dims = [1024, 256, 128, 64, 32]
        self.neck = ConvStack(
            dims_in=[D + 2, 2, 2, 2, 2],
            dims_res=neck_dims,
            dims_out=[None, None, None, None, None],
            resampler_types=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
            num_res_blocks=[0, 2, 2, 2, 0],
        )

        # Points head: input dims match neck res dims, head res dims also match
        head_dims = [1024, 256, 128, 64, 32]
        self.points_head = ConvStack(
            dims_in=[neck_dims[0], neck_dims[1], neck_dims[2], neck_dims[3], neck_dims[4]],
            dims_res=head_dims,
            dims_out=[None, None, None, None, 3],
            resampler_types=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
            num_res_blocks=[0, 1, 1, 1, 0],
        )

        # Mask head: same structure as points
        self.mask_head = ConvStack(
            dims_in=[neck_dims[0], neck_dims[1], neck_dims[2], neck_dims[3], neck_dims[4]],
            dims_res=head_dims,
            dims_out=[None, None, None, None, 1],
            resampler_types=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
            num_res_blocks=[0, 1, 1, 1, 0],
        )

        # Scale head: CLS token -> metric scale
        self.scale_head = ScaleHead(D, D, 1)

        # Remap output config — MoGe-2-vitl checkpoint uses "exp"
        self.remap_output = "exp"

    def _normalized_uv(self, h: int, w: int, aspect_ratio: float) -> mx.array:
        """Generate normalized UV coordinates matching upstream normalized_view_plane_uv().

        Upstream formula (geometry_torch.py):
          span_x = aspect_ratio / sqrt(1 + aspect_ratio^2)
          span_y = 1 / sqrt(1 + aspect_ratio^2)
          u = linspace(-span_x * (w-1)/w, +span_x * (w-1)/w, w)
          v = linspace(-span_y * (h-1)/h, +span_y * (h-1)/h, h)

        Returns [H, W, 2] UV map.
        """
        diag = math.sqrt(1 + aspect_ratio ** 2)
        span_u = aspect_ratio / diag
        span_v = 1.0 / diag

        u = mx.linspace(-span_u * (w - 1) / w, span_u * (w - 1) / w, w)
        v = mx.linspace(-span_v * (h - 1) / h, span_v * (h - 1) / h, h)
        grid_u, grid_v = mx.meshgrid(u, v, indexing="xy")

        return mx.stack([grid_u, grid_v], axis=-1)

    def forward(
        self,
        image: mx.array,
        num_tokens: int = 3600,
    ) -> Dict[str, mx.array]:
        """Neural network forward pass.

        Args:
            image: [B, H, W, 3] in [0, 1] range (channels-last)
            num_tokens: target number of ViT tokens

        Returns:
            dict with 'points', 'normal', 'mask', 'metric_scale'
        """
        B = image.shape[0]
        img_h, img_w = image.shape[1], image.shape[2]
        aspect_ratio = img_w / img_h

        # Compute token grid
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # Encoder
        features, cls_token = self.encoder(image, base_h, base_w)
        # features: [B, D, base_h, base_w] channels-first
        # Convert to channels-last for our conv modules
        features_hw = features.transpose(0, 2, 3, 1)  # [B, H, W, D]

        # Build multi-scale UV features
        in_features = [None] * 5
        in_features[0] = features_hw  # will have UV appended below

        for level in range(5):
            h = base_h * (2 ** level)
            w = base_w * (2 ** level)
            uv = self._normalized_uv(h, w, aspect_ratio)  # [H, W, 2]
            uv = mx.broadcast_to(uv[None], (B, h, w, 2))

            if in_features[level] is None:
                in_features[level] = uv
            else:
                in_features[level] = mx.concatenate([in_features[level], uv], axis=-1)

        # Neck
        neck_features = self.neck(in_features)

        # Heads
        points_out = self.points_head(neck_features)[-1]  # last level
        mask_out = self.mask_head(neck_features)[-1]
        metric_scale = self.scale_head(cls_token)

        # Resize to original resolution
        points_out = _bilinear_resize(points_out, img_h, img_w)
        mask_out = _bilinear_resize(mask_out, img_h, img_w)

        # Remap output — MoGe-2-vitl uses 'exp' mode:
        #   xy' = xy * exp(z),  z' = exp(z)
        # This makes z always positive (the network outputs log-depth)
        # and scales xy by depth (the network outputs normalized lateral coords).
        if self.remap_output == "exp":
            xy = points_out[..., :2]
            z = mx.exp(points_out[..., 2:3])
            points_out = mx.concatenate([xy * z, z], axis=-1)
        elif self.remap_output == "sinh":
            points_out = mx.sinh(points_out)
        elif self.remap_output == "sinh_exp":
            xy = mx.sinh(points_out[..., :2])
            z = mx.exp(points_out[..., 2:3])
            points_out = mx.concatenate([xy, z], axis=-1)
        # 'linear' is identity (no-op)

        mask = mx.sigmoid(mask_out[..., 0])  # [B, H, W]
        metric_scale = mx.exp(metric_scale[:, 0])  # [B]

        return {
            "points": points_out,
            "mask": mask,
            "metric_scale": metric_scale,
        }

    def infer(
        self,
        image: mx.array,
        num_tokens: int = 3600,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
    ) -> Dict[str, mx.array]:
        """User-friendly inference matching upstream MoGe API.

        Args:
            image: [3, H, W] or [B, 3, H, W] in [0, 1] — NOTE: channels-first
                   to match upstream PyTorch API. Internally converts to MLX
                   channels-last.
            num_tokens: number of ViT tokens (default uses resolution_level)
            resolution_level: 0-9, controls token count

        Returns:
            dict with 'points', 'depth', 'intrinsics', 'mask'
        """
        # Handle channels-first input (PyTorch convention)
        if image.ndim == 3:
            image = image[None]  # add batch
        # [B, C, H, W] -> [B, H, W, C]
        image = image.transpose(0, 2, 3, 1)

        img_h, img_w = image.shape[1], image.shape[2]
        aspect_ratio = img_w / img_h

        # Compute num_tokens from resolution_level
        min_tokens, max_tokens = 1200, 3600
        num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward
        output = self.forward(image, num_tokens=num_tokens)
        points = output["points"]
        mask = output["mask"]
        metric_scale = output["metric_scale"]

        # Post-processing: recover focal and shift
        mask_binary = mask > 0.5

        # Recover focal and shift from point map (numpy-side, matches upstream)
        focal, shift = _recover_focal_shift_np(points, mask_binary, img_h, img_w, aspect_ratio)

        # Compute intrinsics
        diag = math.sqrt(1 + aspect_ratio ** 2)
        fx = focal / 2 * diag / aspect_ratio
        fy = focal / 2 * diag
        intrinsics = mx.array([
            [fx, 0, 0.5],
            [0, fy, 0.5],
            [0, 0, 1],
        ])

        # Apply shift
        points_shifted = points.at[..., 2].add(shift)

        # Update mask for negative depth
        mask_binary = mask_binary & (points_shifted[..., 2] > 0)

        depth = points_shifted[..., 2]

        # Force projection: recompute points from depth + intrinsics
        if force_projection:
            points_shifted = _depth_to_points(depth, intrinsics, img_h, img_w)

        # Apply metric scale
        points_shifted = points_shifted * metric_scale[..., None, None, None]
        depth = depth * metric_scale[..., None, None]

        # Apply mask
        if apply_mask:
            inf = mx.array(float("inf"))
            points_shifted = mx.where(mask_binary[..., None], points_shifted, inf)
            depth = mx.where(mask_binary, depth, inf)

        result = {
            "points": points_shifted[0],
            "depth": depth[0],
            "intrinsics": intrinsics,
            "mask": mask_binary[0],
        }
        return result


# ============================================================================
# Utility functions
# ============================================================================

def _pil_resize(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Resize using align_corners=False bilinear.

    For the encoder input resize, we now use _bilinear_resize which
    implements align_corners=False matching upstream F.interpolate.
    This wrapper exists for API compatibility.
    """
    return _bilinear_resize(x, target_h, target_w)


def _bilinear_resize(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear resize for [B, H, W, C] tensor (align_corners=False).

    Matches PyTorch's F.interpolate(align_corners=False) pixel coordinate
    convention: output pixel i maps to input coordinate (i + 0.5) * H/target_h - 0.5.
    """
    if x.shape[1] == target_h and x.shape[2] == target_w:
        return x

    B, H, W, C = x.shape

    # align_corners=False: (i + 0.5) * src_size / dst_size - 0.5
    y_coords = (mx.arange(target_h).astype(mx.float32) + 0.5) * H / target_h - 0.5
    x_coords = (mx.arange(target_w).astype(mx.float32) + 0.5) * W / target_w - 0.5
    grid_y, grid_x = mx.meshgrid(y_coords, x_coords, indexing="ij")

    # Bilinear interpolation
    y0 = mx.floor(grid_y).astype(mx.int32)
    x0 = mx.floor(grid_x).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H - 1)
    x1 = mx.minimum(x0 + 1, W - 1)
    y0 = mx.clip(y0, 0, H - 1)
    x0 = mx.clip(x0, 0, W - 1)

    fy = grid_y - y0.astype(mx.float32)
    fx = grid_x - x0.astype(mx.float32)

    # Gather corners for all batches
    # x shape: [B, H, W, C]
    w00 = ((1 - fy) * (1 - fx))[..., None]
    w01 = ((1 - fy) * fx)[..., None]
    w10 = (fy * (1 - fx))[..., None]
    w11 = (fy * fx)[..., None]

    result = (
        x[:, y0, x0] * w00 +
        x[:, y0, x1] * w01 +
        x[:, y1, x0] * w10 +
        x[:, y1, x1] * w11
    )
    return result


def _recover_focal_shift_np(
    points: mx.array,
    mask: mx.array,
    img_h: int,
    img_w: int,
    aspect_ratio: float,
) -> Tuple[float, float]:
    """Recover focal length and z-shift from predicted point map.

    Uses numpy least-squares, matching upstream MoGe's recover_focal_shift().
    Returns (focal, shift) as Python floats.
    """
    # Downsample for efficiency
    ds = 64
    points_np = np.array(points[0])  # [H, W, 3]
    mask_np = np.array(mask[0])  # [H, W]

    # Simple nearest downsample
    h_idx = np.linspace(0, img_h - 1, ds, dtype=int)
    w_idx = np.linspace(0, img_w - 1, ds, dtype=int)
    points_ds = points_np[np.ix_(h_idx, w_idx)]
    mask_ds = mask_np[np.ix_(h_idx, w_idx)]

    # UV coordinates (must match upstream normalized_view_plane_uv)
    diag = math.sqrt(1 + aspect_ratio ** 2)
    span_u = aspect_ratio / diag
    span_v = 1.0 / diag
    u = np.linspace(-span_u * (ds - 1) / ds, span_u * (ds - 1) / ds, ds)
    v = np.linspace(-span_v * (ds - 1) / ds, span_v * (ds - 1) / ds, ds)
    gu, gv = np.meshgrid(u, v)
    uv = np.stack([gu, gv], axis=-1)

    if mask_ds.any():
        pts = points_ds[mask_ds]
        uv_m = uv[mask_ds]
    else:
        pts = points_ds.reshape(-1, 3)
        uv_m = uv.reshape(-1, 2)

    if len(pts) < 2:
        return 1.0, 0.0

    # Nonlinear focal/shift recovery matching upstream exactly.
    # Solves: min |focal * xy / (z + shift) - uv| via Levenberg-Marquardt.
    from functools import partial
    from scipy.optimize import least_squares

    xy = pts[:, :2]
    z = pts[:, 2]

    def _residual(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (f * xy_proj - uv).ravel()

    try:
        solution = least_squares(
            partial(_residual, uv_m, xy, z), x0=0, ftol=1e-3, method="lm",
        )
        shift = float(solution["x"].squeeze())
        xy_proj = xy / (z + shift)[:, None]
        focal = float((xy_proj * uv_m).sum() / np.square(xy_proj).sum())
    except Exception:
        focal, shift = 1.0, 0.0

    return focal, shift


def _depth_to_points(
    depth: mx.array,
    intrinsics: mx.array,
    img_h: int,
    img_w: int,
) -> mx.array:
    """Convert depth map to point map using camera intrinsics.

    Args:
        depth: [B, H, W]
        intrinsics: [3, 3] normalized intrinsics

    Returns:
        [B, H, W, 3] point map
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = mx.linspace(0.5 / img_w, 1 - 0.5 / img_w, img_w)
    v = mx.linspace(0.5 / img_h, 1 - 0.5 / img_h, img_h)
    grid_u, grid_v = mx.meshgrid(u, v, indexing="xy")

    x = (grid_u - cx) / fx * depth
    y = (grid_v - cy) / fy * depth

    return mx.stack([x, y, depth], axis=-1)
