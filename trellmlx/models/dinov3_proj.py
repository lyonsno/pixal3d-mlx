"""DINOv3 Feature Extractor with View-Aligned Projection for Pixal3D MLX.

Wraps the existing DINOv3ViT backbone with ProjGrid to produce:
1. Global features (CLS + register tokens) for cross-attention
2. View-aligned projected features (3D grid → 2D → sampled) for projection attention

This is the Pixal3D-specific conditioning model. The DINOv3 backbone is
identical to trellis2mlx; only the projection grid and feature routing are new.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .dinov3 import DINOv3ViT, IMAGENET_MEAN, IMAGENET_STD
from ..modules.proj_grid import ProjGrid


class DinoV3ProjFeatureExtractor(nn.Module):
    """DINOv3 with view-aligned projection for Pixal3D conditioning.

    For each stage (SS, Shape 512, Shape 1024, Tex 1024), one instance
    is created with a different grid_resolution matching the stage's
    spatial resolution.

    Outputs:
        global_features: [B, 5, 1024] — CLS + 4 register tokens
        proj_features:   [B, R^3, D] — projected features
            D = embed_dim (1024) without upsampling
            D = embed_dim * 2 (2048) with bilinear upsampling (LR + HR concat)

    The per-block proj_linear lives in ProjectAttention, not here.
    """

    def __init__(
        self,
        dinov3_model: DINOv3ViT,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_bilinear_upsample: bool = False,
        upsample_target_size: int = 128,
    ):
        super().__init__()
        self.dinov3 = dinov3_model
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.patch_size = dinov3_model.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = dinov3_model.hidden_size
        self.num_prefix = dinov3_model.num_prefix_tokens  # 5 (CLS + 4 regs)
        self.use_bilinear_upsample = use_bilinear_upsample
        self.upsample_target_size = upsample_target_size

        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size,
        )

        # Output dimension: 1024 without upsample, 2048 with (LR + HR concat)
        self.proj_channels = self.embed_dim * 2 if use_bilinear_upsample else self.embed_dim

    def __call__(
        self,
        image: mx.array,
        camera_angle_x: mx.array,
        distance: mx.array,
        mesh_scale: mx.array,
        transform_matrix=None,
    ) -> Tuple[mx.array, mx.array]:
        """Extract view-aligned features.

        Args:
            image: [B, H, W, 3] in [0, 1], NOT normalized yet.
            camera_angle_x: [B] FOV in radians.
            distance: [B] camera distance.
            mesh_scale: [B] mesh scale factor.

        Returns:
            (global_features, proj_features):
                global: [B, 5, 1024]
                proj:   [B, R^3, 1024]
        """
        B = image.shape[0]

        # Normalize (ImageNet)
        image_norm = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Extract DINOv3 features
        z = self.dinov3(image_norm)  # [B, 1029, 1024]

        # Split into global (CLS + registers) and patch tokens
        z_global = z[:, :self.num_prefix]  # [B, 5, 1024]
        z_patches = z[:, self.num_prefix:]  # [B, 1024, 1024]

        # Reshape patches to spatial grid [B, h, w, D]
        z_spatial = z_patches.reshape(
            B, self.patch_number, self.patch_number, self.embed_dim
        )

        # LR projection: sample from DINOv3 patch feature map
        z_proj_lr = self.proj_grid(
            z_spatial,
            camera_angle_x,
            distance,
            mesh_scale,
            transform_matrix,
            BHWC=True,
        )  # [B, R^3, 1024]

        if self.use_bilinear_upsample:
            # Bilinear upsample: resize patch features to higher resolution
            # then sample again for HR features. Approximates NAF upsampling.
            # z_spatial is [B, h, w, D], upsample to [B, H', W', D]
            h, w = self.patch_number, self.patch_number
            target_h = target_w = self.upsample_target_size

            # Transpose to [B, D, h, w] for resize, then back
            z_bchw = z_spatial.transpose(0, 3, 1, 2)  # [B, D, h, w]

            # Bilinear resize via MLX — reshape each channel
            # Use numpy for the resize (small tensor, done once)
            z_np = np.array(z_bchw)  # [B, D, h, w]
            from scipy.ndimage import zoom
            scale_h = target_h / h
            scale_w = target_w / w
            z_hr_np = zoom(z_np, (1, 1, scale_h, scale_w), order=1)  # bilinear
            z_hr = mx.array(z_hr_np.astype(np.float32))  # [B, D, H', W']

            # Sample from upsampled features
            z_proj_hr = self.proj_grid(
                z_hr,
                camera_angle_x,
                distance,
                mesh_scale,
                transform_matrix,
                BHWC=False,  # [B, D, H', W']
            )  # [B, R^3, 1024]

            # Concatenate LR + HR (matching NAF output format)
            z_proj = mx.concatenate([z_proj_lr, z_proj_hr], axis=-1)  # [B, R^3, 2048]
        else:
            z_proj = z_proj_lr

        return z_global, z_proj


def preprocess_image(image: Image.Image, size: int = 512) -> mx.array:
    """Preprocess a PIL image for DINOv3.

    Args:
        image: PIL RGB image.
        size: Target size.

    Returns:
        [1, H, W, 3] MLX array in [0, 1].
    """
    image = image.resize((size, size), Image.LANCZOS)
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return mx.array(arr[None])  # [1, H, W, 3]
