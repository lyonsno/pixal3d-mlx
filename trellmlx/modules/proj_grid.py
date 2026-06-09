"""3D Grid Projection for Pixal3D MLX port.

Projects a 3D grid of points to 2D image coordinates using camera parameters,
then samples features from DINOv3 patch feature maps at those locations.

This is the core spatial mechanism behind pixel-aligned back-projection:
explicit pixel-to-voxel correspondence via perspective projection.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _grid_sample_bilinear(fmap: mx.array, queries_ndc: mx.array) -> mx.array:
    """Sample features from a 2D feature map at NDC coordinates via bilinear interpolation.

    Args:
        fmap: [B, C, H, W] feature map.
        queries_ndc: [B, K, 2] coordinates in [-1, 1] NDC space.

    Returns:
        [B, C, K] sampled features.
    """
    B, C, H, W = fmap.shape
    K = queries_ndc.shape[1]

    # NDC [-1,1] → pixel coordinates [0, H-1] / [0, W-1]
    # align_corners=False convention: pixel centers at (i+0.5)/N * 2 - 1
    x = (queries_ndc[..., 0] + 1) * 0.5 * W - 0.5   # [B, K]
    y = (queries_ndc[..., 1] + 1) * 0.5 * H - 0.5   # [B, K]

    # Floor/ceil pixel indices
    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Interpolation weights
    wx = (x - x0.astype(mx.float32))[..., None]  # [B, K, 1]
    wy = (y - y0.astype(mx.float32))[..., None]  # [B, K, 1]

    # Clamp to valid range (border padding)
    x0 = mx.clip(x0, 0, W - 1)
    x1 = mx.clip(x1, 0, W - 1)
    y0 = mx.clip(y0, 0, H - 1)
    y1 = mx.clip(y1, 0, H - 1)

    # fmap is [B, C, H, W] — gather at 4 corners
    # Reshape to [B, C, H*W] for flat indexing
    fmap_flat = fmap.reshape(B, C, H * W)

    def _gather(yy, xx):
        idx = yy * W + xx  # [B, K]
        # Gather: for each batch b, pick fmap_flat[b, :, idx[b, k]]
        # idx shape [B, K] -> expand to [B, C, K] for take_along_axis
        idx_exp = mx.broadcast_to(idx[:, None, :], (B, C, K))
        return mx.take_along_axis(fmap_flat, idx_exp, axis=2)  # [B, C, K]

    f00 = _gather(y0, x0)
    f01 = _gather(y0, x1)
    f10 = _gather(y1, x0)
    f11 = _gather(y1, x1)

    # Bilinear interpolation: [B, C, K]
    # Transpose weights to [B, 1, K] for broadcasting
    wx = wx.transpose(0, 2, 1)  # [B, 1, K]
    wy = wy.transpose(0, 2, 1)  # [B, 1, K]

    result = (f00 * (1 - wx) * (1 - wy) +
              f01 * wx * (1 - wy) +
              f10 * (1 - wx) * wy +
              f11 * wx * wy)

    return result  # [B, C, K]


def project_points_to_image(
    points_3d: mx.array,
    transform_matrix: mx.array,
    camera_angle_x: mx.array,
    resolution: int = 518,
):
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: [B, N, 3] 3D points in [-1, 1].
        transform_matrix: [B, 4, 4] camera transform.
        camera_angle_x: [B] horizontal FOV in radians.
        resolution: image resolution.

    Returns:
        points_2d: [B, N, 2] image coordinates.
        depth: [B, N] depth values.
        valid_mask: [B, N] mask for valid points.
    """
    B = transform_matrix.shape[0]
    N = points_3d.shape[1]

    # Homogeneous coordinates [B, N, 4]
    ones = mx.ones((B, N, 1), dtype=points_3d.dtype)
    points_h = mx.concatenate([points_3d, ones], axis=-1)

    # World to camera: inverse of transform_matrix
    # Use numpy for the inverse (small 4x4 matrix, done once)
    tm_np = np.array(transform_matrix, dtype=np.float64)
    w2c_np = np.linalg.inv(tm_np)
    w2c = mx.array(w2c_np.astype(np.float32))

    # Transform to camera coords: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_cam = (points_h @ w2c.transpose(0, 2, 1))[..., :3]

    x_cam = points_cam[..., 0]
    y_cam = points_cam[..., 1]
    z_cam = points_cam[..., 2]

    # Depth (Blender camera faces -Z)
    depth = -z_cam

    # Camera intrinsics
    fov_half = camera_angle_x / 2.0
    focal_length = 16.0 / mx.tan(fov_half)
    f_pixels = (focal_length * resolution / 32.0)[:, None]  # [B, 1]

    # Perspective projection
    x_ndc = f_pixels * x_cam / (-z_cam + 1e-8)
    y_ndc = f_pixels * y_cam / (-z_cam + 1e-8)

    # Image coordinates
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0

    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)
    )

    points_2d = mx.stack([x_pixel, y_pixel], axis=-1)
    return points_2d, depth, valid_mask


class ProjGrid(nn.Module):
    """3D Grid Projection Module.

    Creates a grid of R^3 points in [-1, 1], projects them to 2D image
    coordinates using camera parameters, and samples DINOv3 patch features
    at those locations.

    Weight mapping: only has register buffers (grid_points, front_view_transform_matrix).
    """

    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

        # Create 3D grid points
        one_dim = np.linspace(-1, 1, grid_resolution).astype(np.float32)
        x, y, z = np.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = np.stack((x, y, z), axis=-1)

        # Rotation: align with Blender coordinate system
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        grid_points = grid_points @ rotation.T
        self._grid_points = mx.array(grid_points.reshape(-1, 3))

        # Default front view transform
        self._front_view_transform = mx.array(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32))

    def __call__(
        self,
        features_map: mx.array,
        camera_angle_x: mx.array,
        distance: mx.array,
        mesh_scale: mx.array,
        transform_matrix=None,
        BHWC: bool = True,
    ) -> mx.array:
        """Project 3D grid and sample features.

        Args:
            features_map: [B, H, W, C] if BHWC else [B, C, H, W].
            camera_angle_x: [B] FOV angle in radians.
            distance: [B] camera distance.
            mesh_scale: [B] mesh scale factor.
            transform_matrix: Optional [B, 4, 4] camera transform.
            BHWC: Whether features_map is BHWC format.

        Returns:
            [B, R^3, C] projected features.
        """
        if BHWC:
            B = features_map.shape[0]
        else:
            B = features_map.shape[0]

        # Scale grid points
        grid_points = mx.broadcast_to(
            self._grid_points[None], (B, self._grid_points.shape[0], 3)
        )
        grid_points = grid_points / mesh_scale[:, None, None] / 2.0

        # Build transform matrix (Pixal3D only uses front-view; custom transform
        # is not used at inference time, matching upstream's assert)
        assert transform_matrix is None, "Custom transform_matrix not supported"
        tm = mx.broadcast_to(
            self._front_view_transform[None], (B, 4, 4)
        )
        # Set camera distance — need mutable copy
        tm_np = np.array(tm)
        for b in range(B):
            tm_np[b, 1, 3] = -float(distance[b])
        tm = mx.array(tm_np.astype(np.float32))

        # Project to image coordinates
        image_points, _depth, _valid = project_points_to_image(
            grid_points, tm, camera_angle_x, self.image_resolution
        )

        # Normalize to [-1, 1] for grid_sample
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            # [B, H, W, C] -> [B, C, H, W]
            features_map = features_map.transpose(0, 3, 1, 2)

        # Sample features
        x = _grid_sample_bilinear(features_map, image_points_norm)  # [B, C, K]
        x = x.transpose(0, 2, 1)  # [B, K, C]

        return x
