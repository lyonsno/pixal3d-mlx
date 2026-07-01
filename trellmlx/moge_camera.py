"""MoGe-2 camera estimation for Pixal3D-MLX.

Estimates camera intrinsics (FOV) from a single image using MoGe-2,
matching the upstream Pixal3D camera conditioning pipeline.

Two backends:
  - MLX (default): pure MLX MoGe-2 port used by generate_pixal3d.py
    unless a different backend is requested.
  - PyTorch/MPS (--pytorch-moge): optional reference backend matching
    upstream Pixal3D. MoGe loads on MPS, infers, and unloads before the
    MLX pipeline starts.

The MoGe model and weights now live in the standalone moge-mlx package.
"""

# Re-export from moge-mlx so existing pixal3d-mlx call sites keep working.
from moge_mlx.camera import (  # noqa: F401
    estimate_camera_params,
    estimate_camera_params_mlx,
    _compute_distance_from_fov,
)
