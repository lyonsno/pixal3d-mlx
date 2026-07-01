"""MoGe-2 camera estimation for Pixal3D-MLX.

Estimates camera intrinsics (FOV) from a single image using MoGe-2,
matching the upstream Pixal3D camera conditioning pipeline.

MLX backend only — pixal3d-mlx is a pure-MLX pipeline. The MoGe model
and weights live in the standalone moge-mlx package.

Flags:
  --no-moge: skip camera estimation, use fixed default FOV (~49°).
  --fov <radians>: manual FOV override (takes priority).
"""

# Re-export from moge-mlx so existing pixal3d-mlx call sites keep working.
from moge_mlx.camera import (  # noqa: F401
    estimate_camera_params_mlx,
    _compute_distance_from_fov,
)
