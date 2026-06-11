"""MoGe-2 camera estimation for Pixal3D-MLX.

Estimates camera intrinsics (FOV) from a single image using MoGe-2,
matching the upstream Pixal3D camera conditioning pipeline. This is a
PyTorch/MPS sidecar — MoGe runs on MPS, then results are passed to
the MLX pipeline as plain Python floats.

The model loads, infers, and unloads before the MLX pipeline starts,
so MoGe and the flow models never share GPU memory.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np


# Match upstream Pixal3D inference.py: ViT-L for camera estimation.
DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl"


def estimate_camera_params(
    image_path: str | Path,
    *,
    model_name: str = DEFAULT_MOGE_MODEL,
    device: str = "mps",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> dict:
    """Estimate camera parameters from an image using MoGe-2.

    Matches upstream Pixal3D's get_camera_params_wild_moge() exactly:
    load MoGe, infer intrinsics, compute FOV and distance.

    Args:
        image_path: Path to the input image.
        model_name: HuggingFace model name for MoGe.
        device: PyTorch device for MoGe inference.
        mesh_scale: Mesh scale factor for distance computation.
        extend_pixel: Pixel extension for distance computation.
        image_resolution: Target resolution for distance computation.

    Returns:
        dict with 'camera_angle_x' (radians), 'distance', 'mesh_scale'.
    """
    import torch
    from PIL import Image

    t0 = time.perf_counter()

    # Load MoGe model.
    print(f"  Loading MoGe-2 ({model_name})...", flush=True)
    from moge.model import import_model_class_by_version
    moge_model = (
        import_model_class_by_version("v2")
        .from_pretrained(model_name)
        .to(device)
        .eval()
    )
    t_load = time.perf_counter() - t0
    print(f"  MoGe loaded ({t_load:.1f}s)", flush=True)

    # Load and preprocess image (matching upstream exactly).
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)

    # Infer intrinsics.
    t_infer = time.perf_counter()
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()

    # Extract FOV from normalized focal length.
    fx_normalized = float(intrinsics[0, 0])
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    # Compute distance using the same geometry as upstream.
    distance = _compute_distance_from_fov(
        camera_angle_x,
        mesh_scale=mesh_scale,
        image_resolution=image_resolution,
        extend_pixel=extend_pixel,
    )

    t_total = time.perf_counter() - t0
    print(
        f"  MoGe camera: FOV={math.degrees(camera_angle_x):.1f}°, "
        f"distance={distance:.4f} ({t_total:.1f}s total)",
        flush=True,
    )

    # Unload MoGe to free GPU memory before MLX pipeline starts.
    del moge_model, output, image_tensor
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "camera_angle_x": camera_angle_x,
        "distance": distance,
        "mesh_scale": mesh_scale,
    }


def _compute_distance_from_fov(
    camera_angle_x: float,
    mesh_scale: float = 1.0,
    image_resolution: int = 512,
    extend_pixel: int = 0,
) -> float:
    """Compute camera distance from FOV.

    Pure math, no PyTorch dependency. Matches upstream Pixal3D's
    distance_from_fov() for the standard grid_point=[-1, 0, 0].
    """
    # Rotation matrix (Blender convention)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    grid_point = np.array([-1.0, 0.0, 0.0]) @ rotation.T
    grid_point = grid_point / mesh_scale / 2.0

    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0

    xt = 0 - extend_pixel
    x_ndc = xt - image_resolution / 2.0

    xw, yw = grid_point[0], grid_point[1]
    distance = f_pixels * xw / x_ndc - yw
    return float(distance)
