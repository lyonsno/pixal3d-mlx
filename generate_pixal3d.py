"""Generate a 3D mesh from a single image using Pixal3D on MLX.

Pixal3D pipeline with projection-based conditioning:
  1. Image → DINOv3 + ProjGrid → global + projected features
  2. Sparse structure flow (proj) → occupancy grid → LR coordinates
  3. LR SLat flow (proj) → denormalize → decoder upsample → HR coordinates
  4. HR SLat flow (proj) → denormalize → full decode → mesh extraction
  5. Texture SLat flow (proj) → texture decode → PBR bake → GLB

Usage:
    PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb
"""

import argparse
import gc
import math
import os
import time

import mlx.core as mx
import numpy as np
from PIL import Image


# === Normalization constants from Pixal3D pipeline.json ===
# (These will be loaded from the actual pipeline.json when available)
# Using TRELLIS.2-4B values as placeholder — will need to verify against Pixal3D
SHAPE_SLAT_MEAN = np.array([
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
    -3.186688, 1.578665,
], dtype=np.float32)

SHAPE_SLAT_STD = np.array([
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
], dtype=np.float32)

TEX_SLAT_MEAN = np.array([
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364,
    0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479,
    -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
    0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717,
    1.213905, 0.971588, -0.023631, 0.10675, 2.021786, 0.250524,
    -0.662387, -0.768862,
], dtype=np.float32)

TEX_SLAT_STD = np.array([
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316,
    2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343,
    2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
    3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999,
    2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849,
    2.786195, 2.77519,
], dtype=np.float32)


# === Camera parameter estimation ===

def compute_distance_from_fov(camera_angle_x, mesh_scale=1.0, image_resolution=512, extend_pixel=0):
    """Compute camera distance from FOV using Pixal3D's projection math.

    Uses the same geometry as Pixal3D's inference.py distance_from_fov().
    """
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    grid_point = np.array([-1.0, 0.0, 0.0]) @ rotation.T
    grid_point = grid_point / mesh_scale / 2.0

    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0

    # Target pixel for the grid corner
    xt = 0 - extend_pixel
    yt = image_resolution - 1 + extend_pixel

    x_ndc = xt - image_resolution / 2.0
    xw, yw = grid_point[0], grid_point[1]

    distance = f_pixels * xw / x_ndc - yw
    return distance


def get_default_camera_params(fov_rad=0.8575560450553894, mesh_scale=1.0):
    """Get default camera parameters (Pixal3D's default front view)."""
    distance = compute_distance_from_fov(fov_rad, mesh_scale)
    return {
        'camera_angle_x': fov_rad,
        'distance': distance,
        'mesh_scale': mesh_scale,
    }


# === Helpers ===

def _denormalize_slat(slat, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD):
    return slat * mx.array(std) + mx.array(mean)


def _normalize_slat(slat, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD):
    return (slat - mx.array(mean)) / mx.array(std)


def _requantize_coords(hr_coords_np, lr_resolution, hr_resolution):
    """Requantize decoder output coords to target resolution."""
    spatial = hr_coords_np[:, 1:4].astype(np.float64)
    spatial = ((spatial + 0.5) / lr_resolution * (hr_resolution // 16)).astype(np.int32)
    result = hr_coords_np.copy()
    result[:, 1:4] = spatial
    return np.unique(result, axis=0)


# === Feature extraction ===

def extract_proj_features(image_path, dinov3_model, grid_resolution, image_size,
                          camera_params, no_rembg=False):
    """Extract DINOv3 + projection features for one stage.

    Returns:
        cond: dict {'global': [1, 5, 1024], 'proj': [1, R^3, 1024]}
        neg_cond: dict with zeros
    """
    from trellmlx.models.dinov3_proj import DinoV3ProjFeatureExtractor, preprocess_image

    extractor = DinoV3ProjFeatureExtractor(
        dinov3_model, image_size=image_size, grid_resolution=grid_resolution,
    )

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    if not no_rembg:
        try:
            from trellmlx.preprocess import preprocess_image as rembg_preprocess
            img = rembg_preprocess(image_path)
        except ImportError:
            pass

    img_tensor = preprocess_image(img, size=image_size)

    fov = mx.array([camera_params['camera_angle_x']])
    dist = mx.array([camera_params['distance']])
    scale = mx.array([camera_params['mesh_scale']])

    z_global, z_proj = extractor(img_tensor, fov, dist, scale)
    mx.eval(z_global, z_proj)

    cond = {'global': z_global, 'proj': z_proj}
    neg_cond = {
        'global': mx.zeros_like(z_global),
        'proj': mx.zeros_like(z_proj),
    }

    return cond, neg_cond


def index_proj_by_coords(cond, neg_cond, coords_4d, grid_resolution):
    """Index full-grid projection features by sparse coordinates.

    For SLat stages, we need per-token projection features indexed by
    the sparse occupancy coordinates.

    Args:
        cond: dict with 'proj' of shape [1, R^3, D]
        neg_cond: dict with 'proj' of shape [1, R^3, D]
        coords_4d: [N, 4] int array (batch, x, y, z)
        grid_resolution: R (grid was R^3)

    Returns:
        sparse_cond, sparse_neg_cond: dicts with 'proj' of shape [1, N, D]
    """
    R = grid_resolution
    z_proj = cond['proj']  # [1, R^3, D]
    z_proj_grid = z_proj.reshape(1, R, R, R, -1)

    batch_idx = coords_4d[:, 0].astype(np.int32)
    x_idx = coords_4d[:, 1].astype(np.int32)
    y_idx = coords_4d[:, 2].astype(np.int32)
    z_idx = coords_4d[:, 3].astype(np.int32)

    # Index into the grid
    z_proj_grid_np = np.array(z_proj_grid)
    z_proj_sparse = z_proj_grid_np[batch_idx, x_idx, y_idx, z_idx]
    z_proj_sparse = mx.array(z_proj_sparse)[None]  # [1, N, D]

    sparse_cond = {'global': cond['global'], 'proj': z_proj_sparse}
    sparse_neg_cond = {
        'global': neg_cond['global'],
        'proj': mx.zeros_like(z_proj_sparse),
    }

    return sparse_cond, sparse_neg_cond


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Pixal3D on MLX: Image to 3D")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="/tmp/pixal3d-mlx-mesh.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual FOV in radians (default: auto ~49 degrees)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--ss-only", action="store_true",
                        help="Run only sparse structure stage (for debugging)")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    t_total = time.perf_counter()

    # Camera parameters
    if args.fov > 0:
        camera_params = get_default_camera_params(fov_rad=args.fov)
        print(f"Manual FOV: {math.degrees(args.fov):.1f} deg, distance: {camera_params['distance']:.4f}")
    else:
        camera_params = get_default_camera_params()
        print(f"Default FOV: {math.degrees(camera_params['camera_angle_x']):.1f} deg, "
              f"distance: {camera_params['distance']:.4f}")

    # Resolve checkpoint paths
    HF_PIXAL3D = os.path.expanduser(
        "~/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts/"
    )

    from trellmlx.weight_loader import load_weights
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.cleanup import cleanup_model, cleanup

    # === DINOv3 backbone (shared across stages) ===
    print("=== Loading DINOv3 backbone ===", flush=True)
    from trellmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights

    DINOV3_WEIGHTS = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/"
        "snapshots/"
    )
    if os.path.isdir(DINOV3_WEIGHTS):
        snapshots = [d for d in os.listdir(DINOV3_WEIGHTS) if not d.startswith(".")]
        if snapshots:
            DINOV3_WEIGHTS = os.path.join(DINOV3_WEIGHTS, snapshots[0])

    dinov3 = DINOv3ViT()
    num_loaded = load_dinov3_weights(dinov3, DINOV3_WEIGHTS)
    print(f"  DINOv3 loaded ({num_loaded} arrays).", flush=True)

    # === Stage 1: Sparse Structure (proj) ===
    print("\n=== Stage 1: Sparse Structure (proj) ===", flush=True)
    from trellmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel
    from trellmlx.models.sparse_structure_decoder import SparseStructureDecoder

    # Extract features for SS stage (grid_resolution=16, image_size=512)
    print("  Extracting proj features (SS, 512, grid=16)...", flush=True)
    ss_cond, ss_neg_cond = extract_proj_features(
        args.image, dinov3, grid_resolution=16, image_size=512,
        camera_params=camera_params, no_rembg=args.no_rembg,
    )

    ss_flow = Pixal3DSparseStructureFlowModel(
        in_channels=8, out_channels=8,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=1024,
        resolution=16,
    )
    load_weights(ss_flow, HF_PIXAL3D + "ss_flow_img_dit_1_3B_64_bf16.safetensors", verbose=False)

    # SS sampler params from Pixal3D pipeline.json
    SS_SAMPLER = dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
                      guidance_interval=(0.6, 1.0), rescale_t=5.0)

    noise = mx.random.normal((1, 8, 16, 16, 16))
    t0 = time.perf_counter()
    z_s = flow_euler_sample(ss_flow, noise, ss_cond, ss_neg_cond, verbose=False, **SS_SAMPLER)
    mx.eval(z_s)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s", flush=True)

    # Decode sparse structure
    # Use TRELLIS.2 decoder (same architecture, different checkpoint)
    HF_LARGE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/"
    )
    ss_dec_ckpt = HF_PIXAL3D + "ss_dec_conv3d_16l8_fp16.safetensors"
    if not os.path.exists(ss_dec_ckpt):
        ss_dec_ckpt = HF_LARGE + "ss_dec_conv3d_16l8_fp16.safetensors"

    ss_dec = SparseStructureDecoder()
    load_weights(ss_dec, ss_dec_ckpt, verbose=False)

    logits = ss_dec(z_s.astype(mx.float32))
    mx.eval(logits)
    decoded = np.array(logits[0, 0] > 0)

    lr_resolution = 32
    ratio = decoded.shape[0] // lr_resolution
    decoded_ds = decoded.reshape(
        lr_resolution, ratio, lr_resolution, ratio, lr_resolution, ratio
    ).any(axis=(1, 3, 5))
    lr_coords = np.argwhere(decoded_ds)
    print(f"  {len(lr_coords)} sparse voxels at {lr_resolution}³", flush=True)

    cleanup_model(ss_flow, ss_dec)

    if args.ss_only:
        print(f"\n  SS-only mode. {len(lr_coords)} voxels found.")
        print(f"  Total: {time.perf_counter()-t_total:.1f}s")
        return

    print(f"\n  Total time: {time.perf_counter()-t_total:.1f}s")
    print("  (Shape/texture stages not yet implemented — SS stage proven!)")


if __name__ == "__main__":
    main()
