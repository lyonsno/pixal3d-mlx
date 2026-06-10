"""Resume Pixal3D pipeline from shape_latent checkpoint.

Skips stages 1-3 (geometry inference, ~35 min) and runs:
  Stage 3: Shape decode
  Stage 4: Texture SLat
  Stage 5: Texture decode
  Stage 6: Texture bake + GLB export

Usage:
    PYTHONPATH=. python resume_pixal3d.py --image photo.png --ckpt outputs/ckpt_dir --output mesh.glb
"""

import argparse
import gc
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from generate_pixal3d import (
    get_default_camera_params, extract_proj_features, index_proj_by_coords,
    _denormalize_slat, _normalize_slat, _requantize_coords,
    SHAPE_SLAT_MEAN, SHAPE_SLAT_STD, TEX_SLAT_MEAN, TEX_SLAT_STD,
)


def main():
    parser = argparse.ArgumentParser(description="Resume Pixal3D from checkpoint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory")
    parser.add_argument("--output", default="outputs/resumed.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--target-faces", type=int, default=500_000)
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--keep-largest", action="store_true")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.metal.reset_peak_memory()
    t_total = time.perf_counter()

    from trellmlx.weight_loader import load_weights
    from trellmlx.samplers import flow_euler_sample
    from trellmlx.cleanup import cleanup_model
    from trellmlx.checkpoint import load_checkpoint

    # Camera
    if args.fov > 0:
        camera_params = get_default_camera_params(fov_rad=args.fov)
    else:
        camera_params = get_default_camera_params()

    HF_PIXAL3D = os.path.expanduser(
        "~/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts/"
    )

    # === Load checkpoint ===
    print("=== Loading shape_latent checkpoint ===", flush=True)
    ckpt = load_checkpoint(args.ckpt, 'shape_latent')
    hr_slat = mx.array(ckpt['hr_slat'])
    quant_coords = ckpt['quant_coords']
    hr_coords_3d = quant_coords[:, 1:4]
    hr_resolution = int(ckpt['hr_resolution'])
    num_tokens = len(quant_coords)
    print(f"  {num_tokens:,} tokens, res {hr_resolution}", flush=True)

    # === Stage 3: Shape Decode ===
    print("\n=== Stage 3: Decode Shape ===", flush=True)
    from trellmlx.models.shape_slat_decoder import SLatDecoder

    shape_dec_ckpt = HF_PIXAL3D + "shape_dec_next_dc_f16c32_fp16.safetensors"
    shape_decoder = SLatDecoder(out_channels=7, pred_subdiv=True)
    load_weights(shape_decoder, shape_dec_ckpt, verbose=False)

    t0 = time.perf_counter()
    dec_out, dec_coords, shape_subs = shape_decoder(
        hr_slat, mx.array(quant_coords), return_subs=True,
    )
    mx.eval(dec_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({dec_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(shape_decoder)
    del shape_decoder
    gc.collect()
    mx.metal.clear_cache()

    # === Mesh Extraction ===
    print("\n=== Mesh Extraction ===", flush=True)
    from trellmlx.mesh_extract import decoder_output_to_mesh

    dec_coords_np = np.array(dec_coords)
    dec_feats_np = np.array(dec_out)
    del dec_out, dec_coords
    gc.collect()

    mesh_grid_size = hr_resolution
    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        dec_feats_np, dec_coords_np, resolution=mesh_grid_size,
    )
    print(f"  Extracted: {time.perf_counter()-t0:.1f}s ({len(vertices):,}V {len(faces):,}F)", flush=True)

    from generate import _cleanup_and_simplify_mesh
    vertices, faces = _cleanup_and_simplify_mesh(
        vertices, faces,
        target_faces=args.target_faces,
        no_cleanup=args.no_cleanup,
        keep_largest=args.keep_largest,
    )

    # === DINOv3 + NAF for texture conditioning ===
    print("\n=== Loading DINOv3 + NAF ===", flush=True)
    from trellmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights
    from trellmlx.models.naf import NAF as NAFModel
    from trellmlx.models.naf_loader import load_naf_weights

    DINOV3_WEIGHTS = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/"
    )
    if os.path.isdir(DINOV3_WEIGHTS):
        snapshots = [d for d in os.listdir(DINOV3_WEIGHTS) if not d.startswith(".")]
        if snapshots:
            DINOV3_WEIGHTS = os.path.join(DINOV3_WEIGHTS, snapshots[0])

    dinov3 = DINOv3ViT()
    load_dinov3_weights(dinov3, DINOV3_WEIGHTS)
    naf = NAFModel()
    load_naf_weights(naf, os.path.join(os.path.dirname(__file__), "weights", "naf_release.safetensors"), verbose=False)
    print("  Loaded.", flush=True)

    # === Stage 4: Texture SLat ===
    print("\n=== Stage 4: Texture SLat ===", flush=True)
    from trellmlx.models.pixal3d_flow import Pixal3DSLatFlowModel

    TEX_SAMPLER = dict(steps=12, guidance_strength=1.0, guidance_rescale=0.0,
                       guidance_interval=(0.6, 0.9), rescale_t=3.0)

    tex_image_size = 1024
    tex_naf_size = 512
    print(f"  Extracting proj features (tex, {tex_image_size}, grid=64, NAF to {tex_naf_size})...", flush=True)
    tex_cond, tex_neg_cond = extract_proj_features(
        args.image, dinov3, grid_resolution=64, image_size=tex_image_size,
        camera_params=camera_params, no_rembg=args.no_rembg,
        use_naf_upsample=True, upsample_target_size=tex_naf_size, naf_model=naf,
    )
    tex_cond_sparse, tex_neg_cond_sparse = index_proj_by_coords(
        tex_cond, tex_neg_cond, quant_coords, grid_resolution=64,
    )

    tex_flow = Pixal3DSLatFlowModel(
        in_channels=64, out_channels=32,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=2048,
    )
    load_weights(tex_flow, HF_PIXAL3D + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", verbose=False)

    shape_cond = _normalize_slat(hr_slat)
    mx.eval(shape_cond)

    tex_noise = mx.random.normal((num_tokens, 32))
    t0 = time.perf_counter()
    tex_slat = flow_euler_sample(
        tex_flow, tex_noise, tex_cond_sparse, tex_neg_cond_sparse,
        verbose=False, coords=mx.array(hr_coords_3d),
        concat_cond=shape_cond,
        **TEX_SAMPLER,
    )
    mx.eval(tex_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

    tex_slat = _denormalize_slat(tex_slat, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD)
    mx.eval(tex_slat)

    cleanup_model(tex_flow)
    del tex_flow, tex_cond, tex_neg_cond, tex_cond_sparse, tex_neg_cond_sparse
    del dinov3, naf
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 5: Texture Decode ===
    print("\n=== Stage 5: Texture Decode ===", flush=True)

    tex_dec_ckpt = HF_PIXAL3D + "tex_dec_next_dc_f16c32_fp16.safetensors"
    if not os.path.exists(tex_dec_ckpt):
        HF_4B = os.path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
            "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
        )
        tex_dec_ckpt = HF_4B + "tex_dec_next_dc_f16c32_fp16.safetensors"

    tex_decoder = SLatDecoder(out_channels=6, pred_subdiv=False)
    load_weights(tex_decoder, tex_dec_ckpt, verbose=False)

    t0 = time.perf_counter()
    tex_out, tex_coords = tex_decoder(
        tex_slat, mx.array(quant_coords), guide_subs=shape_subs,
    )
    mx.eval(tex_out)
    tex_out = tex_out * 0.5 + 0.5
    mx.eval(tex_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({tex_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(tex_decoder)
    del tex_decoder
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 6: Texture Bake + GLB ===
    print("\n=== Stage 6: Texture Bake + GLB ===", flush=True)
    from trellmlx.texture_bake import uv_unwrap, bake_texture
    import trimesh
    from trimesh.visual.material import PBRMaterial

    tex_np = np.array(tex_out)
    tex_coords_spatial = np.array(tex_coords)[:, 1:4]

    t0 = time.perf_counter()
    uv_verts, uv_faces, uvs, vmapping = uv_unwrap(vertices, faces)
    print(f"  UV unwrap: {len(uv_verts):,}V {len(uv_faces):,}F ({time.perf_counter()-t0:.1f}s)", flush=True)

    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_verts, uv_faces, uvs, vmapping,
        tex_coords_spatial, tex_np, mesh_grid_size,
        texture_size=args.texture_size,
        backend="gpu",
    )
    print(f"  Baked textures ({args.texture_size}x{args.texture_size})", flush=True)

    export_verts = uv_verts.copy()
    export_verts[:, 1], export_verts[:, 2] = uv_verts[:, 2].copy(), -uv_verts[:, 1].copy()
    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]

    mesh = trimesh.Trimesh(vertices=export_verts, faces=uv_faces, process=False)
    normals = mesh.vertex_normals

    material = PBRMaterial(
        baseColorTexture=Image.fromarray(base_color),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0, roughnessFactor=1.0,
        alphaMode=alpha_mode, doubleSided=True,
    )

    textured_mesh = trimesh.Trimesh(
        vertices=export_verts, faces=uv_faces, vertex_normals=normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    textured_mesh.export(args.output)
    peak_mem = mx.metal.get_peak_memory() / 1e9
    print(f"\n  Saved: {args.output}")
    print(f"  Total: {time.perf_counter()-t_total:.1f}s")
    print(f"  Peak GPU memory: {peak_mem:.1f} GB")


if __name__ == "__main__":
    main()
