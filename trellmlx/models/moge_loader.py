"""Weight loader for MoGe-2 MLX model.

Loads weights from HuggingFace safetensors (PyTorch format) into the MLX
MoGe model with correct key remapping and Conv2d weight transposition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np


def _transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """Transpose Conv2d weight from PyTorch [O, I, H, W] to MLX [O, H, W, I]."""
    return np.transpose(w, (0, 2, 3, 1))


def _transpose_conv_transpose_weight(w: np.ndarray) -> np.ndarray:
    """Transpose ConvTranspose2d weight from PyTorch [I, O, H, W] to MLX [O, H, W, I]."""
    return np.transpose(w, (1, 2, 3, 0))


def load_moge_weights(model, weights_path: str = None, model_name: str = None,
                      verbose: bool = True) -> int:
    """Load MoGe-2 weights from HuggingFace safetensors into MLX model.

    Args:
        model: MoGeModel instance
        weights_path: Path to safetensors file(s) or HF cache directory.
                      If None, auto-detects from HuggingFace cache.
        model_name: HuggingFace model name (default: "Ruicheng/moge-2-vitl").
                    Use "Ruicheng/moge-2-vitl-normal" for the normal head variant.
        verbose: Print loading progress

    Returns:
        Number of weight arrays loaded
    """
    if model_name is None:
        model_name = "Ruicheng/moge-2-vitl"
    if weights_path is None:
        weights_path = _find_hf_weights(model_name)

    # Load weights — supports both .pt (PyTorch checkpoint) and .safetensors
    pt_weights = _load_pt_weights(weights_path)

    if verbose:
        print(f"  Loaded {len(pt_weights)} PyTorch weight arrays")

    # Build MLX weight dict
    mlx_weights = {}

    # === Encoder normalization buffers ===
    # These are [1, 3, 1, 1] in PyTorch; we store as [1, 1, 1, 3] channels-last
    if "encoder.image_mean" in pt_weights:
        mlx_weights["encoder.image_mean"] = mx.array(
            pt_weights["encoder.image_mean"].transpose(0, 2, 3, 1)
        )
    if "encoder.image_std" in pt_weights:
        mlx_weights["encoder.image_std"] = mx.array(
            pt_weights["encoder.image_std"].transpose(0, 2, 3, 1)
        )

    # === DINOv2 Backbone ===
    _map_backbone(pt_weights, mlx_weights)

    # === Encoder output projections ===
    for i in range(4):
        pt_key = f"encoder.output_projections.{i}"
        mlx_key = f"encoder.output_projections.{i}"
        _map_conv2d(pt_weights, mlx_weights, pt_key, mlx_key)

    # === Neck ===
    _map_convstack(pt_weights, mlx_weights, "neck", "neck",
                   num_levels=5, num_resamplers=4,
                   num_res_blocks=[0, 2, 2, 2, 0],
                   output_levels=[])

    # === Points head ===
    _map_convstack(pt_weights, mlx_weights, "points_head", "points_head",
                   num_levels=5, num_resamplers=4,
                   num_res_blocks=[0, 1, 1, 1, 0],
                   output_levels=[4])

    # === Mask head ===
    _map_convstack(pt_weights, mlx_weights, "mask_head", "mask_head",
                   num_levels=5, num_resamplers=4,
                   num_res_blocks=[0, 1, 1, 1, 0],
                   output_levels=[4])

    # === Normal head (optional, present in moge-2-vitl-normal) ===
    if model.normal_head is not None:
        _map_convstack(pt_weights, mlx_weights, "normal_head", "normal_head",
                       num_levels=5, num_resamplers=4,
                       num_res_blocks=[0, 1, 1, 1, 0],
                       output_levels=[4])

    # === Scale head ===
    # PyTorch: scale_head.0/2/4 (Linear layers at indices 0, 2, 4 in Sequential)
    # MLX: scale_head.fc1/fc2/fc3
    for pt_idx, mlx_name in [(0, "fc1"), (2, "fc2"), (4, "fc3")]:
        mlx_weights[f"scale_head.{mlx_name}.weight"] = mx.array(pt_weights[f"scale_head.{pt_idx}.weight"])
        mlx_weights[f"scale_head.{mlx_name}.bias"] = mx.array(pt_weights[f"scale_head.{pt_idx}.bias"])

    # Load into model
    model.load_weights(list(mlx_weights.items()))

    if verbose:
        print(f"  Mapped {len(mlx_weights)} MLX weight arrays")

    return len(mlx_weights)


def _map_backbone(pt: Dict, mlx: Dict):
    """Map DINOv2 backbone weights."""
    prefix = "encoder.backbone"

    # Embeddings
    mlx[f"{prefix}.cls_token"] = mx.array(pt[f"{prefix}.cls_token"])
    mlx[f"{prefix}.pos_embed"] = mx.array(pt[f"{prefix}.pos_embed"])

    # Patch embed conv: PyTorch [O, I, H, W] -> MLX [O, H, W, I]
    w = pt[f"{prefix}.patch_embed.proj.weight"]
    mlx[f"{prefix}.patch_embed.weight"] = mx.array(_transpose_conv_weight(w))
    mlx[f"{prefix}.patch_embed.bias"] = mx.array(pt[f"{prefix}.patch_embed.proj.bias"])

    # Final norm
    mlx[f"{prefix}.norm.weight"] = mx.array(pt[f"{prefix}.norm.weight"])
    mlx[f"{prefix}.norm.bias"] = mx.array(pt[f"{prefix}.norm.bias"])

    # Transformer blocks
    for i in range(24):
        bp = f"{prefix}.blocks.{i}"

        # Norms
        mlx[f"{bp}.norm1.weight"] = mx.array(pt[f"{bp}.norm1.weight"])
        mlx[f"{bp}.norm1.bias"] = mx.array(pt[f"{bp}.norm1.bias"])
        mlx[f"{bp}.norm2.weight"] = mx.array(pt[f"{bp}.norm2.weight"])
        mlx[f"{bp}.norm2.bias"] = mx.array(pt[f"{bp}.norm2.bias"])

        # Attention (fused QKV)
        mlx[f"{bp}.attn.qkv.weight"] = mx.array(pt[f"{bp}.attn.qkv.weight"])
        mlx[f"{bp}.attn.qkv.bias"] = mx.array(pt[f"{bp}.attn.qkv.bias"])
        mlx[f"{bp}.attn.proj.weight"] = mx.array(pt[f"{bp}.attn.proj.weight"])
        mlx[f"{bp}.attn.proj.bias"] = mx.array(pt[f"{bp}.attn.proj.bias"])

        # Layer scale
        mlx[f"{bp}.ls1"] = mx.array(pt[f"{bp}.ls1.gamma"])
        mlx[f"{bp}.ls2"] = mx.array(pt[f"{bp}.ls2.gamma"])

        # MLP
        mlx[f"{bp}.mlp.fc1.weight"] = mx.array(pt[f"{bp}.mlp.fc1.weight"])
        mlx[f"{bp}.mlp.fc1.bias"] = mx.array(pt[f"{bp}.mlp.fc1.bias"])
        mlx[f"{bp}.mlp.fc2.weight"] = mx.array(pt[f"{bp}.mlp.fc2.weight"])
        mlx[f"{bp}.mlp.fc2.bias"] = mx.array(pt[f"{bp}.mlp.fc2.bias"])


def _map_conv2d(pt: Dict, mlx: Dict, pt_prefix: str, mlx_prefix: str):
    """Map a single Conv2d layer."""
    w = pt[f"{pt_prefix}.weight"]
    mlx[f"{mlx_prefix}.weight"] = mx.array(_transpose_conv_weight(w))
    if f"{pt_prefix}.bias" in pt:
        mlx[f"{mlx_prefix}.bias"] = mx.array(pt[f"{pt_prefix}.bias"])


def _map_convstack(
    pt: Dict, mlx: Dict,
    pt_prefix: str, mlx_prefix: str,
    num_levels: int, num_resamplers: int,
    num_res_blocks: list,
    output_levels: list,
):
    """Map a ConvStack's weights."""

    # Input blocks (1x1 Conv2d)
    for i in range(num_levels):
        pt_key = f"{pt_prefix}.input_blocks.{i}"
        if f"{pt_key}.weight" in pt:
            _map_conv2d(pt, mlx, pt_key, f"{mlx_prefix}.input_blocks.{i}")

    # Resamplers
    for i in range(num_resamplers):
        # Resampler is a Sequential with [ConvTranspose2d or Upsample, Conv2d]
        # ConvTranspose2d: index 0
        pt_ct_key = f"{pt_prefix}.resamplers.{i}.0"
        if f"{pt_ct_key}.weight" in pt:
            w = pt[f"{pt_ct_key}.weight"]
            # Check if it's ConvTranspose2d (shape [I, O, H, W]) or something else
            # ConvTranspose2d weights: [in_channels, out_channels, kH, kW]
            mlx[f"{mlx_prefix}.resamplers.{i}.conv_t.weight"] = mx.array(
                _transpose_conv_transpose_weight(w)
            )
            if f"{pt_ct_key}.bias" in pt:
                mlx[f"{mlx_prefix}.resamplers.{i}.conv_t.bias"] = mx.array(pt[f"{pt_ct_key}.bias"])

        # Conv2d after resampler: index 1
        pt_conv_key = f"{pt_prefix}.resamplers.{i}.1"
        if f"{pt_conv_key}.weight" in pt:
            w = pt[f"{pt_conv_key}.weight"]
            mlx[f"{mlx_prefix}.resamplers.{i}.conv.conv.weight"] = mx.array(
                _transpose_conv_weight(w)
            )
            if f"{pt_conv_key}.bias" in pt:
                mlx[f"{mlx_prefix}.resamplers.{i}.conv.conv.bias"] = mx.array(
                    pt[f"{pt_conv_key}.bias"]
                )

    # Residual blocks
    for level in range(num_levels):
        n_blocks = num_res_blocks[level]
        for b in range(n_blocks):
            pt_base = f"{pt_prefix}.res_blocks.{level}.{b}"
            mlx_base = f"{mlx_prefix}.res_blocks.{level}.{b}"

            # layers.2 -> layers_2.conv (ReplicateConv2d wraps nn.Conv2d)
            pt_key = f"{pt_base}.layers.2"
            if f"{pt_key}.weight" in pt:
                w = pt[f"{pt_key}.weight"]
                mlx[f"{mlx_base}.layers_2.conv.weight"] = mx.array(_transpose_conv_weight(w))
                if f"{pt_key}.bias" in pt:
                    mlx[f"{mlx_base}.layers_2.conv.bias"] = mx.array(pt[f"{pt_key}.bias"])

            # layers.5 -> layers_5.conv
            pt_key = f"{pt_base}.layers.5"
            if f"{pt_key}.weight" in pt:
                w = pt[f"{pt_key}.weight"]
                mlx[f"{mlx_base}.layers_5.conv.weight"] = mx.array(_transpose_conv_weight(w))
                if f"{pt_key}.bias" in pt:
                    mlx[f"{mlx_base}.layers_5.conv.bias"] = mx.array(pt[f"{pt_key}.bias"])

    # Output blocks
    for i in output_levels:
        pt_key = f"{pt_prefix}.output_blocks.{i}"
        if f"{pt_key}.weight" in pt:
            _map_conv2d(pt, mlx, pt_key, f"{mlx_prefix}.output_blocks.{i}")


def _load_pt_weights(weights_path: str) -> Dict[str, np.ndarray]:
    """Load weights from .pt or .safetensors file(s).

    Returns a flat dict of {key: numpy_array}.
    """
    if os.path.isdir(weights_path):
        # Look for model.pt first, then safetensors
        pt_path = os.path.join(weights_path, "model.pt")
        if os.path.exists(pt_path):
            weights_path = pt_path
        else:
            import glob
            st_files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
            if st_files:
                weights_path = st_files[0]  # will handle below
            else:
                raise FileNotFoundError(f"No model.pt or .safetensors in {weights_path}")

    if weights_path.endswith(".pt") or weights_path.endswith(".pth"):
        import torch
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        return {k: v.numpy() for k, v in state_dict.items()}
    else:
        from safetensors import safe_open
        weights = {}
        with safe_open(weights_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights


def _find_hf_weights(model_name: str) -> str:
    """Find HuggingFace model weights in the cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"MoGe weights not found in HF cache. Run:\n"
            f"  huggingface-cli download {model_name}"
        )

    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots):
        versions = [d for d in os.listdir(snapshots) if not d.startswith(".")]
        if versions:
            return os.path.join(snapshots, versions[0])

    return model_dir
