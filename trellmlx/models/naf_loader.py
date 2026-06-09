"""Load NAF weights from safetensors into the MLX NAF model.

Weight mapping:
    PyTorch (upstream)                → MLX
    image_encoder.encoder.N.*         → encoder_1x1.N.*
    image_encoder.sem_encoder.N.*     → encoder_3x3.N.*
    image_encoder.rope.periods        → rope.periods

Conv2d weight transposition: PyTorch [out, in, H, W] → MLX [out, H, W, in]
"""

import mlx.core as mx
import numpy as np
from safetensors import safe_open


_KEY_MAP = {
    "image_encoder.encoder.": "encoder_1x1.",
    "image_encoder.sem_encoder.": "encoder_3x3.",
    "image_encoder.rope.": "rope.",
}


def _remap_key(key: str) -> str:
    for old, new in _KEY_MAP.items():
        if key.startswith(old):
            return new + key[len(old):]
    return key


def load_naf_weights(model, checkpoint_path: str, verbose: bool = True):
    """Load NAF weights from safetensors into MLX model."""
    import mlx.utils

    model_keys = set(k for k, _ in mlx.utils.tree_flatten(model.parameters()))

    weights = {}
    skipped = []

    with safe_open(checkpoint_path, framework="numpy") as sf:
        for ckpt_key in sf.keys():
            mlx_key = _remap_key(ckpt_key)
            tensor = sf.get_tensor(ckpt_key)

            if mlx_key not in model_keys:
                skipped.append(ckpt_key)
                if verbose:
                    print(f"  SKIP {ckpt_key} -> {mlx_key}")
                continue

            # Transpose Conv2d weights: PyTorch [out, in, H, W] → MLX [out, H, W, in]
            if tensor.ndim == 4 and ckpt_key.endswith(".weight"):
                tensor = tensor.transpose(0, 2, 3, 1)

            weights[mlx_key] = mx.array(tensor)

    # Check for missing keys
    loaded_keys = set(weights.keys())
    missing = model_keys - loaded_keys
    if missing and verbose:
        print(f"  {len(missing)} model params not in checkpoint:")
        for k in sorted(missing):
            print(f"    MISSING {k}")

    model.load_weights(list(weights.items()))

    if verbose:
        print(f"  NAF: loaded {len(weights)}/{len(model_keys)} parameters")
        if skipped:
            print(f"  Skipped {len(skipped)} checkpoint keys")

    return skipped
