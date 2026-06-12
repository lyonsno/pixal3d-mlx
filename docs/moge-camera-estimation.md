# MoGe-2 Camera Estimation

Pixal3D-MLX uses [MoGe-2](https://github.com/microsoft/MoGe) (ViT-L) to
estimate camera intrinsics (FOV, distance) from the input image. This replaces
the fixed default FOV (49.1°) with per-image estimates, which improves
projection conditioning for photos, angled shots, and non-centered objects.

## Backends

**MLX (default):** Pure MLX port of MoGe-2 (326M parameters). No PyTorch
dependency. Loads in ~1s, infers in ~2.6s on M4 Max.

**PyTorch/MPS (`--pytorch-moge`):** Uses PyTorch MoGe via MPS. Requires
`moge` package (`pip install moge`). Infers in ~1.9s but loads in ~6s.

**Disabled (`--no-moge`):** Falls back to fixed 49.1° FOV. Use when MoGe
weights are not downloaded or for reproducibility with a known camera.

**Manual (`--fov <radians>`):** Override with a specific FOV. Takes priority
over all other methods.

## Parity

The MLX backend matches PyTorch MoGe within float32 precision:

| Metric | Value |
|--------|-------|
| Raw forward correlation (fp32 vs fp32) | 0.9999973 |
| Raw forward max absolute diff | 0.017 |
| FOV difference | 0.08° |
| Points correlation (post-processing) | 0.99999940 |

### Known precision gap

The remaining 0.017 max difference comes from the image preprocessing resize.
PyTorch uses `F.interpolate(antialias=True)` which applies a low-pass filter
before bilinear sampling. MLX does not yet have antialiased interpolation — a
[PR to add it](https://github.com/lyonsno/mlx/tree/antialias-interpolation) is
under review. Once merged, this gap closes to float32 noise.

For practical use, this difference is negligible: a 0.08° FOV error is below
any camera's actual calibration uncertainty, and the downstream effect on
Pixal3D projection conditioning is undetectable.

## Weights

MoGe-2 ViT-L weights are downloaded from HuggingFace on first use:

```
huggingface-cli download Ruicheng/moge-2-vitl
```

The model checkpoint is ~1.3 GB (float32). It loads into unified memory,
runs inference, and unloads before the Pixal3D flow models start — the two
never share GPU memory.

## Architecture

The MLX MoGe-2 model is a full port of microsoft/MoGe v2:

- **DINOv2-L/14 backbone**: 24 transformer blocks, 1024 dim, 16 heads,
  learned position embeddings, intermediate feature extraction at layers
  5/11/17/23 with per-layer normalization
- **DINOv2Encoder**: 4× Conv2d 1×1 output projections, summed
- **ConvStack neck**: 5-level decoder with ConvTranspose2d resamplers
- **Points/mask heads**: same ConvStack architecture, 3-ch and 1-ch output
- **Scale head**: 3-layer MLP from CLS token
- **Post-processing**: nonlinear focal/shift recovery via Levenberg-Marquardt
  (scipy), depth-to-points projection, metric scale application

Total: 326M parameters, 482 weight arrays loaded from HuggingFace `.pt`
checkpoint.
