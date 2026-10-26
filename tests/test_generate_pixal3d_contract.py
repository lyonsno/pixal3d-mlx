"""Exercise SS entrypoint preparation without loading multi-billion-parameter models."""

import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import generate_pixal3d


class ReachedSampler(Exception):
    pass


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_ss_entrypoint_preserves_conditioning_and_packed_weights(monkeypatch, bits):
    """The actual CLI must deliver fp32 context and executable packed weights."""
    import trellmlx.models.dinov3 as dinov3
    import trellmlx.models.naf as naf
    import trellmlx.models.naf_loader as naf_loader
    import trellmlx.models.pixal3d_flow as pixal_flow
    import trellmlx.samplers as samplers
    import trellmlx.weight_loader as weight_loader

    class TinyFlow(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(64, 64)
            self.apply(lambda x: x.astype(mx.float16))

        def apply(self, fn, *args, **kwargs):
            result = super().apply(fn, *args, **kwargs)
            if isinstance(self.proj, nn.QuantizedLinear):
                # Check before the dictionary cast so the two regressions fail
                # independently on the old entrypoint.
                assert self.proj.weight.dtype == mx.uint32
                output = self.proj(mx.ones((1, 64), dtype=mx.float32))
                assert bool(mx.all(mx.isfinite(output)))
            return result

    cond = {"global": mx.array([[[1.0]]], dtype=mx.float16),
            "proj": mx.array([[[2.0]]], dtype=mx.float16)}
    neg = {key: mx.zeros_like(value) for key, value in cond.items()}

    def sample(model, noise, positive, negative, **kwargs):
        assert set(positive) == set(negative) == {"global", "proj"}
        assert noise.dtype == mx.float32
        for key in cond:
            assert positive[key].dtype == negative[key].dtype == mx.float32
            np.testing.assert_array_equal(np.array(positive[key]), np.array(cond[key]))
            np.testing.assert_array_equal(np.array(negative[key]), np.array(neg[key]))
        assert cond["global"].dtype == mx.float16
        if bits:
            assert isinstance(model.proj, nn.QuantizedLinear)
            assert model.proj.bits == bits
            assert model.proj.weight.dtype == mx.uint32
        else:
            assert model.proj.weight.dtype == mx.float32
        raise ReachedSampler

    monkeypatch.setattr(sys, "argv", ["generate_pixal3d.py", "--image", "unused.png",
                                     "--no-moge", "--resolution", "512",
                                     "--quantize", str(bits)])
    monkeypatch.setattr(dinov3, "DINOv3ViT", lambda: object())
    monkeypatch.setattr(dinov3, "load_dinov3_weights", lambda *a, **k: 0)
    monkeypatch.setattr(naf, "NAF", lambda: object())
    monkeypatch.setattr(naf_loader, "load_naf_weights", lambda *a, **k: None)
    monkeypatch.setattr(weight_loader, "load_weights", lambda *a, **k: None)
    monkeypatch.setattr(pixal_flow, "Pixal3DSparseStructureFlowModel", TinyFlow)
    monkeypatch.setattr(generate_pixal3d, "extract_proj_features", lambda *a, **k: (cond, neg))
    monkeypatch.setattr(samplers, "flow_euler_sample", sample)
    # Preprocessing is exercised in test_pixal3d_conditioning_inputs.py.
    monkeypatch.setattr(generate_pixal3d, "prepare_conditioning_image", lambda image, output, no_rembg=False: image)
    with pytest.raises(ReachedSampler):
        generate_pixal3d.main()


@pytest.mark.parametrize("resolution", [512, 1024])
def test_resume_texture_projection_uses_checkpoint_resolution(monkeypatch, resolution):
    import generate
    import resume_pixal3d
    import trellmlx.checkpoint as checkpoint
    import trellmlx.cleanup as cleanup
    import trellmlx.mesh_extract as mesh_extract
    import trellmlx.models.dinov3 as dinov3
    import trellmlx.models.naf as naf
    import trellmlx.models.naf_loader as naf_loader
    import trellmlx.models.pixal3d_flow as pixal_flow
    import trellmlx.models.shape_slat_decoder as shape_decoder
    import trellmlx.samplers as samplers
    import trellmlx.weight_loader as weight_loader

    class TinyDecoder:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return mx.zeros((1, 7)), mx.zeros((1, 4), dtype=mx.int32), []

    def extract(*args, grid_resolution, **kwargs):
        assert grid_resolution == resolution // 16
        return {}, {}

    def index(*args, grid_resolution, **kwargs):
        assert grid_resolution == resolution // 16
        return {}, {}

    def sample(*args, **kwargs):
        raise ReachedSampler

    monkeypatch.setattr(sys, "argv", ["resume_pixal3d.py", "--image", "unused.png", "--ckpt", "unused"])
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a: {
        "hr_slat": np.zeros((1, 32), dtype=np.float32),
        "quant_coords": np.zeros((1, 4), dtype=np.int32), "hr_resolution": resolution})
    monkeypatch.setattr(shape_decoder, "SLatDecoder", TinyDecoder)
    monkeypatch.setattr(weight_loader, "load_weights", lambda *a, **k: None)
    monkeypatch.setattr(cleanup, "cleanup_model", lambda *a, **k: None)
    monkeypatch.setattr(mesh_extract, "decoder_output_to_mesh", lambda *a, **k: ([], []))
    monkeypatch.setattr(generate, "_cleanup_and_simplify_mesh", lambda v, f, **k: (v, f))
    monkeypatch.setattr(dinov3, "DINOv3ViT", lambda: object())
    monkeypatch.setattr(dinov3, "load_dinov3_weights", lambda *a, **k: 0)
    monkeypatch.setattr(naf, "NAF", lambda: object())
    monkeypatch.setattr(naf_loader, "load_naf_weights", lambda *a, **k: None)
    monkeypatch.setattr(pixal_flow, "Pixal3DSLatFlowModel", lambda **k: object())
    monkeypatch.setattr(resume_pixal3d, "extract_proj_features", extract)
    monkeypatch.setattr(resume_pixal3d, "index_proj_by_coords", index)
    monkeypatch.setattr(samplers, "flow_euler_sample", sample)
    with pytest.raises(ReachedSampler):
        resume_pixal3d.main()


@pytest.mark.parametrize("resolution", [512, 1024])
def test_hr_and_texture_projection_follow_actual_latent_grid(monkeypatch, resolution):
    """Replay main through texture sampling with tiny stage stand-ins.

    Upstream Pixal3DImageTo3DPipeline.run derives both projection overrides
    from actual_hr_resolution // 16, independently of image resolution.
    """
    import generate
    import trellmlx.cleanup as cleanup
    import trellmlx.mesh_extract as mesh_extract
    import trellmlx.models.dinov3 as dinov3
    import trellmlx.models.naf as naf
    import trellmlx.models.naf_loader as naf_loader
    import trellmlx.models.pixal3d_flow as pixal_flow
    import trellmlx.models.shape_slat_decoder as shape_decoder
    import trellmlx.models.sparse_structure_decoder as ss_decoder
    import trellmlx.samplers as samplers
    import trellmlx.weight_loader as weight_loader

    class TinySSDecoder:
        def __call__(self, z):
            return mx.ones((1, 1, 32, 32, 32))

    class TinyShapeDecoder:
        def __init__(self, **kwargs):
            pass

        def upsample(self, *args, **kwargs):
            return mx.array([[0, 0, 0, 0], [0, 511, 511, 511]])

        def __call__(self, *args, **kwargs):
            return mx.zeros((2, 7)), mx.zeros((2, 4), dtype=mx.int32), []

    extracted_grids = []
    indexed_grids = []

    def extract(*args, grid_resolution, **kwargs):
        extracted_grids.append(grid_resolution)
        return {}, {}

    def index(cond, neg, coords, grid_resolution, **kwargs):
        indexed_grids.append(grid_resolution)
        assert np.all(coords[:, 1:] < grid_resolution)
        return {}, {}

    def sample(model, noise, *args, **kwargs):
        if "concat_cond" in kwargs:
            # Both extraction and sparse indexing must use the matching grid.
            assert extracted_grids == [16, 32, resolution // 16, resolution // 16]
            assert indexed_grids == [32, resolution // 16, resolution // 16]
            raise ReachedSampler
        return mx.zeros_like(noise)

    monkeypatch.setattr(sys, "argv", ["generate_pixal3d.py", "--image", "unused.png",
                                     "--no-moge", "--resolution", str(resolution)])
    monkeypatch.setattr(dinov3, "DINOv3ViT", lambda: object())
    monkeypatch.setattr(dinov3, "load_dinov3_weights", lambda *a, **k: 0)
    monkeypatch.setattr(naf, "NAF", lambda: object())
    monkeypatch.setattr(naf_loader, "load_naf_weights", lambda *a, **k: None)
    monkeypatch.setattr(weight_loader, "load_weights", lambda *a, **k: None)
    monkeypatch.setattr(pixal_flow, "Pixal3DSparseStructureFlowModel", lambda **k: nn.Linear(1, 1))
    monkeypatch.setattr(pixal_flow, "Pixal3DSLatFlowModel", lambda **k: object())
    monkeypatch.setattr(ss_decoder, "SparseStructureDecoder", TinySSDecoder)
    monkeypatch.setattr(shape_decoder, "SLatDecoder", TinyShapeDecoder)
    monkeypatch.setattr(cleanup, "cleanup_model", lambda *a, **k: None)
    monkeypatch.setattr(generate_pixal3d, "extract_proj_features", extract)
    monkeypatch.setattr(generate_pixal3d, "index_proj_by_coords", index)
    monkeypatch.setattr(samplers, "flow_euler_sample", sample)
    monkeypatch.setattr(mesh_extract, "decoder_output_to_mesh", lambda *a, **k: ([], []))
    monkeypatch.setattr(generate, "_cleanup_and_simplify_mesh", lambda v, f, **k: (v, f))
    # Preprocessing is exercised in test_pixal3d_conditioning_inputs.py.
    monkeypatch.setattr(generate_pixal3d, "prepare_conditioning_image", lambda image, output, no_rembg=False: image)
    with pytest.raises(ReachedSampler):
        generate_pixal3d.main()
