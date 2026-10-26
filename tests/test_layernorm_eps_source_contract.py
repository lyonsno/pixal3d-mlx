"""Source-contract eps/softplus checks for the Pixal3D path.

Upstream Pixal3D (TencentARC, ~/dev/Pixal3D) uses torch defaults at these
sites, which differ from the 1e-6 used inside transformer/conv blocks:

- Flow terminal norm: ``F.layer_norm(h, h.shape[-1:])`` (eps=1e-5)
  pixal3d/models/sparse_structure_flow.py:293,
  pixal3d/models/structured_latent_flow.py:255
- Shape SLat decoder terminal norm: ``F.layer_norm`` (eps=1e-5)
  pixal3d/models/sc_vaes/sparse_unet_vae.py:380,499
- SS VAE ``ChannelLayerNorm32(channels)`` -> nn.LayerNorm default eps=1e-5
  pixal3d/models/sparse_structure_vae.py:17,33-34,262
- DINOv3 (HF DINOv3ViTModel layer_norm_eps=1e-5) and the extractor's final
  ``F.layer_norm`` (eps=1e-5)
  pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:462
- FDG quad weight ``F.softplus`` (beta=1, threshold=20 -> linear above 20)
  pixal3d/models/sc_vaes/fdg_vae.py:89,102

Block-internal norms stay at 1e-6 (modules/*/transformer/modulated.py).
"""

import mlx.core as mx
import numpy as np
import pytest

BLOCK_EPS = 1e-6
TERMINAL_EPS = 1e-5


def _spy_layernorm(monkeypatch, module):
    calls = []
    real = module._layernorm_noaffine

    def spy(x, eps=BLOCK_EPS):
        calls.append(eps)
        return real(x, eps=eps)

    monkeypatch.setattr(module, "_layernorm_noaffine", spy)
    return calls


def test_pixal3d_ss_flow_terminal_norm_uses_torch_default_eps(monkeypatch):
    from trellmlx.models import pixal3d_flow

    calls = _spy_layernorm(monkeypatch, pixal3d_flow)
    R = 2
    model = pixal3d_flow.Pixal3DSparseStructureFlowModel(
        in_channels=4, out_channels=4, model_channels=32, num_heads=4,
        num_blocks=1, mlp_hidden=64, context_channels=16,
        proj_in_channels=16, resolution=R,
    )
    cond = {"global": mx.ones((1, 3, 16)), "proj": mx.ones((1, R**3, 16))}
    mx.eval(model(mx.ones((1, 4, R, R, R)), mx.array([0.5]), cond))

    assert calls[-1] == TERMINAL_EPS
    assert calls[:-1] and all(e == BLOCK_EPS for e in calls[:-1])


def test_pixal3d_slat_flow_terminal_norm_uses_torch_default_eps(monkeypatch):
    from trellmlx.models import pixal3d_flow

    calls = _spy_layernorm(monkeypatch, pixal3d_flow)
    N = 6
    model = pixal3d_flow.Pixal3DSLatFlowModel(
        in_channels=8, out_channels=8, model_channels=32, num_heads=4,
        num_blocks=1, mlp_hidden=64, context_channels=16, proj_in_channels=16,
    )
    cond = {"global": mx.ones((1, 3, 16)), "proj": mx.ones((1, N, 16))}
    mx.eval(model(mx.ones((N, 8)), mx.array([0.5]), cond))

    assert calls[-1] == TERMINAL_EPS
    assert calls[:-1] and all(e == BLOCK_EPS for e in calls[:-1])


def test_shape_slat_decoder_terminal_norm_uses_torch_default_eps(monkeypatch):
    from trellmlx.models import shape_slat_decoder

    calls = _spy_layernorm(monkeypatch, shape_slat_decoder)
    model = shape_slat_decoder.SLatDecoder(
        out_channels=7, latent_channels=8, model_channels=[16], num_blocks=[0],
    )
    coords = mx.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=mx.int32)
    out, _ = model(mx.ones((2, 8)), coords)
    mx.eval(out)

    assert calls == [TERMINAL_EPS]


def test_sparse_structure_decoder_channel_layernorm_uses_torch_default_eps():
    from trellmlx.models.sparse_structure_decoder import (
        ChannelLayerNorm32, SparseStructureDecoder,
    )

    model = SparseStructureDecoder(
        out_channels=1, latent_channels=8, num_res_blocks=1,
        num_res_blocks_middle=1, channels=[32, 16, 8],
    )
    norms = [m for _, m in model.named_modules()
             if isinstance(m, ChannelLayerNorm32)]
    assert norms
    assert all(n.eps == TERMINAL_EPS for n in norms)


def test_dinov3_layernorms_use_hf_config_eps(monkeypatch):
    from trellmlx.models import dinov3

    model = dinov3.DINOv3ViT(
        hidden_size=32, num_heads=2, num_layers=1, patch_size=16,
        num_register_tokens=1, intermediate_size=64,
    )
    for layer in model.layers:
        assert layer.norm1.eps == TERMINAL_EPS
        assert layer.norm2.eps == TERMINAL_EPS

    eps_seen = []
    real = mx.fast.layer_norm

    def spy(x, weight, bias, eps):
        eps_seen.append(eps)
        return real(x, weight, bias, eps)

    monkeypatch.setattr(mx.fast, "layer_norm", spy)
    mx.eval(model(mx.ones((1, 32, 32, 3))))

    # norm1, norm2, final no-affine norm
    assert len(eps_seen) == 3
    assert all(e == TERMINAL_EPS for e in eps_seen)


def test_fdg_softplus_is_linear_above_torch_threshold():
    from trellmlx.mesh_extract import _softplus

    x = np.array([-5.0, 0.0, 5.0, 19.0, 25.0, 60.0], dtype=np.float32)
    out = _softplus(x)
    np.testing.assert_allclose(out[:4], np.log1p(np.exp(x[:4])), rtol=1e-6)
    np.testing.assert_array_equal(out[4:], x[4:])
    assert np.all(np.isfinite(out))
