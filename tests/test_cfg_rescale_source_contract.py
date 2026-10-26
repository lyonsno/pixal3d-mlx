"""CFG rescale must follow upstream ClassifierFreeGuidanceSamplerMixin.

Upstream (pixal3d/pipelines/samplers/classifier_free_guidance_mixin.py:20-26):
    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)            # no clamp
For SparseTensor/VarLenTensor (modules/sparse/basic.py:295-299) std is the
per-sample population std over all tokens and channels; for dense tensors it
is torch.std (Bessel-corrected) over all non-batch dims.
"""

import mlx.core as mx
import numpy as np

from trellmlx.samplers import flow_euler_sample, _pred_to_xstart, _xstart_to_pred


def _run_one_step(noise, pred_pos, pred_neg, guidance_strength, **kwargs):
    calls = {"n": 0}

    def model(sample, t, conditioning, **kw):
        calls["n"] += 1
        return pred_pos if calls["n"] == 1 else pred_neg

    return flow_euler_sample(
        model, noise, mx.ones((1, 1)), mx.zeros((1, 1)), steps=1,
        guidance_strength=guidance_strength, guidance_rescale=1.0,
        guidance_interval=(0.0, 1.0), rescale_t=1.0, verbose=False, **kwargs)


def _expected(noise, x0_cfg_scale, pred_cfg):
    x0_cfg = _pred_to_xstart(noise, 1.0, pred_cfg, 1e-5)
    pred = _xstart_to_pred(noise, 1.0, x0_cfg * x0_cfg_scale, 1e-5)
    return noise - pred


def test_sparse_tokens_use_sequence_level_population_std():
    noise = mx.zeros((2, 2), dtype=mx.float32)
    pred_pos = mx.array([[0.0, 10.0], [0.0, 10.0]], dtype=mx.float32)
    pred_cfg = mx.array([[0.0, 1.0], [0.0, 100.0]], dtype=mx.float32)
    g = 2.0
    pred_neg = (g * pred_pos - pred_cfg) / (g - 1.0)

    out = _run_one_step(noise, pred_pos, pred_neg, g,
                        coords=mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32))

    x0_pos = np.array(_pred_to_xstart(noise, 1.0, pred_pos, 1e-5))
    x0_cfg = np.array(_pred_to_xstart(noise, 1.0, pred_cfg, 1e-5))
    ratio = x0_pos.std() / x0_cfg.std()  # population std over all tokens+channels
    np.testing.assert_allclose(np.array(out), np.array(_expected(noise, ratio, pred_cfg)),
                               rtol=1e-5, atol=1e-5)


def test_dense_rescale_is_not_clamped():
    noise = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)
    base = mx.array(np.arange(16, dtype=np.float32).reshape(1, 2, 2, 2, 2))
    pred_pos = base * 10.0
    pred_cfg = base  # std ratio = 10, far outside the old [0.5, 2] clamp
    g = 2.0
    pred_neg = (g * pred_pos - pred_cfg) / (g - 1.0)

    out = _run_one_step(noise, pred_pos, pred_neg, g)

    x0_pos = np.array(_pred_to_xstart(noise, 1.0, pred_pos, 1e-5))
    x0_cfg = np.array(_pred_to_xstart(noise, 1.0, pred_cfg, 1e-5))
    ratio = x0_pos.std(ddof=1) / x0_cfg.std(ddof=1)
    np.testing.assert_allclose(np.array(out), np.array(_expected(noise, ratio, pred_cfg)),
                               rtol=1e-4, atol=1e-4)
