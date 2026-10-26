"""NAF neighborhood attention must match NATTEN border semantics.

Upstream NAF (valeoai/NAF src/layers/attentions.py:72) calls
``natten.na2d(q, k, v, kernel_size, dilation)``. NATTEN computes dilated NA as
standard NA inside each interleaved dilation group
(natten src/natten/token_permute/torch_impl.py:137-181), and standard NA keeps
every window fully in-bounds by shifting it inward at borders
(csrc/include/natten/cuda/fna/na_utils.cuh:1246-1254, stride=1):

    start = max(i - r, 0) + (i + r >= n) * (n - r - i - 1)

so border windows hold K distinct keys -- no padded/replicated keys.
"""

import mlx.core as mx
import numpy as np
import pytest

from trellmlx.models.naf import neighborhood_attention


def _natten_start(i, n, K):
    r = K // 2
    return max(i - r, 0) + (i + r >= n) * (n - r - i - 1)


def _natten_neighbors(i, length, K, d):
    g = i % d
    a = i // d
    n = len(range(g, length, d))
    s = _natten_start(a, n, K)
    return [g + (s + j) * d for j in range(K)]


def _reference(q, k, v, K, d, scale):
    B, H, X, Y, _ = q.shape
    out = np.zeros(q.shape[:-1] + (v.shape[-1],), dtype=np.float64)
    for x in range(X):
        rows = _natten_neighbors(x, X, K, d[0])
        for y in range(Y):
            cols = _natten_neighbors(y, Y, K, d[1])
            kn = k[:, :, rows][:, :, :, cols].reshape(B, H, K * K, -1)
            vn = v[:, :, rows][:, :, :, cols].reshape(B, H, K * K, -1)
            logits = np.einsum("bhd,bhnd->bhn", q[:, :, x, y], kn) * scale
            w = np.exp(logits - logits.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            out[:, :, x, y] = np.einsum("bhn,bhnd->bhd", w, vn)
    return out


@pytest.mark.parametrize(
    "X,Y,K,d",
    [
        (7, 6, 3, (1, 1)),     # plain NA, borders shift inward
        (12, 12, 3, (2, 2)),   # dilated, divisible
        (13, 11, 3, (3, 2)),   # dilated, uneven dilation groups
    ],
)
def test_neighborhood_attention_matches_natten_border_windows(X, Y, K, d):
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 2, X, Y, 4)).astype(np.float32)
    k = rng.standard_normal((1, 2, X, Y, 4)).astype(np.float32)
    v = rng.standard_normal((1, 2, X, Y, 5)).astype(np.float32)
    scale = 0.5

    got = neighborhood_attention(
        mx.array(q), mx.array(k), mx.array(v), (K, K), scale, dilation=d
    )
    want = _reference(q, k, v, K, d, scale)
    np.testing.assert_allclose(np.array(got), want, rtol=1e-4, atol=1e-5)


def test_neighborhood_attention_rejects_dilation_group_smaller_than_kernel():
    q = mx.zeros((1, 1, 8, 8, 2))
    with pytest.raises(ValueError):
        neighborhood_attention(q, q, q, (5, 5), 1.0, dilation=(2, 2))
