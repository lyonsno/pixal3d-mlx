import mlx.core as mx

import generate_pixal3d


def test_cast_conditioning_to_dtype_handles_pixal3d_context_dict():
    cond = {
        "global": mx.array([[[1.0]]], dtype=mx.float16),
        "proj": mx.array([[[2.0]]], dtype=mx.float16),
    }

    cast = generate_pixal3d.cast_conditioning_to_dtype(cond, mx.float32)

    assert cast["global"].dtype == mx.float32
    assert cast["proj"].dtype == mx.float32
    assert cond["global"].dtype == mx.float16
