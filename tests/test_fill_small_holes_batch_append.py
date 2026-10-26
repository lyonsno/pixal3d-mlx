"""Regression test ensuring fill_small_holes batches centroid vertex appends."""

import numpy as np

import trellmlx.mesh_cleanup as mc


def _make_disconnected_triangles(num_triangles: int):
    # Each isolated triangle has a 3-edge boundary loop with perimeter < 0.03.
    base = np.array(
        [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.0, 0.003, 0.0]],
        dtype=np.float32,
    )
    offsets = np.arange(num_triangles, dtype=np.float32)[:, None, None] * np.array(
        [[[0.02, 0.0, 0.0]]], dtype=np.float32
    )
    vertices = (base[None, :, :] + offsets).reshape(-1, 3)
    faces = np.arange(num_triangles * 3, dtype=np.int32).reshape(num_triangles, 3)
    return vertices, faces


def test_fill_small_holes_batches_centroid_vstack(monkeypatch):
    vertices, faces = _make_disconnected_triangles(8)
    real_vstack = np.vstack
    vstack_calls = []

    def counting_vstack(tup, *args, **kwargs):
        vstack_calls.append(len(tup))
        return real_vstack(tup, *args, **kwargs)

    monkeypatch.setattr(mc.np, "vstack", counting_vstack)
    out_v, out_f = mc.fill_small_holes(
        vertices, faces, max_hole_perimeter=0.03, verbose=False
    )

    assert out_v.shape == (24 + 8, 3)
    assert out_v.dtype == np.float32
    assert out_f.shape == (8 + 24, 3)
    assert out_f.dtype == np.int32
    assert len(vstack_calls) == 1
