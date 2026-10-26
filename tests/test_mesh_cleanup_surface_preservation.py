"""Cleanup must not delete attached surface detail (trellis2mlx d642ab6).

The old order split non-manifold edges first and then removed components
below a face-count fraction of the largest. That deleted fine attached
sheets and left holes larger than the fill limit. The local cleanup filters
components by area *before* splitting, and later passes do not re-filter
fragments the first split produced.
"""

import numpy as np

from trellmlx.mesh_cleanup import cleanup_mesh


def _make_fine_surface_and_debris():
    # Three sheets share an overloaded edge. Each sheet is below the area
    # cutoff, but their original connected part is above it. Face 4 is
    # detached debris.
    vertices = np.array([
        [0, 0, 0], [0.01, 0, 0], [0, 0.0012, 0],
        [0, -0.0012, 0], [0, 0, 0.0012], [-0.001, 0, 0],
        [1, 0, 0], [1.001, 0, 0], [1, 0.001, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4],
                      [0, 2, 5], [6, 7, 8]], dtype=np.int64)
    return vertices, faces


def _triangles(v, f):
    return {tuple(sorted(map(tuple, t))) for t in v[f]}


def test_cleanup_keeps_fine_surface_but_removes_detached_debris():
    vertices, faces = _make_fine_surface_and_debris()
    out_v, out_f = cleanup_mesh(vertices, faces, max_hole_perimeter=0,
                                do_fix_normals=False, verbose=False)
    assert _triangles(vertices, faces[:4]) <= _triangles(out_v, out_f)
    assert not _triangles(vertices, faces[4:]) & _triangles(out_v, out_f)


def test_repeated_generation_cleanup_preserves_original_fine_surface():
    from generate import _cleanup_and_simplify_mesh

    vertices, faces = _make_fine_surface_and_debris()
    out_v, out_f = _cleanup_and_simplify_mesh(
        vertices, faces, target_faces=0, no_cleanup=False,
        log=lambda *args, **kwargs: None)
    assert _triangles(vertices, faces[:4]) <= _triangles(out_v, out_f)
    assert not _triangles(vertices, faces[4:]) & _triangles(out_v, out_f)
