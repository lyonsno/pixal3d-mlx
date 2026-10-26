"""Cleanup regressions found in review of cf37acf.

F1: hole caps must follow boundary direction. F2: overloaded edges must be
split even when no edge has exactly two faces. Plus later-pass coverage.
"""

from collections import Counter

import numpy as np

from trellmlx.mesh_cleanup import fill_small_holes, repair_non_manifold_edges


def _directed_edge_counts(faces):
    c = Counter()
    for a, b, d in np.asarray(faces).tolist():
        c[(a, b)] += 1
        c[(b, d)] += 1
        c[(d, a)] += 1
    return c


def _winding_consistent(faces):
    # Every edge shared by two faces must be traversed once in each direction.
    c = _directed_edge_counts(faces)
    return all(n == 1 and c.get((b, a), 0) <= 1 for (a, b), n in c.items())


def _max_edge_incidence(faces):
    c = Counter()
    for a, b, d in np.asarray(faces).tolist():
        for e in ((a, b), (b, d), (d, a)):
            c[tuple(sorted(e))] += 1
    return max(c.values())


def _triangles(v, f):
    return {tuple(sorted(map(tuple, t))) for t in np.round(v[f], 7)}


def _open_tetra():
    vertices = np.array([
        [0, 0, 0], [0.006, 0, 0], [0, 0.006, 0], [0, 0, 0.006],
    ], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2]], dtype=np.int64)
    return vertices, faces


def _three_sheets():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    return vertices, faces


# ── F1: hole caps follow boundary direction ──

def test_fill_small_holes_caps_with_consistent_winding():
    v, f = _open_tetra()
    assert _winding_consistent(f)
    out_v, out_f = fill_small_holes(v, f, verbose=False)
    assert len(out_f) > len(f)
    assert _winding_consistent(out_f)
    assert _max_edge_incidence(out_f) == 2  # watertight


def test_generation_helper_below_simplify_threshold_keeps_winding():
    from generate import _cleanup_and_simplify_mesh

    v, f = _open_tetra()
    out_v, out_f = _cleanup_and_simplify_mesh(
        v, f, target_faces=500_000, no_cleanup=False,
        log=lambda *a, **k: None)
    assert _max_edge_incidence(out_f) == 2
    assert _winding_consistent(out_f)


# ── F2: overloaded edges split even with no manifold pairs ──

def test_repair_splits_overloaded_edge_without_manifold_pairs():
    v, f = _three_sheets()
    out_v, out_f = repair_non_manifold_edges(v, f, verbose=False)
    assert _triangles(v, f) == _triangles(out_v, out_f)
    assert _max_edge_incidence(out_f) <= 2


# ── Later cleanup passes do not re-filter split fragments ──

def test_later_cleanup_passes_preserve_split_fine_surface():
    from generate import _cleanup_and_simplify_mesh

    # Each split sheet is below the 1e-5 area cutoff and its boundary
    # perimeter (~0.04) exceeds the 0.03 hole-fill limit, so nothing caps it
    # back above the cutoff; the original attached part (2.7e-5) is kept.
    vertices = np.array([
        [0, 0, 0], [0.02, 0, 0], [0, 0.0009, 0],
        [0, -0.0009, 0], [0, 0, 0.0009], [-0.001, 0, 0],
        [1, 0, 0], [1.001, 0, 0], [1, 0.001, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4],
                      [0, 2, 5], [6, 7, 8]], dtype=np.int64)
    simplify_calls = []

    def identity_simplify(v, f, target_reduction):
        simplify_calls.append(target_reduction)
        return v, f

    out_v, out_f = _cleanup_and_simplify_mesh(
        vertices, faces, target_faces=1, no_cleanup=False,
        simplify=identity_simplify, log=lambda *a, **k: None)

    assert simplify_calls  # coarse and final passes ran, each followed by cleanup
    assert _triangles(vertices, faces[:4]) <= _triangles(out_v, out_f)
    assert not _triangles(vertices, faces[4:]) & _triangles(out_v, out_f)
