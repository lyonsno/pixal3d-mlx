"""GLB export frame must match upstream Pixal3D inference.py."""

import numpy as np

import generate_pixal3d
import resume_pixal3d

# o_voxel postprocess.to_glb: (x, y, z) -> (x, z, -y)
TO_GLB = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
# Pixal3D inference.py:271-278 (upper-left 3x3 of the 4x4 applied to the GLB)
UPSTREAM_ROT = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)


def test_export_frame_matches_upstream_to_glb_then_rotation():
    rng = np.random.default_rng(0)
    v = rng.uniform(-0.5, 0.5, size=(16, 3))
    want = v @ TO_GLB.T @ UPSTREAM_ROT.T

    got = generate_pixal3d.to_pixal3d_glb_frame(v)

    np.testing.assert_allclose(got, want, atol=0)
    np.testing.assert_allclose(got, v * np.array([-1, 1, -1]), atol=0)


def test_export_frame_is_proper_rotation_and_does_not_mutate_input():
    v = np.eye(3)
    before = v.copy()
    m = generate_pixal3d.to_pixal3d_glb_frame(v)
    assert np.isclose(np.linalg.det(m), 1.0)
    np.testing.assert_array_equal(v, before)


def test_resume_exports_with_same_frame():
    assert resume_pixal3d.to_pixal3d_glb_frame is generate_pixal3d.to_pixal3d_glb_frame
