"""Pixal3D conditioning image and camera must follow upstream inference.

Upstream (TencentARC/Pixal3D):
- pipeline.preprocess_image crops the alpha bbox with a 1.1x margin
  (pixal3d/pipelines/pixal3d_image_to_3d.py:173-186);
- inference.py:192-215 runs MoGe-2 on that *preprocessed* image and uses one
  camera for every stage.
"""

import json
import sys

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image


class Reached(Exception):
    pass


def _rgba_fixture(path):
    """64x48 RGBA with an off-center opaque 21x13 object."""
    arr = np.zeros((48, 64, 4), dtype=np.uint8)
    arr[..., :3] = (200, 120, 40)
    arr[10:23, 30:51, 3] = 255
    Image.fromarray(arr, "RGBA").save(path)
    return path


def _upstream_preprocess(image: Image.Image, bg_color=(0, 0, 0)) -> Image.Image:
    """Transcription of upstream Pixal3DImageTo3DPipeline.preprocess_image (alpha path)."""
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)),
                             Image.Resampling.LANCZOS)
    output = image
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    rgb = output[:, :, :3]
    a = output[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    output = rgb * a + bg * (1.0 - a)
    return Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))


def test_projection_features_use_upstream_pixal3d_crop(monkeypatch, tmp_path):
    import generate_pixal3d
    import trellmlx.models.dinov3_proj as dinov3_proj

    image_path = _rgba_fixture(tmp_path / "in.png")
    seen = {}

    def capture_preprocess(img, size):
        seen["image"] = np.array(img.convert("RGB"))
        return mx.zeros((1, 3, size, size))

    class StubExtractor:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a):
            return mx.zeros((1, 5, 4)), mx.zeros((1, 8, 4))

    monkeypatch.setattr(dinov3_proj, "preprocess_image", capture_preprocess)
    monkeypatch.setattr(dinov3_proj, "DinoV3ProjFeatureExtractor", StubExtractor)

    generate_pixal3d.extract_proj_features(
        str(image_path), object(), grid_resolution=2, image_size=8,
        camera_params=generate_pixal3d.get_default_camera_params())

    expected = np.array(_upstream_preprocess(Image.open(image_path)))
    assert seen["image"].shape == expected.shape
    np.testing.assert_array_equal(seen["image"], expected)


def _stub_camera_route(monkeypatch, tmp_path, captured):
    import trellmlx.moge_camera as moge_camera

    def estimator(image_path, **kwargs):
        captured["moge_image"] = np.array(Image.open(image_path).convert("RGB"))
        captured["moge_path"] = str(image_path)
        return {"camera_angle_x": 0.6, "distance": 2.1, "mesh_scale": 1.0}

    monkeypatch.setattr(moge_camera, "estimate_camera_params_mlx", estimator)


def _stub_models(monkeypatch):
    import trellmlx.models.dinov3 as dinov3
    import trellmlx.models.naf as naf
    import trellmlx.models.naf_loader as naf_loader
    import trellmlx.weight_loader as weight_loader

    monkeypatch.setattr(dinov3, "DINOv3ViT", lambda: object())
    monkeypatch.setattr(dinov3, "load_dinov3_weights", lambda *a, **k: 0)
    monkeypatch.setattr(naf, "NAF", lambda: object())
    monkeypatch.setattr(naf_loader, "load_naf_weights", lambda *a, **k: None)
    monkeypatch.setattr(weight_loader, "load_weights", lambda *a, **k: None)


def test_moge_and_stage_conditioning_share_the_preprocessed_image(monkeypatch, tmp_path):
    import generate_pixal3d

    image_path = _rgba_fixture(tmp_path / "in.png")
    captured = {}
    _stub_camera_route(monkeypatch, tmp_path, captured)
    _stub_models(monkeypatch)

    def extract(image, dinov3_model, grid_resolution, image_size, camera_params, no_rembg=False, **k):
        captured["stage_image"] = np.array(Image.open(image).convert("RGB")) if no_rembg else None
        captured["stage_camera"] = dict(camera_params)
        raise Reached

    monkeypatch.setattr(generate_pixal3d, "extract_proj_features", extract)
    monkeypatch.setattr(sys, "argv", ["generate_pixal3d.py", "--image", str(image_path),
                                     "--output", str(tmp_path / "out.glb")])
    with pytest.raises(Reached):
        generate_pixal3d.main()

    expected = np.array(_upstream_preprocess(Image.open(image_path)))
    assert captured["moge_image"].shape == expected.shape
    np.testing.assert_array_equal(captured["moge_image"], expected)
    assert captured["stage_image"] is not None, "stage re-preprocessed instead of using MoGe's image"
    np.testing.assert_array_equal(captured["stage_image"], expected)
    assert captured["stage_camera"]["camera_angle_x"] == pytest.approx(0.6)
    assert captured["stage_camera"]["distance"] == pytest.approx(2.1)


def test_generation_checkpoints_the_camera(monkeypatch, tmp_path):
    import generate_pixal3d

    image_path = _rgba_fixture(tmp_path / "in.png")
    ckpt = tmp_path / "ckpt"
    _stub_camera_route(monkeypatch, tmp_path, {})
    _stub_models(monkeypatch)

    def extract(*a, **k):
        raise Reached

    monkeypatch.setattr(generate_pixal3d, "extract_proj_features", extract)
    monkeypatch.setattr(sys, "argv", ["generate_pixal3d.py", "--image", str(image_path),
                                     "--output", str(tmp_path / "out.glb"),
                                     "--save-checkpoints", str(ckpt)])
    with pytest.raises(Reached):
        generate_pixal3d.main()

    meta = json.loads((ckpt / "camera.json").read_text())
    assert meta["camera_angle_x"] == pytest.approx(0.6)
    assert meta["distance"] == pytest.approx(2.1)
    assert meta["mesh_scale"] == pytest.approx(1.0)


def test_resume_uses_checkpointed_camera(monkeypatch, tmp_path):
    import generate
    import resume_pixal3d
    import trellmlx.cleanup as cleanup
    import trellmlx.mesh_extract as mesh_extract
    import trellmlx.models.shape_slat_decoder as shape_decoder
    import trellmlx.models.pixal3d_flow as pixal_flow
    from trellmlx.checkpoint import save_checkpoint

    ckpt = tmp_path / "ckpt"
    save_checkpoint(str(ckpt), "shape_latent", hr_slat=np.zeros((1, 32), dtype=np.float32),
                    quant_coords=np.zeros((1, 4), dtype=np.int32), hr_resolution=512)
    save_checkpoint(str(ckpt), "camera", camera_angle_x=0.6, distance=2.1, mesh_scale=1.0)
    image_path = _rgba_fixture(tmp_path / "in.png")
    _stub_models(monkeypatch)

    class TinyDecoder:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return mx.zeros((1, 7)), mx.zeros((1, 4), dtype=mx.int32), []

    seen = {}

    def extract(*a, camera_params, **k):
        seen.update(camera_params)
        raise Reached

    monkeypatch.setattr(shape_decoder, "SLatDecoder", TinyDecoder)
    monkeypatch.setattr(cleanup, "cleanup_model", lambda *a, **k: None)
    monkeypatch.setattr(mesh_extract, "decoder_output_to_mesh", lambda *a, **k: ([], []))
    monkeypatch.setattr(generate, "_cleanup_and_simplify_mesh", lambda v, f, **k: (v, f))
    monkeypatch.setattr(pixal_flow, "Pixal3DSLatFlowModel", lambda **k: object())
    monkeypatch.setattr(resume_pixal3d, "extract_proj_features", extract)
    monkeypatch.setattr(sys, "argv", ["resume_pixal3d.py", "--image", str(image_path),
                                     "--ckpt", str(ckpt)])
    with pytest.raises(Reached):
        resume_pixal3d.main()

    assert seen["camera_angle_x"] == pytest.approx(0.6)
    assert seen["distance"] == pytest.approx(2.1)
