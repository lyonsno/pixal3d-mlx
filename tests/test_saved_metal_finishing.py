"""Saved finishing must not replay inference or reinterpret decoded appearance."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


def test_finish_only_dispatches_before_any_model_or_rng_work(monkeypatch, tmp_path):
    import resume_pixal3d

    seen = {}

    def finish(**kwargs):
        seen.update(kwargs)
        return {"status": "complete"}

    def inference_forbidden(*args, **kwargs):
        raise AssertionError("saved finishing must not initialize inference")

    monkeypatch.setitem(sys.modules, "trellmlx.metal_finish", SimpleNamespace(finish=finish))
    monkeypatch.setattr(resume_pixal3d.mx.random, "seed", inference_forbidden)
    monkeypatch.setattr(resume_pixal3d.mx.metal, "reset_peak_memory", inference_forbidden)
    monkeypatch.setattr(sys, "argv", [
        "resume_pixal3d.py", "--finish-only", "--ckpt", str(tmp_path / "checkpoints"),
        "--finishing-config", str(tmp_path / "metal.json"),
        "--finishing-output", str(tmp_path / "finished"),
        "--target-faces", "200000", "--texture-size", "1024",
    ])
    with pytest.raises(SystemExit) as stopped:
        resume_pixal3d.main()
    assert stopped.value.code == 0
    assert seen == {
        "checkpoints": Path(tmp_path / "checkpoints"),
        "output_dir": Path(tmp_path / "finished"),
        "config_path": Path(tmp_path / "metal.json"),
        "target_faces": 200000,
        "texture_size": 1024,
    }


def saved_fields(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "raw_mesh.npz", vertices=np.eye(3, dtype=np.float32), faces=[[0, 1, 2]])
    attrs = np.array([[-.25, .0, .5, .75, 1., 1.25], [.3] * 6], dtype=np.float32)
    coords = np.array([[0, 0, 2, 3], [0, 7, 6, 5]], dtype=np.int32)
    np.savez(checkpoints / "texture.npz", tex_out=attrs, tex_coords=coords)
    (checkpoints / "mesh.json").write_text('{"hr_resolution": 512}')
    return checkpoints, attrs, coords


def test_saved_appearance_is_identity_including_values_outside_unit_interval(tmp_path):
    from trellmlx.metal_finish import prepare_inputs

    checkpoints, attrs, coords = saved_fields(tmp_path)
    result = prepare_inputs(checkpoints, tmp_path / "prepared")
    assert result["grid_size"] == 512
    with np.load(tmp_path / "prepared" / "texture.npz") as data:
        np.testing.assert_array_equal(data["tex_np"], attrs)
        np.testing.assert_array_equal(data["tex_coords_spatial"], coords[:, 1:])
    assert (tmp_path / "prepared" / "mesh_raw.npz").resolve() == checkpoints / "raw_mesh.npz"


@pytest.mark.parametrize("defect", ["conflicting_grid", "missing_grid", "multiple_batches", "outside_grid"])
def test_ambiguous_saved_inputs_fail_instead_of_substituting_defaults(tmp_path, defect):
    from trellmlx.metal_finish import prepare_inputs

    checkpoints, attrs, coords = saved_fields(tmp_path)
    if defect == "conflicting_grid":
        (checkpoints / "raw_mesh.json").write_text('{"mesh_grid_size": 1024}')
    elif defect == "missing_grid":
        (checkpoints / "mesh.json").unlink()
    else:
        coords[0, 0 if defect == "multiple_batches" else 1] = 1 if defect == "multiple_batches" else 512
        np.savez(checkpoints / "texture.npz", tex_out=attrs, tex_coords=coords)
    with pytest.raises(ValueError):
        prepare_inputs(checkpoints, tmp_path / "prepared")


def source_receipt(tmp_path):
    """Synthetic guard fixture; actual Metal conformance is exercised separately."""
    from trellmlx.metal_finish import _digest

    output = tmp_path / "source.glb"
    output.write_bytes(b"guard-only fixture")
    prepared = {"grid_size": 512, "mesh_sha256": "mesh", "texture_sha256": "texture"}
    bindings = dict(source=tmp_path, revision="revision", backend_sha256="backend", prepared=prepared,
                    target_faces=200000, texture_size=1024, band=1., project_back=0., output=output)
    result = {
        "schema": "trellis2mlx.original-metal-remesh.v1", "status": "complete", "failure_phase": None,
        "source": {"root": str(tmp_path), "revision": "revision"}, "backend_manifest_sha256": "backend",
        "requested": {"mesh_sha256": "mesh", "texture_sha256": "texture"},
        "effective": {"accelerator": "Metal custom kernels", "host_tensors": "torch-cpu", "grid_size": 512,
                      "target_faces": 200000, "texture_size": 1024, "band": 1., "project_back": 0.,
                      "alpha_mode": "OPAQUE", "double_sided": False},
        "stages": [{"name": n} for n in ["raw", "filled_original", "rebuilt", "reduced", "uv", "normals"]],
        "output": {"path": str(output), "sha256": _digest(output)},
    }
    return result, bindings


def test_source_receipt_allows_additive_fields(tmp_path):
    from trellmlx.metal_finish import validate_source_result

    result, bindings = source_receipt(tmp_path)
    result["source"]["extra_provenance"] = "compatible"
    result["new_diagnostic"] = 1
    validate_source_result(result, **bindings)


@pytest.mark.parametrize("defect", ["fallback", "wrong_grid", "wrong_input", "partial", "changed_output"])
def test_false_source_closure_is_rejected(tmp_path, defect):
    from trellmlx.metal_finish import validate_source_result

    result, bindings = source_receipt(tmp_path)
    if defect == "fallback":
        result["effective"]["accelerator"] = "CPU fallback"
    elif defect == "wrong_grid":
        result["effective"]["grid_size"] = 1024
    elif defect == "wrong_input":
        result["requested"]["mesh_sha256"] = "stale"
    elif defect == "partial":
        result["stages"].pop()
    else:
        bindings["output"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        validate_source_result(result, **bindings)


def test_pixal_frame_changes_only_geometry_and_normals(tmp_path):
    import trimesh
    from PIL import Image
    from trellmlx.metal_finish import export_pixal_frame

    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]],
                           vertex_normals=np.tile([0., 0., 1.], (3, 1)), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (2, 2), (30, 80, 160, 255)),
            alphaMode="OPAQUE", doubleSided=False))
    original, output = tmp_path / "source.glb", tmp_path / "pixal.glb"
    mesh.export(original)
    before = trimesh.load(original, force="scene", process=False).to_geometry()
    record = export_pixal_frame(original, output)
    after = trimesh.load(output, force="scene", process=False).to_geometry()
    np.testing.assert_array_equal(after.vertices, -before.vertices[:, [0, 2, 1]])
    np.testing.assert_array_equal(after.faces, before.faces)
    np.testing.assert_allclose(after.vertex_normals, -before.vertex_normals[:, [0, 2, 1]], atol=1e-7)
    np.testing.assert_array_equal(after.visual.uv, before.visual.uv)
    np.testing.assert_array_equal(np.asarray(after.visual.material.baseColorTexture),
                                  np.asarray(before.visual.material.baseColorTexture))
    assert after.visual.material.alphaMode == "OPAQUE"
    assert after.visual.material.doubleSided is False
    assert record["faces"] == 1


def test_failure_before_source_report_still_preserves_outer_report(monkeypatch, tmp_path):
    import trellmlx.metal_finish as module

    checkpoints, _, _ = saved_fields(tmp_path)
    source = tmp_path / "trellis"
    source.mkdir()
    (source / "scripts").mkdir()
    (source / "scripts/source_metal_remesh_finishing.py").write_text("# mocked runner")
    backend = tmp_path / "backend.json"
    backend.write_text("{}")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": "pixal3d.original-metal-finishing.v1",
        "source_root": str(source), "source_revision": "revision", "python": sys.executable,
        "backend_manifest": str(backend), "python_path": []}))
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda argv, **kwargs: "revision\n" if "rev-parse" in argv else "")

    def fail(argv, **kwargs):
        kwargs["stderr"].write("optional backend unavailable\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fail)
    output = tmp_path / "finished"
    result = module.finish(checkpoints=checkpoints, output_dir=output, config_path=config,
                           target_faces=200000, texture_size=1024)
    assert result["status"] == "failed"
    assert result["failure_phase"] == "source_finishing"
    assert result["source_exit_code"] == 7
    assert "before writing its report" in result["error"]
    assert json.loads((output / "report.json").read_text())["status"] == "failed"
    assert not (output / "output.glb").exists()


def test_existing_evidence_is_preserved_on_refusal(tmp_path):
    from trellmlx.metal_finish import finish

    output = tmp_path / "finished"
    output.mkdir()
    (output / "report.json").write_text("original evidence")
    result = finish(checkpoints=tmp_path, output_dir=output, config_path=tmp_path / "missing.json",
                    target_faces=200000, texture_size=1024)
    assert result["failure_phase"] == "output_admission"
    assert (output / "report.json").read_text() == "original evidence"
    assert len(list(tmp_path.glob("finished-failure-*.json"))) == 1


def test_generation_rejects_invalid_exporter_before_model_initialization(monkeypatch, tmp_path):
    import generate_pixal3d
    import trellmlx.metal_finish as module

    def reject(path):
        raise ValueError("source revision mismatch")

    def inference_forbidden(*args, **kwargs):
        raise AssertionError("model initialization preceded finalizer admission")

    monkeypatch.setattr(module, "validate_config", reject)
    monkeypatch.setattr(generate_pixal3d.mx.random, "seed", inference_forbidden)
    monkeypatch.setattr(sys, "argv", ["generate_pixal3d.py", "--image", "unused.png",
        "--save-checkpoints", str(tmp_path), "--finishing-config", str(tmp_path / "metal.json")])
    with pytest.raises(ValueError, match="source revision mismatch"):
        generate_pixal3d.main()


def test_duplicate_config_fields_fail_loudly(tmp_path):
    from trellmlx.metal_finish import _read_json

    path = tmp_path / "ambiguous.json"
    path.write_text('{"source_revision": "old", "source_revision": "new"}')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        _read_json(path)
