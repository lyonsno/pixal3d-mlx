"""Reuse TRELLIS2MLX's optional original Metal exporter on saved Pixal fields.

Inference remains MLX. The explicitly configured external interpreter supplies
Torch-facing Metal mesh, BVH, rasterization and sparse-sampling dependencies.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def validate_config(config_path):
    """Reject unavailable or ambiguous exporters before a generation begins."""
    config = _read_json(config_path)
    if config.get("schema") != "pixal3d.original-metal-finishing.v1":
        raise ValueError("explicit original Metal finishing configuration required")
    for key in ("source_root", "python", "backend_manifest"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError(f"absolute {key} required")
        if not Path(config[key]).exists():
            raise ValueError(f"configured {key} does not exist")
    paths = config.get("python_path")
    if not isinstance(paths, list) or not all(isinstance(p, str) and Path(p).is_absolute() for p in paths):
        raise ValueError("explicit absolute python_path list required")
    source = Path(config["source_root"]).resolve()
    if not (source / "scripts/source_metal_remesh_finishing.py").is_file():
        raise ValueError("configured checkout has no original Metal finishing runner")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != config.get("source_revision") or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("exact clean TRELLIS exporter source required")
    band, project = float(config.get("band", 1.0)), float(config.get("project_back", 0.0))
    if not np.isfinite(band) or band <= 0 or not np.isfinite(project) or not 0 <= project <= 1:
        raise ValueError("positive finite band and project_back in [0,1] required")
    return config


def prepare_inputs(checkpoints, destination):
    """Keep decoded appearance values unchanged and verify one spatial batch."""
    checkpoints, destination = Path(checkpoints).resolve(), Path(destination)
    raw, texture = checkpoints / "raw_mesh.npz", checkpoints / "texture.npz"
    grid_values = []
    for name, field in (("raw_mesh.json", "mesh_grid_size"), ("mesh.json", "hr_resolution"),
                        ("shape_latent.json", "hr_resolution")):
        path = checkpoints / name
        if path.exists() and field in (metadata := _read_json(path)):
            grid_values.append(metadata[field])
    if (not grid_values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                              for v in grid_values) or len(set(grid_values)) != 1):
        raise ValueError("checkpoint grid metadata must declare one matching positive resolution")
    grid = grid_values[0]
    inputs = {name: {"path": str(path), "sha256": _digest(path)}
              for name, path in (("raw_mesh", raw), ("texture", texture))}
    with np.load(texture, allow_pickle=False) as saved:
        attrs, coords = saved["tex_out"], saved["tex_coords"]
    if (attrs.ndim != 2 or attrs.shape[1] != 6 or not len(attrs)
            or coords.shape != (len(attrs), 4) or coords.dtype.kind not in "iu"
            or attrs.dtype.kind != "f" or not np.isfinite(attrs).all() or not np.all(coords[:, 0] == 0)
            or coords[:, 1:].min() < 0 or coords[:, 1:].max() >= grid):
        raise ValueError("finite decoded six-channel appearance and one valid coordinate batch required")
    destination.mkdir()
    (destination / "mesh_raw.npz").symlink_to(raw)
    # generate_pixal3d saves AFTER *0.5+0.5. A second affine decode is wrong.
    np.savez(destination / "texture.npz", tex_np=attrs,
             tex_coords_spatial=coords[:, 1:4], mesh_grid_size=grid)
    with np.load(destination / "texture.npz", allow_pickle=False) as prepared:
        if (not np.array_equal(prepared["tex_np"], attrs)
                or not np.array_equal(prepared["tex_coords_spatial"], coords[:, 1:4])):
            raise ValueError("adapted appearance changed")
    return {"grid_size": grid, "inputs": inputs,
            "mesh_sha256": inputs["raw_mesh"]["sha256"],
            "texture_sha256": _digest(destination / "texture.npz"),
            "appearance_transform": "identity: saved tex_out already decoded; zero batch removed"}


def validate_source_result(result, *, source, revision, backend_sha256, prepared,
                           target_faces, texture_size, band, project_back, output):
    """Check the observed source runner's receipt before consuming its product."""
    expected = {"accelerator": "Metal custom kernels", "host_tensors": "torch-cpu",
                "grid_size": prepared["grid_size"], "target_faces": target_faces,
                "texture_size": texture_size, "band": band, "project_back": project_back,
                "alpha_mode": "OPAQUE", "double_sided": False}
    requested = result.get("requested", {})
    if (result.get("schema") != "trellis2mlx.original-metal-remesh.v1"
            or result.get("status") != "complete" or result.get("failure_phase") is not None
            or result.get("source", {}).get("root") != str(source)
            or result.get("source", {}).get("revision") != revision
            or result.get("backend_manifest_sha256") != backend_sha256
            or any(result.get("effective", {}).get(k) != v for k, v in expected.items())
            or requested.get("mesh_sha256") != prepared["mesh_sha256"]
            or requested.get("texture_sha256") != prepared["texture_sha256"]
            or [s.get("name") for s in result.get("stages", [])]
            != ["raw", "filled_original", "rebuilt", "reduced", "uv", "normals"]
            or result.get("output", {}).get("path") != str(output)
            or _digest(output) != result.get("output", {}).get("sha256")):
        raise ValueError("original Metal result does not match requested source, inputs, backend and settings")


def export_pixal_frame(source_glb, output):
    import trimesh

    scene = trimesh.load(source_glb, force="scene", process=False)
    # Source exporter: (x,z,-y). Pixal's final frame: (-x,y,-z).
    scene.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float))
    scene.export(output)
    reopened = trimesh.load(output, force="scene", process=False)
    geometries = list(reopened.geometry.values())
    if not geometries or any(not len(g.faces) or not np.isfinite(g.vertices).all() for g in geometries):
        raise ValueError("empty or nonfinite Pixal export")
    return {"path": str(output), "sha256": _digest(output), "size_bytes": Path(output).stat().st_size,
            "vertices": sum(len(g.vertices) for g in geometries),
            "faces": sum(len(g.faces) for g in geometries), "frame": "(-x,y,-z)"}


def finish(*, checkpoints, output_dir, config_path, target_faces, texture_size):
    output_dir = Path(output_dir).resolve()
    report = {"schema": "pixal3d.saved-metal-finishing.v1", "status": "failed",
              "phase": "output_admission", "model_calls": 0,
              "requested": {"checkpoints": str(checkpoints), "output_dir": str(output_dir),
                            "config_path": str(config_path), "target_faces": target_faces,
                            "texture_size": texture_size}}
    report_path, owns_output = output_dir / "report.json", False

    def persist():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise FileExistsError(f"previous finishing evidence preserved: {output_dir}")
        with (output_dir / ".owner").open("x"):
            pass
        owns_output = True
        persist()
        report["phase"] = "source_admission"
        config = validate_config(config_path)
        paths = config["python_path"]
        source = Path(config["source_root"]).resolve()
        revision = config["source_revision"]
        if target_faces <= 0 or texture_size <= 0:
            raise ValueError("positive target faces and texture size required")
        band, project = float(config.get("band", 1.0)), float(config.get("project_back", 0.0))
        backend = Path(config["backend_manifest"]).resolve()
        report["config_sha256"] = _digest(config_path)
        report["phase"] = "input_admission"
        report["prepared"] = prepared = prepare_inputs(checkpoints, output_dir / "prepared")
        source_output = output_dir / "source"
        argv = [config["python"], "-u", str(source / "scripts/source_metal_remesh_finishing.py"),
                "--checkpoints", str(output_dir / "prepared"), "--output-dir", str(source_output),
                "--mesh-sha256", prepared["mesh_sha256"], "--texture-sha256", prepared["texture_sha256"],
                "--backend-manifest", str(backend), "--expected-revision", revision,
                "--grid-size", str(prepared["grid_size"]), "--target-faces", str(target_faces),
                "--texture-size", str(texture_size), "--band", str(band), "--project-back", str(project)]
        report.update(phase="source_finishing", command=argv)
        persist()
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join([str(source), *paths])}
        with (output_dir / "stdout.log").open("w") as stdout, (output_dir / "stderr.log").open("w") as stderr:
            process = subprocess.run(argv, cwd=source, env=env, stdout=stdout, stderr=stderr)
        report["source_exit_code"] = process.returncode
        if not (source_output / "report.json").exists():
            raise RuntimeError(f"source exporter exited {process.returncode} before writing its report; see stderr.log")
        report["source_result"] = result = _read_json(source_output / "report.json")
        if process.returncode:
            raise RuntimeError(result.get("error", "original Metal exporter failed"))
        validate_source_result(result, source=source, revision=revision, backend_sha256=_digest(backend),
                               prepared=prepared, target_faces=target_faces, texture_size=texture_size,
                               band=band, project_back=project, output=source_output / "output.glb")
        report["phase"] = "pixal_export_frame"
        report["output"] = export_pixal_frame(source_output / "output.glb", output_dir / "output.glb")
        if report["output"]["faces"] != result["output"]["faces"]:
            raise ValueError("Pixal frame conversion changed face count")
        for row in prepared["inputs"].values():
            if _digest(row["path"]) != row["sha256"]:
                raise ValueError("saved prediction changed during finishing")
        report.update(status="complete", phase=None)
    except Exception as error:
        report.update(failure_phase=report["phase"], error=f"{type(error).__name__}: {error}")
        if not owns_output:
            report_path = output_dir.parent / f"{output_dir.name}-failure-{uuid.uuid4().hex}.json"
    persist()
    print(f"{report['status']}: {report_path}", flush=True)
    return report
