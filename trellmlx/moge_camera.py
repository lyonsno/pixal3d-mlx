"""MoGe-2 camera parameter estimation for Pixal3D.

Estimates camera FOV from an input image using MoGe-2 via subprocess.
No torch dependency in the main process — MoGe runs in a separate
Python process using PyTorch/MPS.

If MoGe is not installed, falls back to the default fixed FOV.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np


# Inline script that runs MoGe and prints intrinsics as JSON.
# Runs in a separate process so torch never loads in our MLX process.
_MOGE_SCRIPT = '''
import json
import math
import sys
import numpy as np
from PIL import Image

image_path = sys.argv[1]

import torch
from moge.model.v2 import MoGeModel

model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
model = model.to("mps")
model.eval()

pil_image = Image.open(image_path).convert("RGB")
width, height = pil_image.size
image_np = np.array(pil_image).astype(np.float32) / 255.0
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to("mps")

with torch.no_grad():
    output = model.infer(image_tensor)

intrinsics = output["intrinsics"].squeeze().cpu().numpy()
fx_normalized = float(intrinsics[0, 0])
fx = fx_normalized * width
camera_angle_x = 2 * math.atan(width / (2 * fx))

print(json.dumps({
    "camera_angle_x": camera_angle_x,
    "fx_normalized": fx_normalized,
    "width": width,
    "height": height,
}))
'''


def estimate_camera_fov(image_path: str) -> dict | None:
    """Estimate camera FOV from an image using MoGe-2 subprocess.

    Returns:
        dict with 'camera_angle_x', 'fx_normalized', 'width', 'height'.
        Returns None if MoGe is not available.
    """
    # Write the inline script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_MOGE_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, str(image_path)],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No module named 'moge'" in stderr:
                print("  MoGe: not installed (pip install moge), using default FOV", flush=True)
            else:
                print(f"  MoGe: failed ({stderr[:200]})", flush=True)
            return None

        data = json.loads(result.stdout.strip())
        return data

    except subprocess.TimeoutExpired:
        print("  MoGe: timed out (120s)", flush=True)
        return None
    except (json.JSONDecodeError, Exception) as e:
        print(f"  MoGe: error ({e})", flush=True)
        return None
    finally:
        os.unlink(script_path)


def get_camera_params_with_moge(image_path: str, mesh_scale: float = 1.0,
                                 image_resolution: int = 512, extend_pixel: int = 0) -> dict | None:
    """Estimate full camera params (FOV + distance) using MoGe-2.

    Returns:
        dict with 'camera_angle_x', 'distance', 'mesh_scale'.
        Returns None if MoGe estimation failed.
    """
    moge_result = estimate_camera_fov(image_path)
    if moge_result is None:
        return None

    camera_angle_x = moge_result['camera_angle_x']

    # Compute distance from FOV (same math as generate_pixal3d.py)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    grid_point = np.array([-1.0, 0.0, 0.0]) @ rotation.T
    grid_point = grid_point / mesh_scale / 2.0

    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0

    xt = 0 - extend_pixel
    x_ndc = xt - image_resolution / 2.0
    xw, yw = grid_point[0], grid_point[1]

    distance = f_pixels * xw / x_ndc - yw

    fov_deg = math.degrees(camera_angle_x)
    print(f"  MoGe camera: FOV={fov_deg:.1f}°, distance={distance:.4f}", flush=True)

    return {
        'camera_angle_x': camera_angle_x,
        'distance': distance,
        'mesh_scale': mesh_scale,
    }
