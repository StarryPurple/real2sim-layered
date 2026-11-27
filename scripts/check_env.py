"""Verify PyTorch CUDA, headless MuJoCo (EGL), Blender 4.2, OpenCV, Trimesh, and ffmpeg."""

from __future__ import annotations

import os
import subprocess
import sys

from real2sim_layered.schemas import resolve_blender_bin
from real2sim_layered.sim._mujoco import load_mjcf_model, mj_step, new_mj_data


def check_torch_gpu() -> None:
    import torch

    print(
        f"PyTorch: {torch.__version__} (CUDA built={torch.version.cuda}, avail={torch.cuda.is_available()})"
    )
    assert torch.cuda.is_available(), "CUDA is not available"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    a = torch.randn((256, 256), device=dev)
    b = torch.randn((256, 256), device=dev)
    c = (a @ b).sum().item()
    print(f"GPU[0]:  {name} sm_{cap[0]}{cap[1]}, matmul_ok={c:.2f}")


def check_python_stack() -> None:
    import cv2
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    import scipy
    import trimesh

    print(f"Python:  {sys.version.split()[0]} ({sys.executable})")
    print(f"MuJoCo:  {mujoco.__version__} (MUJOCO_GL={os.environ.get('MUJOCO_GL')})")
    print(f"OpenCV:  {cv2.__version__}")
    print(f"SciPy:   {scipy.__version__}")
    print(f"Trimesh: {trimesh.__version__}")
    print(f"ffmpeg:  {imageio_ffmpeg.get_ffmpeg_exe()}")

    xml = """
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="mass" pos="0 0 0.5">
          <joint name="slide" type="slide" axis="0 0 1" stiffness="65" damping="0.7" springref="0.0"/>
          <geom type="sphere" size="0.05" mass="0.2" rgba="0.1 0.6 0.8 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = load_mjcf_model(xml)
    data = new_mj_data(model)
    for _ in range(500):
        mj_step(model, data)
    qpos = float(data.qpos[0])
    assert -0.04 < qpos < -0.02, f"Unexpected spring equilibrium: {qpos}"

    renderer = mujoco.Renderer(model, height=120, width=160)
    renderer.update_scene(data)
    img = renderer.render()
    renderer.close()
    assert img.shape == (120, 160, 3), f"Unexpected MuJoCo render shape: {img.shape}"
    assert np.std(img) > 1.0, "Blank MuJoCo render"
    print(f"MuJoCo spring-mass + offscreen render OK: qpos={qpos:.4f} m, img={img.shape}")


def check_blender() -> None:
    blender = resolve_blender_bin()
    res = subprocess.run(
        [
            blender,
            "-b",
            "--factory-startup",
            "--python-expr",
            (
                "import bpy; "
                "bpy.ops.mesh.primitive_cube_add(location=(0,0,2)); "
                "obj=bpy.context.active_object; "
                "bpy.ops.rigidbody.object_add(); "
                "bpy.context.scene.frame_end=10; "
                "[bpy.context.scene.frame_set(f) for f in range(1,11)]; "
                "print('BLENDER_RIGID_Z=', round(obj.matrix_world.translation.z, 4))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = [line for line in res.stdout.splitlines() if "BLENDER_RIGID_Z=" in line]
    assert marker, f"Blender test failed:\n{res.stdout}\n{res.stderr}"
    print(f"Blender headless rigid-body OK: {marker[-1]} ({blender})")


if __name__ == "__main__":
    check_torch_gpu()
    check_python_stack()
    check_blender()
    print("ALL CHECKS PASSED")
