"""Procedural Blender 4.2 Python script generator and headless RGBA renderer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_layered.schemas import CoarseSceneSpec

_BLENDER_TEMPLATE = '''"""Auto-generated procedural Blender 4.2 scene builder for real2sim-layered."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

SPEC = json.loads(r"""__SPEC_JSON__""")
FRAME_INDICES = __FRAME_INDICES__
VIEWS = __VIEWS__
RENDER_DIR = Path(r"__RENDER_DIR__")
BLEND_OUT = Path(r"__BLEND_OUT__")
WIDTH = __WIDTH__
HEIGHT = __HEIGHT__
SAMPLES = __SAMPLES__


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(name, rgba):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def add_primitive(obj_spec):
    prim = obj_spec["primitive"]
    hx, hy, hz = obj_spec["size_m"]
    if prim == "sphere":
        r = max(0.01, (hx + hy + hz) / 3.0)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=24, ring_count=16)
    elif prim in ("cylinder", "capsule"):
        r = max(0.01, 0.5 * (hx + hy))
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=max(0.02, 2.0 * hz), vertices=24)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.scale = (max(0.02, 2.0 * hx), max(0.02, 2.0 * hy), max(0.02, 2.0 * hz))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        mod = obj.modifiers.new(name="Bevel", type="BEVEL")
        mod.width = min(hx, hy, hz) * 0.12
        mod.segments = 2

    obj = bpy.context.active_object
    obj.name = obj_spec["object_id"]
    mat = make_material(f"{obj.name}_mat", obj_spec["color_rgba"])
    obj.data.materials.append(mat)
    for kf in obj_spec["keyframes"]:
        frame = int(kf["frame_idx"]) + 1
        obj.location = tuple(kf["position"])
        obj.rotation_euler = tuple(kf["euler_xyz"])
        obj.scale = tuple(kf["scale"])
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)
    return obj


def add_camera(cam_spec):
    cam_data = bpy.data.cameras.new(name=cam_spec["name"])
    cam_data.lens_unit = "FOV"
    cam_data.angle_y = math.radians(float(cam_spec["fov_y_deg"]))
    cam_obj = bpy.data.objects.new(cam_spec["name"], cam_data)
    bpy.context.collection.objects.link(cam_obj)
    pos = Vector(cam_spec["position"])
    target = Vector(cam_spec["target"])
    cam_obj.location = pos
    direction = target - pos
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam_obj


def add_lights():
    for idx, (loc, energy) in enumerate([((0.6, -1.3, 1.8), 180.0), ((-1.1, 1.0, 1.5), 110.0)]):
        light_data = bpy.data.lights.new(name=f"Light_{idx}", type="POINT")
        light_data.energy = energy
        light_obj = bpy.data.objects.new(name=f"Light_{idx}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc


def main():
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.frame_start = 1
    scene.frame_end = max(1, int(SPEC["num_frames"]))

    add_lights()
    for obj_spec in SPEC["objects"]:
        add_primitive(obj_spec)

    cam_objs = {}
    for cam_spec in SPEC["cameras"]:
        cam_objs[cam_spec["name"]] = add_camera(cam_spec)

    if VIEWS and VIEWS[0] in cam_objs:
        scene.camera = cam_objs[VIEWS[0]]

    BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))

    for view_name in VIEWS:
        if view_name not in cam_objs:
            continue
        scene.camera = cam_objs[view_name]
        view_dir = RENDER_DIR / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in FRAME_INDICES:
            scene.frame_set(int(frame_idx) + 1)
            scene.render.filepath = str(view_dir / f"{int(frame_idx):06d}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
'''


def generate_blender_script(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    views: list[str],
    render_dir: Path,
    blend_out: Path,
    width: int = 160,
    height: int = 120,
    samples: int = 6,
) -> str:
    """Compile a CoarseSceneSpec into a standalone Blender 4.2 bpy Python script."""
    script = _BLENDER_TEMPLATE
    replacements = {
        "__SPEC_JSON__": scene.model_dump_json(),
        "__FRAME_INDICES__": json.dumps(frame_indices),
        "__VIEWS__": json.dumps(views),
        "__RENDER_DIR__": str(render_dir.resolve()),
        "__BLEND_OUT__": str(blend_out.resolve()),
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__SAMPLES__": str(samples),
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    return script


def run_blender_scene(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    views: list[str],
    out_dir: Path,
    width: int = 160,
    height: int = 120,
    samples: int = 6,
) -> tuple[Path, Path, dict[str, list[np.ndarray]]]:
    """Generate build_scene.py, execute headless Blender to save scene.blend, and load RGBA renders."""
    blender_bin = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender_bin:
        raise RuntimeError(
            "BLENDER_BIN is not set and 'blender' is not on PATH; run 'source scripts/env.sh'."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "renders"
    blend_out = out_dir / "scene.blend"
    script_path = out_dir / "build_scene.py"

    script_text = generate_blender_script(
        scene=scene,
        frame_indices=frame_indices,
        views=views,
        render_dir=render_dir,
        blend_out=blend_out,
        width=width,
        height=height,
        samples=samples,
    )
    script_path.write_text(script_text, encoding="utf-8")

    subprocess.run(
        [blender_bin, "-b", "--factory-startup", "--python", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not blend_out.is_file():
        raise RuntimeError(f"Blender finished without creating {blend_out}")

    rendered: dict[str, list[np.ndarray]] = {}
    for view_name in views:
        frames: list[np.ndarray] = []
        for frame_idx in frame_indices:
            img_path = render_dir / view_name / f"{frame_idx:06d}.png"
            if not img_path.is_file():
                continue
            with Image.open(img_path) as img:
                rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
                frames.append(rgba)
        if frames:
            rendered[view_name] = frames

    return script_path, blend_out, rendered
