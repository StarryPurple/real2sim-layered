"""Standalone Blender 4.2 worker script executed via `blender -b --python ... -- ...`."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build and render a CoarseSceneSpec in Blender.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="JSON list of integer frame indices.")
    parser.add_argument("--views", required=True, help="JSON list of camera view names.")
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--blend-out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--samples", type=int, default=6)
    return parser.parse_args(argv)


def _make_material(name: str, rgba: list[float]) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def _add_primitive(obj_spec: dict[str, Any]) -> Any:
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
    obj.data.materials.append(_make_material(f"{obj.name}_mat", obj_spec["color_rgba"]))
    for kf in obj_spec["keyframes"]:
        frame = int(kf["frame_idx"]) + 1
        obj.location = tuple(kf["position"])
        obj.rotation_euler = tuple(kf["euler_xyz"])
        obj.scale = tuple(kf["scale"])
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)
    return obj


def _add_camera(cam_spec: dict[str, Any]) -> Any:
    cam_data = bpy.data.cameras.new(name=cam_spec["name"])
    cam_data.lens_unit = "FOV"
    cam_data.angle_y = math.radians(float(cam_spec["fov_y_deg"]))
    cam_obj = bpy.data.objects.new(cam_spec["name"], cam_data)
    bpy.context.collection.objects.link(cam_obj)
    pos = Vector(cam_spec["position"])
    target = Vector(cam_spec["target"])
    cam_obj.location = pos
    cam_obj.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    return cam_obj


def _add_lights() -> None:
    for idx, (loc, energy) in enumerate([((0.6, -1.3, 1.8), 180.0), ((-1.1, 1.0, 1.5), 110.0)]):
        light_data = bpy.data.lights.new(name=f"Light_{idx}", type="POINT")
        light_data.energy = energy
        light_obj = bpy.data.objects.new(name=f"Light_{idx}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc


def main() -> None:
    args = _parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    frame_indices: list[int] = json.loads(args.frames)
    views: list[str] = json.loads(args.views)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.frame_start = 1
    scene.frame_end = max(1, int(spec["num_frames"]))

    _add_lights()
    for obj_spec in spec["objects"]:
        _add_primitive(obj_spec)

    cam_objs = {cam["name"]: _add_camera(cam) for cam in spec["cameras"]}
    if views and views[0] in cam_objs:
        scene.camera = cam_objs[views[0]]

    args.blend_out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))

    for view_name in views:
        if view_name not in cam_objs:
            continue
        scene.camera = cam_objs[view_name]
        view_dir = args.render_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in frame_indices:
            scene.frame_set(int(frame_idx) + 1)
            scene.render.filepath = str(view_dir / f"{int(frame_idx):06d}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
