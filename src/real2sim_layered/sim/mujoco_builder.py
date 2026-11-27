"""MuJoCo MJCF compiler and offscreen kinematic/foreground RGBA renderer."""

from __future__ import annotations

import itertools
import math
from xml.sax.saxutils import escape

import mujoco
import numpy as np

from real2sim_layered.schemas import CameraSpec, CoarseSceneSpec, KeyframePose, SceneObjectSpec
from real2sim_layered.sim._mujoco import (
    joint_qpos_adr,
    load_mjcf_model,
    mj_forward,
    new_mj_data,
)


def _camera_xyaxes(cam: CameraSpec) -> str:
    """Compute MuJoCo camera `xyaxes` (right_xyz up_xyz) looking from position to target."""
    pos = np.array(cam.position, dtype=np.float64)
    target = np.array(cam.target, dtype=np.float64)
    forward = target - pos
    norm = float(np.linalg.norm(forward))
    forward = np.array([0.0, 1.0, 0.0]) if norm < 1e-8 else forward / norm
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= max(1e-8, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    up /= max(1e-8, float(np.linalg.norm(up)))
    vals = [*right.tolist(), *up.tolist()]
    return " ".join(f"{v:.5f}" for v in vals)


def _geom_size_str(obj: SceneObjectSpec) -> str:
    hx, hy, hz = obj.size_m
    if obj.primitive == "sphere":
        r = max(0.01, (hx + hy + hz) / 3.0)
        return f"{r:.4f}"
    if obj.primitive in ("cylinder", "capsule"):
        r = max(0.01, 0.5 * (hx + hy))
        return f"{r:.4f} {max(0.01, hz):.4f}"
    return f"{max(0.01, hx):.4f} {max(0.01, hy):.4f} {max(0.01, hz):.4f}"


def _euler_to_quat_wxyz(euler_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    rx, ry, rz = euler_xyz
    cx, sx = math.cos(0.5 * rx), math.sin(0.5 * rx)
    cy, sy = math.cos(0.5 * ry), math.sin(0.5 * ry)
    cz, sz = math.cos(0.5 * rz), math.sin(0.5 * rz)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


def interpolate_object_pose(
    keyframes: list[KeyframePose],
    frame_idx: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Linearly interpolate (position, euler_xyz) for `frame_idx` across sorted keyframes."""
    ordered = sorted(keyframes, key=lambda k: k.frame_idx)
    if frame_idx <= ordered[0].frame_idx:
        return ordered[0].position, ordered[0].euler_xyz
    if frame_idx >= ordered[-1].frame_idx:
        return ordered[-1].position, ordered[-1].euler_xyz

    for left, right in itertools.pairwise(ordered):
        if left.frame_idx <= frame_idx <= right.frame_idx:
            span = max(1, right.frame_idx - left.frame_idx)
            alpha = (frame_idx - left.frame_idx) / float(span)
            pos = tuple(
                left.position[i] * (1.0 - alpha) + right.position[i] * alpha for i in range(3)
            )
            eul = tuple(
                left.euler_xyz[i] * (1.0 - alpha) + right.euler_xyz[i] * alpha for i in range(3)
            )
            return (pos[0], pos[1], pos[2]), (eul[0], eul[1], eul[2])
    return ordered[-1].position, ordered[-1].euler_xyz


def build_mjcf_xml(scene: CoarseSceneSpec, include_floor: bool = True) -> str:
    """Compile a CoarseSceneSpec into an executable MuJoCo MJCF XML string."""
    lines = [
        f'<mujoco model="{escape(scene.case_id)}">',
        '  <option gravity="0 0 -9.81" timestep="0.005"/>',
        '  <visual><rgba haze="0 0 0 0"/></visual>',
        "  <worldbody>",
        '    <light pos="0.5 -1.2 1.8" dir="-0.2 0.6 -0.8" diffuse="0.9 0.9 0.9"/>',
        '    <light pos="-1.0 1.0 1.5" dir="0.4 -0.4 -0.8" diffuse="0.5 0.5 0.5"/>',
    ]
    if include_floor:
        lines.append(
            f'    <geom name="floor" type="plane" pos="0 0 {scene.floor_z:.4f}" '
            'size="3 3 0.1" rgba="0.95 0.95 0.95 1" friction="0.7 0.01 0.001"/>'
        )
    for cam in scene.cameras:
        pos_s = " ".join(f"{v:.4f}" for v in cam.position)
        xy_s = _camera_xyaxes(cam)
        lines.append(
            f'    <camera name="{escape(cam.name)}" pos="{pos_s}" '
            f'xyaxes="{xy_s}" fovy="{cam.fov_y_deg:.2f}"/>'
        )

    for obj in scene.objects:
        kf0 = sorted(obj.keyframes, key=lambda k: k.frame_idx)[0]
        pos_s = " ".join(f"{v:.4f}" for v in kf0.position)
        quat_s = " ".join(f"{v:.5f}" for v in _euler_to_quat_wxyz(kf0.euler_xyz))
        rgba_s = " ".join(f"{v:.4f}" for v in obj.color_rgba)
        size_s = _geom_size_str(obj)
        safe_id = escape(obj.object_id)
        lines.append(f'    <body name="{safe_id}" pos="{pos_s}" quat="{quat_s}">')
        if obj.is_dynamic:
            lines.append(f'      <freejoint name="{safe_id}_jnt"/>')
        lines.append(
            f'      <geom name="{safe_id}_geom" type="{obj.primitive}" size="{size_s}" '
            f'mass="{obj.mass_kg:.4f}" friction="{obj.friction:.3f} 0.01 0.001" rgba="{rgba_s}"/>'
        )
        lines.append("    </body>")

    lines.extend(["  </worldbody>", "</mujoco>"])
    return "\n".join(lines) + "\n"


def render_coarse_scene_mujoco(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    views: list[str],
    width: int = 160,
    height: int = 120,
) -> dict[str, list[np.ndarray]]:
    """Render kinematic keyframe-interpolated CoarseSceneSpec across views as float32 RGBA."""
    # Omit floor plane during foreground RGBA rendering so alpha matches segmented object masks.
    model = load_mjcf_model(build_mjcf_xml(scene, include_floor=False))
    data = new_mj_data(model)

    rgb_renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer.enable_segmentation_rendering()

    body_qpos_adr: dict[str, int] = {}
    for obj in scene.objects:
        if obj.is_dynamic:
            adr = joint_qpos_adr(model, f"{obj.object_id}_jnt")
            if adr is not None:
                body_qpos_adr[obj.object_id] = adr

    available_cams = {cam.name for cam in scene.cameras}
    active_views = [v for v in views if v in available_cams]
    rendered: dict[str, list[np.ndarray]] = {v: [] for v in active_views}

    try:
        for frame_idx in frame_indices:
            for obj in scene.objects:
                adr = body_qpos_adr.get(obj.object_id)
                if adr is None:
                    continue
                pos, eul = interpolate_object_pose(obj.keyframes, frame_idx)
                data.qpos[adr : adr + 3] = pos
                data.qpos[adr + 3 : adr + 7] = _euler_to_quat_wxyz(eul)
            mj_forward(model, data)

            for view_name in active_views:
                rgb_renderer.update_scene(data, camera=view_name)
                rgb = rgb_renderer.render().astype(np.float32) / 255.0

                seg_renderer.update_scene(data, camera=view_name)
                seg = seg_renderer.render()
                # Segmentation channel 0 holds object ID (-1 for background sky).
                alpha = np.where(seg[..., 0] >= 0, 1.0, 0.0).astype(np.float32)
                rendered[view_name].append(np.concatenate([rgb, alpha[..., None]], axis=-1))
    finally:
        rgb_renderer.close()
        seg_renderer.close()

    return rendered
