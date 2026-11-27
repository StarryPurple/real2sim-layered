"""Layer-1 common-sense Real2Sim synthesizer (G_int -> S_0) with pluggable backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from real2sim_layered.schemas import (
    AssetBackend,
    CameraSpec,
    CoarseSceneSpec,
    InteractionGraph,
    KeyframePose,
    MvoikCase,
    PrimitiveKind,
    RgbaColor,
    SceneObjectSpec,
)
from real2sim_layered.sim import (
    build_mjcf_xml,
    render_coarse_scene_mujoco,
    run_blender_scene,
)
from real2sim_layered.tools import load_rgba_frame, sample_frame_indices

_MATERIAL_PRIORS: dict[str, tuple[float, float]] = {
    "rigid_solid": (0.35, 0.65),
    "soft_elastic": (0.15, 0.85),
    "fruit_cuttable": (0.25, 0.55),
    "paper_thin": (0.05, 0.50),
    "fluid": (0.20, 0.15),
}

_PRIMITIVE_MAP: dict[str, PrimitiveKind] = {
    "box": "box",
    "sphere": "sphere",
    "cylinder": "cylinder",
    "capsule": "capsule",
}

_RegionStats = tuple[float, float, float, float, RgbaColor]


def build_canonical_cameras(
    target: tuple[float, float, float] = (0.0, 0.0, 0.14),
    distance: float = 1.15,
    fov_y_deg: float = 40.0,
) -> list[CameraSpec]:
    """Canonical input + 4 MVOIK-4D evaluation viewpoints around `target`."""
    specs = [
        ("input_cam", -30.0, 14.0),
        ("eval_0", 0.0, 12.0),
        ("eval_1", 90.0, 12.0),
        ("eval_2", 180.0, 12.0),
        ("eval_3", 45.0, 58.0),
    ]
    tx, ty, tz = target
    cameras: list[CameraSpec] = []
    for name, az_deg, el_deg in specs:
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        horiz = distance * math.cos(el)
        # Azimuth 0 places camera on -Y looking toward +Y.
        cameras.append(
            CameraSpec(
                name=name,
                position=(
                    tx + horiz * math.sin(az),
                    ty - horiz * math.cos(az),
                    tz + distance * math.sin(el),
                ),
                target=target,
                fov_y_deg=fov_y_deg,
            )
        )
    return cameras


def _region_stats(
    rgba: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    min_box: float = 0.03,
) -> _RegionStats:
    """Compute normalized (cx, cy, bw, bh, median_rgba) for foreground pixel coordinates."""
    h, w, _ = rgba.shape
    cx = float(np.mean(xs)) / float(w)
    cy = float(np.mean(ys)) / float(h)
    bw = max(min_box, float(xs.max() - xs.min() + 1) / float(w))
    bh = max(min_box, float(ys.max() - ys.min() + 1) / float(h))
    med = np.median(rgba[ys, xs, :3], axis=0)
    return cx, cy, bw, bh, (float(med[0]), float(med[1]), float(med[2]), 1.0)


def _split_object_regions(rgba: np.ndarray, num_objects: int) -> list[_RegionStats]:
    """Partition foreground pixels along horizontal quantile bands for `num_objects` entities."""
    mask = rgba[..., 3] >= 0.5
    if not np.any(mask):
        if num_objects <= 1:
            return [(0.5, 0.65, 0.18, 0.18, (0.5, 0.5, 0.5, 1.0))]
        return [
            (0.35 + 0.3 * i / (num_objects - 1), 0.65, 0.12, 0.12, (0.5, 0.5, 0.5, 1.0))
            for i in range(num_objects)
        ]

    ys, xs = np.nonzero(mask)
    whole = _region_stats(rgba, xs, ys, min_box=0.04)
    if num_objects <= 1:
        return [whole]

    x_edges = np.quantile(xs, np.linspace(0.0, 1.0, num=num_objects + 1))
    regions: list[_RegionStats] = []
    for idx in range(num_objects):
        sel = (xs >= float(x_edges[idx])) & (xs <= float(x_edges[idx + 1]))
        regions.append(_region_stats(rgba, xs[sel], ys[sel]) if np.any(sel) else whole)
    return regions


def _backproject_to_world(
    region: tuple[float, float, float, float],
    depth_offset_m: float,
    aspect: float = 4.0 / 3.0,
    fov_y_deg: float = 40.0,
    cam_distance: float = 1.15,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Back-project normalized 2D bbox `(cx, cy, bw, bh)` into metric 3D `(pos, half_extents)`."""
    cx_norm, cy_norm, bw_norm, bh_norm = region
    half_h_world = cam_distance * math.tan(math.radians(fov_y_deg) * 0.5)
    half_w_world = half_h_world * aspect

    wx = (cx_norm - 0.5) * 2.0 * half_w_world
    # Image y grows downward; map vertical image span into z around tabletop height.
    wz = max(0.04, (0.85 - cy_norm) * 2.0 * half_h_world)
    hx = max(0.025, min(0.25, bw_norm * half_w_world * 0.9))
    hz = max(0.025, min(0.25, bh_norm * half_h_world * 0.9))
    hy = max(0.025, min(0.20, 0.5 * (hx + hz)))
    return (wx, depth_offset_m, wz), (hx, hy, hz)


def synthesize_coarse_scene(
    case: MvoikCase,
    ikg: InteractionGraph,
    backend: AssetBackend,
    max_frames: int = 8,
) -> CoarseSceneSpec:
    """Fuse VLM InteractionGraph (G_int) and RGBA visual cues into CoarseSceneSpec (S_0)."""
    frame_indices = sample_frame_indices(case.num_frames, max_frames=max_frames)
    num_objs = len(ikg.objects)
    aspect = float(case.width) / float(max(1, case.height))

    per_obj_kfs: list[list[KeyframePose]] = [[] for _ in range(num_objs)]
    per_obj_sizes: list[list[tuple[float, float, float]]] = [[] for _ in range(num_objs)]
    per_obj_colors: list[list[RgbaColor]] = [[] for _ in range(num_objs)]

    for frame_idx in frame_indices:
        rgba = load_rgba_frame(case.input_frames[frame_idx], target_size=(160, 120))
        for obj_i, (cx, cy, bw, bh, color) in enumerate(
            _split_object_regions(rgba, num_objects=num_objs)
        ):
            depth_offset = (obj_i - 0.5 * (num_objs - 1)) * 0.04
            pos, half_ext = _backproject_to_world(
                (cx, cy, bw, bh), depth_offset_m=depth_offset, aspect=aspect
            )
            per_obj_kfs[obj_i].append(KeyframePose(frame_idx=frame_idx, position=pos))
            per_obj_sizes[obj_i].append(half_ext)
            per_obj_colors[obj_i].append(color)

    scene_objects: list[SceneObjectSpec] = []
    for obj_i, entity in enumerate(ikg.objects):
        size_arr = np.median(np.array(per_obj_sizes[obj_i], dtype=np.float64), axis=0)
        color_arr = np.median(np.array(per_obj_colors[obj_i], dtype=np.float64), axis=0)
        mass, friction = _MATERIAL_PRIORS.get(entity.material_hint, (0.30, 0.65))
        scene_objects.append(
            SceneObjectSpec(
                object_id=entity.object_id,
                label=entity.label,
                primitive=_PRIMITIVE_MAP.get(entity.primitive_proxy.strip().lower(), "box"),
                size_m=(float(size_arr[0]), float(size_arr[1]), float(size_arr[2])),
                color_rgba=(float(color_arr[0]), float(color_arr[1]), float(color_arr[2]), 1.0),
                mass_kg=mass,
                friction=friction,
                is_dynamic=entity.is_dynamic,
                material_hint=entity.material_hint,
                keyframes=per_obj_kfs[obj_i],
            )
        )

    mean_z = float(np.mean([obj.keyframes[0].position[2] for obj in scene_objects]))
    cameras = build_canonical_cameras(target=(0.0, 0.0, max(0.08, mean_z)))

    return CoarseSceneSpec(
        case_id=case.case_id,
        backend=backend,
        regime=ikg.dominant_regime,
        num_frames=case.num_frames,
        floor_z=0.0,
        cameras=cameras,
        objects=scene_objects,
        notes=[
            f"Synthesized from G_int summary: {ikg.summary}",
            f"VLM predicted regime={ikg.dominant_regime} (confidence={ikg.confidence:.2f})",
        ],
    )


@dataclass(frozen=True)
class Layer1Artifacts:
    """Materialized scene files, MJCF XML for diagnostics, and rendered multi-view frames."""

    scene_spec: CoarseSceneSpec
    spec_path: Path
    mjcf_xml: str
    mjcf_path: Path
    blend_path: Path | None
    rendered_by_view: dict[str, list[np.ndarray]]


def materialize_layer1_scene(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    views: list[str],
    out_dir: Path,
    width: int = 160,
    height: int = 120,
) -> Layer1Artifacts:
    """Compile and execute a Layer-1 CoarseSceneSpec on either sim_proxy or procedural_blender."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "scene_spec.json"
    spec_path.write_text(scene.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # Always emit a companion MJCF XML so deterministic MuJoCo physics diagnostics run uniformly.
    mjcf_xml = build_mjcf_xml(scene, include_floor=True)
    mjcf_path = out_dir / "scene.xml"
    mjcf_path.write_text(mjcf_xml, encoding="utf-8")

    blend_path: Path | None = None
    if scene.backend == AssetBackend.PROCEDURAL_BLENDER:
        blend_path, rendered = run_blender_scene(
            scene=scene,
            frame_indices=frame_indices,
            views=views,
            out_dir=out_dir,
            width=width,
            height=height,
            samples=6,
        )
    else:
        rendered = render_coarse_scene_mujoco(
            scene=scene,
            frame_indices=frame_indices,
            views=views,
            width=width,
            height=height,
        )

    return Layer1Artifacts(
        scene_spec=scene,
        spec_path=spec_path,
        mjcf_xml=mjcf_xml,
        mjcf_path=mjcf_path,
        blend_path=blend_path,
        rendered_by_view=rendered,
    )
