"""Layer-1 common-sense Real2Sim synthesizer (G_int -> S_0) with pluggable backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
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
from real2sim_layered.tools import (
    get_default_mesh_lifter,
    load_rgba_frame,
    sample_frame_indices,
)

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

_RegionStats = tuple[float, float, float, float, float, RgbaColor]


def build_canonical_cameras(
    target: tuple[float, float, float] = (0.0, 0.0, 0.14),
    distance: float = 1.15,
    fov_y_deg: float = 49.1,
) -> list[CameraSpec]:
    """Canonical input + 4 MVOIK-4D evaluation viewpoints (HAT-4D 49.1 deg FOV convention)."""
    specs = [
        ("input_cam", -30.0, 12.0),
        ("eval_0", 0.0, 10.0),
        ("eval_1", 90.0, 10.0),
        ("eval_2", 180.0, 10.0),
        ("eval_3", 45.0, 58.0),
    ]
    tx, ty, tz = target
    cameras: list[CameraSpec] = []
    for name, az_deg, el_deg in specs:
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        horiz = distance * math.cos(el)
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


def _estimate_roll_rad(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate smooth, bounded in-plane principal-axis tilt angle (radians) without step snaps."""
    if xs.size < 16:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(max(1, xs.size))
    eigvals, eigvecs = np.linalg.eigh(cov)
    anisotropy = float(eigvals[1] / max(1e-6, eigvals[0]))
    weight = max(0.0, min(1.0, (anisotropy - 1.35) / 0.85))
    if weight <= 0.0:
        return 0.0
    vx, vy = float(eigvecs[0, 1]), float(eigvecs[1, 1])
    if vx < 0.0:
        vx, vy = -vx, -vy
    angle = math.atan2(-vy, vx)
    return weight * max(-0.45, min(0.45, angle))


def _smooth_keyframes(kfs: list[KeyframePose]) -> list[KeyframePose]:
    """Apply gentle 3-tap temporal smoothing on interior keyframes to suppress per-frame jitter."""
    if len(kfs) <= 2:
        return kfs
    smoothed: list[KeyframePose] = [kfs[0]]
    for i in range(1, len(kfs) - 1):
        p0, p1, p2 = kfs[i - 1], kfs[i], kfs[i + 1]
        pos = tuple(
            0.25 * p0.position[c] + 0.5 * p1.position[c] + 0.25 * p2.position[c] for c in range(3)
        )
        eul = tuple(
            0.25 * p0.euler_xyz[c] + 0.5 * p1.euler_xyz[c] + 0.25 * p2.euler_xyz[c]
            for c in range(3)
        )
        smoothed.append(
            p1.model_copy(
                update={
                    "position": (pos[0], pos[1], pos[2]),
                    "euler_xyz": (eul[0], eul[1], eul[2]),
                }
            )
        )
    smoothed.append(kfs[-1])
    return smoothed


def _region_stats(
    rgba: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    min_box: float = 0.03,
) -> _RegionStats:
    """Compute normalized (cx, cy, bw, bh, roll_rad, median_rgba) for foreground pixels."""
    h, w, _ = rgba.shape
    cx = float(np.mean(xs)) / float(w)
    cy = float(np.mean(ys)) / float(h)
    bw = max(min_box, float(xs.max() - xs.min() + 1) / float(w))
    bh = max(min_box, float(ys.max() - ys.min() + 1) / float(h))
    roll = _estimate_roll_rad(xs, ys)
    med = np.median(rgba[ys, xs, :3], axis=0)
    return cx, cy, bw, bh, roll, (float(med[0]), float(med[1]), float(med[2]), 1.0)


def _split_object_regions(rgba: np.ndarray, num_objects: int) -> list[_RegionStats]:
    """Segment foreground into `num_objects` regions via connected components + color/XY k-means."""
    mask_u8 = (rgba[..., 3] >= 0.5).astype(np.uint8)
    if not np.any(mask_u8):
        if num_objects <= 1:
            return [(0.5, 0.65, 0.18, 0.18, 0.0, (0.5, 0.5, 0.5, 1.0))]
        return [
            (0.35 + 0.3 * i / (num_objects - 1), 0.65, 0.12, 0.12, 0.0, (0.5, 0.5, 0.5, 1.0))
            for i in range(num_objects)
        ]

    ys, xs = np.nonzero(mask_u8)
    whole = _region_stats(rgba, xs, ys, min_box=0.04)
    if num_objects <= 1:
        return [whole]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    min_pixels = max(12, int(xs.size * 0.02))
    components = [
        (int(stats[lbl, cv2.CC_STAT_AREA]), lbl)
        for lbl in range(1, num_labels)
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    if len(components) >= num_objects:
        top_labels = [lbl for _, lbl in sorted(components, reverse=True)[:num_objects]]
        regions = []
        for lbl in top_labels:
            c_ys, c_xs = np.nonzero(labels == lbl)
            regions.append(_region_stats(rgba, c_xs, c_ys))
        return sorted(regions, key=lambda r: r[0])

    # Fallback: deterministic spatial + color k-means when objects visually touch in 2D.
    h, w, _ = rgba.shape
    feats = np.stack(
        [
            xs.astype(np.float32) / float(w),
            ys.astype(np.float32) / float(h),
            rgba[ys, xs, 0] * 0.6,
            rgba[ys, xs, 1] * 0.6,
            rgba[ys, xs, 2] * 0.6,
        ],
        axis=1,
    )
    if feats.shape[0] < num_objects:
        return [whole] * num_objects

    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    best_labels = np.zeros((feats.shape[0], 1), dtype=np.int32)
    _, cluster_ids, _ = cv2.kmeans(
        feats, num_objects, best_labels, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    flat_ids = cluster_ids.reshape(-1)
    regions = []
    for k in range(num_objects):
        sel = flat_ids == k
        regions.append(_region_stats(rgba, xs[sel], ys[sel]) if np.any(sel) else whole)
    return sorted(regions, key=lambda r: r[0])


def _backproject_to_world(
    region: tuple[float, float, float, float],
    depth_offset_m: float,
    aspect: float = 4.0 / 3.0,
    fov_y_deg: float = 49.1,
    cam_distance: float = 1.15,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Back-project normalized 2D bbox `(cx, cy, bw, bh)` into metric 3D `(pos, half_extents)`."""
    cx_norm, cy_norm, bw_norm, bh_norm = region
    half_h_world = cam_distance * math.tan(math.radians(fov_y_deg) * 0.5)
    half_w_world = half_h_world * aspect

    wx = (cx_norm - 0.5) * 2.0 * half_w_world
    wz = max(0.04, (0.85 - cy_norm) * 2.0 * half_h_world)
    hx = max(0.025, min(0.25, bw_norm * half_w_world * 0.85))
    hz = max(0.025, min(0.25, bh_norm * half_h_world * 0.85))
    hy = max(0.025, min(0.20, 0.5 * (hx + hz)))
    return (wx, depth_offset_m, wz), (hx, hy, hz)


def _match_regions_to_previous(
    regions: list[_RegionStats],
    prev_regions: list[_RegionStats] | None,
) -> list[_RegionStats]:
    """Greedily match current frame regions to previous identities via XY + RGB proximity."""
    if prev_regions is None or len(regions) != len(prev_regions) or len(regions) <= 1:
        return regions
    remaining = list(range(len(regions)))
    matched: list[_RegionStats] = []
    for p_cx, p_cy, _, _, _, p_col in prev_regions:
        best_i = min(
            remaining,
            key=lambda idx: (
                (regions[idx][0] - p_cx) ** 2
                + (regions[idx][1] - p_cy) ** 2
                + 0.4 * sum((regions[idx][5][c] - p_col[c]) ** 2 for c in range(3))
            ),
        )
        remaining.remove(best_i)
        matched.append(regions[best_i])
    return matched


def _extract_object_rgba_crops(
    rgba_full: np.ndarray,
    regions: list[_RegionStats],
) -> list[np.ndarray]:
    """Crop per-object segmented RGBA patches from frame 0 for Image-to-3D lifting."""
    h, w, _ = rgba_full.shape
    mask = rgba_full[..., 3] >= 0.5
    if not np.any(mask):
        return [rgba_full for _ in regions]

    if len(regions) <= 1:
        ys, xs = np.nonzero(mask)
        return [rgba_full[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]]

    ys_all, xs_all = np.nonzero(mask)
    pts_x = xs_all.astype(np.float64) / float(w)
    pts_y = ys_all.astype(np.float64) / float(h)
    pts_rgb = rgba_full[ys_all, xs_all, :3].astype(np.float64)

    dists = np.stack(
        [
            (pts_x - r[0]) ** 2
            + (pts_y - r[1]) ** 2
            + 0.35 * np.sum((pts_rgb - np.array(r[5][:3])) ** 2, axis=1)
            for r in regions
        ],
        axis=0,
    )
    assign = np.argmin(dists, axis=0)

    crops: list[np.ndarray] = []
    for obj_i in range(len(regions)):
        sel = assign == obj_i
        if int(sel.sum()) < 24:
            ys, xs = ys_all, xs_all
            obj_mask = mask
        else:
            ys, xs = ys_all[sel], xs_all[sel]
            obj_mask = np.zeros((h, w), dtype=bool)
            obj_mask[ys, xs] = True
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        patch = rgba_full[y0:y1, x0:x1].copy()
        patch[~obj_mask[y0:y1, x0:x1], 3] = 0.0
        crops.append(patch)
    return crops


def synthesize_coarse_scene(
    case: MvoikCase,
    ikg: InteractionGraph,
    backend: AssetBackend,
    max_frames: int = 8,
    asset_dir: Path | None = None,
) -> CoarseSceneSpec:
    """Fuse VLM InteractionGraph (G_int) and RGBA visual cues into CoarseSceneSpec (S_0)."""
    frame_indices = sample_frame_indices(case.num_frames, max_frames=max_frames)
    num_objs = len(ikg.objects)
    aspect = float(case.width) / float(max(1, case.height))

    per_obj_kfs: list[list[KeyframePose]] = [[] for _ in range(num_objs)]
    per_obj_sizes: list[list[tuple[float, float, float]]] = [[] for _ in range(num_objs)]
    per_obj_colors: list[list[RgbaColor]] = [[] for _ in range(num_objs)]
    prev_regions: list[_RegionStats] | None = None
    frame0_regions: list[_RegionStats] = []

    for frame_idx in frame_indices:
        rgba = load_rgba_frame(case.input_frames[frame_idx], target_size=(160, 120))
        regions = _match_regions_to_previous(
            _split_object_regions(rgba, num_objects=num_objs), prev_regions
        )
        if not frame0_regions:
            frame0_regions = regions
        prev_regions = regions
        for obj_i, (cx, cy, bw, bh, roll, color) in enumerate(regions):
            depth_offset = (obj_i - 0.5 * (num_objs - 1)) * 0.05
            pos, half_ext = _backproject_to_world(
                (cx, cy, bw, bh), depth_offset_m=depth_offset, aspect=aspect
            )
            per_obj_kfs[obj_i].append(
                KeyframePose(frame_idx=frame_idx, position=pos, euler_xyz=(0.0, roll, 0.0))
            )
            per_obj_sizes[obj_i].append(half_ext)
            per_obj_colors[obj_i].append(color)

    crops_3d: list[np.ndarray] = []
    resolved_asset_dir = asset_dir or (Path("outputs") / "assets_image_to_3d" / case.case_id)
    if backend == AssetBackend.IMAGE_TO_3D_SIM:
        rgba_full0 = load_rgba_frame(case.input_frames[frame_indices[0]])
        crops_3d = _extract_object_rgba_crops(rgba_full0, frame0_regions)

    scene_objects: list[SceneObjectSpec] = []
    for obj_i, entity in enumerate(ikg.objects):
        size_arr = np.median(np.array(per_obj_sizes[obj_i], dtype=np.float64), axis=0)
        color_arr = np.median(np.array(per_obj_colors[obj_i], dtype=np.float64), axis=0)
        mass, friction = _MATERIAL_PRIORS.get(entity.material_hint, (0.30, 0.65))
        size_m = (float(size_arr[0]), float(size_arr[1]), float(size_arr[2]))
        color_rgba: RgbaColor = (float(color_arr[0]), float(color_arr[1]), float(color_arr[2]), 1.0)
        vis_mesh: str | None = None
        tex_path: str | None = None
        col_meshes: list[str] = []

        if backend == AssetBackend.IMAGE_TO_3D_SIM and obj_i < len(crops_3d):
            asset = get_default_mesh_lifter().lift_rgba_crop(
                rgba_crop=crops_3d[obj_i],
                target_size_m=size_m,
                out_dir=resolved_asset_dir,
                asset_id=f"{case.case_id}_{entity.object_id}",
            )
            vis_mesh = str(asset.visual_mesh_path)
            tex_path = str(asset.texture_path)
            col_meshes = [str(p) for p in asset.collision_mesh_paths]
            size_m = asset.half_extents_m
            color_rgba = asset.mean_rgba

        scene_objects.append(
            SceneObjectSpec(
                object_id=entity.object_id,
                label=entity.label,
                primitive=_PRIMITIVE_MAP.get(entity.primitive_proxy.strip().lower(), "box"),
                size_m=size_m,
                color_rgba=color_rgba,
                mass_kg=mass,
                friction=friction,
                is_dynamic=entity.is_dynamic,
                material_hint=entity.material_hint,
                visual_mesh_path=vis_mesh,
                texture_path=tex_path,
                collision_mesh_paths=col_meshes,
                keyframes=_smooth_keyframes(per_obj_kfs[obj_i]),
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
    """Compile and execute a CoarseSceneSpec on either sim_proxy or procedural_blender."""
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
