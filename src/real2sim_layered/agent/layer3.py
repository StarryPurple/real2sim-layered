"""Layer-3 failure-aware anomaly detection and regime fallback router (R_anom)."""

from __future__ import annotations

import numpy as np

from real2sim_layered.schemas import (
    AnomalyReport,
    CoarseSceneSpec,
    ExecutionMode,
    InteractionGraph,
    MvoikCase,
    PhysicalRegime,
    PhysicsMetrics,
    SceneObjectSpec,
)
from real2sim_layered.sim import render_coarse_scene_mujoco
from real2sim_layered.tools import load_rgba_frame

_NON_RIGID_CV_TRIGGER = 0.14
_TRACKING_ERR_TRIGGER_M = 0.16
_SILHOUETTE_DROP_TRIGGER = 0.05


def _build_grounded_kinematic_ref(
    l1_scene: CoarseSceneSpec,
    l2_scene: CoarseSceneSpec,
) -> CoarseSceneSpec:
    """Apply L2's frame-0 contact-safe grounding offset onto L1 keyframes as the visual reference."""
    ref_objs: list[SceneObjectSpec] = []
    for l1_obj, l2_obj in zip(l1_scene.objects, l2_scene.objects, strict=True):
        dx = l2_obj.keyframes[0].position[0] - l1_obj.keyframes[0].position[0]
        dy = l2_obj.keyframes[0].position[1] - l1_obj.keyframes[0].position[1]
        dz = l2_obj.keyframes[0].position[2] - l1_obj.keyframes[0].position[2]
        kfs = [
            kf.model_copy(
                update={
                    "position": (
                        kf.position[0] + dx,
                        kf.position[1] + dy,
                        kf.position[2] + dz,
                    )
                }
            )
            for kf in l1_obj.keyframes
        ]
        ref_objs.append(l2_obj.model_copy(update={"keyframes": kfs}))
    return l2_scene.model_copy(update={"objects": ref_objs})


def _measure_non_rigid_shape_cv(
    case: MvoikCase, frame_indices: list[int]
) -> tuple[float, list[tuple[float, float]]]:
    """Compute foreground area/aspect variation (label-free non-rigidity signal) and (w, h) ratios."""
    areas: list[float] = []
    extents: list[tuple[float, float]] = []
    for f_idx in frame_indices:
        rgba = load_rgba_frame(case.input_frames[f_idx], target_size=(160, 120))
        mask = rgba[..., 3] >= 0.5
        areas.append(float(mask.sum()))
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            bw = float(xs.max() - xs.min() + 1)
            bh = float(ys.max() - ys.min() + 1)
            extents.append((bw, bh))
        else:
            extents.append((1.0, 1.0))

    area_arr = np.array(areas, dtype=np.float64)
    mean_area = max(1.0, float(np.mean(area_arr)))
    area_cv = float(np.std(area_arr) / mean_area)

    aspect_arr = np.array([bw / max(1.0, bh) for bw, bh in extents], dtype=np.float64)
    aspect_cv = float(np.std(aspect_arr) / max(1e-4, float(np.mean(aspect_arr))))
    combined_cv = 0.6 * area_cv + 0.4 * aspect_cv

    ref_w, ref_h = extents[0]
    rel_extents = [
        (
            max(0.5, min(1.6, bw / max(1.0, ref_w))),
            max(0.5, min(1.6, bh / max(1.0, ref_h))),
        )
        for bw, bh in extents
    ]
    return combined_cv, rel_extents


def _input_view_silhouette_iou(
    scene: CoarseSceneSpec,
    case: MvoikCase,
    frame_indices: list[int],
) -> float:
    """Compare rendered `input_cam` silhouette against observed MVOIK-4D input RGBA masks."""
    rendered = render_coarse_scene_mujoco(
        scene=scene,
        frame_indices=frame_indices,
        views=["input_cam"],
        width=160,
        height=120,
    ).get("input_cam", [])
    if not rendered:
        return 0.0
    ious: list[float] = []
    for pred, f_idx in zip(rendered, frame_indices, strict=True):
        gt = load_rgba_frame(case.input_frames[f_idx], target_size=(160, 120))
        pm = pred[..., 3] >= 0.5
        gm = gt[..., 3] >= 0.5
        union = int(np.logical_or(pm, gm).sum())
        ious.append(1.0 if union == 0 else int(np.logical_and(pm, gm).sum()) / float(union))
    return float(np.mean(ious))


def _choose_fallback_mode(ikg: InteractionGraph, shape_cv: float) -> ExecutionMode:
    if ikg.dominant_regime == PhysicalRegime.TOPOLOGICAL_SPLIT:
        return "fallback_topological"
    if ikg.dominant_regime == PhysicalRegime.MULTIPHYSICS_OOD:
        return "fallback_kinematic"
    if ikg.dominant_regime == PhysicalRegime.DEFORMABLE or shape_cv >= _NON_RIGID_CV_TRIGGER:
        return "fallback_deformable"
    return "fallback_kinematic"


def _apply_fallback_scene(
    l1_scene: CoarseSceneSpec,
    l2_scene: CoarseSceneSpec,
    mode: ExecutionMode,
    rel_extents: list[tuple[float, float]],
    frame_indices: list[int],
) -> CoarseSceneSpec:
    """Keep L2 contact-safe grounding while applying regime-appropriate fallback kinematics/scales."""
    extent_by_frame = dict(zip(frame_indices, rel_extents, strict=True))
    fallback_objs: list[SceneObjectSpec] = []

    for l1_obj, l2_obj in zip(l1_scene.objects, l2_scene.objects, strict=True):
        # Preserve L2's contact-resolved frame-0 3D grounding offset so penetration stays eliminated.
        dx = l2_obj.keyframes[0].position[0] - l1_obj.keyframes[0].position[0]
        dy = l2_obj.keyframes[0].position[1] - l1_obj.keyframes[0].position[1]
        dz = l2_obj.keyframes[0].position[2] - l1_obj.keyframes[0].position[2]
        min_z = l2_obj.keyframes[0].position[2]

        new_kfs = []
        for kf in l1_obj.keyframes:
            sx, sz = extent_by_frame.get(kf.frame_idx, (1.0, 1.0))
            scale = (sx, 0.5 * (sx + sz), sz) if mode == "fallback_deformable" else (1.0, 1.0, 1.0)
            new_kfs.append(
                kf.model_copy(
                    update={
                        "position": (
                            kf.position[0] + dx,
                            kf.position[1] + dy,
                            max(min_z, kf.position[2] + dz),
                        ),
                        "scale": scale,
                    }
                )
            )
        fallback_objs.append(l2_obj.model_copy(update={"keyframes": new_kfs}))

    notes = [
        *l2_scene.notes,
        f"Layer-3 anomaly fallback activated -> mode={mode}",
    ]
    return l2_scene.model_copy(update={"objects": fallback_objs, "notes": notes})


def detect_and_apply_fallback(
    case: MvoikCase,
    ikg: InteractionGraph,
    l1_scene: CoarseSceneSpec,
    l2_scene: CoarseSceneSpec,
    l2_phys: PhysicsMetrics,
    frame_indices: list[int],
) -> tuple[CoarseSceneSpec, AnomalyReport]:
    """Detect rigid-solver mismatch using label-free visual+physics residuals and route fallback."""
    shape_cv, rel_extents = _measure_non_rigid_shape_cv(case, frame_indices)
    grounded_ref = _build_grounded_kinematic_ref(l1_scene, l2_scene)
    ref_iou = _input_view_silhouette_iou(grounded_ref, case, frame_indices)
    l2_iou = _input_view_silhouette_iou(l2_scene, case, frame_indices)
    sil_drop = ref_iou - l2_iou
    track_err = l2_phys.trajectory_tracking_err_m

    reasons: list[str] = []
    if not l2_phys.stable:
        reasons.append("MuJoCo solver diverged")
    if sil_drop > _SILHOUETTE_DROP_TRIGGER:
        reasons.append(f"rigid dynamics degraded grounded silhouette IoU by {sil_drop:.3f}")
    if track_err > _TRACKING_ERR_TRIGGER_M:
        reasons.append(f"rigid trajectory drifted from observation ({track_err:.3f}m)")
    if (
        ikg.dominant_regime
        in (
            PhysicalRegime.DEFORMABLE,
            PhysicalRegime.TOPOLOGICAL_SPLIT,
            PhysicalRegime.MULTIPHYSICS_OOD,
        )
        and shape_cv > 0.05
    ):
        reasons.append(
            f"VLM non-rigid prior ({ikg.dominant_regime}) corroborated by shape CV ({shape_cv:.2f})"
        )

    if not reasons:
        return (
            l2_scene,
            AnomalyReport(
                triggered=False,
                non_rigid_shape_cv=shape_cv,
                physics_tracking_err_m=track_err,
                silhouette_drop=sil_drop,
                fallback_mode="physics_sim",
                reason="Rigid local physics agrees with visual observation.",
            ),
        )

    mode = _choose_fallback_mode(ikg, shape_cv)
    repaired_scene = _apply_fallback_scene(l1_scene, l2_scene, mode, rel_extents, frame_indices)
    return (
        repaired_scene,
        AnomalyReport(
            triggered=True,
            non_rigid_shape_cv=shape_cv,
            physics_tracking_err_m=track_err,
            silhouette_drop=sil_drop,
            fallback_mode=mode,
            reason="; ".join(reasons),
        ),
    )
