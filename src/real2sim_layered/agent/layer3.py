"""Layer-3 failure-aware anomaly detection, optical-flow non-rigidity, and regime fallback."""

from __future__ import annotations

import itertools

import cv2
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
from real2sim_layered.sim import (
    build_mjcf_xml,
    evaluate_mujoco_scene,
    render_coarse_scene_mujoco,
)
from real2sim_layered.tools import load_rgba_frame

_NON_RIGID_CV_TRIGGER = 0.14
_FLOW_DEFORM_TRIGGER = 0.08
_TRACKING_ERR_TRIGGER_M = 0.14
_SILHOUETTE_DROP_TRIGGER = 0.04


def _measure_non_rigid_signals(
    case: MvoikCase,
    frame_indices: list[int],
) -> tuple[float, float, list[tuple[float, float]]]:
    """Compute label-free (shape_cv, flow_nonrigid_residual, rel_extents) over input RGBA frames."""
    areas: list[float] = []
    extents: list[tuple[float, float]] = []
    grays: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for f_idx in frame_indices:
        rgba = load_rgba_frame(case.input_frames[f_idx], target_size=(160, 120))
        mask = rgba[..., 3] >= 0.5
        masks.append(mask)
        rgb_u8 = np.clip(rgba[..., :3] * rgba[..., 3:4] * 255.0, 0, 255).astype(np.uint8)
        grays.append(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY))
        areas.append(float(mask.sum()))
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            extents.append((float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
        else:
            extents.append((1.0, 1.0))

    area_arr = np.array(areas, dtype=np.float64)
    area_cv = float(np.std(area_arr) / max(1.0, float(np.mean(area_arr))))
    aspect_arr = np.array([bw / max(1.0, bh) for bw, bh in extents], dtype=np.float64)
    aspect_cv = float(np.std(aspect_arr) / max(1e-4, float(np.mean(aspect_arr))))
    shape_cv = 0.5 * area_cv + 0.5 * aspect_cv

    # Dense optical flow intra-component non-rigid residual (avoids multi-body rigid false positives).
    flow_residuals: list[float] = []
    for (g0, m0), (g1, m1) in itertools.pairwise(zip(grays, masks, strict=True)):
        union_mask = np.logical_or(m0, m1)
        if int(union_mask.sum()) < 20:
            continue
        flow = np.zeros((g0.shape[0], g0.shape[1], 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, flow, 0.5, 2, 11, 2, 5, 1.1, 0)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            m0.astype(np.uint8), connectivity=8
        )
        for comp_id in range(1, num_labels):
            if int(stats[comp_id, cv2.CC_STAT_AREA]) < 40:
                continue
            comp_mask = labels == comp_id
            fg_flow = flow[comp_mask]
            mean_flow = fg_flow.mean(axis=0, keepdims=True)
            residual = np.linalg.norm(fg_flow - mean_flow, axis=-1)
            diag = max(
                10.0,
                float(
                    np.hypot(
                        float(stats[comp_id, cv2.CC_STAT_WIDTH]),
                        float(stats[comp_id, cv2.CC_STAT_HEIGHT]),
                    )
                ),
            )
            flow_residuals.append(float(np.mean(residual)) / diag)

    flow_nonrigid = float(np.max(flow_residuals)) if flow_residuals else 0.0

    ref_w, ref_h = extents[0]
    rel_extents = [
        (
            max(0.55, min(1.5, bw / max(1.0, ref_w))),
            max(0.55, min(1.5, bh / max(1.0, ref_h))),
        )
        for bw, bh in extents
    ]
    return shape_cv, flow_nonrigid, rel_extents


def _input_view_silhouette_iou(
    scene: CoarseSceneSpec,
    case: MvoikCase,
    frame_indices: list[int],
) -> float:
    rendered = render_coarse_scene_mujoco(
        scene=scene, frame_indices=frame_indices, views=["input_cam"], width=160, height=120
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


def _choose_fallback_mode(
    ikg: InteractionGraph,
    shape_cv: float,
    flow_nonrigid: float,
) -> ExecutionMode:
    if ikg.dominant_regime == PhysicalRegime.TOPOLOGICAL_SPLIT:
        return "fallback_topological"
    if ikg.dominant_regime == PhysicalRegime.MULTIPHYSICS_OOD:
        return "fallback_kinematic"
    if (
        ikg.dominant_regime == PhysicalRegime.DEFORMABLE
        or shape_cv >= _NON_RIGID_CV_TRIGGER
        or flow_nonrigid >= _FLOW_DEFORM_TRIGGER
    ):
        return "fallback_deformable"
    return "fallback_kinematic"


def _apply_topological_split(
    base_objs: list[SceneObjectSpec],
    floor_z: float,
) -> list[SceneObjectSpec]:
    """Split the primary cuttable/largest dynamic object into two separating sub-halves."""
    target_idx = next(
        (i for i, o in enumerate(base_objs) if "fruit" in o.material_hint or "cut" in o.label),
        0,
    )
    split_objs: list[SceneObjectSpec] = []
    for idx, obj in enumerate(base_objs):
        if idx != target_idx:
            split_objs.append(obj)
            continue
        hx, hy, hz = obj.size_m
        half_hx = max(0.015, hx * 0.48)
        min_z = floor_z + hz + 0.004
        mid_k = len(obj.keyframes) // 2
        for side, suffix in ((-1.0, "part_a"), (1.0, "part_b")):
            sub_kfs = []
            for k_i, kf in enumerate(obj.keyframes):
                sep_progress = (
                    0.0
                    if k_i < mid_k
                    else (k_i - mid_k + 1) / float(max(1, len(obj.keyframes) - mid_k))
                )
                offset_x = side * (half_hx + 0.002 + 0.015 * sep_progress)
                sub_kfs.append(
                    kf.model_copy(
                        update={
                            "position": (
                                kf.position[0] + offset_x,
                                kf.position[1],
                                max(min_z, kf.position[2]),
                            )
                        }
                    )
                )
            split_objs.append(
                obj.model_copy(
                    update={
                        "object_id": f"{obj.object_id}_{suffix}",
                        "size_m": (half_hx, hy, hz),
                        "mass_kg": max(0.05, obj.mass_kg * 0.5),
                        "keyframes": sub_kfs,
                    }
                )
            )
    return split_objs


def _build_fallback_scene(
    l1_scene: CoarseSceneSpec,
    l2_scene: CoarseSceneSpec,
    mode: ExecutionMode,
    rel_extents: list[tuple[float, float]],
    frame_indices: list[int],
    prefer_l2_poses: bool = False,
) -> CoarseSceneSpec:
    """Keep L2 contact-safe floor & depth separation while applying regime-specific fallback."""
    extent_by_frame = dict(zip(frame_indices, rel_extents, strict=True))
    fallback_objs: list[SceneObjectSpec] = []

    for l1_obj, l2_obj in zip(l1_scene.objects, l2_scene.objects, strict=True):
        use_l2 = prefer_l2_poses and mode != "fallback_kinematic"
        src_kfs = l2_obj.keyframes if use_l2 else l1_obj.keyframes
        dx = 0.0 if use_l2 else l2_obj.keyframes[0].position[0] - l1_obj.keyframes[0].position[0]
        dy = 0.0 if use_l2 else l2_obj.keyframes[0].position[1] - l1_obj.keyframes[0].position[1]
        dz = 0.0 if use_l2 else l2_obj.keyframes[0].position[2] - l1_obj.keyframes[0].position[2]
        hz = l2_obj.size_m[2]

        new_kfs = []
        for kf in src_kfs:
            sx, sz = extent_by_frame.get(kf.frame_idx, (1.0, 1.0))
            if mode == "fallback_deformable":
                scale = (sx, 0.5 * (sx + sz), sz)
                squash_dz = -0.5 * max(0.0, 1.0 - sz) * hz
                min_z = l2_scene.floor_z + hz + 0.002
            else:
                scale = (1.0, 1.0, 1.0)
                squash_dz = 0.0
                min_z = l2_obj.keyframes[0].position[2]
            new_kfs.append(
                kf.model_copy(
                    update={
                        "position": (
                            kf.position[0] + dx,
                            kf.position[1] + dy,
                            max(min_z, kf.position[2] + dz + squash_dz),
                        ),
                        "scale": scale,
                    }
                )
            )
        fallback_objs.append(l2_obj.model_copy(update={"keyframes": new_kfs}))

    if mode == "fallback_topological":
        fallback_objs = _apply_topological_split(fallback_objs, floor_z=l2_scene.floor_z)

    return l2_scene.model_copy(
        update={
            "objects": fallback_objs,
            "notes": [*l2_scene.notes, f"Layer-3 anomaly fallback activated -> mode={mode}"],
        }
    )


def detect_and_apply_fallback(
    case: MvoikCase,
    ikg: InteractionGraph,
    l1_scene: CoarseSceneSpec,
    l2_scene: CoarseSceneSpec,
    l2_phys: PhysicsMetrics,
    frame_indices: list[int],
) -> tuple[CoarseSceneSpec, AnomalyReport]:
    """Detect rigid-solver mismatch via shape/flow/tracking residuals and gate candidate commit."""
    shape_cv, flow_nonrigid, rel_extents = _measure_non_rigid_signals(case, frame_indices)
    l1_iou = _input_view_silhouette_iou(l1_scene, case, frame_indices)
    l2_iou = _input_view_silhouette_iou(l2_scene, case, frame_indices)
    sil_drop = l1_iou - l2_iou
    track_err = l2_phys.trajectory_tracking_err_m

    reasons: list[str] = []
    if not l2_phys.stable:
        reasons.append("MuJoCo solver diverged")
    if sil_drop > _SILHOUETTE_DROP_TRIGGER:
        reasons.append(f"rigid physics degraded input silhouette IoU by {sil_drop:.3f}")
    if track_err > _TRACKING_ERR_TRIGGER_M:
        reasons.append(f"rigid trajectory drifted ({track_err:.3f}m)")
    if flow_nonrigid >= _FLOW_DEFORM_TRIGGER:
        reasons.append(f"non-rigid optical flow residual ({flow_nonrigid:.3f})")
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

    primary_mode = _choose_fallback_mode(ikg, shape_cv, flow_nonrigid)
    candidate_modes: list[ExecutionMode] = [primary_mode]
    if primary_mode != "fallback_kinematic":
        candidate_modes.append("fallback_kinematic")

    last_diag = ""
    for mode in candidate_modes:
        candidate = _build_fallback_scene(
            l1_scene,
            l2_scene,
            mode,
            rel_extents,
            frame_indices,
            prefer_l2_poses=(l2_iou >= l1_iou),
        )
        cand_iou = _input_view_silhouette_iou(candidate, case, frame_indices)
        cand_phys = evaluate_mujoco_scene(
            build_mjcf_xml(candidate, include_floor=True),
            num_steps=160,
            scene=l1_scene,
            delivered_scene=candidate,
        )
        phys_ok = (
            cand_phys.stable
            and cand_phys.max_penetration_m <= l2_phys.max_penetration_m + 0.015
            and (cand_phys.counterfactual_survival or not l2_phys.counterfactual_survival)
        )
        if phys_ok and cand_iou >= l2_iou - 0.005:
            return (
                candidate,
                AnomalyReport(
                    triggered=True,
                    non_rigid_shape_cv=shape_cv,
                    physics_tracking_err_m=track_err,
                    silhouette_drop=sil_drop,
                    fallback_mode=mode,
                    reason="; ".join(reasons),
                ),
            )
        last_diag = (
            f"mode='{mode}' cand_iou={cand_iou:.3f} vs l2_iou={l2_iou:.3f}, "
            f"cand_pen={cand_phys.max_penetration_m:.3f} vs l2_pen={l2_phys.max_penetration_m:.3f}"
        )

    return (
        l2_scene,
        AnomalyReport(
            triggered=False,
            non_rigid_shape_cv=shape_cv,
            physics_tracking_err_m=track_err,
            silhouette_drop=sil_drop,
            fallback_mode="physics_sim",
            reason=f"Fallback rejected by pre-commit gate ({last_diag}); retained L2.",
        ),
    )
