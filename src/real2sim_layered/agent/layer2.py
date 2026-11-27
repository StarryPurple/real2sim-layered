"""Layer-2 local physics escalation (E_phys) with support-plane alignment and visual guard."""

from __future__ import annotations

import itertools
import math

import numpy as np

from real2sim_layered.schemas import (
    CoarseSceneSpec,
    EscalationWindow,
    InteractionGraph,
    KeyframePose,
    MvoikCase,
    PhysicalRegime,
    PhysicsMetrics,
    SceneObjectSpec,
)
from real2sim_layered.sim import (
    build_mjcf_xml,
    evaluate_mujoco_scene,
    render_coarse_scene_mujoco,
    simulate_mujoco_keyframes,
)
from real2sim_layered.tools import load_rgba_frame

_CONTACT_MARGIN_M = 0.004
_PENETRATION_TARGET_M = 0.015
_FLOATING_TARGET = 0.25


def detect_escalation_windows(
    case: MvoikCase,
    ikg: InteractionGraph,
    frame_indices: list[int],
) -> list[EscalationWindow]:
    """Select contact/interaction windows via union of VLM G_int and visual motion cues."""
    vlm_windows: list[tuple[int, int]] = []
    for interval in ikg.intervals:
        if (
            interval.requires_physics_escalation
            or interval.contact_expected
            or interval.regime in (PhysicalRegime.RIGID, PhysicalRegime.OCCLUDED_KINEMATIC)
        ):
            start = max(0, min(case.num_frames - 1, interval.start_frame))
            end = max(start, min(case.num_frames - 1, interval.end_frame))
            vlm_windows.append((start, end))

    centroids: list[tuple[float, float]] = []
    areas: list[float] = []
    for f_idx in frame_indices:
        rgba = load_rgba_frame(case.input_frames[f_idx], target_size=(160, 120))
        mask = rgba[..., 3] >= 0.5
        areas.append(float(mask.mean()))
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            centroids.append((float(np.mean(xs)) / 160.0, float(np.mean(ys)) / 120.0))
        else:
            centroids.append((0.5, 0.5))

    visual_active: list[int] = []
    for idx, (f_prev, f_cur) in enumerate(itertools.pairwise(frame_indices)):
        c0, c1 = centroids[idx], centroids[idx + 1]
        disp = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        a0, a1 = max(1e-4, areas[idx]), areas[idx + 1]
        area_delta = abs(a1 - a0) / a0
        if disp > 0.025 or area_delta > 0.12:
            visual_active.extend([f_prev, f_cur])

    vis_window = (min(visual_active), max(visual_active)) if visual_active else None

    if vlm_windows and vis_window is not None:
        return [
            EscalationWindow(
                start_frame=min(min(w[0] for w in vlm_windows), vis_window[0]),
                end_frame=max(max(w[1] for w in vlm_windows), vis_window[1]),
                source="both",
                reason="Union of VLM interaction interval and visual motion/area trigger.",
            )
        ]
    if vlm_windows:
        return [
            EscalationWindow(
                start_frame=min(w[0] for w in vlm_windows),
                end_frame=max(w[1] for w in vlm_windows),
                source="vlm_ikg",
                reason="Flagged by VLM InteractionGraph contact/physics interval.",
            )
        ]
    if vis_window is not None:
        return [
            EscalationWindow(
                start_frame=vis_window[0],
                end_frame=vis_window[1],
                source="visual_motion",
                reason="Triggered by VLM-independent foreground displacement/area change.",
            )
        ]
    return [
        EscalationWindow(
            start_frame=0,
            end_frame=max(0, case.num_frames - 1),
            source="visual_motion",
            reason="Default full-clip verification window.",
        )
    ]


def _rotated_half_extents(
    obj: SceneObjectSpec,
    euler_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Conservative axis-aligned half-extents accounting for pitch/roll tilt."""
    hx, hy, hz = obj.size_m
    pitch = abs(euler_xyz[1])
    cp, sp = abs(math.cos(pitch)), abs(math.sin(pitch))
    return (hx * cp + hz * sp, hy, hx * sp + hz * cp)


def _resolve_initial_contacts(scene: CoarseSceneSpec, separation_scale: float) -> CoarseSceneSpec:
    """Anchor support plane `floor_z` under observed objects and clear pairwise AABB overlaps."""
    updated_objs: list[SceneObjectSpec] = [obj.model_copy(deep=True) for obj in scene.objects]

    # 1. Estimate support plane directly beneath the lowest frame-0 object bottom so grounding
    # removes underground penetration WITHOUT shifting objects upward out of camera framing.
    frame0_bottoms = [
        obj.keyframes[0].position[2] - _rotated_half_extents(obj, obj.keyframes[0].euler_xyz)[2]
        for obj in updated_objs
    ]
    calibrated_floor_z = min(frame0_bottoms) - _CONTACT_MARGIN_M

    for obj in updated_objs:
        repaired_kfs: list[KeyframePose] = []
        for kf in obj.keyframes:
            _, _, eff_hz = _rotated_half_extents(obj, kf.euler_xyz)
            min_z = calibrated_floor_z + eff_hz + _CONTACT_MARGIN_M
            x, y, z = kf.position
            repaired_kfs.append(kf.model_copy(update={"position": (x, y, max(min_z, z))}))
        obj.keyframes = repaired_kfs

    # 2. Separate pairwise overlaps on frame 0 (preferring depth Y to preserve input-view X/Z).
    num_objs = len(updated_objs)
    offsets = [np.zeros(3, dtype=np.float64) for _ in range(num_objs)]
    extents = [
        np.array(_rotated_half_extents(obj, obj.keyframes[0].euler_xyz), dtype=np.float64)
        for obj in updated_objs
    ]
    for _ in range(16):
        moved = False
        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                pi = np.array(updated_objs[i].keyframes[0].position, dtype=np.float64) + offsets[i]
                pj = np.array(updated_objs[j].keyframes[0].position, dtype=np.float64) + offsets[j]
                delta = pj - pi
                overlap = (extents[i] + extents[j] + _CONTACT_MARGIN_M) - np.abs(delta)
                if np.all(overlap > 0.0):
                    axis = (
                        1
                        if overlap[1] <= min(overlap[0], overlap[2]) * 2.0
                        else int(np.argmin(overlap))
                    )
                    sign = 1.0 if delta[axis] >= 0.0 else -1.0
                    shift = 0.6 * float(overlap[axis]) * separation_scale * sign
                    offsets[i][axis] -= shift
                    offsets[j][axis] += shift
                    moved = True
        if not moved:
            break

    for idx, obj in enumerate(updated_objs):
        off = offsets[idx]
        if float(np.linalg.norm(off)) < 1e-6:
            continue
        repaired_kfs = []
        for kf in obj.keyframes:
            _, _, eff_hz = _rotated_half_extents(obj, kf.euler_xyz)
            min_z = calibrated_floor_z + eff_hz + _CONTACT_MARGIN_M
            repaired_kfs.append(
                kf.model_copy(
                    update={
                        "position": (
                            kf.position[0] + float(off[0]),
                            kf.position[1] + float(off[1]),
                            max(min_z, kf.position[2] + float(off[2])),
                        )
                    }
                )
            )
        obj.keyframes = repaired_kfs

    return scene.model_copy(update={"floor_z": calibrated_floor_z, "objects": updated_objs})


def _blend_window_trajectories(
    grounded: CoarseSceneSpec,
    simulated: CoarseSceneSpec,
    windows: list[EscalationWindow],
    sim_weight: float,
) -> CoarseSceneSpec:
    """Apply MuJoCo-simulated poses inside escalated windows, retaining grounded poses outside."""

    def _in_window(frame_idx: int) -> bool:
        return any(w.start_frame <= frame_idx <= w.end_frame for w in windows)

    blended_objs: list[SceneObjectSpec] = []
    for base_obj, sim_obj in zip(grounded.objects, simulated.objects, strict=True):
        sim_by_frame = {kf.frame_idx: kf for kf in sim_obj.keyframes}
        new_kfs: list[KeyframePose] = []
        for kf in base_obj.keyframes:
            sim_kf = sim_by_frame.get(kf.frame_idx)
            if sim_kf is None or not _in_window(kf.frame_idx):
                new_kfs.append(kf)
                continue
            bx, by, bz = kf.position
            sx, sy, sz = sim_kf.position
            be = kf.euler_xyz
            se = sim_kf.euler_xyz
            blended_eul = (
                (1.0 - sim_weight) * be[0] + sim_weight * se[0],
                (1.0 - sim_weight) * be[1] + sim_weight * se[1],
                (1.0 - sim_weight) * be[2] + sim_weight * se[2],
            )
            _, _, eff_hz = _rotated_half_extents(base_obj, blended_eul)
            min_z = grounded.floor_z + eff_hz + _CONTACT_MARGIN_M
            new_kfs.append(
                kf.model_copy(
                    update={
                        "position": (
                            (1.0 - sim_weight) * bx + sim_weight * sx,
                            (1.0 - sim_weight) * by + sim_weight * sy,
                            max(min_z, (1.0 - sim_weight) * bz + sim_weight * sz),
                        ),
                        "euler_xyz": blended_eul,
                    }
                )
            )
        blended_objs.append(base_obj.model_copy(update={"keyframes": new_kfs}))
    return grounded.model_copy(update={"objects": blended_objs})


def _input_view_iou(scene: CoarseSceneSpec, case: MvoikCase, frame_indices: list[int]) -> float:
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


def run_global_rigid_physics(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
) -> tuple[CoarseSceneSpec, PhysicsMetrics]:
    """Ablation arm (`L1_global_phys`): naive global rigid-body rollout with no contact repair."""
    sim_scene = simulate_mujoco_keyframes(scene, frame_indices)
    xml = build_mjcf_xml(sim_scene, include_floor=True)
    phys = evaluate_mujoco_scene(xml, num_steps=200, scene=scene, delivered_scene=sim_scene)
    return sim_scene, phys


def run_layer2_escalation(
    scene: CoarseSceneSpec,
    windows: list[EscalationWindow],
    frame_indices: list[int],
    max_iters: int = 3,
    case: MvoikCase | None = None,
) -> tuple[CoarseSceneSpec, PhysicsMetrics, int]:
    """Iterative simulator-in-the-loop local physics escalation with visual reprojection guard."""
    best_scene = scene
    best_phys = evaluate_mujoco_scene(
        build_mjcf_xml(scene, include_floor=True),
        num_steps=200,
        scene=scene,
        delivered_scene=scene,
    )
    ref_iou = _input_view_iou(scene, case, frame_indices) if case is not None else 0.0
    iters_run = 0

    for step_idx in range(1, max_iters + 1):
        iters_run = step_idx
        grounded = _resolve_initial_contacts(best_scene, separation_scale=1.0 + 0.15 * step_idx)
        sim_rollout = simulate_mujoco_keyframes(grounded, frame_indices)

        # Pick highest physics blend weight whose input-view silhouette does not collapse.
        chosen_cand = grounded
        for weight in (0.55, 0.30, 0.12):
            cand = _blend_window_trajectories(grounded, sim_rollout, windows, sim_weight=weight)
            if case is None:
                chosen_cand = cand
                break
            cand_iou = _input_view_iou(cand, case, frame_indices)
            if cand_iou >= ref_iou - 0.03:
                chosen_cand = cand
                break
            chosen_cand = cand

        cand_xml = build_mjcf_xml(chosen_cand, include_floor=True)
        cand_phys = evaluate_mujoco_scene(
            cand_xml, num_steps=200, scene=scene, delivered_scene=chosen_cand
        )

        if (
            cand_phys.max_penetration_m <= best_phys.max_penetration_m + 1e-4
            or cand_phys.counterfactual_survival
        ):
            best_scene = chosen_cand
            best_phys = cand_phys

        if (
            best_phys.stable
            and best_phys.max_penetration_m < _PENETRATION_TARGET_M
            and best_phys.floating_frame_ratio < _FLOATING_TARGET
            and best_phys.counterfactual_survival
        ):
            break

    notes = [
        *best_scene.notes,
        f"Layer-2 escalation: {iters_run} iter(s), floor_z={best_scene.floor_z:.4f}, "
        f"max_pen={best_phys.max_penetration_m:.4f}m, cf_ok={best_phys.counterfactual_survival}",
    ]
    return best_scene.model_copy(update={"notes": notes}), best_phys, iters_run
