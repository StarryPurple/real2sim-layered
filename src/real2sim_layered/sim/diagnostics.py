"""Deterministic MuJoCo physical diagnostics (delivered trajectory + counterfactual checks)."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from real2sim_layered.schemas import CoarseSceneSpec, KeyframePose, PhysicsMetrics
from real2sim_layered.sim._mujoco import (
    joint_qpos_adr,
    load_mjcf_model,
    mj_forward,
    mj_step,
    new_mj_data,
)

_POSITION_BOUND_M = 20.0
_FLOATING_HEIGHT_M = 0.03
_FLOATING_SPEED_MPS = 0.05


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


def _bodies_within_bounds(data: Any, dynamic_body_ids: list[int]) -> int | None:
    """Return the first body ID outside scene bounds, or None if all are within bounds."""
    for body_id in dynamic_body_ids:
        if float(np.linalg.norm(data.xpos[body_id])) > _POSITION_BOUND_M:
            return body_id
    return None


def _step_contact_summary(model: Any, data: Any) -> tuple[float, set[int]]:
    """Return (max_penetration_m, body_ids_in_contact) for the current MuJoCo state."""
    max_pen = 0.0
    touching_bodies: set[int] = set()
    for c_idx in range(int(data.ncon)):
        contact = data.contact[c_idx]
        dist = float(contact.dist)
        if dist < 0.0:
            max_pen = max(max_pen, -dist)
        touching_bodies.add(int(model.geom_bodyid[contact.geom1]))
        touching_bodies.add(int(model.geom_bodyid[contact.geom2]))
    return max_pen, touching_bodies


def _body_half_heights(
    model: Any,
    dynamic_body_ids: list[int],
    scene: CoarseSceneSpec | None = None,
) -> dict[int, float]:
    """Estimate vertical half-extent per dynamic body from scene spec or attached geoms."""
    half_z: dict[int, float] = {bid: 0.05 for bid in dynamic_body_ids}
    scene_hz_by_name = (
        {obj.object_id: float(obj.size_m[2]) for obj in scene.objects} if scene else {}
    )
    for bid in dynamic_body_ids:
        bname = str(model.body(bid).name)
        if bname in scene_hz_by_name:
            half_z[bid] = max(0.01, scene_hz_by_name[bname])
            continue
    for geom_id in range(int(model.ngeom)):
        bid = int(model.geom_bodyid[geom_id])
        if bid not in half_z or str(model.body(bid).name) in scene_hz_by_name:
            continue
        size = model.geom_size[geom_id]
        hz = float(size[2]) if float(size[2]) > 0.0 else float(size[0])
        if hz <= 0.0:
            hz = float(model.geom_rbound[geom_id])
        half_z[bid] = max(0.01, hz)
    return half_z


def _count_floating_bodies(
    model: Any,
    data: Any,
    dynamic_body_ids: list[int],
    touching_bodies: set[int],
    body_hz: dict[int, float],
    floor_z: float = 0.0,
) -> int:
    """Count dynamic bodies that are unsupported, elevated above floor, and nearly stationary."""
    floating = 0
    for bid in dynamic_body_ids:
        if bid in touching_bodies:
            continue
        z_bottom = float(data.xpos[bid][2]) - body_hz[bid] - floor_z
        dof_adr = int(model.body_dofadr[bid])
        dof_num = int(model.body_dofnum[bid])
        speed = (
            float(np.linalg.norm(data.qvel[dof_adr : dof_adr + min(3, dof_num)]))
            if dof_num > 0
            else 0.0
        )
        if z_bottom > _FLOATING_HEIGHT_M and speed < _FLOATING_SPEED_MPS:
            floating += 1
    return floating


def _interp_kf_pos(
    keyframes: list[KeyframePose],
    frame_idx: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ordered = sorted(keyframes, key=lambda k: k.frame_idx)
    if frame_idx <= ordered[0].frame_idx:
        return ordered[0].position, ordered[0].euler_xyz
    if frame_idx >= ordered[-1].frame_idx:
        return ordered[-1].position, ordered[-1].euler_xyz
    for left, right in itertools.pairwise(ordered):
        if left.frame_idx <= frame_idx <= right.frame_idx:
            alpha = (frame_idx - left.frame_idx) / float(max(1, right.frame_idx - left.frame_idx))
            pos = tuple(
                left.position[i] * (1.0 - alpha) + right.position[i] * alpha for i in range(3)
            )
            eul = tuple(
                left.euler_xyz[i] * (1.0 - alpha) + right.euler_xyz[i] * alpha for i in range(3)
            )
            return (pos[0], pos[1], pos[2]), (eul[0], eul[1], eul[2])
    return ordered[-1].position, ordered[-1].euler_xyz


def _evaluate_delivered_keyframes(
    model: Any,
    delivered_scene: CoarseSceneSpec,
    ref_scene: CoarseSceneSpec | None,
) -> tuple[float, float, float]:
    """Evaluate contact penetration, floating, and tracking error along delivered keyframes."""
    data = new_mj_data(model)
    dynamic_body_ids = [
        body_id for body_id in range(1, model.nbody) if model.body_dofnum[body_id] > 0
    ]
    body_hz = _body_half_heights(model, dynamic_body_ids, scene=delivered_scene)
    qpos_adrs = {
        obj.object_id: joint_qpos_adr(model, f"{obj.object_id}_jnt")
        for obj in delivered_scene.objects
        if obj.is_dynamic
    }
    body_id_by_obj = {
        obj.object_id: int(model.jnt_bodyid[int(model.joint(f"{obj.object_id}_jnt").id)])
        for obj in delivered_scene.objects
        if obj.is_dynamic and qpos_adrs.get(obj.object_id) is not None
    }

    frame_set = sorted({kf.frame_idx for obj in delivered_scene.objects for kf in obj.keyframes})
    if not frame_set:
        return 0.0, 0.0, 0.0

    max_pen = 0.0
    float_body_frames = 0
    total_body_frames = 0
    track_errs: list[float] = []
    ref_by_id = {obj.object_id: obj for obj in ref_scene.objects} if ref_scene else {}

    for idx, frame_idx in enumerate(frame_set):
        data.qvel[:] = 0.0
        for obj in delivered_scene.objects:
            adr = qpos_adrs.get(obj.object_id)
            if adr is None:
                continue
            pos, eul = _interp_kf_pos(obj.keyframes, frame_idx)
            data.qpos[adr : adr + 3] = pos
            data.qpos[adr + 3 : adr + 7] = _euler_to_quat_wxyz(eul)
            bid = body_id_by_obj.get(obj.object_id)
            if bid is not None and len(frame_set) >= 2:
                prev_f = frame_set[max(0, idx - 1)]
                next_f = frame_set[min(len(frame_set) - 1, idx + 1)]
                p_prev, _ = _interp_kf_pos(obj.keyframes, prev_f)
                p_next, _ = _interp_kf_pos(obj.keyframes, next_f)
                dt_s = max(1e-3, (next_f - prev_f) / max(1.0, delivered_scene.fps))
                dof_adr = int(model.body_dofadr[bid])
                data.qvel[dof_adr : dof_adr + 3] = (
                    np.array(p_next, dtype=np.float64) - np.array(p_prev, dtype=np.float64)
                ) / dt_s
            ref_obj = ref_by_id.get(obj.object_id)
            if ref_obj is not None:
                r_pos, _ = _interp_kf_pos(ref_obj.keyframes, frame_idx)
                track_errs.append(
                    float(
                        np.linalg.norm(
                            np.array(pos, dtype=np.float64) - np.array(r_pos, dtype=np.float64)
                        )
                    )
                )

        mj_forward(model, data)
        pen, touching = _step_contact_summary(model, data)
        max_pen = max(max_pen, pen)
        if dynamic_body_ids:
            float_body_frames += _count_floating_bodies(
                model,
                data,
                dynamic_body_ids,
                touching,
                body_hz,
                floor_z=delivered_scene.floor_z,
            )
            total_body_frames += len(dynamic_body_ids)

    float_ratio = float_body_frames / float(max(1, total_body_frames))
    mean_track_err = float(np.mean(track_errs)) if track_errs else 0.0
    return max_pen, float_ratio, mean_track_err


def _step_and_collect(
    model: Any,
    num_steps: int,
    perturb_vel: float = 0.0,
    floor_z: float = 0.0,
    scene: CoarseSceneSpec | None = None,
) -> tuple[bool, bool, float, float, float, list[str]]:
    """Run free MuJoCo rollout from initial state and collect per-body stability diagnostics."""
    data = new_mj_data(model)
    if perturb_vel != 0.0 and model.nv > 0:
        data.qvel[:] = perturb_vel

    dynamic_body_ids = [
        body_id for body_id in range(1, model.nbody) if model.body_dofnum[body_id] > 0
    ]
    body_hz = _body_half_heights(model, dynamic_body_ids, scene=scene)

    max_penetration = 0.0
    floating_body_steps = 0
    total_body_steps = 0
    finite_ok = True
    bounds_ok = True
    notes: list[str] = []
    vel_norms: list[float] = []

    for step in range(num_steps):
        mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            finite_ok = False
            notes.append(f"Non-finite state detected at step {step}")
            break

        escaped_body = _bodies_within_bounds(data, dynamic_body_ids)
        if escaped_body is not None:
            bounds_ok = False
            notes.append(f"Body {escaped_body} exited scene bounds at step {step}")
            break

        step_pen, touching = _step_contact_summary(model, data)
        max_penetration = max(max_penetration, step_pen)
        vel_norms.append(float(np.linalg.norm(data.qvel)) if model.nv > 0 else 0.0)

        if dynamic_body_ids:
            floating_body_steps += _count_floating_bodies(
                model, data, dynamic_body_ids, touching, body_hz, floor_z=floor_z
            )
            total_body_steps += len(dynamic_body_ids)

    floating_ratio = floating_body_steps / float(max(1, total_body_steps))
    tail = vel_norms[max(0, len(vel_norms) - max(1, len(vel_norms) // 5)) :]
    mean_tail_speed = float(np.mean(tail)) if tail else 0.0
    support_stability = float(math.exp(-mean_tail_speed)) if (finite_ok and bounds_ok) else 0.0
    return finite_ok, bounds_ok, max_penetration, floating_ratio, support_stability, notes


def evaluate_mujoco_scene(
    mjcf_xml: str,
    num_steps: int = 200,
    counterfactual_perturb_vel: float = 0.05,
    scene: CoarseSceneSpec | None = None,
    delivered_scene: CoarseSceneSpec | None = None,
) -> PhysicsMetrics:
    """Verify delivered multi-frame trajectory plus free-rollout counterfactual survival."""
    model = load_mjcf_model(mjcf_xml)
    active_scene = delivered_scene or scene
    floor_z = active_scene.floor_z if active_scene is not None else 0.0
    finite_ok, bounds_ok, roll_pen, roll_float, stability, notes = _step_and_collect(
        model, num_steps=num_steps, perturb_vel=0.0, floor_z=floor_z, scene=active_scene
    )

    if delivered_scene is not None:
        traj_pen, traj_float, track_err = _evaluate_delivered_keyframes(
            model, delivered_scene=delivered_scene, ref_scene=scene
        )
        max_pen = max(traj_pen, roll_pen)
        float_ratio = traj_float
    else:
        max_pen = roll_pen
        float_ratio = roll_float
        track_err = 0.0

    cf_finite, cf_bounds, cf_pen, cf_float, _, _ = _step_and_collect(
        model,
        num_steps=num_steps,
        perturb_vel=counterfactual_perturb_vel,
        floor_z=floor_z,
        scene=active_scene,
    )
    counterfactual_ok = cf_finite and cf_bounds and cf_pen < 0.03 and cf_float < 0.5
    if max_pen > 0.02:
        notes.append(f"Excessive contact penetration: {max_pen:.4f} m")
    if float_ratio > 0.25:
        notes.append(f"High per-body floating ratio: {float_ratio:.2%}")
    if not counterfactual_ok:
        notes.append("Scene failed small-velocity counterfactual perturbation check")

    return PhysicsMetrics(
        simulator="mujoco",
        num_steps=num_steps,
        stable=finite_ok and bounds_ok,
        object_permanence_ok=finite_ok and bounds_ok,
        max_penetration_m=max_pen,
        floating_frame_ratio=float_ratio,
        support_stability_score=stability,
        trajectory_tracking_err_m=track_err,
        counterfactual_survival=counterfactual_ok,
        notes=notes,
    )
