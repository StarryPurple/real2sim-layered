"""Deterministic MuJoCo physical diagnostics (per-body contact, floating, trajectory tracking)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from real2sim_layered.schemas import CoarseSceneSpec, PhysicsMetrics
from real2sim_layered.sim._mujoco import load_mjcf_model, mj_step, new_mj_data

_POSITION_BOUND_M = 20.0
_FLOATING_HEIGHT_M = 0.03
_FLOATING_SPEED_MPS = 0.05


def _bodies_within_bounds(data: Any, dynamic_body_ids: list[int]) -> int | None:
    """Return the first body ID outside scene bounds, or None if all are within bounds."""
    for body_id in dynamic_body_ids:
        if float(np.linalg.norm(data.xpos[body_id])) > _POSITION_BOUND_M:
            return body_id
    return None


def _step_contact_summary(model: Any, data: Any) -> tuple[float, set[int]]:
    """Return (max_penetration_m, body_ids_in_contact) for the current MuJoCo step."""
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


def _body_half_heights(model: Any, dynamic_body_ids: list[int]) -> dict[int, float]:
    """Estimate vertical half-extent per dynamic body from its attached geoms."""
    half_z: dict[int, float] = {bid: 0.05 for bid in dynamic_body_ids}
    for geom_id in range(int(model.ngeom)):
        bid = int(model.geom_bodyid[geom_id])
        if bid not in half_z:
            continue
        size = model.geom_size[geom_id]
        # Sphere uses size[0]; box/cylinder/capsule use size[2] or size[1].
        hz = float(size[2]) if float(size[2]) > 0.0 else float(size[0])
        half_z[bid] = max(0.01, hz)
    return half_z


def _count_floating_bodies(
    model: Any,
    data: Any,
    dynamic_body_ids: list[int],
    touching_bodies: set[int],
    body_hz: dict[int, float],
) -> int:
    """Count dynamic bodies that are unsupported, elevated above floor, and nearly stationary."""
    floating = 0
    for bid in dynamic_body_ids:
        if bid in touching_bodies:
            continue
        z_bottom = float(data.xpos[bid][2]) - body_hz[bid]
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


def _interpolate_target_positions(
    scene: CoarseSceneSpec,
    sim_time_s: float,
) -> list[np.ndarray]:
    """Return observed target positions for dynamic objects at `sim_time_s`."""
    frame_f = max(0.0, min(float(scene.num_frames - 1), sim_time_s * scene.fps))
    targets: list[np.ndarray] = []
    for obj in scene.objects:
        if not obj.is_dynamic:
            continue
        kfs = sorted(obj.keyframes, key=lambda k: k.frame_idx)
        if frame_f <= kfs[0].frame_idx:
            targets.append(np.array(kfs[0].position, dtype=np.float64))
            continue
        if frame_f >= kfs[-1].frame_idx:
            targets.append(np.array(kfs[-1].position, dtype=np.float64))
            continue
        pos = np.array(kfs[-1].position, dtype=np.float64)
        for idx in range(len(kfs) - 1):
            left, right = kfs[idx], kfs[idx + 1]
            if left.frame_idx <= frame_f <= right.frame_idx:
                alpha = (frame_f - left.frame_idx) / float(max(1, right.frame_idx - left.frame_idx))
                pos = (1.0 - alpha) * np.array(left.position, dtype=np.float64) + alpha * np.array(
                    right.position, dtype=np.float64
                )
                break
        targets.append(pos)
    return targets


def _step_and_collect(
    model: Any,
    num_steps: int,
    perturb_vel: float = 0.0,
    scene: CoarseSceneSpec | None = None,
) -> tuple[bool, bool, float, float, float, float, list[str]]:
    """Run MuJoCo rollout with per-body contact/floating and optional trajectory tracking."""
    data = new_mj_data(model)
    if perturb_vel != 0.0 and model.nv > 0:
        data.qvel[:] = perturb_vel

    dynamic_body_ids = [
        body_id for body_id in range(1, model.nbody) if model.body_dofnum[body_id] > 0
    ]
    body_hz = _body_half_heights(model, dynamic_body_ids)

    max_penetration = 0.0
    floating_body_steps = 0
    total_body_steps = 0
    finite_ok = True
    bounds_ok = True
    notes: list[str] = []
    vel_norms: list[float] = []
    track_errs: list[float] = []

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
                model, data, dynamic_body_ids, touching, body_hz
            )
            total_body_steps += len(dynamic_body_ids)

        if scene is not None and dynamic_body_ids and step % 5 == 0:
            targets = _interpolate_target_positions(scene, float(data.time))
            for bid, target in zip(dynamic_body_ids, targets, strict=False):
                track_errs.append(float(np.linalg.norm(data.xpos[bid] - target)))

    floating_ratio = floating_body_steps / float(max(1, total_body_steps))
    tail = vel_norms[max(0, len(vel_norms) - max(1, len(vel_norms) // 5)) :]
    mean_tail_speed = float(np.mean(tail)) if tail else 0.0
    support_stability = float(math.exp(-mean_tail_speed)) if (finite_ok and bounds_ok) else 0.0
    mean_track_err = float(np.mean(track_errs)) if track_errs else 0.0

    if max_penetration > 0.02:
        notes.append(f"Excessive contact penetration: {max_penetration:.4f} m")
    if floating_ratio > 0.25:
        notes.append(f"High per-body floating ratio: {floating_ratio:.2%}")

    return (
        finite_ok,
        bounds_ok,
        max_penetration,
        floating_ratio,
        support_stability,
        mean_track_err,
        notes,
    )


def evaluate_mujoco_scene(
    mjcf_xml: str,
    num_steps: int = 200,
    counterfactual_perturb_vel: float = 0.05,
    scene: CoarseSceneSpec | None = None,
) -> PhysicsMetrics:
    """Deterministically verify a MuJoCo MJCF scene with per-body and counterfactual checks."""
    model = load_mjcf_model(mjcf_xml)
    finite_ok, bounds_ok, max_pen, float_ratio, stability, track_err, notes = _step_and_collect(
        model, num_steps=num_steps, perturb_vel=0.0, scene=scene
    )
    cf_finite, cf_bounds, cf_pen, cf_float, _, _, _ = _step_and_collect(
        model, num_steps=num_steps, perturb_vel=counterfactual_perturb_vel, scene=None
    )
    counterfactual_ok = cf_finite and cf_bounds and cf_pen < 0.03 and cf_float < 0.5
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
