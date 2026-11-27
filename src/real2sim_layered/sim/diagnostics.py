"""Deterministic MuJoCo physical diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from real2sim_layered.schemas import PhysicsMetrics
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


def _step_max_penetration(data: Any) -> float:
    """Return maximum penetration depth (m) across active contacts in the current step."""
    max_pen = 0.0
    for c_idx in range(int(data.ncon)):
        dist = float(data.contact[c_idx].dist)
        if dist < 0.0:
            max_pen = max(max_pen, -dist)
    return max_pen


def _step_and_collect(
    model: Any,
    num_steps: int,
    perturb_vel: float = 0.0,
) -> tuple[bool, bool, float, float, float, list[str]]:
    """Run MuJoCo rollout and return stability, permanence, penetration, floating, settling, notes."""
    data = new_mj_data(model)
    if perturb_vel != 0.0 and model.nv > 0:
        data.qvel[:] = perturb_vel

    dynamic_body_ids = [
        body_id for body_id in range(1, model.nbody) if model.body_dofnum[body_id] > 0
    ]

    max_penetration = 0.0
    floating_frames = 0
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

        max_penetration = max(max_penetration, _step_max_penetration(data))
        vel_norm = float(np.linalg.norm(data.qvel)) if model.nv > 0 else 0.0
        vel_norms.append(vel_norm)

        if dynamic_body_ids:
            min_z = min(float(data.xpos[b][2]) for b in dynamic_body_ids)
            if (
                int(data.ncon) == 0
                and min_z > _FLOATING_HEIGHT_M
                and vel_norm < _FLOATING_SPEED_MPS
            ):
                floating_frames += 1

    steps_run = max(1, len(vel_norms))
    floating_ratio = floating_frames / float(steps_run)

    tail = vel_norms[max(0, len(vel_norms) - max(1, steps_run // 5)) :]
    mean_tail_speed = float(np.mean(tail)) if tail else 0.0
    # Smoothly map residual settling velocity into [0, 1] stability score.
    support_stability = float(math.exp(-mean_tail_speed)) if (finite_ok and bounds_ok) else 0.0
    if max_penetration > 0.02:
        notes.append(f"Excessive contact penetration: {max_penetration:.4f} m")
    if floating_ratio > 0.25:
        notes.append(f"High unsupported floating ratio: {floating_ratio:.2%}")

    return finite_ok, bounds_ok, max_penetration, floating_ratio, support_stability, notes


def evaluate_mujoco_scene(
    mjcf_xml: str,
    num_steps: int = 200,
    counterfactual_perturb_vel: float = 0.05,
) -> PhysicsMetrics:
    """Deterministically verify a MuJoCo MJCF scene and run a counterfactual perturbation check."""
    model = load_mjcf_model(mjcf_xml)
    finite_ok, bounds_ok, max_pen, float_ratio, stability, notes = _step_and_collect(
        model, num_steps=num_steps, perturb_vel=0.0
    )
    cf_finite, cf_bounds, cf_pen, _, _, _ = _step_and_collect(
        model, num_steps=num_steps, perturb_vel=counterfactual_perturb_vel
    )
    counterfactual_ok = cf_finite and cf_bounds and cf_pen < 0.05
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
        counterfactual_survival=counterfactual_ok,
        notes=notes,
    )
