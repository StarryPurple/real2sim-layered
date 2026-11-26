"""Physics simulation backends and deterministic verifiers (MuJoCo & Blender)."""

from __future__ import annotations

from real2sim_layered.sim.diagnostics import (
    evaluate_mujoco_scene,
    render_mujoco_rgba_sequence,
)

__all__ = [
    "evaluate_mujoco_scene",
    "render_mujoco_rgba_sequence",
]
