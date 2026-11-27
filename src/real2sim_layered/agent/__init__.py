"""Layered Real2Sim agent modules (Layer 1 synthesis, Layer 2 escalation, Layer 3 fallback)."""

from __future__ import annotations

from real2sim_layered.agent.layer1 import (
    Layer1Artifacts,
    build_canonical_cameras,
    materialize_layer1_scene,
    synthesize_coarse_scene,
)
from real2sim_layered.agent.layer2 import (
    detect_escalation_windows,
    run_global_rigid_physics,
    run_layer2_escalation,
)
from real2sim_layered.agent.layer3 import detect_and_apply_fallback

__all__ = [
    "Layer1Artifacts",
    "build_canonical_cameras",
    "detect_and_apply_fallback",
    "detect_escalation_windows",
    "materialize_layer1_scene",
    "run_global_rigid_physics",
    "run_layer2_escalation",
    "synthesize_coarse_scene",
]
