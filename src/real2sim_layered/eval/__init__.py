"""Dual-axis visual and physical evaluation harness."""

from __future__ import annotations

from real2sim_layered.eval.metrics import (
    EvalRunMeta,
    compute_frame_visual_metrics,
    evaluate_case_dual_axis,
    evaluate_multiview_rgba,
    format_reports_markdown,
)

__all__ = [
    "EvalRunMeta",
    "compute_frame_visual_metrics",
    "evaluate_case_dual_axis",
    "evaluate_multiview_rgba",
    "format_reports_markdown",
]
