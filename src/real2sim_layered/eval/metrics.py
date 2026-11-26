"""Dual-axis evaluation: multi-view RGBA visual metrics + physical diagnostics."""

from __future__ import annotations

import math

import numpy as np

from real2sim_layered.schemas import (
    DualAxisReport,
    MvoikCase,
    PhysicsMetrics,
    ViewVisualMetrics,
    VisualMetrics,
)
from real2sim_layered.sim import evaluate_mujoco_scene

_ALPHA_THRESHOLD = 0.5
_MAX_PSNR_DB = 50.0


def _downsample_patch_vector(rgba: np.ndarray, grid: int = 16) -> np.ndarray:
    """Compute a normalized spatial-color patch descriptor over premultiplied RGB."""
    height, width, _ = rgba.shape
    premul = rgba[..., :3] * rgba[..., 3:4]
    y_edges = np.linspace(0, height, num=grid + 1, dtype=int)
    x_edges = np.linspace(0, width, num=grid + 1, dtype=int)
    patches = np.zeros((grid, grid, 3), dtype=np.float32)
    for yi in range(grid):
        for xi in range(grid):
            block = premul[
                y_edges[yi] : max(y_edges[yi] + 1, y_edges[yi + 1]),
                x_edges[xi] : max(x_edges[xi] + 1, x_edges[xi + 1]),
            ]
            patches[yi, xi] = block.mean(axis=(0, 1))
    vec = patches.reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


def compute_frame_visual_metrics(
    pred_rgba: np.ndarray,
    gt_rgba: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute (silhouette_iou, masked_psnr_db, masked_mae, patch_cosine_sim) for one frame pair."""
    if pred_rgba.shape != gt_rgba.shape or pred_rgba.ndim != 3 or pred_rgba.shape[-1] != 4:
        raise ValueError(
            f"Expected matching (H, W, 4) RGBA arrays, got {pred_rgba.shape} and {gt_rgba.shape}"
        )

    pred_mask = pred_rgba[..., 3] >= _ALPHA_THRESHOLD
    gt_mask = gt_rgba[..., 3] >= _ALPHA_THRESHOLD
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    iou = 1.0 if union == 0 else inter / float(union)

    eval_mask = np.logical_or(pred_mask, gt_mask)
    if not np.any(eval_mask):
        mae = 0.0
        psnr = _MAX_PSNR_DB
    else:
        pred_rgb = pred_rgba[..., :3] * pred_rgba[..., 3:4]
        gt_rgb = gt_rgba[..., :3] * gt_rgba[..., 3:4]
        diff = pred_rgb[eval_mask] - gt_rgb[eval_mask]
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff * diff))
        psnr = _MAX_PSNR_DB if mse <= 1e-10 else min(_MAX_PSNR_DB, 10.0 * math.log10(1.0 / mse))

    pred_vec = _downsample_patch_vector(pred_rgba)
    gt_vec = _downsample_patch_vector(gt_rgba)
    cosine = float(np.clip(np.dot(pred_vec, gt_vec), -1.0, 1.0))
    return iou, psnr, mae, cosine


def evaluate_multiview_rgba(
    pred_by_view: dict[str, list[np.ndarray]],
    gt_by_view: dict[str, list[np.ndarray]],
) -> VisualMetrics:
    """Aggregate visual fidelity metrics across all shared evaluation views."""
    shared_views = sorted(set(pred_by_view.keys()) & set(gt_by_view.keys()))
    if not shared_views:
        raise ValueError("No overlapping views between prediction and ground truth.")

    per_view: list[ViewVisualMetrics] = []
    for view_name in shared_views:
        preds = pred_by_view[view_name]
        gts = gt_by_view[view_name]
        n = min(len(preds), len(gts))
        if n == 0:
            raise ValueError(f"Empty frame list for view '{view_name}'")
        ious: list[float] = []
        psnrs: list[float] = []
        maes: list[float] = []
        cosines: list[float] = []
        for idx in range(n):
            iou, psnr, mae, cos = compute_frame_visual_metrics(preds[idx], gts[idx])
            ious.append(iou)
            psnrs.append(psnr)
            maes.append(mae)
            cosines.append(cos)
        per_view.append(
            ViewVisualMetrics(
                view_name=view_name,
                num_frames=n,
                silhouette_iou=float(np.mean(ious)),
                masked_psnr_db=float(np.mean(psnrs)),
                masked_mae=float(np.mean(maes)),
                patch_cosine_sim=float(np.mean(cosines)),
            )
        )

    return VisualMetrics(
        mean_silhouette_iou=float(np.mean([v.silhouette_iou for v in per_view])),
        mean_masked_psnr_db=float(np.mean([v.masked_psnr_db for v in per_view])),
        mean_masked_mae=float(np.mean([v.masked_mae for v in per_view])),
        mean_patch_cosine_sim=float(np.mean([v.patch_cosine_sim for v in per_view])),
        per_view=per_view,
    )


def evaluate_case_dual_axis(
    case: MvoikCase,
    method_tag: str,
    pred_by_view: dict[str, list[np.ndarray]] | None = None,
    gt_by_view: dict[str, list[np.ndarray]] | None = None,
    mjcf_xml: str | None = None,
    physics_steps: int = 200,
    wall_time_s: float = 0.0,
    vlm_tokens_used: int = 0,
) -> DualAxisReport:
    """Compute a DualAxisReport combining visual metrics and MuJoCo physics diagnostics."""
    visual: VisualMetrics | None = None
    if pred_by_view is not None and gt_by_view is not None:
        visual = evaluate_multiview_rgba(pred_by_view, gt_by_view)

    physics: PhysicsMetrics | None = None
    if mjcf_xml is not None:
        physics = evaluate_mujoco_scene(mjcf_xml, num_steps=physics_steps)

    return DualAxisReport(
        case_id=case.case_id,
        regime=case.regime,
        method_tag=method_tag,
        visual=visual,
        physics=physics,
        wall_time_s=wall_time_s,
        vlm_tokens_used=vlm_tokens_used,
    )


def format_reports_markdown(reports: list[DualAxisReport]) -> str:
    """Render a compact Markdown summary table suitable for .agents/records/."""
    header = (
        "| Case | Regime | Method | IoU | PSNR (dB) | CosSim | Stable | MaxPen (m) | Float% | CF-OK | Tokens |\n"
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    rows = [header]
    for rep in reports:
        iou = f"{rep.visual.mean_silhouette_iou:.3f}" if rep.visual else "—"
        psnr = f"{rep.visual.mean_masked_psnr_db:.1f}" if rep.visual else "—"
        cos = f"{rep.visual.mean_patch_cosine_sim:.3f}" if rep.visual else "—"
        stable = ("yes" if rep.physics.stable else "no") if rep.physics else "—"
        pen = f"{rep.physics.max_penetration_m:.4f}" if rep.physics else "—"
        flt = f"{rep.physics.floating_frame_ratio * 100:.1f}%" if rep.physics else "—"
        cf = ("yes" if rep.physics.counterfactual_survival else "no") if rep.physics else "—"
        rows.append(
            f"| `{rep.case_id}` | `{rep.regime}` | `{rep.method_tag}` | "
            f"{iou} | {psnr} | {cos} | {stable} | {pen} | {flt} | {cf} | {rep.vlm_tokens_used} |"
        )
    return "\n".join(rows)
