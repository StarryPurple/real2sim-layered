#!/usr/bin/env python3
"""Run 30-case H2 feedback-modality ablation: visual_only vs physics_only vs dual_axis."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats

from real2sim_layered.agent import (
    detect_and_apply_fallback,
    detect_escalation_windows,
    materialize_layer1_scene,
    run_layer2_escalation,
    synthesize_coarse_scene,
)
from real2sim_layered.eval import EvalRunMeta, evaluate_case_dual_axis
from real2sim_layered.schemas import (
    AssetBackend,
    CoarseSceneSpec,
    DualAxisReport,
    ExecutionMode,
    InteractionGraph,
    MvoikCase,
    RuntimeConfig,
)
from real2sim_layered.sim import build_mjcf_xml, evaluate_mujoco_scene, render_coarse_scene_mujoco
from real2sim_layered.tools import list_cases, load_case, load_rgba_frame, sample_frame_indices


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


def _load_gt_views(
    case: MvoikCase, indices: list[int], target_size: tuple[int, int] = (160, 120)
) -> dict[str, list[np.ndarray]]:
    return {
        view_name: [load_rgba_frame(frames[idx], target_size=target_size) for idx in indices]
        for view_name, frames in sorted(case.eval_views.items())
    }


def _run_visual_only_arm(
    case: MvoikCase, l1_scene: CoarseSceneSpec, eval_indices: list[int]
) -> CoarseSceneSpec:
    """Visual-only refinement: optimize vertical offset strictly by input-view silhouette IoU."""
    best_scene = l1_scene
    best_iou = _input_view_iou(l1_scene, case, eval_indices)
    for dz in (-0.06, -0.03, 0.03, 0.06):
        shifted_objs = [
            obj.model_copy(
                update={
                    "keyframes": [
                        kf.model_copy(
                            update={
                                "position": (kf.position[0], kf.position[1], kf.position[2] + dz)
                            }
                        )
                        for kf in obj.keyframes
                    ]
                }
            )
            for obj in l1_scene.objects
        ]
        cand = l1_scene.model_copy(update={"objects": shifted_objs})
        iou = _input_view_iou(cand, case, eval_indices)
        if iou > best_iou + 1e-4:
            best_iou = iou
            best_scene = cand
    return best_scene


def _run_physics_only_arm(
    case: MvoikCase,
    ikg: InteractionGraph,
    l1_scene: CoarseSceneSpec,
    eval_indices: list[int],
) -> tuple[CoarseSceneSpec, ExecutionMode, bool]:
    """Physics-only refinement: L2 contact escalation without input-view visual silhouette guard."""
    windows = detect_escalation_windows(case, ikg, eval_indices)
    phys_scene, phys_metrics, _ = run_layer2_escalation(
        l1_scene, windows=windows, frame_indices=eval_indices, case=None
    )
    if not phys_metrics.stable or phys_metrics.max_penetration_m > 0.08:
        cand = l1_scene.model_copy(update={"floor_z": phys_scene.floor_z})
        cand_phys = evaluate_mujoco_scene(
            build_mjcf_xml(cand, include_floor=True),
            num_steps=160,
            scene=l1_scene,
            delivered_scene=cand,
        )
        if cand_phys.stable and cand_phys.max_penetration_m < phys_metrics.max_penetration_m:
            return cand, "fallback_kinematic", True
    return phys_scene, "physics_sim", False


@dataclass(frozen=True)
class _CaseCtx:
    case: MvoikCase
    ref_scene: CoarseSceneSpec
    eval_indices: list[int]
    gt_by_view: dict[str, list[np.ndarray]]


def _eval_arm(
    ctx: _CaseCtx,
    scene: CoarseSceneSpec,
    method_tag: str,
    mode: ExecutionMode,
    triggered: bool,
    out_dir: Path,
    t0: float,
) -> DualAxisReport:
    art = materialize_layer1_scene(
        scene=scene,
        frame_indices=ctx.eval_indices,
        views=list(ctx.gt_by_view.keys()),
        out_dir=out_dir,
        width=160,
        height=120,
    )
    rep = evaluate_case_dual_axis(
        case=ctx.case,
        method_tag=method_tag,
        pred_by_view=art.rendered_by_view,
        gt_by_view=ctx.gt_by_view,
        mjcf_xml=art.mjcf_xml,
        scene=ctx.ref_scene,
        delivered_scene=scene,
        meta=EvalRunMeta(
            physics_steps=200,
            execution_mode=mode,
            anomaly_triggered=triggered,
        ),
    )
    return rep.model_copy(update={"wall_time_s": time.perf_counter() - t0})


def main() -> None:
    parser = argparse.ArgumentParser(description="30-case H2 feedback-modality ablation.")
    parser.add_argument("--ikg-dir", type=Path, default=Path("outputs/ikg_all30_qwen35_4b_roi"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ablation_all30_feedback_h2"))
    parser.add_argument("--max-frames", type=int, default=6)
    args = parser.parse_args()

    cfg = RuntimeConfig.from_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cids = list_cases(cfg.dataset_root)
    reports: list[DualAxisReport] = []

    for cid in cids:
        case = load_case(cfg.dataset_root, cid)
        ikg = InteractionGraph.model_validate_json(
            (args.ikg_dir / f"{cid}.json").read_text(encoding="utf-8")
        )
        eval_indices = sample_frame_indices(case.num_frames, max_frames=args.max_frames)
        gt_by_view = _load_gt_views(case, eval_indices)
        l1_scene = synthesize_coarse_scene(
            case=case, ikg=ikg, backend=AssetBackend.SIM_PROXY, max_frames=args.max_frames
        )
        ctx = _CaseCtx(case, l1_scene, eval_indices, gt_by_view)

        # Arm 1: visual_only
        t1 = time.perf_counter()
        vis_scene = _run_visual_only_arm(case, l1_scene, eval_indices)
        reports.append(
            _eval_arm(
                ctx,
                vis_scene,
                "visual_only",
                "kinematic",
                False,
                args.out_dir / "visual_only" / cid,
                t1,
            )
        )

        # Arm 2: physics_only (no visual silhouette guard in L2/L3)
        t2 = time.perf_counter()
        phys_scene, phys_mode, phys_trig = _run_physics_only_arm(case, ikg, l1_scene, eval_indices)
        reports.append(
            _eval_arm(
                ctx,
                phys_scene,
                "physics_only",
                phys_mode,
                phys_trig,
                args.out_dir / "physics_only" / cid,
                t2,
            )
        )

        # Arm 3: dual_axis (full L1+L2+L3 with counterfactual-guarded pre-commit gate)
        t3 = time.perf_counter()
        windows = detect_escalation_windows(case, ikg, eval_indices)
        l2_scene, l2_phys, _ = run_layer2_escalation(l1_scene, windows, eval_indices, case=case)
        l3_scene, anom = detect_and_apply_fallback(
            case=case,
            ikg=ikg,
            l1_scene=l1_scene,
            l2_scene=l2_scene,
            l2_phys=l2_phys,
            frame_indices=eval_indices,
        )
        reports.append(
            _eval_arm(
                ctx,
                l3_scene,
                "dual_axis",
                anom.fallback_mode,
                anom.triggered,
                args.out_dir / "dual_axis" / cid,
                t3,
            )
        )
        print(f"[DONE] {cid:32s} dual_axis mode={anom.fallback_mode} trig={anom.triggered}")

    out_json = args.out_dir / "reports.json"
    out_json.write_text(
        json.dumps([r.model_dump(mode="json") for r in reports], indent=2) + "\n",
        encoding="utf-8",
    )

    by_arm: dict[str, list[DualAxisReport]] = defaultdict(list)
    for r in reports:
        by_arm[r.method_tag].append(r)

    print("\n=== 30-CASE H2 FEEDBACK-MODALITY ABLATION SUMMARY (sim_proxy) ===")
    for arm in ("visual_only", "physics_only", "dual_axis"):
        rs = by_arm[arm]
        iou = np.mean([r.visual.mean_silhouette_iou for r in rs if r.visual])
        psnr = np.mean([r.visual.mean_masked_psnr_db for r in rs if r.visual])
        cos = np.mean([r.visual.mean_patch_cosine_sim for r in rs if r.visual])
        pen = np.mean([r.physics.max_penetration_m for r in rs if r.physics])
        trk = np.mean([r.physics.trajectory_tracking_err_m for r in rs if r.physics])
        cf = np.mean(
            [1.0 if (r.physics and r.physics.counterfactual_survival) else 0.0 for r in rs]
        )
        print(
            f"{arm:14s} IoU={iou:.4f} PSNR={psnr:.2f} Cos={cos:.4f} "
            f"MaxPen={pen:.4f} TrackErr={trk:.4f} CF={cf:.1%}"
        )

    vis_pen = [r.physics.max_penetration_m for r in by_arm["visual_only"] if r.physics]
    dual_pen = [r.physics.max_penetration_m for r in by_arm["dual_axis"] if r.physics]
    phys_iou = [r.visual.mean_silhouette_iou for r in by_arm["physics_only"] if r.visual]
    dual_iou = [r.visual.mean_silhouette_iou for r in by_arm["dual_axis"] if r.visual]
    p_pen = float(cast(Any, stats.wilcoxon(vis_pen, dual_pen)).pvalue)
    p_iou = float(cast(Any, stats.wilcoxon(dual_iou, phys_iou)).pvalue)
    print(
        f"Wilcoxon p: MaxPen(dual < visual_only)={p_pen:.2e} | IoU(dual > physics_only)={p_iou:.2e}"
    )


if __name__ == "__main__":
    main()
