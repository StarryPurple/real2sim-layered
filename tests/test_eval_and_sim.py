"""Unit tests for MuJoCo physical diagnostics and dual-axis evaluation metrics."""

from __future__ import annotations

import numpy as np

from real2sim_layered.eval import (
    compute_frame_visual_metrics,
    evaluate_multiview_rgba,
    format_reports_markdown,
)
from real2sim_layered.schemas import DualAxisReport, PhysicalRegime
from real2sim_layered.sim import evaluate_mujoco_scene

_STABLE_BLOCK_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.055">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.2" rgba="0.2 0.5 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FLOATING_ZERO_G_XML = """
<mujoco>
  <option gravity="0 0 0" timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="floater" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_visual_metrics_identical_and_disjoint() -> None:
    gt = np.zeros((64, 64, 4), dtype=np.float32)
    gt[16:48, 16:48, :3] = [0.8, 0.2, 0.1]
    gt[16:48, 16:48, 3] = 1.0

    iou, psnr, mae, cos = compute_frame_visual_metrics(gt, gt)
    assert iou == 1.0
    assert psnr == 50.0
    assert mae == 0.0
    assert cos > 0.999

    shifted = np.zeros_like(gt)
    shifted[0:16, 0:16, :3] = [0.1, 0.9, 0.2]
    shifted[0:16, 0:16, 3] = 1.0
    iou_bad, _, mae_bad, _ = compute_frame_visual_metrics(shifted, gt)
    assert iou_bad == 0.0
    assert mae_bad > 0.1

    mv = evaluate_multiview_rgba({"eval_0": [gt], "eval_1": [gt]}, {"eval_0": [gt], "eval_1": [gt]})
    assert mv.mean_silhouette_iou == 1.0
    assert len(mv.per_view) == 2


def test_mujoco_diagnostics_distinguish_stable_vs_floating() -> None:
    stable = evaluate_mujoco_scene(_STABLE_BLOCK_XML, num_steps=120)
    assert stable.stable is True
    assert stable.object_permanence_ok is True
    assert stable.max_penetration_m < 0.02
    assert stable.floating_frame_ratio < 0.1
    assert stable.counterfactual_survival is True

    floating = evaluate_mujoco_scene(_FLOATING_ZERO_G_XML, num_steps=80)
    assert floating.floating_frame_ratio > 0.9
    assert any("floating" in note.lower() for note in floating.notes)


def test_format_reports_markdown() -> None:
    phys = evaluate_mujoco_scene(_STABLE_BLOCK_XML, num_steps=40)
    rep = DualAxisReport(
        case_id="0004_brick_fall_1",
        regime=PhysicalRegime.RIGID,
        method_tag="unit_test",
        physics=phys,
        vlm_tokens_used=420,
    )
    table = format_reports_markdown([rep])
    assert "0004_brick_fall_1" in table
    assert "unit_test" in table
    assert "420" in table
