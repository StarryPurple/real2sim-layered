"""Dual-axis evaluation schemas (visual fidelity + physical plausibility)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from real2sim_layered.schemas.dataset import PhysicalRegime


class ViewVisualMetrics(BaseModel):
    """Per-view visual comparison metrics against MVOIK-4D RGBA ground truth."""

    view_name: str
    num_frames: int = Field(ge=1)
    silhouette_iou: float = Field(ge=0.0, le=1.0)
    masked_psnr_db: float
    masked_mae: float = Field(ge=0.0, le=1.0)
    patch_cosine_sim: float = Field(ge=-1.0, le=1.0)


class VisualMetrics(BaseModel):
    """Aggregated multi-view visual fidelity metrics for one case."""

    mean_silhouette_iou: float = Field(ge=0.0, le=1.0)
    mean_masked_psnr_db: float
    mean_masked_mae: float = Field(ge=0.0, le=1.0)
    mean_patch_cosine_sim: float = Field(ge=-1.0, le=1.0)
    per_view: list[ViewVisualMetrics] = Field(default_factory=list)


class PhysicsMetrics(BaseModel):
    """Deterministic physical verification diagnostics from MuJoCo/Blender."""

    simulator: str = "mujoco"
    num_steps: int = Field(ge=1)
    stable: bool
    object_permanence_ok: bool
    max_penetration_m: float = Field(ge=0.0)
    floating_frame_ratio: float = Field(ge=0.0, le=1.0)
    support_stability_score: float = Field(ge=0.0, le=1.0)
    counterfactual_survival: bool
    notes: list[str] = Field(default_factory=list)


class DualAxisReport(BaseModel):
    """Combined visual + physical evaluation report for one MVOIK-4D case."""

    case_id: str
    regime: PhysicalRegime
    method_tag: str
    visual: VisualMetrics | None = None
    physics: PhysicsMetrics | None = None
    wall_time_s: float = Field(default=0.0, ge=0.0)
    vlm_tokens_used: int = Field(default=0, ge=0)
