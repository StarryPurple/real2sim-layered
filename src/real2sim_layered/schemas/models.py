"""Pure Pydantic domain contracts: dataset, IKG (G_int), scene (S_0), layered trace, and eval."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]
RgbaColor = tuple[float, float, float, float]
PrimitiveKind = Literal["box", "sphere", "cylinder", "capsule"]
WindowSource = Literal["vlm_ikg", "visual_motion", "both"]
ExecutionMode = Literal[
    "kinematic",
    "physics_sim",
    "fallback_deformable",
    "fallback_topological",
    "fallback_kinematic",
]


# ---------------------------------------------------------------------------
# Dataset & physical regime taxonomy
# ---------------------------------------------------------------------------


class PhysicalRegime(StrEnum):
    """Five physically heterogeneous interaction regimes in MVOIK-4D."""

    RIGID = "rigid"
    OCCLUDED_KINEMATIC = "occluded_kinematic"
    DEFORMABLE = "deformable"
    TOPOLOGICAL_SPLIT = "topological_split"
    MULTIPHYSICS_OOD = "multiphysics_ood"


class MvoikCase(BaseModel):
    """Metadata and frame paths for one MVOIK-4D interaction case."""

    case_id: str
    split: str = "with_gt"
    regime: PhysicalRegime
    input_dir: Path
    eval_gt_dir: Path
    input_frames: list[Path]
    eval_views: dict[str, list[Path]]
    num_frames: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ProbeCaseSpec(BaseModel):
    """One case entry in the cross-regime probe manifest."""

    case_id: str
    regime: PhysicalRegime
    rationale: str


class ProbeManifest(BaseModel):
    """Canonical 5-case cross-regime probe split for rapid iteration."""

    name: str = "mvoik4d_with_gt_probe_v1"
    split: str = "with_gt"
    cases: list[ProbeCaseSpec]


# ---------------------------------------------------------------------------
# Interaction Knowledge Graph (G_int)
# ---------------------------------------------------------------------------


class AssetBackend(StrEnum):
    """Selectable geometry backends compared under H3."""

    PROCEDURAL_BLENDER = "procedural_blender"
    SIM_PROXY = "sim_proxy"
    IMAGE_TO_3D_SIM = "image_to_3d_sim"


class ObjectEntity(BaseModel):
    """One participating object (or tool/hand) identified in the video."""

    object_id: str = Field(description="Short snake_case identifier, e.g., 'red_cup' or 'knife'.")
    label: str = Field(description="Semantic object category or description.")
    primitive_proxy: str = Field(
        default="box",
        description="Coarse geometry primitive: box, sphere, cylinder, capsule, or mesh.",
    )
    is_dynamic: bool = Field(
        default=True,
        description="Whether this object moves or deforms during the clip.",
    )
    material_hint: str = Field(
        default="rigid_solid",
        description="Coarse material prior: rigid_solid, soft_elastic, fruit_cuttable, paper_thin, or fluid.",
    )


class InteractionInterval(BaseModel):
    """One temporal segment of multi-object interaction."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    subject_id: str
    object_id: str
    action: str = Field(
        description="Action verb phrase, e.g., 'pushes', 'covers', 'presses', 'cuts'."
    )
    contact_expected: bool = Field(default=True)
    occlusion_expected: bool = Field(default=False)
    requires_physics_escalation: bool = Field(
        default=False,
        description="True when gravity, collision, support transfer, or tipping requires local physics verification.",
    )
    regime: PhysicalRegime = Field(default=PhysicalRegime.RIGID)

    @model_validator(mode="after")
    def _ensure_ordered_interval(self) -> InteractionInterval:
        if self.end_frame < self.start_frame:
            self.start_frame, self.end_frame = self.end_frame, self.start_frame
        return self


class InteractionGraph(BaseModel):
    """Layer-1 structured Interaction Knowledge Graph (G_int) extracted by VLM."""

    case_id: str
    summary: str = Field(description="One-sentence physical description of the scene interaction.")
    dominant_regime: PhysicalRegime
    recommended_backend: AssetBackend = Field(default=AssetBackend.SIM_PROXY)
    objects: list[ObjectEntity] = Field(min_length=1)
    intervals: list[InteractionInterval] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Coarse 4D scene specification (S_0) and Layered Execution Trace
# ---------------------------------------------------------------------------


class KeyframePose(BaseModel):
    """One timed 3D pose keyframe for a scene object."""

    frame_idx: int = Field(ge=0)
    position: Vec3
    euler_xyz: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


class SceneObjectSpec(BaseModel):
    """Coarse geometry, material, and kinematic trajectory for one scene object."""

    object_id: str
    label: str
    primitive: PrimitiveKind = "box"
    size_m: Vec3 = Field(
        default=(0.08, 0.08, 0.08),
        description="Half-extents (hx, hy, hz) in meters.",
    )
    color_rgba: RgbaColor = (0.5, 0.5, 0.5, 1.0)
    mass_kg: float = Field(default=0.3, gt=0.0)
    friction: float = Field(default=0.6, ge=0.0, le=2.0)
    is_dynamic: bool = True
    material_hint: str = "rigid_solid"
    keyframes: list[KeyframePose] = Field(min_length=1)


class CameraSpec(BaseModel):
    """Pinhole camera pose targeting the scene center."""

    name: str
    position: Vec3
    target: Vec3 = (0.0, 0.0, 0.12)
    fov_y_deg: float = Field(default=49.1, gt=5.0, lt=140.0)


class CoarseSceneSpec(BaseModel):
    """Layer-1/2/3 editable 4D scene hypothesis compiled to MJCF or Blender."""

    case_id: str
    backend: AssetBackend
    regime: PhysicalRegime
    num_frames: int = Field(ge=1)
    fps: float = Field(default=24.0, gt=0.0)
    floor_z: float = 0.0
    cameras: list[CameraSpec] = Field(min_length=1)
    objects: list[SceneObjectSpec] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class EscalationWindow(BaseModel):
    """Temporal window selected for Layer-2 local physics escalation."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    source: WindowSource
    reason: str


class AnomalyReport(BaseModel):
    """Layer-3 failure-aware anomaly detection and fallback routing decision."""

    triggered: bool
    non_rigid_shape_cv: float = Field(default=0.0, ge=0.0)
    physics_tracking_err_m: float = Field(default=0.0, ge=0.0)
    silhouette_drop: float = Field(default=0.0)
    fallback_mode: ExecutionMode = "physics_sim"
    reason: str = ""


# ---------------------------------------------------------------------------
# Dual-axis evaluation reports
# ---------------------------------------------------------------------------


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
    """Deterministic physical verification diagnostics from MuJoCo."""

    simulator: str = "mujoco"
    num_steps: int = Field(ge=1)
    stable: bool
    object_permanence_ok: bool
    max_penetration_m: float = Field(ge=0.0)
    floating_frame_ratio: float = Field(ge=0.0, le=1.0)
    support_stability_score: float = Field(ge=0.0, le=1.0)
    trajectory_tracking_err_m: float = Field(default=0.0, ge=0.0)
    counterfactual_survival: bool
    notes: list[str] = Field(default_factory=list)


class DualAxisReport(BaseModel):
    """Combined visual + physical evaluation report for one MVOIK-4D case."""

    case_id: str
    regime: PhysicalRegime
    method_tag: str
    execution_mode: ExecutionMode = "kinematic"
    anomaly_triggered: bool = False
    visual: VisualMetrics | None = None
    physics: PhysicsMetrics | None = None
    wall_time_s: float = Field(default=0.0, ge=0.0)
    vlm_tokens_used: int = Field(default=0, ge=0)
