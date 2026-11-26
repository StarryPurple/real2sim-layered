"""Structured Interaction Knowledge Graph (G_int) schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from real2sim_layered.schemas.dataset import PhysicalRegime


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


class InteractionGraph(BaseModel):
    """Layer-1 structured Interaction Knowledge Graph (G_int) extracted by VLM."""

    case_id: str
    summary: str = Field(description="One-sentence physical description of the scene interaction.")
    dominant_regime: PhysicalRegime
    recommended_backend: AssetBackend = Field(default=AssetBackend.SIM_PROXY)
    objects: list[ObjectEntity] = Field(min_length=1)
    intervals: list[InteractionInterval] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
