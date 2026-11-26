"""Dataset and probe manifest schemas for MVOIK-4D."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


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
