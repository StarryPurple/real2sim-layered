"""Coarse 4D scene specification (S_0) shared by MuJoCo and Blender backends."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from real2sim_layered.schemas.dataset import PhysicalRegime
from real2sim_layered.schemas.ikg import AssetBackend

Vec3 = tuple[float, float, float]
RgbaColor = tuple[float, float, float, float]
PrimitiveKind = Literal["box", "sphere", "cylinder", "capsule"]


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
    fov_y_deg: float = Field(default=42.0, gt=5.0, lt=140.0)


class CoarseSceneSpec(BaseModel):
    """Layer-1 editable 4D scene hypothesis (S_0) compiled to MJCF or Blender."""

    case_id: str
    backend: AssetBackend
    regime: PhysicalRegime
    num_frames: int = Field(ge=1)
    fps: float = Field(default=24.0, gt=0.0)
    floor_z: float = 0.0
    cameras: list[CameraSpec] = Field(min_length=1)
    objects: list[SceneObjectSpec] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)
