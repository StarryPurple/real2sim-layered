"""MVOIK-4D dataset loader, regime taxonomy, and 5-case probe manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_layered.schemas import MvoikCase, PhysicalRegime, ProbeCaseSpec, ProbeManifest

DEFAULT_PROBE_MANIFEST = ProbeManifest(
    name="mvoik4d_with_gt_probe_v1",
    split="with_gt",
    cases=[
        ProbeCaseSpec(
            case_id="0004_brick_fall_1",
            regime=PhysicalRegime.RIGID,
            rationale="Pure rigid-body gravity, tipping, and ground contact.",
        ),
        ProbeCaseSpec(
            case_id="0029_cups_uncover_object",
            regime=PhysicalRegime.OCCLUDED_KINEMATIC,
            rationale="Multi-object containment, occlusion, and object permanence.",
        ),
        ProbeCaseSpec(
            case_id="0014_press_ball_1",
            regime=PhysicalRegime.DEFORMABLE,
            rationale="Elastic contact compression where rigid-body physics fails.",
        ),
        ProbeCaseSpec(
            case_id="0000_cut_apple_1",
            regime=PhysicalRegime.TOPOLOGICAL_SPLIT,
            rationale="Cutting induces topological split into distinct sub-objects.",
        ),
        ProbeCaseSpec(
            case_id="0039_syringe_draw_eject_cola",
            regime=PhysicalRegime.MULTIPHYSICS_OOD,
            rationale="Fluid drawing/ejection outside rigid or elastic solver scope.",
        ),
    ],
)


def infer_regime_from_case_id(case_id: str) -> PhysicalRegime:
    """Classify an MVOIK-4D case ID into one of the five physical regimes."""
    name = case_id.lower()
    if any(tok in name for tok in ("cut_", "peel_")):
        return PhysicalRegime.TOPOLOGICAL_SPLIT
    if any(tok in name for tok in ("press_", "squeeze_", "fold_", "bend_", "stretch_")):
        return PhysicalRegime.DEFORMABLE
    if any(tok in name for tok in ("shuffle", "cups_", "cup_", "cover", "uncover", "box_open")):
        return PhysicalRegime.OCCLUDED_KINEMATIC
    if any(tok in name for tok in ("match_lit", "syringe", "pour_", "blow_", "cola", "fanta")):
        return PhysicalRegime.MULTIPHYSICS_OOD
    return PhysicalRegime.RIGID


def list_cases(dataset_root: Path, split: str = "with_gt") -> list[str]:
    """Return sorted case IDs present in `<dataset_root>/<split>/input`."""
    input_root = dataset_root / split / "input"
    if not input_root.is_dir():
        raise FileNotFoundError(f"MVOIK-4D split input directory not found: {input_root}")
    return sorted(p.name for p in input_root.iterdir() if p.is_dir())


def load_case(dataset_root: Path, case_id: str, split: str = "with_gt") -> MvoikCase:
    """Load frame paths, resolution, and regime metadata for one MVOIK-4D case."""
    input_dir = dataset_root / split / "input" / case_id
    eval_gt_dir = dataset_root / split / "eval_gt" / case_id
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Case input directory not found: {input_dir}")

    input_frames = sorted(input_dir.glob("*.png"))
    if not input_frames:
        raise ValueError(f"No PNG frames found in {input_dir}")

    with Image.open(input_frames[0]) as first_img:
        width, height = first_img.size

    eval_views: dict[str, list[Path]] = {}
    if eval_gt_dir.is_dir():
        for view_dir in sorted(p for p in eval_gt_dir.iterdir() if p.is_dir()):
            view_frames = sorted(view_dir.glob("*.png"))
            if view_frames:
                eval_views[view_dir.name] = view_frames

    return MvoikCase(
        case_id=case_id,
        split=split,
        regime=infer_regime_from_case_id(case_id),
        input_dir=input_dir,
        eval_gt_dir=eval_gt_dir,
        input_frames=input_frames,
        eval_views=eval_views,
        num_frames=len(input_frames),
        width=width,
        height=height,
    )


def load_probe_cases(
    dataset_root: Path,
    manifest: ProbeManifest = DEFAULT_PROBE_MANIFEST,
) -> list[MvoikCase]:
    """Load all cases defined in a probe manifest."""
    cases: list[MvoikCase] = []
    for spec in manifest.cases:
        loaded = load_case(dataset_root, spec.case_id, split=manifest.split)
        if loaded.regime != spec.regime:
            loaded = loaded.model_copy(update={"regime": spec.regime})
        cases.append(loaded)
    return cases


def sample_frame_indices(num_frames: int, max_frames: int = 8) -> list[int]:
    """Deterministically sample evenly spaced frame indices covering [0, num_frames - 1]."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if num_frames <= max_frames:
        return list(range(num_frames))
    raw = np.linspace(0, num_frames - 1, num=max_frames)
    return sorted({round(float(idx)) for idx in raw})


def load_rgba_frame(
    path: Path,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a PNG frame as float32 RGBA in [0, 1] with shape (H, W, 4)."""
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        if target_size is not None and rgba.size != target_size:
            rgba = rgba.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(rgba, dtype=np.float32) / 255.0
    return arr
