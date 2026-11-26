"""Unit and data tests for MVOIK-4D dataset loader and probe manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from real2sim_layered.schemas import PhysicalRegime
from real2sim_layered.tools import (
    DEFAULT_PROBE_MANIFEST,
    infer_regime_from_case_id,
    list_cases,
    load_case,
    load_probe_cases,
    load_rgba_frame,
    sample_frame_indices,
)


def test_regime_inference_covers_all_five_regimes() -> None:
    assert infer_regime_from_case_id("0004_brick_fall_1") == PhysicalRegime.RIGID
    assert infer_regime_from_case_id("0029_cups_uncover_object") == (
        PhysicalRegime.OCCLUDED_KINEMATIC
    )
    assert infer_regime_from_case_id("0014_press_ball_1") == PhysicalRegime.DEFORMABLE
    assert infer_regime_from_case_id("0000_cut_apple_1") == PhysicalRegime.TOPOLOGICAL_SPLIT
    assert infer_regime_from_case_id("0039_syringe_draw_eject_cola") == (
        PhysicalRegime.MULTIPHYSICS_OOD
    )


def test_probe_manifest_spans_all_five_regimes() -> None:
    regimes = {spec.regime for spec in DEFAULT_PROBE_MANIFEST.cases}
    assert len(DEFAULT_PROBE_MANIFEST.cases) == 5
    assert regimes == set(PhysicalRegime)


def test_sample_frame_indices_deterministic_endpoints() -> None:
    assert sample_frame_indices(4, max_frames=8) == [0, 1, 2, 3]
    sampled = sample_frame_indices(32, max_frames=6)
    assert sampled[0] == 0
    assert sampled[-1] == 31
    assert len(sampled) == 6
    assert sampled == sorted(sampled)


def test_load_case_synthetic(tmp_path: Path) -> None:
    case_id = "0004_brick_fall_1"
    in_dir = tmp_path / "with_gt" / "input" / case_id
    ev_dir = tmp_path / "with_gt" / "eval_gt" / case_id / "eval_0"
    in_dir.mkdir(parents=True)
    ev_dir.mkdir(parents=True)

    arr = np.zeros((32, 48, 4), dtype=np.uint8)
    arr[8:24, 12:36, :] = [200, 100, 50, 255]
    img = Image.fromarray(arr, mode="RGBA")
    for idx in range(3):
        img.save(in_dir / f"{idx:06d}.png")
        img.save(ev_dir / f"{idx:06d}.png")

    assert list_cases(tmp_path) == [case_id]
    case = load_case(tmp_path, case_id)
    assert case.num_frames == 3
    assert (case.width, case.height) == (48, 32)
    assert "eval_0" in case.eval_views

    loaded = load_rgba_frame(case.input_frames[0], target_size=(24, 16))
    assert loaded.shape == (16, 24, 4)
    assert loaded.dtype == np.float32
    assert 0.0 <= float(loaded.min()) <= float(loaded.max()) <= 1.0


def test_load_live_probe_cases_when_dataset_present() -> None:
    dataset_root = Path("data/MVOIK-4D")
    if not (dataset_root / "with_gt" / "input").is_dir():
        pytest.skip("Local MVOIK-4D/with_gt dataset not present")
    cases = load_probe_cases(dataset_root)
    assert len(cases) == 5
    for case in cases:
        assert case.num_frames >= 30
        assert set(case.eval_views.keys()) == {"eval_0", "eval_1", "eval_2", "eval_3"}
