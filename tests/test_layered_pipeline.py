"""Unit and integration tests for Layer-2 local physics escalation and Layer-3 anomaly fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_layered.agent import (
    build_canonical_cameras,
    detect_and_apply_fallback,
    detect_escalation_windows,
    run_layer2_escalation,
)
from real2sim_layered.schemas import (
    AssetBackend,
    CoarseSceneSpec,
    InteractionGraph,
    InteractionInterval,
    KeyframePose,
    MvoikCase,
    ObjectEntity,
    PhysicalRegime,
    SceneObjectSpec,
)
from real2sim_layered.sim import build_mjcf_xml, evaluate_mujoco_scene


def _make_frames_case(tmp_path: Path, squash: bool = False) -> MvoikCase:
    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True)
    frames: list[Path] = []
    for idx in range(4):
        arr = np.zeros((60, 80, 4), dtype=np.uint8)
        if squash:
            # Strongly compress height across frames to simulate elastic squashing.
            half_h = max(4, 18 - idx * 4)
            arr[45 - half_h : 45, 25 - idx * 2 : 55 + idx * 2, :] = [50, 200, 90, 255]
        else:
            shift = idx * 3
            arr[25:45, 22 + shift : 42 + shift, :] = [40, 180, 70, 255]
        p = in_dir / f"{idx:06d}.png"
        Image.fromarray(arr, mode="RGBA").save(p)
        frames.append(p)

    return MvoikCase(
        case_id="0014_press_ball_1" if squash else "0004_brick_fall_1",
        regime=PhysicalRegime.DEFORMABLE if squash else PhysicalRegime.RIGID,
        input_dir=in_dir,
        eval_gt_dir=tmp_path / "eval_gt",
        input_frames=frames,
        eval_views={},
        num_frames=4,
        width=80,
        height=60,
    )


def test_interval_validator_normalizes_inverted_bounds() -> None:
    iv = InteractionInterval(
        start_frame=10,
        end_frame=3,
        subject_id="a",
        object_id="b",
        action="hits",
    )
    assert iv.start_frame == 3
    assert iv.end_frame == 10


def test_visual_motion_trigger_fires_even_when_vlm_omits_physics(tmp_path: Path) -> None:
    case = _make_frames_case(tmp_path, squash=True)
    ikg = InteractionGraph(
        case_id=case.case_id,
        summary="VLM mistakenly skips physics escalation.",
        dominant_regime=PhysicalRegime.RIGID,
        objects=[ObjectEntity(object_id="obj", label="obj")],
        intervals=[
            InteractionInterval(
                start_frame=0,
                end_frame=3,
                subject_id="obj",
                object_id="obj",
                action="idle",
                contact_expected=False,
                requires_physics_escalation=False,
                regime=PhysicalRegime.DEFORMABLE,
            )
        ],
    )
    windows = detect_escalation_windows(case, ikg, frame_indices=[0, 1, 2, 3])
    assert len(windows) == 1
    assert windows[0].source == "visual_motion"


def test_layer2_repairs_overlapping_and_ground_penetrating_scene() -> None:
    overlapping_scene = CoarseSceneSpec(
        case_id="overlap_test",
        backend=AssetBackend.SIM_PROXY,
        regime=PhysicalRegime.OCCLUDED_KINEMATIC,
        num_frames=4,
        cameras=build_canonical_cameras(),
        objects=[
            SceneObjectSpec(
                object_id="cup_a",
                label="cup a",
                primitive="box",
                size_m=(0.08, 0.08, 0.08),
                keyframes=[KeyframePose(frame_idx=i, position=(0.0, 0.0, 0.02)) for i in range(4)],
            ),
            SceneObjectSpec(
                object_id="cup_b",
                label="cup b",
                primitive="box",
                size_m=(0.08, 0.08, 0.08),
                keyframes=[KeyframePose(frame_idx=i, position=(0.01, 0.0, 0.02)) for i in range(4)],
            ),
        ],
    )
    before = evaluate_mujoco_scene(build_mjcf_xml(overlapping_scene, include_floor=True))
    assert before.max_penetration_m > 0.05

    from real2sim_layered.schemas import EscalationWindow

    windows = [EscalationWindow(start_frame=0, end_frame=3, source="both", reason="contact")]
    repaired, after, iters = run_layer2_escalation(
        overlapping_scene, windows=windows, frame_indices=[0, 1, 2, 3], max_iters=3
    )
    assert iters >= 1
    assert after.max_penetration_m < 0.02
    assert after.counterfactual_survival is True
    assert len(repaired.objects) == 2


def test_layer3_triggers_deformable_fallback_on_squashing_clip(tmp_path: Path) -> None:
    case = _make_frames_case(tmp_path, squash=True)
    ikg = InteractionGraph(
        case_id=case.case_id,
        summary="Elastic ball pressed flat.",
        dominant_regime=PhysicalRegime.DEFORMABLE,
        objects=[ObjectEntity(object_id="ball", label="ball", primitive_proxy="sphere")],
        intervals=[
            InteractionInterval(
                start_frame=0,
                end_frame=3,
                subject_id="ball",
                object_id="ball",
                action="compresses",
                regime=PhysicalRegime.DEFORMABLE,
            )
        ],
    )
    l1_scene = CoarseSceneSpec(
        case_id=case.case_id,
        backend=AssetBackend.SIM_PROXY,
        regime=PhysicalRegime.DEFORMABLE,
        num_frames=4,
        cameras=build_canonical_cameras(),
        objects=[
            SceneObjectSpec(
                object_id="ball",
                label="ball",
                primitive="sphere",
                size_m=(0.07, 0.07, 0.07),
                keyframes=[KeyframePose(frame_idx=i, position=(0.0, 0.0, 0.08)) for i in range(4)],
            )
        ],
    )
    l2_phys = evaluate_mujoco_scene(
        build_mjcf_xml(l1_scene, include_floor=True), num_steps=80, scene=l1_scene
    )
    l3_scene, report = detect_and_apply_fallback(
        case=case,
        ikg=ikg,
        l1_scene=l1_scene,
        l2_scene=l1_scene,
        l2_phys=l2_phys,
        frame_indices=[0, 1, 2, 3],
    )
    assert report.triggered is True
    assert report.fallback_mode == "fallback_deformable"
    assert report.non_rigid_shape_cv > 0.1
    assert l3_scene.objects[0].keyframes[-1].scale != (1.0, 1.0, 1.0)
