"""Unit and integration tests for Layer-1 synthesis, MuJoCo MJCF, and Blender builders."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from real2sim_layered.agent import materialize_layer1_scene, synthesize_coarse_scene
from real2sim_layered.schemas import (
    AssetBackend,
    InteractionGraph,
    InteractionInterval,
    MvoikCase,
    ObjectEntity,
    PhysicalRegime,
)
from real2sim_layered.sim import (
    BLENDER_WORKER_PATH,
    build_blender_command,
    build_mjcf_xml,
    evaluate_mujoco_scene,
    interpolate_object_pose,
)


def _make_synthetic_case(tmp_path: Path) -> tuple[MvoikCase, InteractionGraph]:
    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True)
    frames: list[Path] = []
    for idx in range(4):
        arr = np.zeros((60, 80, 4), dtype=np.uint8)
        x_shift = idx * 4
        arr[25:45, 20 + x_shift : 45 + x_shift, :] = [40, 180, 70, 255]
        p = in_dir / f"{idx:06d}.png"
        Image.fromarray(arr, mode="RGBA").save(p)
        frames.append(p)

    case = MvoikCase(
        case_id="0004_brick_fall_1",
        regime=PhysicalRegime.RIGID,
        input_dir=in_dir,
        eval_gt_dir=tmp_path / "eval_gt",
        input_frames=frames,
        eval_views={},
        num_frames=4,
        width=80,
        height=60,
    )
    ikg = InteractionGraph(
        case_id="0004_brick_fall_1",
        summary="A green block shifts and falls.",
        dominant_regime=PhysicalRegime.RIGID,
        recommended_backend=AssetBackend.SIM_PROXY,
        objects=[
            ObjectEntity(
                object_id="green_block",
                label="green block",
                primitive_proxy="box",
                is_dynamic=True,
                material_hint="rigid_solid",
            )
        ],
        intervals=[
            InteractionInterval(
                start_frame=0,
                end_frame=3,
                subject_id="green_block",
                object_id="green_block",
                action="falls",
                regime=PhysicalRegime.RIGID,
            )
        ],
        confidence=0.9,
    )
    return case, ikg


def test_layer1_sim_proxy_compiles_renders_and_verifies(tmp_path: Path) -> None:
    case, ikg = _make_synthetic_case(tmp_path)
    scene = synthesize_coarse_scene(case, ikg, backend=AssetBackend.SIM_PROXY, max_frames=4)
    assert len(scene.objects) == 1
    assert len(scene.objects[0].keyframes) == 4
    # Median foreground color should preserve the green channel dominance.
    r, g, b, _ = scene.objects[0].color_rgba
    assert g > r
    assert g > b

    pos_mid, _, scl_mid = interpolate_object_pose(scene.objects[0].keyframes, 1)
    assert len(pos_mid) == 3
    assert scl_mid == (1.0, 1.0, 1.0)

    xml = build_mjcf_xml(scene, include_floor=True)
    phys = evaluate_mujoco_scene(xml, num_steps=60)
    assert phys.stable is True
    assert phys.object_permanence_ok is True

    artifacts = materialize_layer1_scene(
        scene=scene,
        frame_indices=[0, 3],
        views=["eval_0", "eval_1"],
        out_dir=tmp_path / "out_sim",
        width=80,
        height=60,
    )
    assert artifacts.mjcf_path.is_file()
    assert artifacts.spec_path.is_file()
    assert set(artifacts.rendered_by_view.keys()) == {"eval_0", "eval_1"}
    assert len(artifacts.rendered_by_view["eval_0"]) == 2
    assert artifacts.rendered_by_view["eval_0"][0].shape == (60, 80, 4)


def test_layer1_procedural_blender_builds_blend_and_renders(tmp_path: Path) -> None:
    case, ikg = _make_synthetic_case(tmp_path)
    scene = synthesize_coarse_scene(
        case, ikg, backend=AssetBackend.PROCEDURAL_BLENDER, max_frames=3
    )
    assert BLENDER_WORKER_PATH.is_file()
    cmd = build_blender_command(
        spec_path=tmp_path / "scene_spec.json",
        frame_indices=[0],
        views=["eval_0"],
        out_dir=tmp_path / "out_blender",
        blender_bin="/usr/bin/blender",
    )
    assert str(BLENDER_WORKER_PATH) in cmd
    assert "--spec" in cmd

    blender = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender:
        pytest.skip("Blender binary not configured in environment")

    artifacts = materialize_layer1_scene(
        scene=scene,
        frame_indices=[0],
        views=["eval_0"],
        out_dir=tmp_path / "out_blender",
        width=80,
        height=60,
    )
    assert artifacts.blend_path is not None
    assert artifacts.blend_path.is_file()
    assert "eval_0" in artifacts.rendered_by_view
    assert artifacts.rendered_by_view["eval_0"][0].shape == (60, 80, 4)
    assert float(artifacts.rendered_by_view["eval_0"][0][..., 3].mean()) > 0.0
