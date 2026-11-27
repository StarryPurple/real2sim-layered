"""Unit and integration tests for image_to_3d_sim CoACD decomposition and MuJoCo mesh execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from real2sim_layered.agent import build_canonical_cameras
from real2sim_layered.schemas import (
    AssetBackend,
    CoarseSceneSpec,
    KeyframePose,
    PhysicalRegime,
    SceneObjectSpec,
)
from real2sim_layered.sim import (
    build_mjcf_xml,
    evaluate_mujoco_scene,
    render_coarse_scene_mujoco,
)
from real2sim_layered.tools import decompose_and_export_mesh


def test_decompose_and_export_mesh_runs_in_mujoco(tmp_path: Path) -> None:
    """CoACD-decomposed textured mesh compiles to MJCF, renders RGBA, and simulates cleanly."""
    base = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    colors = np.zeros((len(base.vertices), 4), dtype=np.uint8)
    colors[:, 0] = 220
    colors[:, 1] = 60
    colors[:, 2] = 40
    colors[:, 3] = 255
    mesh = trimesh.Trimesh(vertices=base.vertices, faces=base.faces, vertex_colors=colors)

    asset = decompose_and_export_mesh(
        raw_mesh=mesh,
        target_size_m=(0.07, 0.07, 0.07),
        out_dir=tmp_path / "assets",
        asset_id="apple_mesh",
        align_triposr_frame=True,
    )
    assert asset.visual_mesh_path.is_file()
    assert asset.texture_path.is_file()
    assert len(asset.collision_mesh_paths) >= 1
    assert all(p.is_file() for p in asset.collision_mesh_paths)
    loaded_vis = trimesh.load(asset.visual_mesh_path, process=False)
    assert isinstance(loaded_vis, trimesh.Trimesh)
    assert float(loaded_vis.volume) > 0.0

    scene = CoarseSceneSpec(
        case_id="mesh_smoke",
        backend=AssetBackend.IMAGE_TO_3D_SIM,
        regime=PhysicalRegime.RIGID,
        num_frames=3,
        floor_z=0.0,
        cameras=build_canonical_cameras(),
        objects=[
            SceneObjectSpec(
                object_id="apple",
                label="red apple",
                primitive="sphere",
                size_m=asset.half_extents_m,
                color_rgba=asset.mean_rgba,
                visual_mesh_path=str(asset.visual_mesh_path),
                texture_path=str(asset.texture_path),
                collision_mesh_paths=[str(p) for p in asset.collision_mesh_paths],
                keyframes=[
                    KeyframePose(frame_idx=i, position=(0.0, 0.0, asset.half_extents_m[2] + 0.005))
                    for i in range(3)
                ],
            )
        ],
    )

    xml_phys = build_mjcf_xml(scene, include_floor=True)
    assert "<asset>" in xml_phys
    assert "apple_col_0" in xml_phys

    phys = evaluate_mujoco_scene(xml_phys, num_steps=80, scene=scene, delivered_scene=scene)
    assert phys.stable is True
    assert phys.max_penetration_m < 0.02
    assert phys.counterfactual_survival is True

    rendered = render_coarse_scene_mujoco(
        scene, frame_indices=[0, 2], views=["eval_0"], width=80, height=60
    )
    assert len(rendered["eval_0"]) == 2
    fg_mask = rendered["eval_0"][0][..., 3] >= 0.5
    assert int(fg_mask.sum()) > 20
    # Red textured apple should have stronger R than B on foreground pixels.
    fg_rgb = rendered["eval_0"][0][fg_mask, :3]
    assert float(fg_rgb[:, 0].mean()) > float(fg_rgb[:, 2].mean()) + 0.15
