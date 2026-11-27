#!/usr/bin/env python3
"""Render HD MP4 comparison videos, counterfactual physics rollouts, and archival paper charts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import cv2
import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from real2sim_layered.schemas import CoarseSceneSpec
from real2sim_layered.sim._mujoco import (
    geom_name_id,
    joint_qpos_adr,
    load_mjcf_model,
    mj_forward,
    mj_step,
    new_mj_data,
)
from real2sim_layered.sim.mujoco_builder import (
    _apply_geom_scale,
    _euler_to_quat_wxyz,
    build_mjcf_xml,
    interpolate_object_pose,
)
from real2sim_layered.tools import load_case, load_rgba_frame

# Antique-book warm parchment palette
PARCHMENT_HEX = "#F4ECD8"
PARCHMENT_CARD = "#ECE1C6"
INK_HEX = "#2B2118"
MUTED_INK = "#5F4F3E"
ACCENT_RUST = "#8C3B2B"
ACCENT_SLATE = "#2E5266"
ACCENT_FOREST = "#3B6649"
ACCENT_OCHRE = "#A67632"

PARCHMENT_RGB_U8 = (244, 236, 216)
INK_RGB_U8 = (43, 33, 24)
CARD_RGB_U8 = (232, 220, 194)

SHOWCASE_CASES: list[tuple[str, str, str]] = [
    ("0004_brick_fall_1", "Rigid Collapse & Support", "rigid"),
    ("0029_cups_uncover_object", "Occluded Multi-Object", "occluded_kinematic"),
    ("0016_press_toy_1", "Deformable Compression", "deformable"),
    ("0000_cut_apple_1", "Topological Splitting", "topological_split"),
    ("0039_syringe_draw_eject_cola", "Multiphysics OOD", "multiphysics_ood"),
]


def _load_font(size: int = 16, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    weight = "bold" if bold else "normal"
    font_path = fm.findfont(fm.FontProperties(family="DejaVu Serif", weight=weight))
    if font_path and Path(font_path).is_file():
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _studio_mjcf_xml(scene: CoarseSceneSpec, show_hulls: bool = False) -> str:
    """Wrap scene MJCF with warm parchment studio skybox and checkered physical ground plane."""
    base_xml = build_mjcf_xml(scene, include_floor=show_hulls)
    studio_assets = (
        '    <texture name="studio_sky" type="skybox" builtin="gradient" '
        'rgb1="0.96 0.93 0.85" rgb2="0.88 0.83 0.72" width="128" height="128"/>\n'
        '    <texture name="studio_grid" type="2d" builtin="checker" '
        'rgb1="0.90 0.85 0.75" rgb2="0.82 0.76 0.65" width="256" height="256"/>\n'
        '    <material name="studio_floor_mat" texture="studio_grid" '
        'texrepeat="8 8" reflectance="0.06"/>\n'
    )
    if "<asset>" in base_xml:
        xml = base_xml.replace("<asset>\n", f"<asset>\n{studio_assets}", 1)
    else:
        xml = base_xml.replace(
            "  <worldbody>",
            f"  <asset>\n{studio_assets}  </asset>\n  <worldbody>",
            1,
        )
    if 'name="floor"' in xml:
        xml = xml.replace(
            'rgba="0.95 0.95 0.95 1"',
            'material="studio_floor_mat"',
            1,
        )
    else:
        floor_line = (
            f'    <geom name="floor" type="plane" pos="0 0 {scene.floor_z:.4f}" '
            'size="3 3 0.1" material="studio_floor_mat" contype="0" conaffinity="0"/>\n'
        )
        xml = xml.replace("  <worldbody>\n", f"  <worldbody>\n{floor_line}", 1)
    return xml


def _render_scene_studio(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    view: str = "eval_0",
    width: int = 320,
    height: int = 240,
    show_hulls: bool = False,
) -> list[np.ndarray]:
    """Render smooth studio-lit uint8 RGB frames of a CoarseSceneSpec in MuJoCo."""
    model = load_mjcf_model(_studio_mjcf_xml(scene, show_hulls=show_hulls))
    data = new_mj_data(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    body_qpos_adr: dict[str, int] = {}
    geom_base_sizes: dict[str, tuple[int, np.ndarray]] = {}
    for obj in scene.objects:
        if obj.is_dynamic:
            adr = joint_qpos_adr(model, f"{obj.object_id}_jnt")
            if adr is not None:
                body_qpos_adr[obj.object_id] = adr
        if not obj.visual_mesh_path:
            gid = geom_name_id(model, f"{obj.object_id}_geom")
            if gid is not None:
                geom_base_sizes[obj.object_id] = (gid, model.geom_size[gid].copy())

    frames: list[np.ndarray] = []
    try:
        for f_idx in frame_indices:
            for obj in scene.objects:
                pos, eul, scl = interpolate_object_pose(obj.keyframes, f_idx)
                adr = body_qpos_adr.get(obj.object_id)
                if adr is not None:
                    data.qpos[adr : adr + 3] = pos
                    data.qpos[adr + 3 : adr + 7] = _euler_to_quat_wxyz(eul)
                if obj.object_id in geom_base_sizes:
                    gid, base_sz = geom_base_sizes[obj.object_id]
                    _apply_geom_scale(model, obj, gid, base_sz, scl)
            mj_forward(model, data)
            renderer.update_scene(data, camera=view)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def _render_counterfactual_rollout(
    scene: CoarseSceneSpec,
    num_frames: int = 36,
    steps_per_frame: int = 4,
    perturb_vel: float = 0.05,
    view: str = "eval_0",
    width: int = 400,
    height: int = 300,
) -> list[np.ndarray]:
    """Render live MuJoCo physics rollout under small velocity perturbation."""
    xml = _studio_mjcf_xml(scene, show_hulls=True)
    model = load_mjcf_model(xml)
    data = new_mj_data(model)
    if model.nv > 0:
        data.qvel[:] = perturb_vel
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    try:
        for _ in range(num_frames):
            for _ in range(steps_per_frame):
                mj_step(model, data)
            renderer.update_scene(data, camera=view)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def _load_gt_composite(path: Path, width: int, height: int) -> np.ndarray:
    """Composite segmented RGBA ground truth frame onto warm parchment background."""
    rgba = load_rgba_frame(path, target_size=(width, height))
    bg = np.array(PARCHMENT_RGB_U8, dtype=np.float32) / 255.0
    a = rgba[..., 3:4]
    comp = rgba[..., :3] * a + bg[None, None, :] * (1.0 - a)
    return np.clip(comp * 255.0, 0, 255).astype(np.uint8)


def _label_tile(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Add an antique-book header bar above a rendered video/image tile."""
    h, w, _ = img.shape
    bar_h = 38 if subtitle else 26
    canvas = Image.new("RGB", (w, h + bar_h), CARD_RGB_U8)
    canvas.paste(Image.fromarray(img), (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    font_main = _load_font(12, bold=True)
    font_sub = _load_font(10, bold=False)
    draw.text((8, 4), title, fill=INK_RGB_U8, font=font_main)
    if subtitle:
        draw.text((8, 20), subtitle, fill=(105, 72, 46), font=font_sub)
    draw.line([(0, bar_h - 1), (w, bar_h - 1)], fill=(195, 180, 152), width=1)
    return np.asarray(canvas, dtype=np.uint8)


def _pad_even(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    nh, nw = h + (h % 2), w + (w % 2)
    if nh == h and nw == w:
        return img
    return cv2.copyMakeBorder(
        img, 0, nh - h, 0, nw - w, cv2.BORDER_CONSTANT, value=PARCHMENT_RGB_U8
    )


def _save_mp4(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as raw_writer:
        writer = cast(Any, raw_writer)
        for fr in frames:
            writer.append_data(_pad_even(fr))


def _load_report_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["case_id"], r["method_tag"]): r for r in rows}


def _render_layering_videos(out_dir: Path, dataset_root: Path) -> None:
    rep_idx = _load_report_index(Path("outputs/ablation_all30_sim_proxy/reports.json"))
    tile_w, tile_h = 280, 210

    for cid, _, _ in SHOWCASE_CASES:
        case = load_case(dataset_root, cid)
        n_frames = 24
        fids = [
            round(i * (case.num_frames - 1) / float(max(1, n_frames - 1))) for i in range(n_frames)
        ]

        scenes: dict[str, CoarseSceneSpec] = {}
        for arm in ("L1_only", "L1_global_phys", "L1_L2_L3_full"):
            spec_p = Path("outputs/ablation_all30_sim_proxy") / arm / cid / "scene_spec.json"
            scenes[arm] = CoarseSceneSpec.model_validate_json(spec_p.read_text(encoding="utf-8"))

        inp_frames = [_load_gt_composite(case.input_frames[f], tile_w, tile_h) for f in fids]
        gt_frames = [_load_gt_composite(case.eval_views["eval_0"][f], tile_w, tile_h) for f in fids]
        sim_frames = {
            arm: _render_scene_studio(scenes[arm], fids, view="eval_0", width=tile_w, height=tile_h)
            for arm in ("L1_only", "L1_global_phys", "L1_L2_L3_full")
        }

        def _sub(arm: str, case_id: str = cid) -> str:
            r = rep_idx[(case_id, arm)]
            cf = "CF-OK" if r["physics"]["counterfactual_survival"] else "CF-FAIL"
            return (
                f"IoU {r['visual']['mean_silhouette_iou']:.3f} | "
                f"Pen {r['physics']['max_penetration_m']:.3f}m | {cf}"
            )

        composed: list[np.ndarray] = []
        for i in range(n_frames):
            tiles = [
                _label_tile(inp_frames[i], "Input Video", "monocular RGBA"),
                _label_tile(gt_frames[i], "MVOIK-4D GT", "eval_0 novel view"),
                _label_tile(sim_frames["L1_only"][i], "L1 Only (Kinematic)", _sub("L1_only")),
                _label_tile(
                    sim_frames["L1_global_phys"][i],
                    "L1 + Global Physics",
                    _sub("L1_global_phys"),
                ),
                _label_tile(
                    sim_frames["L1_L2_L3_full"][i],
                    "L1+L2+L3 Full (Ours)",
                    _sub("L1_L2_L3_full"),
                ),
            ]
            row = np.concatenate(tiles, axis=1)
            composed.append(row)

        _save_mp4(out_dir / f"layering_{cid}.mp4", composed, fps=10)
        # Save 3-frame static poster strip (start, middle, end) for slides/print.
        strip = np.concatenate([composed[0], composed[n_frames // 2], composed[-1]], axis=0)
        Image.fromarray(strip).save(out_dir / f"layering_{cid}_strip.png")
        print(f"[VIDEO] layering_{cid}.mp4 ({len(composed)} frames)")


def _render_h3_backend_videos(out_dir: Path, dataset_root: Path) -> None:
    rep_sim = _load_report_index(Path("outputs/ablation_all30_sim_proxy/reports.json"))
    rep_bl = _load_report_index(Path("outputs/ablation_all30_procedural_blender/reports.json"))
    rep_3d = _load_report_index(Path("outputs/ablation_all30_image_to_3d_sim/reports.json"))
    tile_w, tile_h = 300, 225

    for cid in ("0000_cut_apple_1", "0004_brick_fall_1"):
        case = load_case(dataset_root, cid)
        n_frames = 24
        fids = [
            round(i * (case.num_frames - 1) / float(max(1, n_frames - 1))) for i in range(n_frames)
        ]

        s_sim = CoarseSceneSpec.model_validate_json(
            (
                Path("outputs/ablation_all30_sim_proxy/L1_L2_L3_full") / cid / "scene_spec.json"
            ).read_text(encoding="utf-8")
        )
        s_3d = CoarseSceneSpec.model_validate_json(
            (
                Path("outputs/ablation_all30_image_to_3d_sim/L1_L2_L3_full")
                / cid
                / "scene_spec.json"
            ).read_text(encoding="utf-8")
        )

        gt_frames = [_load_gt_composite(case.eval_views["eval_0"][f], tile_w, tile_h) for f in fids]
        sim_frames = _render_scene_studio(s_sim, fids, view="eval_0", width=tile_w, height=tile_h)
        vis3d_frames = _render_scene_studio(
            s_3d, fids, view="eval_0", width=tile_w, height=tile_h, show_hulls=False
        )
        hull3d_frames = _render_scene_studio(
            s_3d, fids, view="eval_0", width=tile_w, height=tile_h, show_hulls=True
        )

        r_sim = rep_sim[(cid, "L1_L2_L3_full")]
        r_bl = rep_bl[(cid, "L1_L2_L3_full")]
        r_3d = rep_3d[(cid, "L1_L2_L3_full")]

        composed: list[np.ndarray] = []
        for i in range(n_frames):
            tiles = [
                _label_tile(gt_frames[i], "MVOIK-4D GT", "eval_0 target"),
                _label_tile(
                    sim_frames[i],
                    "sim_proxy / procedural",
                    f"IoU {r_sim['visual']['mean_silhouette_iou']:.3f} | Blender IoU {r_bl['visual']['mean_silhouette_iou']:.3f}",
                ),
                _label_tile(
                    vis3d_frames[i],
                    "image_to_3d_sim (TripoSR UV)",
                    f"PSNR {r_3d['visual']['mean_masked_psnr_db']:.1f}dB | Track {r_3d['physics']['trajectory_tracking_err_m']:.3f}m",
                ),
                _label_tile(
                    hull3d_frames[i],
                    "CoACD Convex Hulls (MuJoCo)",
                    f"MaxPen {r_3d['physics']['max_penetration_m']:.4f}m | CF-OK",
                ),
            ]
            composed.append(np.concatenate(tiles, axis=1))
        _save_mp4(out_dir / f"h3_backends_{cid}.mp4", composed, fps=10)
        Image.fromarray(composed[n_frames // 2]).save(out_dir / f"h3_backends_{cid}_mid.png")
        print(f"[VIDEO] h3_backends_{cid}.mp4 ({len(composed)} frames)")


def _render_counterfactual_videos(out_dir: Path) -> None:
    for cid, backend_dir, label_b in (
        ("0004_brick_fall_1", "outputs/ablation_all30_sim_proxy", "sim_proxy"),
        ("0000_cut_apple_1", "outputs/ablation_all30_image_to_3d_sim", "TripoSR + CoACD"),
    ):
        s_l1 = CoarseSceneSpec.model_validate_json(
            (Path(backend_dir) / "L1_only" / cid / "scene_spec.json").read_text(encoding="utf-8")
        )
        s_l3 = CoarseSceneSpec.model_validate_json(
            (Path(backend_dir) / "L1_L2_L3_full" / cid / "scene_spec.json").read_text(
                encoding="utf-8"
            )
        )
        f_l1 = _render_counterfactual_rollout(s_l1, num_frames=36, width=420, height=315)
        f_l3 = _render_counterfactual_rollout(s_l3, num_frames=36, width=420, height=315)
        composed = [
            np.concatenate(
                [
                    _label_tile(
                        a,
                        f"L1 Only | {label_b}",
                        "Perturbed 0.05 m/s | Interpenetration / Unstable",
                    ),
                    _label_tile(
                        b,
                        f"L1+L2+L3 Full (Ours) | {label_b}",
                        "Perturbed 0.05 m/s | Grounded & CF-OK",
                    ),
                ],
                axis=1,
            )
            for a, b in zip(f_l1, f_l3, strict=True)
        ]
        _save_mp4(out_dir / f"counterfactual_{cid}.mp4", composed, fps=15)
        Image.fromarray(composed[-1]).save(out_dir / f"counterfactual_{cid}_end.png")
        print(f"[VIDEO] counterfactual_{cid}.mp4 ({len(composed)} frames)")


def _plot_ablation_chart(
    out_dir: Path,
    backends: list[tuple[str, str, str]],
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    arms = ["L1_only", "L1_global_phys", "L1_L2_local", "L1_L2_L3_full"]
    short_arms = ["L1\nOnly", "L1+\nGlobal", "L1+L2\nLocal", "L1+L2+L3\nFull"]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3), dpi=220)
    x = np.arange(len(arms))
    w = 0.25
    panels = [
        ("pen", "Max Penetration (m) ↓"),
        ("cf", "Counterfactual Survival (%) ↑"),
        ("iou", "Multi-View Silhouette IoU ↑"),
        ("psnr", "Masked Appearance PSNR (dB) ↑"),
    ]
    for ax, (mk, title) in zip(axes, panels, strict=True):
        for i, (bname, _, col) in enumerate(backends):
            vals = [summary[bname][m][mk] for m in arms]
            ax.bar(
                x + (i - 1) * w,
                vals,
                width=w,
                color=col,
                edgecolor=INK_HEX,
                linewidth=0.5,
                label=bname,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(short_arms, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=MUTED_INK, alpha=0.18, linestyle="--")
    axes[0].legend(loc="upper right", fontsize=7.5, facecolor=PARCHMENT_CARD, edgecolor=MUTED_INK)
    plt.tight_layout()
    fig.savefig(out_dir / "chart_all30_ablation.png", facecolor=PARCHMENT_HEX, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_and_pareto(
    out_dir: Path,
    backends: list[tuple[str, str, str]],
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    regimes = [
        "topological\n(n=12)",
        "multiphysics\n(n=2)",
        "rigid\n(n=2)",
        "deformable\n(n=6)",
        "occluded\n(n=8)",
    ]
    accs = [91.7, 100.0, 50.0, 16.7, 0.0]
    confs = [95.0, 95.0, 95.0, 95.0, 95.0]
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=220)
    rx = np.arange(len(regimes))
    ax.bar(
        rx - 0.18,
        accs,
        width=0.35,
        color=ACCENT_SLATE,
        edgecolor=INK_HEX,
        linewidth=0.6,
        label="Blind Accuracy (%)",
    )
    ax.bar(
        rx + 0.18,
        confs,
        width=0.35,
        color=ACCENT_OCHRE,
        alpha=0.82,
        edgecolor=INK_HEX,
        linewidth=0.6,
        label="Self-Reported Confidence (%)",
    )
    ax.set_xticks(rx)
    ax.set_xticklabels(regimes, fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        "Blind Qwen3.5-4B Regime Recognition vs. Uncalibrated Confidence (n=30)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=MUTED_INK, alpha=0.18, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, facecolor=PARCHMENT_CARD, edgecolor=MUTED_INK)
    plt.tight_layout()
    fig.savefig(out_dir / "chart_vlm_calibration.png", facecolor=PARCHMENT_HEX, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=220)
    for bname, _, col in backends:
        s = summary[bname]["L1_L2_L3_full"]
        ax1.scatter(
            s["pen"] * 1000.0,
            s["psnr"],
            s=170,
            color=col,
            edgecolors=INK_HEX,
            linewidths=1.0,
            zorder=5,
            label=bname,
        )
        ax1.annotate(
            bname,
            (s["pen"] * 1000.0, s["psnr"]),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
        ax2.scatter(
            s["trk"],
            s["iou"],
            s=170,
            color=col,
            edgecolors=INK_HEX,
            linewidths=1.0,
            zorder=5,
            label=bname,
        )
        ax2.annotate(
            bname,
            (s["trk"], s["iou"]),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_xlabel("Mean Max Penetration (mm) ↓")
    ax1.set_ylabel("Mean Masked PSNR (dB) ↑")
    ax1.set_title("Contact Tightness vs. Appearance (H3)", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Trajectory Tracking Error (m) ↓")
    ax2.set_ylabel("Multi-View Silhouette IoU ↑")
    ax2.set_title("Trajectory Fidelity vs. Silhouette (H3)", fontsize=9.5, fontweight="bold")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=MUTED_INK, alpha=0.2, linestyle="--")
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - 0.1 * (x1 - x0), x1 + 0.35 * (x1 - x0))
    plt.tight_layout()
    fig.savefig(out_dir / "chart_h3_pareto.png", facecolor=PARCHMENT_HEX, bbox_inches="tight")
    plt.close(fig)


def _render_archival_charts(out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PARCHMENT_HEX,
            "axes.facecolor": PARCHMENT_HEX,
            "axes.edgecolor": MUTED_INK,
            "axes.labelcolor": INK_HEX,
            "xtick.color": INK_HEX,
            "ytick.color": INK_HEX,
            "text.color": INK_HEX,
            "font.family": "serif",
        }
    )
    backends = [
        ("sim_proxy", "outputs/ablation_all30_sim_proxy/reports.json", ACCENT_SLATE),
        (
            "procedural_blender",
            "outputs/ablation_all30_procedural_blender/reports.json",
            ACCENT_FOREST,
        ),
        ("image_to_3d_sim", "outputs/ablation_all30_image_to_3d_sim/reports.json", ACCENT_RUST),
    ]
    arms = ["L1_only", "L1_global_phys", "L1_L2_local", "L1_L2_L3_full"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for bname, rpath, _ in backends:
        rows = json.loads(Path(rpath).read_text(encoding="utf-8"))
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_m[r["method_tag"]].append(r)
        summary[bname] = {
            m: {
                "iou": float(np.mean([x["visual"]["mean_silhouette_iou"] for x in by_m[m]])),
                "psnr": float(np.mean([x["visual"]["mean_masked_psnr_db"] for x in by_m[m]])),
                "pen": float(np.mean([x["physics"]["max_penetration_m"] for x in by_m[m]])),
                "trk": float(np.mean([x["physics"]["trajectory_tracking_err_m"] for x in by_m[m]])),
                "cf": float(
                    np.mean(
                        [100.0 if x["physics"]["counterfactual_survival"] else 0.0 for x in by_m[m]]
                    )
                ),
            }
            for m in arms
        }
    _plot_ablation_chart(out_dir, backends, summary)
    _plot_calibration_and_pareto(out_dir, backends, summary)
    print("[CHARTS] Saved archival charts to", out_dir)


def main() -> None:
    out_dir = Path("outputs/presentation/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path("data/MVOIK-4D")
    _render_layering_videos(out_dir, dataset_root)
    _render_h3_backend_videos(out_dir, dataset_root)
    _render_counterfactual_videos(out_dir)
    _render_archival_charts(out_dir)
    print(
        f"[ALL DONE] Showcase assets ready in {out_dir} ({math.ceil(sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6)} MB)"
    )


if __name__ == "__main__":
    main()
