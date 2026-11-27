"""Headless Blender 4.2 scene materializer and multi-view RGBA renderer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_layered.schemas import CoarseSceneSpec, resolve_blender_bin

BLENDER_WORKER_PATH = Path(__file__).with_name("blender_worker.py")


def build_blender_command(
    spec_path: Path,
    frame_indices: list[int],
    views: list[str],
    out_dir: Path,
    render_size: tuple[int, int] = (160, 120),
    samples: int = 6,
    blender_bin: str | None = None,
) -> list[str]:
    """Build the headless Blender 4.2 command line invoking the static worker script."""
    width, height = render_size
    resolved_out = out_dir.resolve()
    return [
        resolve_blender_bin(blender_bin),
        "-b",
        "--factory-startup",
        "--python",
        str(BLENDER_WORKER_PATH),
        "--",
        "--spec",
        str(spec_path.resolve()),
        "--frames",
        json.dumps(frame_indices),
        "--views",
        json.dumps(views),
        "--render-dir",
        str(resolved_out / "renders"),
        "--blend-out",
        str(resolved_out / "scene.blend"),
        "--width",
        str(width),
        "--height",
        str(height),
        "--samples",
        str(samples),
    ]


def run_blender_scene(
    scene: CoarseSceneSpec,
    frame_indices: list[int],
    views: list[str],
    out_dir: Path,
    width: int = 160,
    height: int = 120,
    samples: int = 6,
) -> tuple[Path, dict[str, list[np.ndarray]]]:
    """Execute headless Blender 4.2 to save `scene.blend` and load rendered RGBA views."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "scene_spec.json"
    if not spec_path.is_file():
        spec_path.write_text(scene.model_dump_json(indent=2) + "\n", encoding="utf-8")

    render_dir = out_dir / "renders"
    blend_out = out_dir / "scene.blend"
    cmd = build_blender_command(
        spec_path=spec_path,
        frame_indices=frame_indices,
        views=views,
        out_dir=out_dir,
        render_size=(width, height),
        samples=samples,
    )
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not blend_out.is_file():
        raise RuntimeError(f"Blender finished without creating {blend_out}")

    rendered: dict[str, list[np.ndarray]] = {}
    for view_name in views:
        frames: list[np.ndarray] = []
        for frame_idx in frame_indices:
            img_path = render_dir / view_name / f"{frame_idx:06d}.png"
            if not img_path.is_file():
                continue
            with Image.open(img_path) as img:
                frames.append(np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0)
        if frames:
            rendered[view_name] = frames

    return blend_out, rendered
