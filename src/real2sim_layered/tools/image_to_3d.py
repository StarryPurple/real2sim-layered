"""Image-to-3D mesh lifting (TripoSR) + CoACD convex collision decomposition for MuJoCo."""

from __future__ import annotations

import importlib
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure

from real2sim_layered.schemas import (
    RgbaColor,
    Vec3,
    resolve_geom3d_device,
    resolve_triposr_dir,
)

_VIT_V5_KEY_RULES: tuple[tuple[str, str], ...] = (
    (".encoder.layer.", ".layers."),
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
)


@dataclass(frozen=True)
class ImageTo3DAsset:
    """Generated visual OBJ + baked UV texture + CoACD convex collision hulls for one object."""

    visual_mesh_path: Path
    texture_path: Path
    collision_mesh_paths: list[Path]
    mean_rgba: RgbaColor
    half_extents_m: Vec3


def _install_triposr_shims(triposr_dir: Path) -> None:
    """Provide pure-Python shims for torchmcubes and rembg so upstream TripoSR runs unmodified."""
    if "torchmcubes" not in sys.modules:
        shim_mc = types.ModuleType("torchmcubes")

        def _skimage_marching_cubes(
            volume: torch.Tensor, level: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            verts, faces, _, _ = measure.marching_cubes(
                volume.detach().cpu().numpy(), level=float(level)
            )
            # TripoSR's MarchingCubeHelper flips [2, 1, 0]; pre-flip so final order stays (x, y, z).
            verts_zyx = np.ascontiguousarray(verts[:, [2, 1, 0]], dtype=np.float32)
            faces_i64 = np.ascontiguousarray(faces, dtype=np.int64)
            return torch.from_numpy(verts_zyx), torch.from_numpy(faces_i64)

        cast(Any, shim_mc).marching_cubes = _skimage_marching_cubes
        sys.modules["torchmcubes"] = shim_mc

    sys.modules.setdefault("rembg", types.ModuleType("rembg"))
    triposr_str = str(triposr_dir)
    if triposr_str not in sys.path:
        sys.path.insert(0, triposr_str)


def _remap_vit_v5_state_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Map transformers v4 ViT keys in TripoSR checkpoints to transformers v5 module names."""
    remapped: dict[str, Any] = {}
    for key, val in ckpt.items():
        new_key = key
        if new_key.startswith("image_tokenizer.model."):
            for old_part, new_part in _VIT_V5_KEY_RULES:
                new_key = new_key.replace(old_part, new_part)
        remapped[new_key] = val
    return remapped


def _prepare_triposr_pil(rgba_crop: np.ndarray, foreground_ratio: float = 0.85) -> Image.Image:
    """Pad an RGBA foreground crop onto a square 50%-gray canvas matching TripoSR's prior."""
    h, w, _ = rgba_crop.shape
    side = max(32, round(max(h, w) / max(0.2, foreground_ratio)))
    canvas = np.full((side, side, 3), 0.5, dtype=np.float32)
    oy = (side - h) // 2
    ox = (side - w) // 2
    alpha = np.clip(rgba_crop[..., 3:4], 0.0, 1.0)
    rgb = np.clip(rgba_crop[..., :3], 0.0, 1.0)
    canvas[oy : oy + h, ox : ox + w] = rgb * alpha + 0.5 * (1.0 - alpha)
    return Image.fromarray(np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _compute_spherical_uv(verts: np.ndarray) -> np.ndarray:
    """Compute equirectangular spherical UV coordinates in [0, 1]^2 for centered mesh vertices."""
    radii = np.linalg.norm(verts, axis=1).clip(min=1e-6)
    u = (np.arctan2(verts[:, 1], verts[:, 0]) / (2.0 * np.pi) + 0.5) % 1.0
    v = 1.0 - np.arccos(np.clip(verts[:, 2] / radii, -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=1).astype(np.float32)


def _bake_spherical_texture(
    uvs: np.ndarray,
    vertex_rgb_u8: np.ndarray,
    tex_size: int = 128,
) -> Image.Image:
    """Bake per-vertex RGB colors onto a gap-free 2D PNG texture using nearest-neighbor EDT fill."""
    acc = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    cnt = np.zeros((tex_size, tex_size), dtype=np.int32)
    px = np.clip((uvs[:, 0] * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    py = np.clip(((1.0 - uvs[:, 1]) * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    np.add.at(acc, (py, px), vertex_rgb_u8.astype(np.float64))
    np.add.at(cnt, (py, px), 1)

    filled = cnt > 0
    tex = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    if np.any(filled):
        tex[filled] = np.clip(acc[filled] / cnt[filled, None], 0.0, 255.0).astype(np.uint8)
        edt_res = distance_transform_edt(~filled, return_distances=False, return_indices=True)
        indices = cast(np.ndarray, edt_res)
        tex = tex[indices[0], indices[1]]
    else:
        tex[:] = 128
    return Image.fromarray(tex, mode="RGB")


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None) -> None:
    """Write a clean Wavefront OBJ (with optional `vt` texture coordinates) for MuJoCo."""
    lines: list[str] = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    if uvs is not None:
        lines.extend(f"vt {uv[0]:.6f} {uv[1]:.6f}" for uv in uvs)
        lines.extend(
            f"f {int(f[0]) + 1}/{int(f[0]) + 1} "
            f"{int(f[1]) + 1}/{int(f[1]) + 1} "
            f"{int(f[2]) + 1}/{int(f[2]) + 1}"
            for f in faces
        )
    else:
        lines.extend(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}" for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_vertex_rgb_u8(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return `(N, 3)` uint8 vertex RGB colors, defaulting to neutral gray when absent."""
    visual = cast(Any, mesh.visual)
    vc = getattr(visual, "vertex_colors", None)
    if vc is not None:
        arr = np.asarray(vc)
        if arr.ndim == 2 and arr.shape[0] == len(mesh.vertices) and arr.shape[1] >= 3:
            return (
                np.clip(arr[:, :3] * 255.0, 0.0, 255.0).astype(np.uint8)
                if arr.dtype.kind == "f" and float(arr.max()) <= 1.0
                else np.clip(arr[:, :3], 0, 255).astype(np.uint8)
            )
    return np.full((len(mesh.vertices), 3), 140, dtype=np.uint8)


def decompose_and_export_mesh(
    raw_mesh: trimesh.Trimesh,
    target_size_m: Vec3,
    out_dir: Path,
    asset_id: str,
    align_triposr_frame: bool = False,
    coacd_threshold: float = 0.08,
    max_convex_hulls: int = 6,
) -> ImageTo3DAsset:
    """Align, scale, UV-bake, and CoACD-decompose a Trimesh into MuJoCo visual + collision OBJs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(raw_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(raw_mesh.faces, dtype=np.int64).copy()
    if align_triposr_frame:
        # TripoSR places input view along +X and width along +Y; rotate +90 deg Z to match MuJoCo -Y cam.
        verts = np.stack([verts[:, 1], -verts[:, 0], verts[:, 2]], axis=1)

    # Ensure outward-facing triangle winding (prevents MuJoCo OpenGL backface culling hollow shells).
    if float(trimesh.Trimesh(vertices=verts, faces=faces, process=False).volume) < 0.0:
        faces = np.ascontiguousarray(faces[:, [0, 2, 1]])

    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extents = np.maximum(1e-4, vmax - vmin)
    target_full = np.array([2.0 * target_size_m[0], 2.0 * target_size_m[1], 2.0 * target_size_m[2]])
    # Match observed X/Z metric extents while preserving learned 3D depth proportion along Y.
    sx = float(target_full[0] / extents[0])
    sz = float(target_full[2] / extents[2])
    sy = math.sqrt(max(1e-6, sx * sz))
    verts = (verts - center) * np.array([sx, sy, sz], dtype=np.float64)
    half_ext = 0.5 * (verts.max(axis=0) - verts.min(axis=0))
    half_ext_m: Vec3 = (
        max(0.015, float(half_ext[0])),
        max(0.015, float(half_ext[1])),
        max(0.015, float(half_ext[2])),
    )

    vertex_rgb_u8 = _extract_vertex_rgb_u8(raw_mesh)
    mean_rgb = vertex_rgb_u8.mean(axis=0) / 255.0
    mean_rgba: RgbaColor = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]), 1.0)

    uvs = _compute_spherical_uv(verts)
    tex_path = (out_dir / f"{asset_id}_texture.png").resolve()
    _bake_spherical_texture(uvs, vertex_rgb_u8).save(tex_path)

    vis_path = (out_dir / f"{asset_id}_visual.obj").resolve()
    _write_obj(vis_path, verts, faces, uvs=uvs)

    # Lazy-import coacd after torch so libgomp TLS initializes under PyTorch without SIGSEGV.
    import coacd

    coacd_api = cast(Any, coacd)
    coacd_api.set_log_level("error")
    parts = coacd_api.run_coacd(
        coacd_api.Mesh(verts, faces),
        threshold=coacd_threshold,
        max_convex_hull=max_convex_hulls,
    )
    col_paths: list[Path] = []
    for idx, (p_verts, p_faces) in enumerate(parts):
        pv = np.asarray(p_verts, dtype=np.float64)
        pf = np.asarray(p_faces, dtype=np.int64)
        if len(pv) < 4 or len(pf) < 4:
            continue
        hp = (out_dir / f"{asset_id}_hull_{idx}.obj").resolve()
        _write_obj(hp, pv, pf, uvs=None)
        col_paths.append(hp)

    if not col_paths:
        hull = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
        hp = (out_dir / f"{asset_id}_hull_0.obj").resolve()
        _write_obj(hp, np.asarray(hull.vertices), np.asarray(hull.faces), uvs=None)
        col_paths.append(hp)

    meta = {
        "visual_mesh_path": str(vis_path),
        "texture_path": str(tex_path),
        "collision_mesh_paths": [str(p) for p in col_paths],
        "mean_rgba": list(mean_rgba),
        "half_extents_m": list(half_ext_m),
    }
    (out_dir / f"{asset_id}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return ImageTo3DAsset(
        visual_mesh_path=vis_path,
        texture_path=tex_path,
        collision_mesh_paths=col_paths,
        mean_rgba=mean_rgba,
        half_extents_m=half_ext_m,
    )


def _load_cached_asset(out_dir: Path, asset_id: str) -> ImageTo3DAsset | None:
    meta_path = out_dir / f"{asset_id}_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vis_path = Path(meta["visual_mesh_path"])
    tex_path = Path(meta["texture_path"])
    col_paths = [Path(p) for p in meta["collision_mesh_paths"]]
    if not vis_path.is_file() or not tex_path.is_file() or not all(p.is_file() for p in col_paths):
        return None
    mc = meta["mean_rgba"]
    he = meta["half_extents_m"]
    return ImageTo3DAsset(
        visual_mesh_path=vis_path,
        texture_path=tex_path,
        collision_mesh_paths=col_paths,
        mean_rgba=(float(mc[0]), float(mc[1]), float(mc[2]), float(mc[3])),
        half_extents_m=(float(he[0]), float(he[1]), float(he[2])),
    )


class TripoSRMeshLifter:
    """Lazy-loaded GPU TripoSR + CoACD asset lifter with disk caching."""

    def __init__(
        self,
        repo_id: str = "stabilityai/TripoSR",
        device: str | None = None,
        mc_resolution: int = 128,
    ) -> None:
        self.repo_id = repo_id
        pref_device = device or resolve_geom3d_device()
        self.device = pref_device if torch.cuda.is_available() else "cpu"
        self.mc_resolution = mc_resolution
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        triposr_dir = resolve_triposr_dir()
        if not (triposr_dir / "tsr" / "system.py").is_file():
            raise FileNotFoundError(
                f"TripoSR sources not found at {triposr_dir}; clone VAST-AI-Research/TripoSR there."
            )
        _install_triposr_shims(triposr_dir)
        tsr_mod = importlib.import_module("tsr.system")
        cfg_path = hf_hub_download(self.repo_id, "config.yaml")
        wt_path = hf_hub_download(self.repo_id, "model.ckpt")
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)
        model = tsr_mod.TSR(cfg)
        raw_ckpt = torch.load(wt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(_remap_vit_v5_state_dict(raw_ckpt), strict=True)
        model.renderer.set_chunk_size(8192)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def lift_rgba_crop(
        self,
        rgba_crop: np.ndarray,
        target_size_m: Vec3,
        out_dir: Path,
        asset_id: str,
    ) -> ImageTo3DAsset:
        """Lift a segmented RGBA object crop into a cached MuJoCo visual + CoACD collision asset."""
        cached = _load_cached_asset(out_dir, asset_id)
        if cached is not None:
            return cached

        model = self._ensure_loaded()
        pil_img = _prepare_triposr_pil(rgba_crop)
        with torch.no_grad():
            codes = model([pil_img], device=self.device)
            meshes = model.extract_mesh(
                codes,
                has_vertex_color=True,
                resolution=self.mc_resolution,
                threshold=25.0,
            )
        return decompose_and_export_mesh(
            raw_mesh=meshes[0],
            target_size_m=target_size_m,
            out_dir=out_dir,
            asset_id=asset_id,
            align_triposr_frame=True,
        )


_DEFAULT_LIFTER: list[TripoSRMeshLifter] = []


def get_default_mesh_lifter() -> TripoSRMeshLifter:
    """Return a process-wide cached TripoSRMeshLifter so GPU weights load at most once."""
    if not _DEFAULT_LIFTER:
        _DEFAULT_LIFTER.append(TripoSRMeshLifter())
    return _DEFAULT_LIFTER[0]
