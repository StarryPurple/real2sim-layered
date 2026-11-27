"""Image-to-3D full-360 mesh lifting (Hunyuan3D-2mini-turbo) + photo projection + CoACD hulls."""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from scipy.ndimage import distance_transform_edt

from real2sim_layered.schemas import (
    RgbaColor,
    Vec3,
    resolve_geom3d_device,
    resolve_hunyuan3d_dir,
)

_DINOV2_V5_KEY_RULES: tuple[tuple[str, str], ...] = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
    (".mlp.weights_out.", ".mlp.down_proj."),
)


@dataclass(frozen=True)
class ImageTo3DAsset:
    """Generated visual OBJ + 360-projected UV texture + CoACD convex collision hulls."""

    visual_mesh_path: Path
    texture_path: Path
    collision_mesh_paths: list[Path]
    mean_rgba: RgbaColor
    half_extents_m: Vec3


def _install_hunyuan3d_shims(hunyuan_dir: Path) -> None:
    """Provide pure-Python shims for optional postprocessor deps and transformers v5 DINOv2 keys."""
    if "pymeshlab" not in sys.modules:
        shim_pm = types.ModuleType("pymeshlab")
        cast(Any, shim_pm).MeshSet = object
        sys.modules["pymeshlab"] = shim_pm
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))

    h_str = str(hunyuan_dir)
    if h_str not in sys.path:
        sys.path.insert(0, h_str)

    cond_mod = importlib.import_module("hy3dgen.shapegen.models.conditioner")
    encoder_cls = cond_mod.SingleImageEncoder
    if getattr(encoder_cls, "_real2sim_v5_patched", False):
        return

    orig_load = encoder_cls.load_state_dict

    def _load_v5_state_dict(
        self: Any,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        remapped: dict[str, Any] = {}
        for key, val in state_dict.items():
            if ".mlp.weights_in." in key:
                gate_w, up_w = val.chunk(2, dim=0)
                remapped[key.replace(".mlp.weights_in.", ".mlp.gate_proj.")] = gate_w
                remapped[key.replace(".mlp.weights_in.", ".mlp.up_proj.")] = up_w
            else:
                new_key = key
                for old_part, new_part in _DINOV2_V5_KEY_RULES:
                    new_key = new_key.replace(old_part, new_part)
                remapped[new_key] = val
        return orig_load(self, remapped, strict=strict, assign=assign)

    encoder_cls.load_state_dict = _load_v5_state_dict
    encoder_cls._real2sim_v5_patched = True


def _prepare_rgba_square_pil(rgba_crop: np.ndarray, foreground_ratio: float = 0.85) -> Image.Image:
    """Pad a segmented RGBA crop onto a transparent square canvas for Hunyuan3D-2mini."""
    h, w, _ = rgba_crop.shape
    side = max(64, round(max(h, w) / max(0.2, foreground_ratio)))
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = np.clip(rgba_crop * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGBA")


def _compute_spherical_uv(verts: np.ndarray) -> np.ndarray:
    """Compute equirectangular spherical UV coordinates in [0, 1]^2 for centered mesh vertices."""
    radii = np.linalg.norm(verts, axis=1).clip(min=1e-6)
    u = (np.arctan2(verts[:, 1], verts[:, 0]) / (2.0 * np.pi) + 0.5) % 1.0
    v = 1.0 - np.arccos(np.clip(verts[:, 2] / radii, -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=1).astype(np.float32)


def _compute_planar_uv(verts: np.ndarray) -> np.ndarray:
    """Project centered (X right, Z up) coordinates symmetrically onto [0, 1]^2 image UVs."""
    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    u = np.clip((verts[:, 0] - xmin) / max(1e-6, xmax - xmin), 0.0, 1.0)
    v = np.clip((verts[:, 2] - zmin) / max(1e-6, zmax - zmin), 0.0, 1.0)
    return np.stack([u, v], axis=1).astype(np.float32)


def _build_eroded_photo_texture(
    rgba_crop: np.ndarray, tex_size: int = 256
) -> tuple[Image.Image, RgbaColor]:
    """Create a border-artifact-free 2D texture by eroding anti-aliased silhouette rims + EDT fill."""
    fg = (rgba_crop[..., 3] >= 0.5).astype(np.uint8)
    rgb_u8 = np.clip(rgba_crop[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)

    # Erode 2-3 px off the silhouette boundary so dark anti-aliased rims never smear onto 90/270 deg sides.
    k_size = 5 if min(fg.shape) >= 32 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(fg, kernel, iterations=1)
    core = eroded > 0 if int(eroded.sum()) >= 24 else (fg > 0)

    if np.any(core):
        med = np.median(rgba_crop[core, :3], axis=0)
        mean_rgba: RgbaColor = (float(med[0]), float(med[1]), float(med[2]), 1.0)
        if not np.all(core):
            edt_res = distance_transform_edt(~core, return_distances=False, return_indices=True)
            indices = cast(np.ndarray, edt_res)
            filled = rgb_u8[indices[0], indices[1]]
        else:
            filled = rgb_u8
    else:
        mean_rgba = (0.55, 0.55, 0.55, 1.0)
        filled = np.full_like(rgb_u8, 140)

    img = Image.fromarray(filled, mode="RGB").resize(
        (tex_size, tex_size), Image.Resampling.BILINEAR
    )
    return img, mean_rgba


def _bake_spherical_texture(
    uvs: np.ndarray,
    vertex_rgb_u8: np.ndarray,
    tex_size: int = 256,
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


def _export_coacd_hulls(
    verts: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    asset_id: str,
    coacd_threshold: float,
    max_convex_hulls: int,
) -> list[Path]:
    """Run CoACD convex decomposition (after torch is loaded) and write watertight collision OBJs."""
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
        if float(trimesh.Trimesh(vertices=pv, faces=pf, process=False).volume) < 0.0:
            pf = np.ascontiguousarray(pf[:, [0, 2, 1]])
        hp = (out_dir / f"{asset_id}_hull_{idx}.obj").resolve()
        _write_obj(hp, pv, pf, uvs=None)
        col_paths.append(hp)

    if not col_paths:
        hull = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
        hp = (out_dir / f"{asset_id}_hull_0.obj").resolve()
        _write_obj(hp, np.asarray(hull.vertices), np.asarray(hull.faces), uvs=None)
        col_paths.append(hp)
    return col_paths


def decompose_and_export_mesh(
    raw_mesh: trimesh.Trimesh,
    target_size_m: Vec3,
    out_dir: Path,
    asset_id: str,
    align_yup_to_zup: bool = False,
    source_rgba_crop: np.ndarray | None = None,
    coacd_threshold: float = 0.08,
    max_convex_hulls: int = 6,
) -> ImageTo3DAsset:
    """Align, scale uniformly, texture-project, and CoACD-decompose a Trimesh for MuJoCo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(raw_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(raw_mesh.faces, dtype=np.int64).copy()
    if align_yup_to_zup:
        # Hunyuan3D canonical Y-up (X right, Y up, Z front) -> MuJoCo Z-up (X right, Y depth, Z up).
        verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)

    # Ensure outward-facing triangle winding (prevents MuJoCo OpenGL backface culling hollow shells).
    if float(trimesh.Trimesh(vertices=verts, faces=faces, process=False).volume) < 0.0:
        faces = np.ascontiguousarray(faces[:, [0, 2, 1]])

    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extents = np.maximum(1e-4, vmax - vmin)
    target_full = np.array([2.0 * target_size_m[0], 2.0 * target_size_m[1], 2.0 * target_size_m[2]])
    # Scale uniformly so learned 360-degree 3D aspect ratios (e.g. sleek knife vs round apple) stay intact.
    scale_u = float(max(target_full[0] / extents[0], target_full[2] / extents[2]))
    scale_u = min(
        scale_u, float(min(target_full[0] / extents[0], target_full[2] / extents[2])) * 1.35
    )
    verts = (verts - center) * scale_u

    half_ext = 0.5 * (verts.max(axis=0) - verts.min(axis=0))
    half_ext_m: Vec3 = (
        max(0.012, float(half_ext[0])),
        max(0.012, float(half_ext[1])),
        max(0.012, float(half_ext[2])),
    )

    tex_path = (out_dir / f"{asset_id}_texture.png").resolve()
    if source_rgba_crop is not None:
        uvs = _compute_planar_uv(verts)
        tex_img, mean_rgba = _build_eroded_photo_texture(source_rgba_crop, tex_size=256)
        tex_img.save(tex_path)
    else:
        vertex_rgb_u8 = _extract_vertex_rgb_u8(raw_mesh)
        mean_rgb = vertex_rgb_u8.mean(axis=0) / 255.0
        mean_rgba = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]), 1.0)
        uvs = _compute_spherical_uv(verts)
        _bake_spherical_texture(uvs, vertex_rgb_u8).save(tex_path)

    vis_path = (out_dir / f"{asset_id}_visual.obj").resolve()
    _write_obj(vis_path, verts, faces, uvs=uvs)
    col_paths = _export_coacd_hulls(
        verts, faces, out_dir, asset_id, coacd_threshold, max_convex_hulls
    )

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


class Hunyuan3DMeshLifter:
    """Lazy-loaded GPU Hunyuan3D-2mini-turbo (5-step FlashVDM) + CoACD asset lifter."""

    def __init__(
        self,
        repo_id: str = "tencent/Hunyuan3D-2mini",
        subfolder: str = "hunyuan3d-dit-v2-mini-turbo",
        device: str | None = None,
        octree_resolution: int = 160,
        num_inference_steps: int = 5,
    ) -> None:
        self.repo_id = repo_id
        self.subfolder = subfolder
        pref_device = device or resolve_geom3d_device()
        self.device = pref_device if torch.cuda.is_available() else "cpu"
        self.octree_resolution = octree_resolution
        self.num_inference_steps = num_inference_steps
        self._pipe: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        hunyuan_dir = resolve_hunyuan3d_dir()
        if not (hunyuan_dir / "hy3dgen" / "shapegen" / "pipelines.py").is_file():
            raise FileNotFoundError(
                f"Hunyuan3D-2 sources not found at {hunyuan_dir}; clone Tencent/Hunyuan3D-2 there."
            )
        _install_hunyuan3d_shims(hunyuan_dir)
        local_snap = snapshot_download(
            self.repo_id,
            allow_patterns=[
                "config.json",
                f"{self.subfolder}/*.yaml",
                f"{self.subfolder}/*.safetensors",
                "hunyuan3d-vae-v2-mini-turbo/*.yaml",
                "hunyuan3d-vae-v2-mini-turbo/*.safetensors",
            ],
        )
        shapegen = importlib.import_module("hy3dgen.shapegen")
        pipe = shapegen.Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            local_snap,
            subfolder=self.subfolder,
            use_safetensors=True,
            device=self.device,
        )
        pipe.enable_flashvdm()
        self._pipe = pipe
        return pipe

    def lift_rgba_crop(
        self,
        rgba_crop: np.ndarray,
        target_size_m: Vec3,
        out_dir: Path,
        asset_id: str,
    ) -> ImageTo3DAsset:
        """Lift a segmented RGBA object crop into a full-360 watertight MuJoCo visual + CoACD asset."""
        cached = _load_cached_asset(out_dir, asset_id)
        if cached is not None:
            return cached

        pipe = self._ensure_loaded()
        pil_img = _prepare_rgba_square_pil(rgba_crop)
        with torch.no_grad():
            meshes = pipe(
                image=pil_img,
                num_inference_steps=self.num_inference_steps,
                octree_resolution=self.octree_resolution,
                num_chunks=20000,
                generator=torch.manual_seed(42),
                output_type="trimesh",
            )
        return decompose_and_export_mesh(
            raw_mesh=meshes[0],
            target_size_m=target_size_m,
            out_dir=out_dir,
            asset_id=asset_id,
            align_yup_to_zup=True,
            source_rgba_crop=rgba_crop,
        )


_DEFAULT_LIFTER: list[Hunyuan3DMeshLifter] = []


def get_default_mesh_lifter() -> Hunyuan3DMeshLifter:
    """Return a process-wide cached Hunyuan3DMeshLifter so GPU weights load at most once."""
    if not _DEFAULT_LIFTER:
        _DEFAULT_LIFTER.append(Hunyuan3DMeshLifter())
    return _DEFAULT_LIFTER[0]
