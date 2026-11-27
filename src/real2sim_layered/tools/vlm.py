"""Unified OpenAI-compatible VLM client with Pydantic structured IKG extraction."""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from real2sim_layered.schemas import (
    AssetBackend,
    InteractionGraph,
    MvoikCase,
    PhysicalRegime,
    VlmConfig,
)
from real2sim_layered.tools.mvoik4d import sample_frame_indices

_SYSTEM_PROMPT = """\
You are a physical 4D scene understanding agent for Real2Sim reconstruction.
Given ordered RGBA key frames from a monocular video of multi-object interactions,
output ONLY a single valid JSON object conforming strictly to the requested schema.
Do not wrap prose outside the JSON object.

Regime definitions:
- "rigid": gravity, tipping, sliding, collision, or support transfer between rigid solids.
- "occluded_kinematic": rigid objects covering, uncovering, or shuffling other objects (occlusion / permanence).
- "deformable": elastic or soft-body compression, squeezing, bending, or folding under contact.
- "topological_split": cutting, slicing, or peeling that splits one object into multiple parts.
- "multiphysics_ood": combustion, liquids, fluid jets, or airflow outside rigid/elastic mechanics.

Backend definitions:
- "procedural_blender": parametric procedural primitives in Blender.
- "sim_proxy": simulation-ready primitive/convex proxies in MuJoCo/Blender.
- "image_to_3d_sim": detailed mesh lifting with convex decomposition.
"""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class _CompletionsEndpoint(Protocol):
    def create(self, **_: Any) -> Any: ...


class _ChatNamespace(Protocol):
    @property
    def completions(self) -> _CompletionsEndpoint: ...


class ChatCompletionsClient(Protocol):
    """Minimal structural interface satisfied by `openai.OpenAI` and unit-test doubles."""

    @property
    def chat(self) -> _ChatNamespace: ...


def compute_union_alpha_bbox(
    paths: list[Path],
    alpha_threshold: int = 16,
    pad_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    """Compute a shared (x0, y0, x1, y1) foreground bounding box across sampled RGBA frames."""
    x0: int | None = None
    y0: int | None = None
    x1 = 0
    y1 = 0
    img_w = 0
    img_h = 0

    lut = [255 if val >= alpha_threshold else 0 for val in range(256)]
    for path in paths:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            img_w, img_h = rgba.size
            bbox = rgba.getchannel("A").point(lut).getbbox()
            if bbox is None:
                continue
            bx0, by0, bx1, by1 = bbox
            x0 = bx0 if x0 is None else min(x0, bx0)
            y0 = by0 if y0 is None else min(y0, by0)
            x1 = max(x1, bx1)
            y1 = max(y1, by1)

    if x0 is None or y0 is None or img_w == 0 or img_h == 0:
        return None
    pad_x = round(max(1, x1 - x0) * pad_ratio)
    pad_y = round(max(1, y1 - y0) * pad_ratio)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img_w, x1 + pad_x),
        min(img_h, y1 + pad_y),
    )


def encode_frame_data_url(
    path: Path,
    max_side: int = 512,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> str:
    """Composite an RGBA frame onto white background, optionally crop to ROI, and JPEG-encode."""
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        if crop_bbox is not None:
            rgba = rgba.crop(crop_bbox)
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(bg, rgba).convert("RGB")
        width, height = composited.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            composited = composited.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BILINEAR,
            )
        buf = io.BytesIO()
        composited.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (handles optional markdown code fences)."""
    cleaned = raw_text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    fence = _JSON_BLOCK_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in VLM output: {raw_text[:200]!r}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Parsed VLM JSON payload must be an object.")
    return parsed


def _anonymize_case_id(case_id: str) -> str:
    """Strip semantic action/object words from MVOIK-4D folder names to avoid prompt leakage."""
    prefix = case_id.split("_", 1)[0]
    return f"clip_{prefix}" if prefix.isdigit() else "clip_unknown"


def _build_ikg_prompt(case: MvoikCase, sampled_indices: list[int]) -> str:
    blind_id = _anonymize_case_id(case.case_id)
    regimes = " | ".join(PhysicalRegime)
    backends = " | ".join(AssetBackend)
    schema_hint = {
        "case_id": blind_id,
        "summary": "One-sentence description of what physically happens.",
        "dominant_regime": regimes,
        "recommended_backend": backends,
        "objects": [
            {
                "object_id": "snake_case_id",
                "label": "short object label",
                "primitive_proxy": "box | sphere | cylinder | capsule | mesh",
                "is_dynamic": True,
                "material_hint": "rigid_solid | soft_elastic | fruit_cuttable | paper_thin | fluid",
            }
        ],
        "intervals": [
            {
                "start_frame": 0,
                "end_frame": case.num_frames - 1,
                "subject_id": "snake_case_id",
                "object_id": "snake_case_id",
                "action": "verb phrase",
                "contact_expected": True,
                "occlusion_expected": False,
                "requires_physics_escalation": True,
                "regime": regimes,
            }
        ],
        "confidence": 0.8,
    }
    return (
        f"Clip ID: {blind_id}\n"
        f"Total frames: {case.num_frames} (sampled frame indices shown in order: {sampled_indices}).\n"
        "Infer everything strictly from the provided visual frames.\n"
        "Analyze the interacting objects, contact/occlusion intervals, and physical regime.\n"
        "Return ONLY valid JSON matching this schema example (replace placeholder values):\n"
        f"{json.dumps(schema_hint, indent=2)}"
    )


@dataclass(frozen=True)
class VlmExtractionResult:
    """Validated InteractionGraph plus token accounting and raw response text."""

    graph: InteractionGraph
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_text: str


class OpenAICompatVlmClient:
    """Unified OpenAI-compatible VLM client targeting local Qwen3.5-4B or remote APIs."""

    def __init__(self, config: VlmConfig, client: ChatCompletionsClient | None = None) -> None:
        self.config = config
        self._client: Any = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_s,
        )

    def extract_ikg(
        self,
        case: MvoikCase,
        max_frames: int = 6,
        image_max_side: int = 512,
        max_attempts: int = 2,
    ) -> VlmExtractionResult:
        """Sample frames from `case`, query the VLM, and validate an InteractionGraph."""
        indices = sample_frame_indices(case.num_frames, max_frames=max_frames)
        sampled_paths = [case.input_frames[idx] for idx in indices]
        crop_bbox = compute_union_alpha_bbox(sampled_paths)
        content: list[dict[str, Any]] = [{"type": "text", "text": _build_ikg_prompt(case, indices)}]
        for frame_path in sampled_paths:
            data_url = encode_frame_data_url(
                frame_path, max_side=image_max_side, crop_bbox=crop_bbox
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        last_err: Exception | None = None
        total_prompt = 0
        total_completion = 0
        raw_text = ""

        for _ in range(max_attempts):
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            usage = response.usage
            if usage is not None:
                total_prompt += int(usage.prompt_tokens or 0)
                total_completion += int(usage.completion_tokens or 0)
            raw_text = response.choices[0].message.content or ""
            try:
                payload = extract_json_payload(raw_text)
                payload["case_id"] = case.case_id
                graph = InteractionGraph.model_validate(payload)
                return VlmExtractionResult(
                    graph=graph,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    total_tokens=total_prompt + total_completion,
                    raw_text=raw_text,
                )
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous output failed schema validation: {exc}. "
                            "Return ONLY a corrected JSON object with valid enum values."
                        ),
                    }
                )

        raise RuntimeError(f"VLM IKG extraction failed after {max_attempts} attempts: {last_err}")
