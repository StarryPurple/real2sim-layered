"""Unit tests for VLM config loading, JSON parsing, and structured IKG retry logic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from real2sim_layered.schemas import MvoikCase, PhysicalRegime, VlmConfig
from real2sim_layered.tools.vlm import (
    OpenAICompatVlmClient,
    _anonymize_case_id,
    compute_union_alpha_bbox,
    extract_json_payload,
)


def test_anonymize_case_id_prevents_prompt_leakage() -> None:
    assert _anonymize_case_id("0000_cut_apple_1") == "clip_0000"
    assert _anonymize_case_id("0014_press_ball_1") == "clip_0014"
    assert "apple" not in _anonymize_case_id("0000_cut_apple_1")


def test_compute_union_alpha_bbox_crops_tightly(tmp_path: Path) -> None:
    p = tmp_path / "f0.png"
    arr = np.zeros((100, 200, 4), dtype=np.uint8)
    arr[40:60, 80:120, :] = 255
    Image.fromarray(arr, mode="RGBA").save(p)
    bbox = compute_union_alpha_bbox([p], pad_ratio=0.1)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert 70 <= x0 <= 80 and 120 <= x1 <= 130
    assert 35 <= y0 <= 40 and 60 <= y1 <= 65


def test_extract_json_payload_handles_think_and_fences() -> None:
    raw = '<think>reasoning...</think>\n```json\n{"a": 1, "b": "ok"}\n```'
    assert extract_json_payload(raw) == {"a": 1, "b": "ok"}


def test_vlm_config_loads_dotenv(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("VLM_MODEL", raising=False)
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        'VLM_MODEL="Qwen/Qwen3.5-4B"\nVLM_BASE_URL=http://127.0.0.1:8009/v1\n',
        encoding="utf-8",
    )
    cfg = VlmConfig.from_env(env_file=env_file)
    assert cfg.model == "Qwen/Qwen3.5-4B"
    assert cfg.base_url == "http://127.0.0.1:8009/v1"


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **_: Any) -> Any:
        text = self._responses[self.calls]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def test_vlm_client_retries_and_validates_ikg(tmp_path: Path) -> None:
    img_path = tmp_path / "000000.png"
    arr = np.full((32, 32, 4), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="RGBA").save(img_path)

    case = MvoikCase(
        case_id="0004_brick_fall_1",
        regime=PhysicalRegime.RIGID,
        input_dir=tmp_path,
        eval_gt_dir=tmp_path,
        input_frames=[img_path],
        eval_views={},
        num_frames=1,
        width=32,
        height=32,
    )

    valid_json = json.dumps(
        {
            "case_id": "0004_brick_fall_1",
            "summary": "A brick tips and falls onto the table.",
            "dominant_regime": "rigid",
            "recommended_backend": "sim_proxy",
            "objects": [
                {
                    "object_id": "brick",
                    "label": "rectangular brick",
                    "primitive_proxy": "box",
                    "is_dynamic": True,
                    "material_hint": "rigid_solid",
                }
            ],
            "intervals": [
                {
                    "start_frame": 0,
                    "end_frame": 0,
                    "subject_id": "brick",
                    "object_id": "brick",
                    "action": "falls onto support",
                    "contact_expected": True,
                    "occlusion_expected": False,
                    "requires_physics_escalation": True,
                    "regime": "rigid",
                }
            ],
            "confidence": 0.9,
        }
    )

    fake_completions = _FakeCompletions(["not valid json", valid_json])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    vlm = OpenAICompatVlmClient(VlmConfig(), client=fake_client)  # pyright: ignore[reportArgumentType]  # test double

    result = vlm.extract_ikg(case, max_frames=1, max_attempts=2)
    assert fake_completions.calls == 2
    assert result.graph.dominant_regime == PhysicalRegime.RIGID
    assert result.total_tokens == 300
