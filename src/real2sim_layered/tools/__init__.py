"""Dataset loaders and external model/VLM tool adapters."""

from __future__ import annotations

from real2sim_layered.tools.mvoik4d import (
    DEFAULT_PROBE_MANIFEST,
    infer_regime_from_case_id,
    list_cases,
    load_case,
    load_probe_cases,
    load_rgba_frame,
    sample_frame_indices,
)
from real2sim_layered.tools.vlm import OpenAICompatVlmClient, VlmExtractionResult

__all__ = [
    "DEFAULT_PROBE_MANIFEST",
    "OpenAICompatVlmClient",
    "VlmExtractionResult",
    "infer_regime_from_case_id",
    "list_cases",
    "load_case",
    "load_probe_cases",
    "load_rgba_frame",
    "sample_frame_indices",
]
