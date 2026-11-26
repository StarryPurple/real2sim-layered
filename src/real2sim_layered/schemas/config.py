"""Single-source runtime and VLM configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _parse_dotenv(env_file: Path) -> dict[str, str]:
    """Parse KEY=VALUE pairs from a local .env file without mutating os.environ."""
    values: dict[str, str] = {}
    if not env_file.is_file():
        return values
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cleaned_key = key.strip()
        cleaned_val = val.strip().strip('"').strip("'")
        if cleaned_key and cleaned_val:
            values[cleaned_key] = cleaned_val
    return values


def _resolve_setting(key: str, dotenv: dict[str, str], default: str) -> str:
    env_val = os.environ.get(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return dotenv.get(key, default)


class VlmConfig(BaseModel):
    """Configuration for an OpenAI-compatible Vision-Language Model endpoint."""

    base_url: str = Field(default="http://127.0.0.1:8000/v1")
    api_key: str = Field(default="local-vlm-key")
    model: str = Field(default="Qwen/Qwen3.5-4B")
    timeout_s: float = Field(default=120.0, gt=0.0)
    max_tokens: int = Field(default=1536, ge=64)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> VlmConfig:
        dotenv = _parse_dotenv(env_file) if env_file is not None else {}
        return cls(
            base_url=_resolve_setting("VLM_BASE_URL", dotenv, "http://127.0.0.1:8000/v1"),
            api_key=_resolve_setting(
                "VLM_API_KEY",
                dotenv,
                _resolve_setting("OPENAI_API_KEY", dotenv, "local-vlm-key"),
            ),
            model=_resolve_setting("VLM_MODEL", dotenv, "Qwen/Qwen3.5-4B"),
            timeout_s=float(_resolve_setting("VLM_TIMEOUT_S", dotenv, "120.0")),
            max_tokens=int(_resolve_setting("VLM_MAX_TOKENS", dotenv, "1536")),
            temperature=float(_resolve_setting("VLM_TEMPERATURE", dotenv, "0.0")),
        )


class RuntimeConfig(BaseModel):
    """Top-level workspace paths and VLM runtime settings."""

    project_root: Path
    dataset_root: Path
    outputs_root: Path
    vlm: VlmConfig

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> RuntimeConfig:
        root = (project_root or Path(os.environ.get("REAL2SIM_ROOT", "."))).resolve()
        env_file = root / ".env"
        dotenv = _parse_dotenv(env_file)
        dataset_root = Path(
            _resolve_setting("MVOIK4D_ROOT", dotenv, str(root / "data" / "MVOIK-4D"))
        )
        outputs_root = Path(_resolve_setting("REAL2SIM_OUTPUTS", dotenv, str(root / "outputs")))
        return cls(
            project_root=root,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            vlm=VlmConfig.from_env(env_file=env_file),
        )
