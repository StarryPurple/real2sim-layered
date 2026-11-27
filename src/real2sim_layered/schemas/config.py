"""Single-source runtime, VLM, proxy, and external-binary configuration loader."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def parse_dotenv(env_file: Path) -> dict[str, str]:
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


def load_hf_token(env_file: Path | None = None) -> str | None:
    """Load HF_TOKEN from environment or a local .env file without echoing it."""
    dotenv = parse_dotenv(env_file) if env_file is not None else {}
    token = _resolve_setting("HF_TOKEN", dotenv, "")
    return token or None


def clear_proxy_env() -> None:
    """Remove proxy environment variables so HuggingFace/local requests connect directly."""
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def resolve_blender_bin(configured: str | None = None) -> str:
    """Resolve the Blender 4.2 executable path from config, BLENDER_BIN, or PATH."""
    blender_bin = configured or os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender_bin:
        raise RuntimeError(
            "BLENDER_BIN is not set and 'blender' is not on PATH; run 'source scripts/env.sh'."
        )
    return blender_bin


def resolve_triposr_dir(configured: Path | None = None) -> Path:
    """Resolve the local TripoSR repository directory from config, TRIPOSR_DIR, or workspace."""
    if configured is not None:
        return configured.resolve()
    env_dir = os.environ.get("TRIPOSR_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip()).resolve()
    root = Path(os.environ.get("REAL2SIM_ROOT", ".")).resolve()
    return (root / "tools" / "TripoSR").resolve()


def resolve_geom3d_device() -> str:
    """Resolve preferred PyTorch device for Image-to-3D mesh lifting."""
    env_dev = os.environ.get("REAL2SIM_GEOM3D_DEVICE")
    if env_dev and env_dev.strip():
        return env_dev.strip()
    return "cuda:0"


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
        dotenv = parse_dotenv(env_file) if env_file is not None else {}
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
        dotenv = parse_dotenv(env_file)
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
