"""Lightweight OpenAI-compatible local VLM server (Qwen/Qwen3.5-4B via Transformers)."""

from __future__ import annotations

import argparse
import base64
import io
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from real2sim_layered.schemas import clear_proxy_env, load_hf_token


def _decode_image_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, b64_part = url.split(",", 1)
        raw = base64.b64decode(b64_part)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    path = Path(url.removeprefix("file://"))
    return Image.open(path).convert("RGB")


def _convert_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI multimodal chat messages into HF processor conversation dicts."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "image_url":
                img_spec = item.get("image_url", {})
                url = img_spec.get("url", "") if isinstance(img_spec, dict) else str(img_spec)
                parts.append({"type": "image", "image": _decode_image_url(url)})
        converted.append({"role": role, "content": parts})
    return converted


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    temperature: float = 0.0


def build_app(
    model_id: str,
    device: str,
    enable_thinking: bool,
    token: str | None,
) -> FastAPI:
    print(f"Loading VLM '{model_id}' onto {device} (bf16) ...", flush=True)
    t0 = time.time()
    processor: Any = AutoProcessor.from_pretrained(model_id, token=token)
    model: Any = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        token=token,
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded '{model_id}' in {time.time() - t0:.1f}s.", flush=True)

    infer_lock = threading.Lock()
    app = FastAPI(title="real2sim-layered local VLM server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_id, "device": device}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        try:
            hf_messages = _convert_openai_messages(req.messages)
            try:
                inputs = processor.apply_chat_template(
                    hf_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                inputs = processor.apply_chat_template(
                    hf_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }
            prompt_len = int(inputs["input_ids"].shape[-1])
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": req.max_tokens,
                "do_sample": req.temperature > 0.0,
            }
            if req.temperature > 0.0:
                gen_kwargs["temperature"] = req.temperature

            with infer_lock, torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)

            new_tokens = output_ids[0][prompt_len:]
            completion_len = int(new_tokens.shape[-1])
            text = processor.decode(new_tokens, skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_len,
                    "total_tokens": prompt_len + completion_len,
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local HuggingFace VLM via OpenAI API.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="HuggingFace model ID or path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--device", default="cuda:2", help="Target torch device (e.g. cuda:2).")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable reasoning <think> blocks (disabled by default for deterministic JSON).",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Unset HTTP(S)_PROXY before downloading model weights from HuggingFace.",
    )
    args = parser.parse_args()

    if args.no_proxy:
        clear_proxy_env()

    root = Path(os.environ.get("REAL2SIM_ROOT", ".")).resolve()
    app = build_app(
        model_id=args.model,
        device=args.device,
        enable_thinking=args.enable_thinking,
        token=load_hf_token(root / ".env"),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
