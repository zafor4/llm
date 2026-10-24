"""OpenAI-compatible External API Gateway for the House of Intelligence.

Allows any external tool, script, LangChain app, or IDE extension to consume
the active models via standard `/v1/chat/completions` and `/v1/models` routes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
import torch
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.api.routers import models as model_state

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible Gateway"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False


@router.get("/models")
def list_models_openai_format():
    """Returns models in standard OpenAI `/v1/models` response format."""
    model_list = model_state.list_models()
    data = []
    for m in model_list.get("models", []):
        data.append({
            "id": m["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": m.get("provider", "house-of-intelligence"),
            "permission": [],
            "root": m["id"],
            "parent": None,
        })

    return {
        "object": "list",
        "data": data,
    }


@router.post("/chat/completions")
def create_chat_completion(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Standard OpenAI-compatible Chat Completion endpoint."""
    target_model = req.model or model_state.ACTIVE_MODEL_NAME

    # Ensure model is active
    if target_model != model_state.ACTIVE_MODEL_NAME:
        # Attempt dynamic hot-swap if requested model exists
        model_state.load_checkpoint(target_model)

    active_model = model_state.ACTIVE_MODEL
    active_tok = model_state.ACTIVE_TOKENIZER

    if active_model is None or active_tok is None:
        raise HTTPException(
            status_code=503,
            detail="No intelligence model is currently loaded in memory."
        )

    # Format user prompt from messages
    prompt = ""
    for msg in req.messages:
        prompt += f"{msg.role.capitalize()}: {msg.content}\n"
    prompt += "Assistant: "

    # Tokenize and infer
    device = model_state.DEVICE
    tokens = active_tok.encode(prompt)
    prompt_token_count = len(tokens)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out_tokens = active_model.generate(
            idx,
            max_new_tokens=min(req.max_tokens or 128, 256),
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )

    completion_tokens = out_tokens[0].tolist()[prompt_token_count:]
    generated_text = active_tok.decode(completion_tokens).strip()

    # Build standard OpenAI response structure
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": target_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": len(completion_tokens),
            "total_tokens": prompt_token_count + len(completion_tokens),
        },
    }
