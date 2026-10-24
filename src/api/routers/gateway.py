"""OpenAI-compatible gateway plus local API key management."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.hub.activity import log_event
from src.hub.keys import create_key, list_keys, revoke_key, validate_bearer
from src.hub.runtime import active_model_id, format_chat_prompt, generate_text, loaded_models
from src.hub.registry import list_models as catalog_list

router = APIRouter(tags=["OpenAI Compatible Gateway"])


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


class KeyCreateRequest(BaseModel):
    label: str = "shared"


@router.get("/v1/models")
def list_models_openai_format():
    data = []
    for m in catalog_list():
        data.append(
            {
                "id": m["id"],
                "object": "model",
                "created": int(m.get("created_at") or time.time()),
                "owned_by": m.get("provider", "intelligence-hub"),
                "permission": [],
                "root": m["id"],
                "parent": None,
            }
        )
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
def create_chat_completion(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not validate_bearer(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")

    target = req.model or active_model_id()
    if not target:
        raise HTTPException(status_code=503, detail="No intelligence model is currently loaded.")

    user_text = ""
    history = []
    system_prompt = None
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_text = msg.content
            history.append({"role": msg.role, "content": msg.content})
        else:
            history.append({"role": msg.role, "content": msg.content})
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    prompt = format_chat_prompt(user_text or "Hello", history=history, system_prompt=system_prompt)
    try:
        result = generate_text(
            prompt,
            model_id=target,
            max_tokens=min(req.max_tokens or 128, 512),
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event("gateway", f"/v1 completion via {result['model']}", meta={"latency": result["latency_sec"]})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": result["tokens_requested"],
            "total_tokens": result["tokens_requested"],
        },
    }


@router.get("/api/gateway/status")
def gateway_status():
    return {
        "base_url": "http://127.0.0.1:8000/v1",
        "active_model": active_model_id(),
        "loaded_models": loaded_models(),
        "keys": list_keys(),
    }


@router.post("/api/gateway/keys")
def issue_key(req: KeyCreateRequest):
    return {"success": True, "key": create_key(req.label)}


@router.delete("/api/gateway/keys/{key_id}")
def delete_key(key_id: str):
    if not revoke_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"success": True}
