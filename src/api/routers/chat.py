"""Playground and product chat completions via the hub runtime."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.hub.activity import log_event
from src.hub.runtime import active_model_id, format_chat_prompt, generate_text

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 64
    system_prompt: Optional[str] = None
    model: Optional[str] = None


@router.post("")
@router.post("/generate")
def generate_chat_response(req: ChatRequest):
    if not (req.model or active_model_id()):
        return {
            "role": "assistant",
            "content": "No model is loaded. Open Models, load a checkpoint or import a provider, then retry.",
            "latency_sec": 0.0,
            "tokens_generated": 0,
            "model_used": None,
        }

    prompt = format_chat_prompt(
        req.message,
        history=[m.model_dump() for m in (req.history or [])],
        system_prompt=req.system_prompt,
    )
    try:
        result = generate_text(
            prompt,
            model_id=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_event("chat", f"Completion via {result['model']}", meta={"latency": result["latency_sec"]})
    return {
        "role": "assistant",
        "content": result["text"],
        "latency_sec": result["latency_sec"],
        "tokens_generated": result["tokens_requested"],
        "model_used": result["model"],
        "provider": result["provider"],
    }
