"""Chat router for conversational AI and direct text generation."""

from __future__ import annotations

import time
import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from src.api.routers.models import ACTIVE_MODEL, ACTIVE_TOKENIZER, ACTIVE_MODEL_NAME, DEVICE

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


@router.post("")
def generate_chat_response(req: ChatRequest):
    """Generate conversational response using active SmallGPT model."""
    start_time = time.time()
    
    if ACTIVE_MODEL is None or ACTIVE_TOKENIZER is None:
        return {
            "role": "assistant",
            "content": "No active SmallGPT model checkpoint is currently loaded. Please load a checkpoint from the Models Manager.",
            "latency_sec": 0.0,
            "tokens_generated": 0,
            "model_used": ACTIVE_MODEL_NAME,
        }

    # Format conversational prompt
    prompt = f"# User: {req.message.strip()}\n# Assistant:\n"
    tokens = ACTIVE_TOKENIZER.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out_tokens = ACTIVE_MODEL.generate(
            idx,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    full_output = ACTIVE_TOKENIZER.decode(out_tokens[0].tolist())
    continuation = full_output[len(prompt):].strip()
    if not continuation:
        continuation = full_output.strip()

    elapsed = time.time() - start_time

    return {
        "role": "assistant",
        "content": continuation,
        "latency_sec": round(elapsed, 3),
        "tokens_generated": req.max_tokens,
        "model_used": ACTIVE_MODEL_NAME,
    }
