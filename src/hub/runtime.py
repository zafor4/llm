"""In-memory model runtime: load, unload, activate, and generate."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.data.tokenizer import BPETokenizerWrapper
from src.hub.activity import log_event
from src.hub.paths import CHECKPOINTS_DIR, DEFAULT_TOKENIZER
from src.hub.providers import (
    ProviderError,
    huggingface_generate,
    load_huggingface,
    ollama_generate,
)
from src.hub.registry import get_model, list_models, upsert_model
from src.hub.resources import get_device
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT


@dataclass
class EngineSlot:
    model_id: str
    provider: str
    kind: str
    device: str
    loaded_at: float
    model: Any = None
    tokenizer: Any = None
    host: str = ""
    origin: str = ""
    capabilities: List[str] = field(default_factory=list)

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "provider": self.provider,
            "kind": self.kind,
            "device": self.device,
            "host": self.host,
            "origin": self.origin,
            "capabilities": self.capabilities,
            "loaded_at": self.loaded_at,
        }


_LOADED: Dict[str, EngineSlot] = {}
_ACTIVE_ID: Optional[str] = None


def loaded_models() -> List[Dict[str, Any]]:
    return [slot.public() for slot in _LOADED.values()]


def active_model_id() -> Optional[str]:
    return _ACTIVE_ID


def get_slot(model_id: Optional[str] = None) -> Optional[EngineSlot]:
    mid = model_id or _ACTIVE_ID
    if not mid:
        return None
    return _LOADED.get(mid)


def set_active(model_id: str) -> EngineSlot:
    if model_id not in _LOADED:
        raise RuntimeError(f"Model '{model_id}' is not loaded in memory.")
    global _ACTIVE_ID
    _ACTIVE_ID = model_id
    log_event("runtime", f"Activated '{model_id}'", meta={"id": model_id})
    return _LOADED[model_id]


def unload(model_id: str) -> bool:
    global _ACTIVE_ID
    slot = _LOADED.pop(model_id, None)
    if slot is None:
        return False
    slot.model = None
    slot.tokenizer = None
    if _ACTIVE_ID == model_id:
        _ACTIVE_ID = next(iter(_LOADED), None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_event("runtime", f"Unloaded '{model_id}' from memory", meta={"id": model_id})
    return True


def _load_smallgpt(record: Dict[str, Any]) -> EngineSlot:
    folder = CHECKPOINTS_DIR / record["id"]
    ckpt = folder / "best_model.pt"
    if not ckpt.exists():
        ckpt = folder / "final_model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"No checkpoint weights found for '{record['id']}'.")
    if not DEFAULT_TOKENIZER.exists():
        raise RuntimeError("Missing tokenizer results/bpe_500.json")

    device = get_device()
    tokenizer = BPETokenizerWrapper.load(DEFAULT_TOKENIZER)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(checkpoint["config"])
    model = SmallGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return EngineSlot(
        model_id=record["id"],
        provider="local",
        kind="smallgpt",
        device=device,
        loaded_at=time.time(),
        model=model,
        tokenizer=tokenizer,
        origin=record.get("relative_path") or str(folder),
        capabilities=["chat", "complete"],
    )


def _load_huggingface(record: Dict[str, Any]) -> EngineSlot:
    device = get_device()
    origin = record.get("origin") or record["id"]
    model, tokenizer = load_huggingface(origin, device)
    return EngineSlot(
        model_id=record["id"],
        provider="huggingface",
        kind="causal-lm",
        device=device,
        loaded_at=time.time(),
        model=model,
        tokenizer=tokenizer,
        origin=origin,
        capabilities=["chat", "complete", "plan"],
    )


def _load_ollama(record: Dict[str, Any]) -> EngineSlot:
    host = record.get("host") or "http://127.0.0.1:11434"
    origin = record.get("origin") or record["name"]
    return EngineSlot(
        model_id=record["id"],
        provider="ollama",
        kind="ollama",
        device="remote",
        loaded_at=time.time(),
        host=host,
        origin=origin,
        capabilities=["chat", "complete", "plan"],
    )


def load_model(model_id: str, activate: bool = True) -> EngineSlot:
    if model_id in _LOADED:
        if activate:
            set_active(model_id)
        return _LOADED[model_id]

    record = get_model(model_id)
    if record is None:
        # Allow loading a local folder that was just trained and not yet synced
        list_models()
        record = get_model(model_id)
    if record is None:
        raise RuntimeError(f"Unknown model '{model_id}'. Register or train it first.")

    provider = record.get("provider")
    if provider == "local":
        slot = _load_smallgpt(record)
    elif provider == "huggingface":
        slot = _load_huggingface(record)
    elif provider == "ollama":
        slot = _load_ollama(record)
    else:
        raise RuntimeError(f"Unsupported provider '{provider}'.")

    _LOADED[model_id] = slot
    if activate or _ACTIVE_ID is None:
        set_active(model_id)
    log_event("runtime", f"Loaded '{model_id}' ({provider})", meta={"id": model_id, "provider": provider})
    return slot


def bootstrap_default() -> Optional[str]:
    """Load a local checkpoint on API boot if one exists."""
    models = list_models()
    preferred = ["GPT-Mini-256", "GPT-Small-256", "GPT-Micro-128", "GPT-Mini-512"]
    by_id = {m["id"]: m for m in models}
    for name in preferred:
        if name in by_id and by_id[name].get("provider") == "local":
            try:
                load_model(name)
                return name
            except Exception as exc:
                log_event("runtime", f"Boot load failed for {name}: {exc}", level="warn")
    for rec in models:
        if rec.get("provider") == "local":
            try:
                load_model(rec["id"])
                return rec["id"]
            except Exception:
                continue
    return None


def generate_text(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    slot = get_slot(model_id)
    if slot is None:
        if model_id:
            slot = load_model(model_id)
        elif _ACTIVE_ID:
            slot = get_slot(_ACTIVE_ID)
    if slot is None:
        raise RuntimeError("No model is loaded. Load a checkpoint or import a provider model.")

    started = time.time()
    if slot.provider == "local":
        tokens = slot.tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=slot.device)
        with torch.no_grad():
            out = slot.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        full = slot.tokenizer.decode(out[0].tolist())
        text = full[len(prompt) :].strip() or full.strip()
    elif slot.provider == "huggingface":
        text = huggingface_generate(
            slot.model,
            slot.tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=slot.device,
        )
    elif slot.provider == "ollama":
        text = ollama_generate(
            slot.host,
            slot.origin,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ProviderError(f"Cannot generate with provider '{slot.provider}'.")

    elapsed = time.time() - started
    return {
        "text": text,
        "model": slot.model_id,
        "provider": slot.provider,
        "latency_sec": round(elapsed, 3),
        "tokens_requested": max_tokens,
    }


def format_chat_prompt(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    lines: List[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt.strip()}")
    for turn in history or []:
        role = (turn.get("role") or "user").capitalize()
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"User: {message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def register_huggingface(model_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    display = name or model_id.split("/")[-1]
    record_id = display.replace(" ", "-")
    return upsert_model(
        {
            "id": record_id,
            "name": display,
            "provider": "huggingface",
            "kind": "causal-lm",
            "status": "registered",
            "origin": model_id,
            "capabilities": ["chat", "complete", "plan"],
            "tags": ["huggingface"],
            "notes": f"Imported from Hugging Face Hub: {model_id}",
            "metrics": {},
        }
    )


def register_ollama(model_name: str, host: str = "http://127.0.0.1:11434") -> Dict[str, Any]:
    record_id = f"ollama-{model_name.replace(':', '-')}"
    return upsert_model(
        {
            "id": record_id,
            "name": model_name,
            "provider": "ollama",
            "kind": "ollama",
            "status": "registered",
            "origin": model_name,
            "host": host.rstrip("/"),
            "capabilities": ["chat", "complete", "plan"],
            "tags": ["ollama"],
            "notes": f"Ollama model at {host}",
            "metrics": {},
        }
    )
