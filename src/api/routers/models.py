"""Model catalog, import, load/unload, activate, and metadata."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.hub.activity import log_event
from src.hub.providers import ollama_ping
from src.hub.registry import get_model, list_models as catalog_list, remove_model, update_model
from src.hub.runtime import (
    active_model_id,
    load_model,
    loaded_models,
    register_huggingface,
    register_ollama,
    set_active,
    unload,
)

router = APIRouter(prefix="/api/models", tags=["Models"])


class LoadModelRequest(BaseModel):
    model_name: str
    activate: bool = True


class ImportHFRequest(BaseModel):
    model_id: str = Field(..., description="Hugging Face repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    name: Optional[str] = None
    load_now: bool = False


class ImportOllamaRequest(BaseModel):
    model_name: str
    host: str = "http://127.0.0.1:11434"
    load_now: bool = True


class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


def _public_catalog():
    loaded = {m["id"] for m in loaded_models()}
    active = active_model_id()
    models = []
    for rec in catalog_list():
        item = dict(rec)
        item["is_active"] = rec["id"] == active
        item["is_loaded"] = rec["id"] in loaded
        item["type"] = {
            "local": "Local SmallGPT Checkpoint",
            "huggingface": "Hugging Face Causal LM",
            "ollama": "Ollama Local Service",
        }.get(rec.get("provider"), rec.get("kind"))
        item["val_perplexity"] = (rec.get("metrics") or {}).get("val_perplexity")
        item["file_size_mb"] = (rec.get("metrics") or {}).get("file_size_mb")
        item["params_formatted"] = rec.get("params_formatted")
        item["tokenizer_type"] = "BPE-500" if rec.get("provider") == "local" else rec.get("provider")
        item["best_val_ppl"] = item["val_perplexity"]
        item["checkpoint_type"] = item["type"]
        models.append(item)
    return models


@router.get("")
@router.get("/")
@router.get("/list")
def list_models():
    models = _public_catalog()
    return {
        "status": "success",
        "active_model": active_model_id(),
        "current_model": active_model_id(),
        "loaded_models": loaded_models(),
        "models": models,
        "available_models": models,
    }


@router.post("/load")
def switch_model(req: LoadModelRequest):
    try:
        slot = load_model(req.model_name, activate=req.activate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "status": "success",
        "message": f"Loaded {req.model_name}",
        "active_model": active_model_id(),
        "loaded": slot.public(),
    }


@router.post("/unload")
def unload_model(req: LoadModelRequest):
    ok = unload(req.model_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"'{req.model_name}' is not loaded.")
    return {"success": True, "status": "success", "active_model": active_model_id()}


@router.post("/activate")
def activate_model(req: LoadModelRequest):
    try:
        set_active(req.model_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "status": "success", "active_model": active_model_id()}


@router.post("/import/huggingface")
def import_huggingface(req: ImportHFRequest):
    record = register_huggingface(req.model_id, req.name)
    loaded = None
    if req.load_now:
        try:
            loaded = load_model(record["id"]).public()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Registered but load failed: {exc}") from exc
    return {"success": True, "status": "success", "model": record, "loaded": loaded}


@router.post("/import/ollama")
def import_ollama(req: ImportOllamaRequest):
    ping = ollama_ping(req.host)
    if not ping.get("ok"):
        raise HTTPException(status_code=400, detail=f"Ollama not reachable at {req.host}: {ping.get('error')}")
    if ping.get("models") and req.model_name not in ping["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_name}' not in Ollama. Available: {ping['models']}",
        )
    record = register_ollama(req.model_name, req.host)
    loaded = load_model(record["id"]).public() if req.load_now else None
    return {"success": True, "status": "success", "model": record, "loaded": loaded, "ping": ping}


@router.get("/providers/ollama")
def probe_ollama(host: str = "http://127.0.0.1:11434"):
    return ollama_ping(host)


@router.patch("/{model_id}")
def patch_model(model_id: str, req: UpdateModelRequest):
    rec = update_model(model_id, req.model_dump(exclude_none=True))
    if rec is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"success": True, "model": rec}


@router.delete("/{model_id}")
def delete_model(model_id: str, delete_files: bool = False):
    if get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model_id in {m["id"] for m in loaded_models()}:
        unload(model_id)
    ok = remove_model(model_id, delete_files=delete_files)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not remove model")
    log_event("registry", f"Deleted catalog entry {model_id}")
    return {"success": True}
