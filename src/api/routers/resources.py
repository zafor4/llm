"""Resource manager: host capacity, what is assigned, include/release engines."""

from __future__ import annotations

import gc
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.hub.activity import log_event
from src.hub.registry import list_models
from src.hub.resources import get_resource_snapshot
from src.hub.runtime import active_model_id, load_model, loaded_models, set_active, unload
from src.api.routers.trainer import TRAIN_STATE

router = APIRouter(prefix="/api/resources", tags=["Resources"])


class ResourceModelRequest(BaseModel):
    model_name: str
    activate: bool = False


@router.get("")
@router.get("/")
def resource_board():
    snap = get_resource_snapshot()
    loaded = loaded_models()
    catalog = list_models()
    loaded_ids = {m["id"] for m in loaded}
    available = [m for m in catalog if m["id"] not in loaded_ids]
    return {
        "host": snap,
        "active_model": active_model_id(),
        "assigned": loaded,
        "available": available,
        "training": {
            "status": TRAIN_STATE.get("status"),
            "is_training": TRAIN_STATE.get("is_training"),
            "model_name": TRAIN_STATE.get("model_name"),
            "peak_vram_mb": TRAIN_STATE.get("peak_vram_mb"),
            "current_step": TRAIN_STATE.get("current_step"),
            "total_steps": TRAIN_STATE.get("total_steps"),
        },
    }


@router.post("/include")
def include_model(req: ResourceModelRequest):
    try:
        slot = load_model(req.model_name, activate=req.activate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_event("resources", f"Included '{req.model_name}' in memory", meta={"id": req.model_name})
    return {"success": True, "assigned": slot.public(), "active_model": active_model_id()}


@router.post("/release")
def release_model(req: ResourceModelRequest):
    if not unload(req.model_name):
        raise HTTPException(status_code=404, detail=f"'{req.model_name}' is not in memory.")
    return {"success": True, "active_model": active_model_id()}


@router.post("/release-all")
def release_all():
    ids = [m["id"] for m in loaded_models()]
    for mid in ids:
        unload(mid)
    _free_cache()
    log_event("resources", f"Released {len(ids)} engines from memory")
    return {"success": True, "released": ids}


@router.post("/activate")
def activate(req: ResourceModelRequest):
    try:
        set_active(req.model_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "active_model": active_model_id()}


@router.post("/free-cache")
def free_cache():
    freed = _free_cache()
    log_event("resources", "Freed unused GPU cache")
    return {"success": True, "host": get_resource_snapshot(), "note": freed}


def _free_cache() -> str:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "GPU cache cleared"
    return "No GPU cache to clear"
