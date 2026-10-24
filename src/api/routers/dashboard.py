"""Control-plane dashboard: vitals, loaded engines, jobs, and activity."""

from __future__ import annotations

from fastapi import APIRouter

from src.hub.activity import list_events
from src.hub.registry import list_models
from src.hub.resources import get_resource_snapshot
from src.hub.runtime import active_model_id, loaded_models
from src.api.routers.trainer import TRAIN_STATE, job_history

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/vitals")
def get_system_vitals():
    snap = get_resource_snapshot()
    models = list_models()
    snap.update(
        {
            "checkpoints_count": sum(1 for m in models if m.get("provider") == "local"),
            "registered_models": len(models),
            "loaded_count": len(loaded_models()),
            "active_checkpoint": active_model_id(),
            "active_model": active_model_id(),
            "training_status": TRAIN_STATE.get("status", "IDLE"),
        }
    )
    return snap


@router.get("/overview")
def get_hub_overview():
    return {
        "vitals": get_system_vitals(),
        "loaded_models": loaded_models(),
        "active_model": active_model_id(),
        "catalog": list_models(),
        "training": TRAIN_STATE,
        "recent_jobs": job_history()[:8],
        "activity": list_events(limit=20),
    }
