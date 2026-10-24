"""Persistent model catalog: local SmallGPT checkpoints, Hugging Face, and Ollama."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.hub.activity import log_event
from src.hub.paths import CHECKPOINTS_DIR, REGISTRY_PATH, ensure_hub_dirs


def _empty_store() -> Dict[str, Any]:
    return {"models": {}, "updated_at": time.time()}


def _load_store() -> Dict[str, Any]:
    ensure_hub_dirs()
    if not REGISTRY_PATH.exists():
        return _empty_store()
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return _empty_store()


def _save_store(store: Dict[str, Any]) -> None:
    ensure_hub_dirs()
    store["updated_at"] = time.time()
    REGISTRY_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _checkpoint_files(folder: Path) -> Dict[str, Any]:
    best = folder / "best_model.pt"
    final = folder / "final_model.pt"
    hist = folder / "training_history.json"
    val_ppl = None
    last_step = None
    if hist.exists():
        try:
            rows = json.loads(hist.read_text(encoding="utf-8"))
            if rows:
                val_ppl = rows[-1].get("val_perplexity")
                last_step = rows[-1].get("step")
        except (json.JSONDecodeError, OSError):
            pass
    weight = best if best.exists() else final
    size_mb = round(weight.stat().st_size / (1024 * 1024), 2) if weight.exists() else 0.0
    return {
        "has_best": best.exists(),
        "has_final": final.exists(),
        "weight_file": weight.name if weight.exists() else None,
        "file_size_mb": size_mb,
        "val_perplexity": val_ppl,
        "last_step": last_step,
        "path": str(folder),
        "relative_path": f"results/checkpoints/{folder.name}",
    }


def _scan_local_checkpoints() -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    if not CHECKPOINTS_DIR.exists():
        return found
    for folder in sorted(CHECKPOINTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if not ((folder / "best_model.pt").exists() or (folder / "final_model.pt").exists()):
            continue
        stats = _checkpoint_files(folder)
        found[folder.name] = {
            "id": folder.name,
            "name": folder.name,
            "provider": "local",
            "kind": "smallgpt",
            "status": "ready",
            "capabilities": ["chat", "complete", "finetune"],
            "device_hint": "local-weights",
            "origin": folder.name,
            "notes": "",
            "tags": ["local", "smallgpt"],
            "metrics": {
                "val_perplexity": stats["val_perplexity"],
                "last_step": stats["last_step"],
                "file_size_mb": stats["file_size_mb"],
            },
            "path": stats["path"],
            "relative_path": stats["relative_path"],
            "weight_file": stats["weight_file"],
            "updated_at": folder.stat().st_mtime,
        }
    return found


def sync_registry() -> Dict[str, Dict[str, Any]]:
    """Merge disk checkpoints with persisted remote/provider entries."""
    store = _load_store()
    locals_found = _scan_local_checkpoints()
    merged = dict(store.get("models") or {})

    # Drop stale local entries that no longer have weights
    for mid, rec in list(merged.items()):
        if rec.get("provider") == "local" and mid not in locals_found:
            del merged[mid]

    for mid, rec in locals_found.items():
        prev = merged.get(mid, {})
        rec["notes"] = prev.get("notes", "")
        rec["tags"] = prev.get("tags") or rec["tags"]
        rec["created_at"] = prev.get("created_at") or time.time()
        merged[mid] = rec

    store["models"] = merged
    _save_store(store)
    return merged


def list_models() -> List[Dict[str, Any]]:
    models = sync_registry()
    return sorted(models.values(), key=lambda m: (m.get("provider", ""), m.get("name", "")))


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    return sync_registry().get(model_id)


def upsert_model(record: Dict[str, Any]) -> Dict[str, Any]:
    store = _load_store()
    mid = record["id"]
    current = store["models"].get(mid, {})
    current.update(record)
    current.setdefault("created_at", time.time())
    current["updated_at"] = time.time()
    store["models"][mid] = current
    _save_store(store)
    log_event("registry", f"Registered model '{mid}' ({current.get('provider')})", meta={"id": mid})
    return current


def update_model(model_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    store = _load_store()
    if model_id not in store["models"]:
        return None
    allowed = {"name", "notes", "tags", "status"}
    for key, value in patch.items():
        if key in allowed:
            store["models"][model_id][key] = value
    store["models"][model_id]["updated_at"] = time.time()
    _save_store(store)
    return store["models"][model_id]


def remove_model(model_id: str, delete_files: bool = False) -> bool:
    store = _load_store()
    rec = store["models"].pop(model_id, None)
    if rec is None:
        return False
    _save_store(store)
    if delete_files and rec.get("provider") == "local":
        folder = CHECKPOINTS_DIR / model_id
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)
    log_event("registry", f"Removed model '{model_id}'", meta={"id": model_id, "deleted_files": delete_files})
    return True
