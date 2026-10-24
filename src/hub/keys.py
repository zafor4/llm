"""Local API keys for sharing the OpenAI-compatible gateway."""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict, List, Optional

from src.hub.activity import log_event
from src.hub.paths import KEYS_PATH, ensure_hub_dirs


def _load() -> List[Dict[str, Any]]:
    if not KEYS_PATH.exists():
        return []
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(keys: List[Dict[str, Any]]) -> None:
    ensure_hub_dirs()
    KEYS_PATH.write_text(json.dumps(keys, indent=2), encoding="utf-8")


def list_keys() -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in item.items() if k != "secret"}
        | {"secret_preview": (item.get("secret") or "")[:8] + "…"}
        for item in _load()
    ]


def create_key(label: str = "default") -> Dict[str, Any]:
    keys = _load()
    item = {
        "id": secrets.token_hex(4),
        "label": label,
        "secret": "hk_" + secrets.token_urlsafe(24),
        "created_at": time.time(),
        "enabled": True,
    }
    keys.append(item)
    _save(keys)
    log_event("gateway", f"Issued API key '{label}'", meta={"id": item["id"]})
    return item


def revoke_key(key_id: str) -> bool:
    keys = _load()
    next_keys = [k for k in keys if k.get("id") != key_id]
    if len(next_keys) == len(keys):
        return False
    _save(next_keys)
    log_event("gateway", f"Revoked API key {key_id}")
    return True


def validate_bearer(header: Optional[str]) -> bool:
    """Allow unauthenticated local calls. If a Bearer token is sent, it must match."""
    if not header:
        return True
    token = header.replace("Bearer", "").strip()
    if not token:
        return True
    return any(k.get("enabled") and k.get("secret") == token for k in _load())
