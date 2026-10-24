"""Thread-safe activity log for hub, training, inference, and agent events."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

from src.hub.paths import ACTIVITY_PATH, ensure_hub_dirs

_LOCK = threading.Lock()
_EVENTS: deque[Dict[str, Any]] = deque(maxlen=300)
_LOADED = False


def _persist() -> None:
    ensure_hub_dirs()
    ACTIVITY_PATH.write_text(json.dumps(list(_EVENTS), indent=2), encoding="utf-8")


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    if ACTIVITY_PATH.exists():
        try:
            data = json.loads(ACTIVITY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data[-300:]:
                    _EVENTS.append(item)
        except (json.JSONDecodeError, OSError):
            pass
    _LOADED = True


def log_event(
    kind: str,
    message: str,
    *,
    level: str = "info",
    source: str = "hub",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _load()
    event = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "kind": kind,
        "level": level,
        "source": source,
        "message": message,
        "meta": meta or {},
    }
    with _LOCK:
        _EVENTS.append(event)
        _persist()
    return event


def list_events(limit: int = 80, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    _load()
    with _LOCK:
        items = list(_EVENTS)
    if kind:
        items = [e for e in items if e.get("kind") == kind]
    return list(reversed(items[-limit:]))
