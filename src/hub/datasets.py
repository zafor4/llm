"""Persistent dataset library: include, store, preview, and load training texts."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.ingest import IngestResult, parse_path, parse_text
from src.hub.activity import log_event
from src.hub.paths import DATASETS_DIR, DATASETS_REGISTRY_PATH, ensure_hub_dirs

BUILTIN_CODE_ID = "builtin-code-algorithms"


def _empty_store() -> Dict[str, Any]:
    return {"datasets": {}, "updated_at": time.time()}


def _load_store() -> Dict[str, Any]:
    ensure_hub_dirs()
    if not DATASETS_REGISTRY_PATH.exists():
        return _empty_store()
    try:
        data = json.loads(DATASETS_REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("datasets"), dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return _empty_store()


def _save_store(store: Dict[str, Any]) -> None:
    ensure_hub_dirs()
    store["updated_at"] = time.time()
    DATASETS_REGISTRY_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "").strip()).strip("-._")
    return (cleaned[:80] or "dataset").lower()


def _unique_id(store: Dict[str, Any], name: str, preferred: Optional[str] = None) -> str:
    base = _slug(preferred or name)
    candidate = base
    n = 2
    while candidate in store["datasets"]:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _folder(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def _write_samples(folder: Path, samples: List[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "samples.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")
    return dest


def _record_from_ingest(
    dataset_id: str,
    name: str,
    ingest: IngestResult,
    *,
    source: str,
    sector: str,
    notes: str,
    tags: List[str],
    origin: str,
    original_name: str = "",
) -> Dict[str, Any]:
    folder = _folder(dataset_id)
    samples_path = _write_samples(folder, ingest.samples)
    size_mb = round(samples_path.stat().st_size / (1024 * 1024), 3)
    now = time.time()
    return {
        "id": dataset_id,
        "name": name.strip() or dataset_id,
        "source": source,
        "sector": sector or "general",
        "format": ingest.format,
        "schema": ingest.schema,
        "status": "ready",
        "sample_count": ingest.sample_count,
        "char_count": ingest.char_count,
        "file_size_mb": size_mb,
        "path": str(folder),
        "relative_path": f"results/datasets/{dataset_id}",
        "samples_file": "samples.jsonl",
        "origin": origin,
        "original_name": original_name,
        "source_files": ingest.source_files,
        "notes": notes or "",
        "tags": tags or [],
        "preview": ingest.preview(),
        "warnings": ingest.warnings,
        "used_by": [],
        "created_at": now,
        "updated_at": now,
    }


def include_samples(
    *,
    name: str,
    samples: List[str],
    sector: str = "general",
    notes: str = "",
    tags: Optional[List[str]] = None,
    dataset_id: Optional[str] = None,
    source: str = "import",
    origin: str = "import",
    format_name: str = "jsonl",
    schema: str = "text",
    original_name: str = "",
    source_files: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cleaned = [s.strip() for s in samples if s and str(s).strip()]
    if not cleaned:
        raise ValueError("No training samples found.")
    ingest = IngestResult(
        samples=cleaned,
        format=format_name,
        schema=schema,
        source_files=source_files or [],
        warnings=warnings or [],
    )
    store = _load_store()
    mid = dataset_id if dataset_id and dataset_id not in store["datasets"] else _unique_id(store, name, dataset_id)
    record = _record_from_ingest(
        mid,
        name,
        ingest,
        source=source,
        sector=sector,
        notes=notes,
        tags=tags or [],
        origin=origin,
        original_name=original_name,
    )
    store["datasets"][mid] = record
    _save_store(store)
    log_event("dataset", f"Included dataset '{mid}' ({record['sample_count']} samples)", meta={"id": mid})
    return record


def include_from_text(
    *,
    name: str,
    text: str,
    sector: str = "general",
    notes: str = "",
    tags: Optional[List[str]] = None,
    format_hint: str = "auto",
    split: str = "auto",
    filename: str = "pasted.txt",
    dataset_id: Optional[str] = None,
    source: str = "paste",
    origin: str = "paste",
) -> Dict[str, Any]:
    ingest = parse_text(text, filename=filename, format_hint=format_hint, split=split, sector=sector)
    if not ingest.samples:
        raise ValueError("No training samples found. Check the format or paste more text.")
    return include_samples(
        name=name,
        samples=ingest.samples,
        sector=sector,
        notes=notes,
        tags=tags,
        dataset_id=dataset_id,
        source=source,
        origin=origin,
        format_name=ingest.format,
        schema=ingest.schema,
        original_name=filename,
        source_files=ingest.source_files,
        warnings=ingest.warnings,
    )


def include_from_path(
    *,
    name: str,
    path: str,
    sector: str = "general",
    notes: str = "",
    tags: Optional[List[str]] = None,
    format_hint: str = "auto",
    split: str = "auto",
) -> Dict[str, Any]:
    src = Path(path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Nothing exists at {src}")
    ingest = parse_path(src, format_hint=format_hint, split=split, sector=sector)
    if not ingest.samples:
        raise ValueError("No training samples found at that path.")
    store = _load_store()
    mid = _unique_id(store, name or src.stem)
    record = _record_from_ingest(
        mid,
        name or src.stem,
        ingest,
        source="path",
        sector=sector,
        notes=notes,
        tags=tags or [],
        origin=str(src),
        original_name=src.name,
    )
    dest = _folder(mid)
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest / f"original{src.suffix.lower() or '.txt'}")
    store["datasets"][mid] = record
    _save_store(store)
    log_event("dataset", f"Included dataset '{mid}' from path", meta={"id": mid, "origin": str(src)})
    return record


def include_from_bytes(
    *,
    name: str,
    raw: bytes,
    filename: str,
    sector: str = "general",
    notes: str = "",
    tags: Optional[List[str]] = None,
    format_hint: str = "auto",
    split: str = "auto",
) -> Dict[str, Any]:
    from src.data.ingest import parse_bytes

    ingest = parse_bytes(raw, filename, format_hint=format_hint, split=split, sector=sector)
    if not ingest.samples:
        raise ValueError("No training samples found in that file.")
    store = _load_store()
    mid = _unique_id(store, name or Path(filename).stem)
    record = _record_from_ingest(
        mid,
        name or Path(filename).stem,
        ingest,
        source="upload",
        sector=sector,
        notes=notes,
        tags=tags or [],
        origin=filename,
        original_name=filename,
    )
    dest = _folder(mid)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".bin"
    (dest / f"original{suffix}").write_bytes(raw)
    store["datasets"][mid] = record
    _save_store(store)
    log_event("dataset", f"Included uploaded dataset '{mid}'", meta={"id": mid, "file": filename})
    return record


def ensure_builtin_datasets() -> None:
    """Register the built-in code corpus so training always has at least one dataset."""
    store = _load_store()
    samples = _folder(BUILTIN_CODE_ID) / "samples.jsonl"
    if BUILTIN_CODE_ID in store["datasets"] and samples.exists():
        return
    if BUILTIN_CODE_ID in store["datasets"]:
        store["datasets"].pop(BUILTIN_CODE_ID, None)
        _save_store(store)
    from src.data.dataset import generate_synthetic_code_corpus

    texts = generate_synthetic_code_corpus(num_samples=600, seed=42)
    include_samples(
        name="Built-in code algorithms",
        samples=texts,
        sector="code",
        notes="Shipped with the hub. Use this or include your own data instead.",
        tags=["builtin", "code"],
        dataset_id=BUILTIN_CODE_ID,
        source="builtin",
        origin="builtin",
        format_name="text",
        schema="code",
        original_name="builtin-code-algorithms",
    )


def list_datasets() -> List[Dict[str, Any]]:
    ensure_builtin_datasets()
    store = _load_store()
    return sorted(store["datasets"].values(), key=lambda d: d.get("updated_at") or 0, reverse=True)


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    ensure_builtin_datasets()
    return _load_store()["datasets"].get(dataset_id)


def update_dataset(dataset_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    store = _load_store()
    if dataset_id not in store["datasets"]:
        return None
    allowed = {"name", "notes", "tags", "sector"}
    for key, value in patch.items():
        if key in allowed:
            store["datasets"][dataset_id][key] = value
    store["datasets"][dataset_id]["updated_at"] = time.time()
    _save_store(store)
    return store["datasets"][dataset_id]


def remove_dataset(dataset_id: str, delete_files: bool = True) -> bool:
    store = _load_store()
    rec = store["datasets"].pop(dataset_id, None)
    if rec is None:
        return False
    _save_store(store)
    if delete_files:
        folder = _folder(dataset_id)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    log_event("dataset", f"Removed dataset '{dataset_id}'", meta={"id": dataset_id})
    return True


def load_dataset_texts(dataset_id: str) -> List[str]:
    rec = get_dataset(dataset_id)
    if rec is None:
        raise FileNotFoundError(f"Dataset '{dataset_id}' is not in the library.")
    path = _folder(dataset_id) / (rec.get("samples_file") or "samples.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Samples for '{dataset_id}' are missing on disk.")
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            text = obj.get("text") if isinstance(obj, dict) else None
            if text:
                texts.append(str(text))
    return texts


def mark_dataset_used(dataset_id: str, model_name: str) -> None:
    store = _load_store()
    rec = store["datasets"].get(dataset_id)
    if rec is None:
        return
    used = list(rec.get("used_by") or [])
    if model_name not in used:
        used.append(model_name)
    rec["used_by"] = used[-40:]
    rec["updated_at"] = time.time()
    store["datasets"][dataset_id] = rec
    _save_store(store)
