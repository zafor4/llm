"""Dataset library: include any text dataset, inspect it, and attach it to training."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.data.ingest import SECTORS
from src.hub.datasets import (
    ensure_builtin_datasets,
    get_dataset,
    include_from_bytes,
    include_from_path,
    include_from_text,
    list_datasets,
    load_dataset_texts,
    remove_dataset,
    update_dataset,
)

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

MAX_UPLOAD_BYTES = 80 * 1024 * 1024


class IncludeTextRequest(BaseModel):
    name: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    sector: str = "general"
    notes: str = ""
    tags: List[str] = Field(default_factory=list)
    format_hint: str = "auto"
    split: str = "auto"
    filename: str = "pasted.txt"


class IncludePathRequest(BaseModel):
    name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    sector: str = "general"
    notes: str = ""
    tags: List[str] = Field(default_factory=list)
    format_hint: str = "auto"
    split: str = "auto"


class UpdateDatasetRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    sector: Optional[str] = None


def _normalize_sector(sector: str) -> str:
    value = (sector or "general").lower().strip()
    return value if value in SECTORS else "general"


@router.get("")
@router.get("/")
def get_datasets():
    ensure_builtin_datasets()
    return {"datasets": list_datasets(), "sectors": list(SECTORS)}


@router.get("/{dataset_id}")
def get_one_dataset(dataset_id: str):
    rec = get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    return rec


@router.get("/{dataset_id}/samples")
def get_dataset_samples(dataset_id: str, offset: int = 0, limit: int = 20):
    rec = get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    try:
        texts = load_dataset_texts(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    start = max(0, offset)
    end = start + max(1, min(limit, 100))
    return {
        "id": dataset_id,
        "total": len(texts),
        "offset": start,
        "samples": texts[start:end],
    }


@router.post("/include")
def include_text(req: IncludeTextRequest):
    try:
        record = include_from_text(
            name=req.name,
            text=req.text,
            sector=_normalize_sector(req.sector),
            notes=req.notes,
            tags=req.tags,
            format_hint=req.format_hint,
            split=req.split,
            filename=req.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "dataset": record}


@router.post("/include-path")
def include_path(req: IncludePathRequest):
    try:
        record = include_from_path(
            name=req.name,
            path=req.path,
            sector=_normalize_sector(req.sector),
            notes=req.notes,
            tags=req.tags,
            format_hint=req.format_hint,
            split=req.split,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "dataset": record}


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    sector: str = Form("general"),
    notes: str = Form(""),
    format_hint: str = Form("auto"),
    split: str = Form("auto"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than 80 MB.")
    filename = file.filename or "upload.txt"
    try:
        record = include_from_bytes(
            name=name or filename,
            raw=raw,
            filename=filename,
            sector=_normalize_sector(sector),
            notes=notes,
            format_hint=format_hint,
            split=split,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "dataset": record}


@router.patch("/{dataset_id}")
def patch_dataset(dataset_id: str, req: UpdateDatasetRequest):
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    if "sector" in patch:
        patch["sector"] = _normalize_sector(patch["sector"])
    rec = update_dataset(dataset_id, patch)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    return rec


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    rec = get_dataset(dataset_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    if rec.get("source") == "builtin":
        raise HTTPException(status_code=400, detail="The built-in dataset stays in the library.")
    if not remove_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' was not found.")
    return {"success": True, "id": dataset_id}
