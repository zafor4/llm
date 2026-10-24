"""Training jobs: from-scratch, continue/finetune, live telemetry, cancel, history."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from src.data.dataset import CausalLMDataset, build_and_split_corpus, generate_synthetic_code_corpus
from src.data.tokenizer import BPETokenizerWrapper
from src.hub.activity import log_event
from src.hub.paths import CHECKPOINTS_DIR, DEFAULT_TOKENIZER, JOBS_PATH, ensure_hub_dirs
from src.hub.registry import sync_registry
from src.hub.resources import get_device
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.training.trainer import GPTTrainer, TrainingArgs

router = APIRouter(prefix="/api/trainer", tags=["Trainer"])

TRAIN_STATE: Dict[str, Any] = {
    "is_training": False,
    "cancel_requested": False,
    "job_id": "",
    "model_name": "",
    "mode": "scratch",
    "base_model": "",
    "current_step": 0,
    "total_steps": 0,
    "current_loss": 0.0,
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "tokens_per_sec": 0.0,
    "peak_vram_mb": 0.0,
    "status": "IDLE",
    "error_message": "",
    "history": [],
}

_TRAINER: Optional[GPTTrainer] = None
_LOCK = threading.Lock()


class TrainStartRequest(BaseModel):
    model_name: str = "Custom-GPT-Micro"
    mode: str = Field("scratch", description="scratch | continue")
    base_model: Optional[str] = None
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    block_size: int = 128
    learning_rate: float = 0.0005
    batch_size: int = 16
    max_steps: int = 100
    corpus_text: Optional[str] = ""


def job_history() -> List[Dict[str, Any]]:
    if not JOBS_PATH.exists():
        return []
    try:
        data = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
        return list(reversed(data)) if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _append_job(record: Dict[str, Any]) -> None:
    ensure_hub_dirs()
    rows = []
    if JOBS_PATH.exists():
        try:
            rows = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(record)
    JOBS_PATH.write_text(json.dumps(rows[-80:], indent=2), encoding="utf-8")


def _apply_log(log: Dict[str, Any]) -> None:
    TRAIN_STATE["current_step"] = log.get("step", TRAIN_STATE["current_step"])
    TRAIN_STATE["current_loss"] = round(float(log.get("train_loss") or 0.0), 4)
    TRAIN_STATE["val_loss"] = round(float(log.get("val_loss") or 0.0), 4)
    TRAIN_STATE["val_perplexity"] = round(float(log.get("val_perplexity") or 0.0), 2)
    TRAIN_STATE["tokens_per_sec"] = round(float(log.get("tokens_per_sec") or 0.0), 1)
    if "peak_gpu_mb" in log:
        TRAIN_STATE["peak_vram_mb"] = round(float(log["peak_gpu_mb"]), 1)
    if not log.get("partial"):
        TRAIN_STATE["history"].append({k: v for k, v in log.items() if k != "partial"})


def _load_base_checkpoint(name: str, device: str) -> tuple[SmallGPT, GPTConfig]:
    folder = CHECKPOINTS_DIR / name
    ckpt = folder / "best_model.pt"
    if not ckpt.exists():
        ckpt = folder / "final_model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"Base checkpoint '{name}' not found.")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(payload["config"])
    model = SmallGPT(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, config


def _training_worker(req: TrainStartRequest, job_id: str):
    global _TRAINER
    TRAIN_STATE.update(
        {
            "is_training": True,
            "cancel_requested": False,
            "job_id": job_id,
            "status": "TRAINING",
            "model_name": req.model_name,
            "mode": req.mode,
            "base_model": req.base_model or "",
            "total_steps": req.max_steps,
            "current_step": 0,
            "history": [],
            "error_message": "",
        }
    )
    log_event("training", f"Started {req.mode} job for '{req.model_name}'", meta={"job_id": job_id})

    try:
        if req.corpus_text and len(req.corpus_text.strip()) > 80:
            corpus = [s.strip() for s in req.corpus_text.split("\n\n") if len(s.strip()) > 10]
        else:
            corpus = generate_synthetic_code_corpus(num_samples=600, seed=42)

        train_texts, val_texts, _ = build_and_split_corpus(corpus)
        if DEFAULT_TOKENIZER.exists():
            tokenizer = BPETokenizerWrapper.load(DEFAULT_TOKENIZER)
        else:
            tokenizer = BPETokenizerWrapper(vocab_size=500).train_from_iterator(corpus)

        train_tokens = [tok for t in train_texts for tok in tokenizer.encode(t, add_special_tokens=True)]
        val_tokens = [tok for t in val_texts for tok in tokenizer.encode(t, add_special_tokens=True)]

        device = get_device()
        if req.mode == "continue" and req.base_model:
            model, gpt_config = _load_base_checkpoint(req.base_model, device)
            block_size = gpt_config.block_size
        else:
            gpt_config = GPTConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=req.block_size,
                n_layer=req.n_layer,
                n_head=req.n_head,
                n_embd=req.n_embd,
                dropout=0.1,
                tie_weights=True,
            )
            model = SmallGPT(gpt_config).to(device)
            block_size = req.block_size

        train_loader = DataLoader(
            CausalLMDataset(train_tokens, block_size=block_size),
            batch_size=req.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            CausalLMDataset(val_tokens, block_size=block_size),
            batch_size=req.batch_size,
            shuffle=False,
        )

        ckpt_dir = CHECKPOINTS_DIR / req.model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArgs(
            learning_rate=req.learning_rate,
            min_learning_rate=req.learning_rate * 0.1,
            warmup_steps=max(5, req.max_steps // 10),
            max_steps=req.max_steps,
            eval_interval=max(10, req.max_steps // 4),
            batch_size=req.batch_size,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            seed=42,
        )
        trainer = GPTTrainer(model, args, train_loader, val_loader)
        _TRAINER = trainer
        summary = trainer.train(
            on_log=_apply_log,
            should_stop=lambda: TRAIN_STATE.get("cancel_requested", False),
        )
        sync_registry()
        cancelled = TRAIN_STATE.get("cancel_requested")
        TRAIN_STATE["status"] = "CANCELLED" if cancelled else "COMPLETED"
        _append_job(
            {
                "job_id": job_id,
                "model_name": req.model_name,
                "mode": req.mode,
                "base_model": req.base_model,
                "status": TRAIN_STATE["status"],
                "best_val_loss": summary.get("best_val_loss"),
                "steps": summary.get("final_step"),
                "finished_at": time.time(),
            }
        )
        log_event("training", f"Job {TRAIN_STATE['status']} for '{req.model_name}'", meta={"job_id": job_id})
    except Exception as exc:
        TRAIN_STATE["status"] = "ERROR"
        TRAIN_STATE["error_message"] = str(exc)
        _append_job(
            {
                "job_id": job_id,
                "model_name": req.model_name,
                "status": "ERROR",
                "error": str(exc),
                "finished_at": time.time(),
            }
        )
        log_event("training", f"Training error: {exc}", level="error", meta={"job_id": job_id})
    finally:
        TRAIN_STATE["is_training"] = False
        _TRAINER = None


@router.post("/start")
def start_training(req: TrainStartRequest):
    if TRAIN_STATE["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    if req.mode == "continue" and not req.base_model:
        raise HTTPException(status_code=400, detail="Continue mode requires base_model.")
    job_id = f"job-{int(time.time())}"
    thread = threading.Thread(target=_training_worker, args=(req, job_id), daemon=True)
    thread.start()
    return {"success": True, "status": "success", "job_id": job_id, "message": f"Training started for '{req.model_name}'"}


@router.post("/cancel")
def cancel_training():
    if not TRAIN_STATE["is_training"]:
        raise HTTPException(status_code=400, detail="No training job is running.")
    TRAIN_STATE["cancel_requested"] = True
    if _TRAINER is not None:
        _TRAINER.request_stop()
    log_event("training", "Cancel requested", meta={"job_id": TRAIN_STATE.get("job_id")})
    return {"success": True, "status": "success"}


@router.get("/status")
def get_training_status():
    return TRAIN_STATE


@router.get("/jobs")
def get_training_jobs():
    return {"jobs": job_history()}
