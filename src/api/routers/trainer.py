"""Model training router for background pretraining, fine-tuning, and metric streaming."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from src.data.dataset import CausalLMDataset, build_and_split_corpus, generate_synthetic_code_corpus
from src.data.tokenizer import BPETokenizerWrapper
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.training.trainer import GPTTrainer, TrainingArgs

router = APIRouter(prefix="/api/trainer", tags=["Trainer"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_STATE = {
    "is_training": False,
    "model_name": "",
    "current_step": 0,
    "total_steps": 0,
    "current_loss": 0.0,
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "tokens_per_sec": 0.0,
    "peak_vram_mb": 0.0,
    "status": "IDLE",  # IDLE, TRAINING, COMPLETED, ERROR
    "error_message": "",
    "history": [],
}


class TrainStartRequest(BaseModel):
    model_name: str = "Custom-GPT-Micro"
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    block_size: int = 128
    learning_rate: float = 0.0005
    batch_size: int = 16
    max_steps: int = 100
    corpus_text: Optional[str] = ""


def _training_worker(req: TrainStartRequest):
    global TRAIN_STATE
    TRAIN_STATE["is_training"] = True
    TRAIN_STATE["status"] = "TRAINING"
    TRAIN_STATE["model_name"] = req.model_name
    TRAIN_STATE["total_steps"] = req.max_steps
    TRAIN_STATE["current_step"] = 0
    TRAIN_STATE["history"] = []
    TRAIN_STATE["error_message"] = ""

    try:
        # 1. Corpus
        if req.corpus_text and len(req.corpus_text.strip()) > 80:
            corpus = [s.strip() for s in req.corpus_text.split("\n\n") if len(s.strip()) > 10]
        else:
            corpus = generate_synthetic_code_corpus(num_samples=600, seed=42)

        train_texts, val_texts, test_texts = build_and_split_corpus(corpus)

        # 2. Tokenizer
        tok_path = PROJECT_ROOT / "results" / "bpe_500.json"
        if tok_path.exists():
            tokenizer = BPETokenizerWrapper.load(tok_path)
        else:
            tokenizer = BPETokenizerWrapper(vocab_size=500).train_from_iterator(corpus)

        train_tokens = [tok for t in train_texts for tok in tokenizer.encode(t, add_special_tokens=True)]
        val_tokens = [tok for t in val_texts for tok in tokenizer.encode(t, add_special_tokens=True)]

        # 3. Model
        gpt_config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=req.block_size,
            n_layer=req.n_layer,
            n_head=req.n_head,
            n_embd=req.n_embd,
            dropout=0.1,
            tie_weights=True,
        )
        model = SmallGPT(gpt_config).to(DEVICE)

        train_loader = DataLoader(CausalLMDataset(train_tokens, block_size=req.block_size), batch_size=req.batch_size, shuffle=True)
        val_loader = DataLoader(CausalLMDataset(val_tokens, block_size=req.block_size), batch_size=req.batch_size, shuffle=False)

        ckpt_dir = PROJECT_ROOT / "results" / "checkpoints" / req.model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArgs(
            learning_rate=req.learning_rate,
            min_learning_rate=req.learning_rate * 0.1,
            warmup_steps=15,
            max_steps=req.max_steps,
            eval_interval=25,
            batch_size=req.batch_size,
            device=DEVICE,
            checkpoint_dir=str(ckpt_dir),
            seed=42,
        )

        trainer = GPTTrainer(model, args, train_loader, val_loader)

        # Execute training
        summary = trainer.train()

        for log in summary["history"]:
            TRAIN_STATE["current_step"] = log["step"]
            TRAIN_STATE["current_loss"] = round(log["train_loss"], 4)
            TRAIN_STATE["val_loss"] = round(log["val_loss"], 4)
            TRAIN_STATE["val_perplexity"] = round(log["val_perplexity"], 2)
            TRAIN_STATE["tokens_per_sec"] = round(log["tokens_per_sec"], 1)
            TRAIN_STATE["peak_vram_mb"] = round(log["peak_gpu_mb"], 1)
            TRAIN_STATE["history"].append(log)

        TRAIN_STATE["status"] = "COMPLETED"
    except Exception as e:
        TRAIN_STATE["status"] = "ERROR"
        TRAIN_STATE["error_message"] = str(e)
    finally:
        TRAIN_STATE["is_training"] = False


@router.post("/start")
def start_training(req: TrainStartRequest):
    """Start training a new SmallGPT model in a background thread."""
    if TRAIN_STATE["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress.")

    thread = threading.Thread(target=_training_worker, args=(req,), daemon=True)
    thread.start()
    return {"success": True, "message": f"Training job started for '{req.model_name}'"}


@router.get("/status")
def get_training_status():
    """Retrieve live training loss, step count, and perplexity."""
    return TRAIN_STATE
