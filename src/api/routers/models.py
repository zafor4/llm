"""Models router for managing local SmallGPT checkpoints and external providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.data.tokenizer import BPETokenizerWrapper
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT

router = APIRouter(prefix="/api/models", tags=["Models"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-memory model singletons
ACTIVE_MODEL_NAME = "GPT-Mini-256"
ACTIVE_MODEL: Optional[SmallGPT] = None
ACTIVE_TOKENIZER: Optional[BPETokenizerWrapper] = None


class LoadModelRequest(BaseModel):
    model_name: str


def load_checkpoint(model_name: str) -> bool:
    global ACTIVE_MODEL, ACTIVE_TOKENIZER, ACTIVE_MODEL_NAME
    ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / model_name / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / model_name / "final_model.pt"

    tok_path = PROJECT_ROOT / "results" / "bpe_500.json"
    if not ckpt_path.exists() or not tok_path.exists():
        return False

    try:
        tokenizer = BPETokenizerWrapper.load(tok_path)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        config = GPTConfig.from_dict(checkpoint["config"])
        model = SmallGPT(config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        ACTIVE_MODEL = model
        ACTIVE_TOKENIZER = tokenizer
        ACTIVE_MODEL_NAME = model_name
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False


# Initial boot load
load_checkpoint(ACTIVE_MODEL_NAME)


@router.get("/list")
def list_models():
    """List all available local checkpoints and providers."""
    checkpoints_dir = PROJECT_ROOT / "results" / "checkpoints"
    models = []

    if checkpoints_dir.exists():
        for d in sorted(checkpoints_dir.iterdir()):
            if d.is_dir() and ((d / "best_model.pt").exists() or (d / "final_model.pt").exists()):
                val_ppl = "N/A"
                hist_file = d / "training_history.json"
                if hist_file.exists():
                    try:
                        h = json.loads(hist_file.read_text())
                        if h:
                            val_ppl = f"{h[-1].get('val_perplexity', 'N/A')}"
                    except Exception:
                        pass

                models.append({
                    "id": d.name,
                    "name": d.name,
                    "type": "Local SmallGPT Checkpoint",
                    "provider": "PyTorch (Local)",
                    "val_perplexity": val_ppl,
                    "is_active": d.name == ACTIVE_MODEL_NAME,
                    "device": DEVICE,
                })

    # Connected Provider slots
    models.append({
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5-0.5B-Instruct",
        "type": "Pretrained Causal LM",
        "provider": "Hugging Face",
        "val_perplexity": "N/A",
        "is_active": False,
        "device": DEVICE,
    })
    models.append({
        "id": "ollama-local",
        "name": "Ollama Local Service",
        "type": "Local API Service (port 11434)",
        "provider": "Ollama",
        "val_perplexity": "N/A",
        "is_active": False,
        "device": "Localhost",
    })

    return {
        "active_model": ACTIVE_MODEL_NAME,
        "models": models,
    }


@router.post("/load")
def switch_model(req: LoadModelRequest):
    """Switch active inference model."""
    success = load_checkpoint(req.model_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint '{req.model_name}'")
    return {"success": True, "active_model": ACTIVE_MODEL_NAME}
