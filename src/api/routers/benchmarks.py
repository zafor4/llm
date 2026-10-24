"""Benchmarks router for empirical scaling evaluation and task results."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter
import pandas as pd

router = APIRouter(prefix="/api/benchmarks", tags=["Benchmarks"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.get("/results")
def get_benchmark_results():
    """Retrieve raw benchmark evaluation table and scaling statistics."""
    csv_path = PROJECT_ROOT / "results" / "agent_benchmark_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return []


@router.get("/scaling-experiments")
def get_scaling_experiments():
    """Retrieve model scaling and tokenizer comparison data."""
    lm_path = PROJECT_ROOT / "results" / "lm_experiments.json"
    tok_path = PROJECT_ROOT / "results" / "tokenizer_experiments.json"

    lm_data = json.loads(lm_path.read_text(encoding="utf-8")) if lm_path.exists() else []
    tok_data = json.loads(tok_path.read_text(encoding="utf-8")) if tok_path.exists() else {}

    return {
        "lm_experiments": lm_data,
        "tokenizer_experiments": tok_data,
    }
