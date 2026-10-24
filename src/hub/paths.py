"""Canonical filesystem locations for the intelligence hub."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
REGISTRY_PATH = RESULTS_DIR / "hub_registry.json"
ACTIVITY_PATH = RESULTS_DIR / "hub_activity.json"
JOBS_PATH = RESULTS_DIR / "training_jobs.json"
KEYS_PATH = RESULTS_DIR / "api_keys.json"
DEFAULT_TOKENIZER = RESULTS_DIR / "bpe_500.json"


def ensure_hub_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
