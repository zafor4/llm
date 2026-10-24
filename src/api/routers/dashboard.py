"""Dashboard router providing system vitals, GPU metrics, and platform telemetry."""

from __future__ import annotations

import os
import psutil
import torch
from fastapi import APIRouter
from pathlib import Path

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.get("/vitals")
def get_system_vitals():
    """Retrieve real-time GPU, CPU, Memory, and Platform metrics."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host)"

    total_vram = 0.0
    used_vram = 0.0
    if cuda_available:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        used_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)

    # System Memory & CPU
    ram = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Check available checkpoints
    checkpoints_dir = PROJECT_ROOT / "results" / "checkpoints"
    checkpoint_count = 0
    if checkpoints_dir.exists():
        checkpoint_count = len([d for d in checkpoints_dir.iterdir() if d.is_dir()])

    return {
        "status": "HEALTHY",
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_total_mb": round(total_vram, 1),
        "vram_used_mb": round(used_vram, 1),
        "ram_total_gb": round(ram.total / (1024**3), 1),
        "ram_used_gb": round(ram.used / (1024**3), 1),
        "cpu_usage_percent": cpu_percent,
        "checkpoints_count": checkpoint_count,
    }
