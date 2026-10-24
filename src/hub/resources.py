"""Host and accelerator resource snapshot for the control plane."""

from __future__ import annotations

import os
from typing import Any, Dict

import psutil
import torch


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU (Host)"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_resource_snapshot() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    total_vram = 0.0
    used_vram = 0.0
    reserved_vram = 0.0
    if cuda:
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 * 1024)
        used_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved_vram = torch.cuda.memory_reserved(0) / (1024 * 1024)

    ram = psutil.virtual_memory()
    disk_root = "C:\\" if os.name == "nt" else "/"
    disk = psutil.disk_usage(disk_root)
    return {
        "status": "HEALTHY",
        "pytorch_version": torch.__version__,
        "cuda_available": cuda,
        "device_name": device_name(),
        "device": get_device(),
        "vram_total_mb": round(total_vram, 1),
        "vram_used_mb": round(used_vram, 1),
        "vram_reserved_mb": round(reserved_vram, 1),
        "vram_free_mb": round(max(0.0, total_vram - reserved_vram), 1),
        "ram_total_gb": round(ram.total / (1024**3), 1),
        "ram_used_gb": round(ram.used / (1024**3), 1),
        "ram_percent": ram.percent,
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count": psutil.cpu_count() or 1,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
    }
