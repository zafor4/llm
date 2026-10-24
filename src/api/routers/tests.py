"""Tests router for automated test execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/api/tests", tags=["Tests"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.post("/run")
def run_pytest_suite():
    """Execute pytest test suite in workspace sandbox."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--no-header"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        output = proc.stdout if proc.stdout else proc.stderr
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "output": output,
        }
    except Exception as e:
        return {"success": False, "exit_code": -1, "output": str(e)}
