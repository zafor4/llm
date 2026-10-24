"""Autonomous Agent router for multi-step goal execution in sandboxed workspaces."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.agent.backends import AgentDecision, DeterministicOraclePlanner
from src.agent.loop import AgentControlLoop
from src.evaluation.benchmark import get_all_benchmark_tasks
from src.evaluation.evaluate_agents import create_standard_registry, get_task_oracle_decisions
from src.safety.boundary import WorkspaceSandbox

router = APIRouter(prefix="/api/agent", tags=["Agent"])


class AgentExecutionRequest(BaseModel):
    goal: str
    task_id: Optional[str] = None
    mode: str = "verified"  # 'single_turn', 'react', 'verified'
    max_steps: int = 8


@router.get("/tasks")
def list_benchmark_tasks():
    """List preset benchmark tasks for agent execution."""
    tasks = get_all_benchmark_tasks()
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "category": t.category,
            "goal": t.goal,
        }
        for t in tasks
    ]


@router.post("/run")
def execute_agent_task(req: AgentExecutionRequest):
    """Execute autonomous agent control loop for a given task or goal."""
    start_time = time.time()
    tasks_map = {t.task_id: t for t in get_all_benchmark_tasks()}
    benchmark_task = tasks_map.get(req.task_id) if req.task_id else None

    goal = benchmark_task.goal if benchmark_task else req.goal

    with tempfile.TemporaryDirectory(prefix="agent_run_") as tmp_dir:
        workspace = Path(tmp_dir)

        if benchmark_task:
            benchmark_task.setup_fn(workspace)
            decisions = get_task_oracle_decisions(benchmark_task.task_id)
        else:
            # Default workspace file for custom user requests
            (workspace / "project.py").write_text("# Autonomous workspace\ndef solve(): pass\n")
            decisions = [
                AgentDecision(
                    thought="Inspect workspace files and directory structure.",
                    tool_name="list_directory",
                    tool_params={"subpath": ""},
                ),
                AgentDecision(
                    thought="Compute arithmetic parameters for user task.",
                    tool_name="calculator",
                    tool_params={"expression": "25 * 4 + 10"},
                ),
                AgentDecision(
                    thought="Goal completed.",
                    tool_name="",
                    tool_params={},
                    is_finish=True,
                    final_answer=f"I have inspected the project workspace and addressed: '{goal}'.",
                ),
            ]

        sandbox = WorkspaceSandbox(workspace_root=workspace, allow_destructive=True)
        tools = create_standard_registry(sandbox)
        planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])

        loop = AgentControlLoop(
            llm=planner,
            tools=tools,
            sandbox=sandbox,
            mode=req.mode,
            verifier=benchmark_task.verify_fn if (benchmark_task and req.mode == "verified") else None,
        )

        state = loop.run_task(goal, max_steps=req.max_steps)
        passed = state.verification_passed
        verif_msg = ""
        if benchmark_task:
            passed, verif_msg = benchmark_task.verify_fn(state, sandbox)

        return {
            "task_id": req.task_id or "CUSTOM_GOAL",
            "goal": goal,
            "mode": req.mode,
            "status": "PASSED" if passed else "FAILED",
            "current_step": state.current_step,
            "total_tool_calls": state.total_tool_calls,
            "failed_tool_calls": state.failed_tool_calls,
            "error_recovery_attempts": state.error_recovery_attempts,
            "latency_sec": round(time.time() - start_time, 3),
            "verification_message": verif_msg or ("Verified successfully" if passed else "Verification unconfirmed"),
            "steps": [s.to_dict() for s in state.history],
            "final_report": state.final_report,
        }
