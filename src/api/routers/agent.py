"""Agent playground: oracle benchmarks plus hub-backed live planning."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.agent.backends import AgentDecision, DeterministicOraclePlanner, HubRuntimePlanner
from src.agent.loop import AgentControlLoop
from src.evaluation.benchmark import get_all_benchmark_tasks
from src.evaluation.evaluate_agents import create_standard_registry, get_task_oracle_decisions
from src.hub.activity import log_event
from src.hub.runtime import active_model_id, get_slot
from src.safety.boundary import WorkspaceSandbox

router = APIRouter(prefix="/api/agent", tags=["Agent"])


class AgentExecutionRequest(BaseModel):
    goal: str = ""
    task_id: Optional[str] = None
    mode: str = "verified"
    max_steps: int = 8
    planner: str = "auto"
    model: Optional[str] = None


@router.get("/tasks")
def list_benchmark_tasks():
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


def _resolve_planner(req: AgentExecutionRequest, benchmark_task, decisions):
    planner_name = (req.planner or "auto").lower()
    model_id = req.model or active_model_id()
    slot = get_slot(model_id) if model_id else None

    if planner_name == "oracle" or (planner_name == "auto" and benchmark_task is not None):
        return DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions]), "oracle"

    if planner_name in {"auto", "active", "hub"} and slot and "plan" in (slot.capabilities or []):
        return HubRuntimePlanner(model_id=slot.model_id), f"hub:{slot.model_id}"

    if planner_name in {"active", "hub"} and slot:
        return HubRuntimePlanner(model_id=slot.model_id), f"hub:{slot.model_id}"

    if decisions:
        return DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions]), "oracle-fallback"

    fallback = [
        AgentDecision(
            thought="Inspect workspace files and directory structure.",
            tool_name="list_directory",
            tool_params={"subpath": ""},
        ),
        AgentDecision(
            thought="Goal noted. Completing inspection.",
            tool_name="",
            tool_params={},
            is_finish=True,
            final_answer=(
                f"Inspected workspace for: '{req.goal}'. "
                "Load an instruct model (Ollama or Hugging Face) and set planner to Active for live planning."
            ),
        ),
    ]
    return DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in fallback]), "inspect-fallback"


@router.post("/run")
def execute_agent_task(req: AgentExecutionRequest):
    start_time = time.time()
    tasks_map = {t.task_id: t for t in get_all_benchmark_tasks()}
    benchmark_task = tasks_map.get(req.task_id) if req.task_id else None
    goal = benchmark_task.goal if benchmark_task else req.goal
    if not goal:
        return {"status": "FAILED", "error": "Provide a task_id or a custom goal."}

    with tempfile.TemporaryDirectory(prefix="agent_run_") as tmp_dir:
        workspace = Path(tmp_dir)
        if benchmark_task:
            benchmark_task.setup_fn(workspace)
            decisions = get_task_oracle_decisions(benchmark_task.task_id)
        else:
            (workspace / "project.py").write_text("# Autonomous workspace\ndef solve():\n    pass\n")
            decisions = []

        sandbox = WorkspaceSandbox(workspace_root=workspace, allow_destructive=True)
        tools = create_standard_registry(sandbox)
        planner, planner_used = _resolve_planner(req, benchmark_task, decisions)

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

        log_event(
            "agent",
            f"Agent run {req.task_id or 'CUSTOM'} via {planner_used}: {'PASSED' if passed else 'FAILED'}",
            meta={"steps": state.current_step, "planner": planner_used},
        )
        return {
            "task_id": req.task_id or "CUSTOM_GOAL",
            "goal": goal,
            "mode": req.mode,
            "planner": planner_used,
            "status": "PASSED" if passed else ("COMPLETED" if state.status == "COMPLETED" else "FAILED"),
            "current_step": state.current_step,
            "total_tool_calls": state.total_tool_calls,
            "failed_tool_calls": state.failed_tool_calls,
            "error_recovery_attempts": state.error_recovery_attempts,
            "latency_sec": round(time.time() - start_time, 3),
            "verification_message": verif_msg or state.status,
            "steps": [s.to_dict() for s in state.history],
            "final_report": state.final_report,
        }
