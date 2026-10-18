"""Unit tests for AgentControlLoop, task state, and error recovery."""

import pytest
from pathlib import Path
from src.agent.backends import AgentDecision, DeterministicOraclePlanner
from src.agent.loop import AgentControlLoop
from src.evaluation.evaluate_agents import create_standard_registry
from src.safety.boundary import WorkspaceSandbox


def test_agent_single_turn_mode(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path)
    tools = create_standard_registry(sandbox)

    planner = DeterministicOraclePlanner()
    planner.set_custom_solver(
        lambda s: AgentDecision(
            thought="Direct response",
            tool_name="",
            tool_params={},
            is_finish=True,
            final_answer="Calculated answer is 42.",
        )
    )

    loop = AgentControlLoop(llm=planner, tools=tools, sandbox=sandbox, mode="single_turn")
    state = loop.run_task("Compute something")

    assert state.status == "COMPLETED"
    assert state.current_step == 1
    assert state.total_tool_calls == 1
    assert "42" in state.final_report


def test_agent_multi_step_and_tool_execution(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path, allow_destructive=True)
    tools = create_standard_registry(sandbox)

    # 2-step plan: Step 1 writes a file, Step 2 reads it back
    decisions = [
        AgentDecision(
            thought="Write output.txt",
            tool_name="write_file",
            tool_params={"path": "output.txt", "content": "agentic system"},
        ),
        AgentDecision(
            thought="Read output.txt",
            tool_name="read_file",
            tool_params={"path": "output.txt"},
        ),
        AgentDecision(
            thought="Done",
            tool_name="",
            tool_params={},
            is_finish=True,
            final_answer="File written and verified.",
        ),
    ]

    planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])
    loop = AgentControlLoop(llm=planner, tools=tools, sandbox=sandbox, mode="react")
    state = loop.run_task("Write and verify file")

    assert state.status == "COMPLETED"
    assert (tmp_path / "output.txt").exists()
    assert (tmp_path / "output.txt").read_text() == "agentic system"
    assert state.failed_tool_calls == 0


def test_agent_error_recovery_on_failed_tool(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path, allow_destructive=True)
    tools = create_standard_registry(sandbox)

    # Step 1 fails with divide by zero; Step 2 recovers with valid expression
    decisions = [
        AgentDecision(
            thought="Attempting division by zero",
            tool_name="calculator",
            tool_params={"expression": "100 / 0"},
        ),
        AgentDecision(
            thought="Recovering with safe division",
            tool_name="calculator",
            tool_params={"expression": "100 / 2"},
        ),
        AgentDecision(
            thought="Finished",
            tool_name="",
            tool_params={},
            is_finish=True,
            final_answer="Answer is 50",
        ),
    ]

    planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])
    loop = AgentControlLoop(llm=planner, tools=tools, sandbox=sandbox, mode="verified")
    state = loop.run_task("Perform calculations")

    assert state.status == "COMPLETED"
    assert state.failed_tool_calls == 1
    assert state.error_recovery_attempts == 1
    assert state.history[0].observation_success is False
    assert state.history[1].observation_success is True


def test_agent_budget_exhaustion(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path)
    tools = create_standard_registry(sandbox)

    # Infinitely repeats a benign tool call
    planner = DeterministicOraclePlanner()
    planner.set_custom_solver(
        lambda s: AgentDecision(
            thought="Keep searching",
            tool_name="calculator",
            tool_params={"expression": "1 + 1"},
            is_finish=False,
        )
    )

    loop = AgentControlLoop(llm=planner, tools=tools, sandbox=sandbox, mode="react")
    state = loop.run_task("Never-ending goal", max_steps=3)

    assert state.status == "FAILED_BUDGET"
    assert state.current_step == 3
