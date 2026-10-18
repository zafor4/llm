"""Autonomous Agent Control Loop with Observe-Plan-Act-Inspect-Reflect-Verify cycle."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.backends import AgentDecision, BaseAgentLLM
from src.agent.state import TaskState
from src.safety.boundary import WorkspaceSandbox
from src.tools.base import ToolRegistry, ToolResult


DEFAULT_SYSTEM_PROMPT = """You are an autonomous AI research and software engineering agent.
Your objective is to accomplish multi-step goals methodically within an isolated workspace.

Operating Rules:
1. Break down the user goal into concrete, verifiable steps.
2. Formulate explicit thoughts before selecting actions.
3. Observe actual tool outputs and check return codes/assertions.
4. If a tool fails, reflect on the exact error message and revise your approach.
5. Verify your solution against explicit success criteria before declaring completion.
6. Never fabricate tool observations or execution results.
"""


class AgentControlLoop:
    """Manages iterative execution, tool invocation, error recovery, and verification."""

    def __init__(
        self,
        llm: BaseAgentLLM,
        tools: ToolRegistry,
        sandbox: WorkspaceSandbox,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verifier: Optional[Callable[[TaskState, WorkspaceSandbox], Tuple[bool, str]]] = None,
        mode: str = "verified",  # 'single_turn', 'react', 'verified'
    ):
        self.llm = llm
        self.tools = tools
        self.sandbox = sandbox
        self.system_prompt = system_prompt
        self.verifier = verifier
        self.mode = mode.lower()

    def run_task(self, goal: str, max_steps: int = 10) -> TaskState:
        """Execute the agent control loop for a given task goal."""
        state = TaskState(goal=goal, max_steps=max_steps, status="RUNNING")
        start_time = time.time()

        # Handle Single-Turn mode (Mode A)
        if self.mode == "single_turn":
            decision = self.llm.plan_step(state, self.tools.get_all_schemas(), self.system_prompt)
            state.add_step(
                thought=decision.thought,
                tool="none",
                params={},
                observation=decision.final_answer or "Single response generated.",
                success=True,
            )
            state.status = "COMPLETED"
            state.final_report = self._generate_final_report(state, decision.final_answer or "")
            state.total_latency_sec = time.time() - start_time
            return state

        # Iterative Loop (Mode B: react, Mode C: verified)
        while state.current_step < state.max_steps:
            # 1. PLAN / REASON: Get decision from LLM backend
            decision = self.llm.plan_step(state, self.tools.get_all_schemas(), self.system_prompt)

            # Check if LLM signaled completion
            if decision.is_finish or not decision.tool_name:
                # 2. VERIFY COMPLETION (Mode C only)
                if self.mode == "verified" and self.verifier is not None:
                    passed, verifier_msg = self.verifier(state, self.sandbox)
                    if passed:
                        state.verification_passed = True
                        state.status = "COMPLETED"
                        state.final_report = self._generate_final_report(state, decision.final_answer or verifier_msg)
                        break
                    else:
                        # Verification failed: trigger error recovery loop
                        state.error_recovery_attempts += 1
                        reflection = f"Verification check failed: {verifier_msg}. Correcting implementation."
                        state.add_step(
                            thought="Attempted to finish, but post-condition verification failed.",
                            tool="verification_check",
                            params={},
                            observation=verifier_msg,
                            success=False,
                            reflection=reflection,
                            verification_passed=False,
                        )
                        continue
                else:
                    # In standard ReAct without explicit verifier
                    state.status = "COMPLETED"
                    state.final_report = self._generate_final_report(state, decision.final_answer or "Task completed.")
                    break

            # 3. SELECT ACTION & EXECUTE TOOL
            tool_result = self.tools.execute(decision.tool_name, decision.tool_params)

            # 4. INSPECT RESULT & WRAP UNTRUSTED OBSERVATION
            wrapped_obs = self.sandbox.wrap_untrusted_observation(
                tool_result.output if tool_result.success else (tool_result.error or "Unknown error"),
                source=decision.tool_name,
            )

            # 5. REFLECT ON ERROR IF FAILED
            reflection = None
            if not tool_result.success:
                state.error_recovery_attempts += 1
                reflection = f"Tool '{decision.tool_name}' failed with error: {tool_result.error}. Planning remediation."

            # 6. UPDATE STATE
            state.add_step(
                thought=decision.thought,
                tool=decision.tool_name,
                params=decision.tool_params,
                observation=wrapped_obs,
                success=tool_result.success,
                reflection=reflection,
            )

        # Handle budget exhaustion
        if state.status == "RUNNING":
            state.status = "FAILED_BUDGET"
            state.final_report = self._generate_final_report(
                state, f"Resource budget exhausted after {state.current_step} steps without confirmed completion."
            )

        state.total_latency_sec = time.time() - start_time
        return state

    def _generate_final_report(self, state: TaskState, summary: str) -> str:
        lines = [
            "# AGENT TASK EXECUTION REPORT",
            f"**Goal:** {state.goal}",
            f"**Status:** {state.status}",
            f"**Verification Passed:** {state.verification_passed}",
            f"**Total Steps / Tool Calls:** {state.current_step} steps ({state.total_tool_calls} tool calls)",
            f"**Failed Tool Calls:** {state.failed_tool_calls}",
            f"**Error Recovery Cycles:** {state.error_recovery_attempts}",
            "\n## Completed Work & Evidence",
            summary,
            "\n## Step-by-Step Execution Trace",
        ]
        for s in state.history:
            status_str = "PASS" if s.observation_success else "FAIL"
            lines.append(f"- **Step {s.step_index}:** {s.thought}")
            lines.append(f"  Action: `{s.action_tool}({s.action_params})` -> [{status_str}]")
            if s.reflection:
                lines.append(f"  *Self-Correction:* {s.reflection}")

        return "\n".join(lines)
