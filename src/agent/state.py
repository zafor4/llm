"""Agent task state, execution history tracking, and step records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Record of a single cycle within the agent control loop."""

    step_index: int
    thought: str
    action_tool: str
    action_params: Dict[str, Any]
    observation: str
    observation_success: bool
    reflection: Optional[str] = None
    verification_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    """Maintains active state, budget, and diagnostic history for an agent run."""

    goal: str
    max_steps: int = 10
    current_step: int = 0
    status: str = "INITIALIZED"  # INITIALIZED, RUNNING, COMPLETED, FAILED_BUDGET, BLOCKED
    history: List[StepRecord] = field(default_factory=list)
    final_report: Optional[str] = None
    verification_passed: bool = False
    
    # Quantitative counters
    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    error_recovery_attempts: int = 0
    total_latency_sec: float = 0.0

    def add_step(
        self,
        thought: str,
        tool: str,
        params: Dict[str, Any],
        observation: str,
        success: bool,
        reflection: Optional[str] = None,
        verification_passed: bool = False,
    ) -> StepRecord:
        self.current_step += 1
        self.total_tool_calls += 1
        if not success:
            self.failed_tool_calls += 1

        record = StepRecord(
            step_index=self.current_step,
            thought=thought,
            action_tool=tool,
            action_params=params,
            observation=observation,
            observation_success=success,
            reflection=reflection,
            verification_passed=verification_passed,
        )
        self.history.append(record)
        return record

    def get_context_summary(self, max_recent_steps: int = 5) -> str:
        """Construct a concise observation history for the next agent planning step."""
        lines = [f"Goal: {self.goal}", f"Step: {self.current_step}/{self.max_steps}"]
        recent = self.history[-max_recent_steps:] if len(self.history) > max_recent_steps else self.history
        for s in recent:
            status_str = "SUCCESS" if s.observation_success else "FAILED"
            lines.append(f"\n[Step {s.step_index}] Thought: {s.thought}")
            lines.append(f"Action: {s.action_tool}({s.action_params}) -> {status_str}")
            # Truncate long observations to preserve context
            obs_preview = s.observation[:400] + ("..." if len(s.observation) > 400 else "")
            lines.append(f"Observation:\n{obs_preview}")
            if s.reflection:
                lines.append(f"Reflection: {s.reflection}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "status": self.status,
            "verification_passed": self.verification_passed,
            "total_tool_calls": self.total_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "error_recovery_attempts": self.error_recovery_attempts,
            "total_latency_sec": self.total_latency_sec,
            "history": [s.to_dict() for s in self.history],
            "final_report": self.final_report,
        }
