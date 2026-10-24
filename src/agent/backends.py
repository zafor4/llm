"""Pluggable Language Model interfaces for the autonomous agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from src.agent.state import TaskState


@dataclass
class AgentDecision:
    """Structured decision returned by the agent model."""

    thought: str
    tool_name: str
    tool_params: Dict[str, Any]
    is_finish: bool = False
    final_answer: Optional[str] = None


class BaseAgentLLM:
    """Abstract interface for agent planners."""

    def plan_step(
        self,
        state: TaskState,
        tool_schemas: List[Dict[str, Any]],
        system_prompt: str,
    ) -> AgentDecision:
        raise NotImplementedError


class DeterministicOraclePlanner(BaseAgentLLM):
    """Deterministic, rule-based planner used for reproducible benchmarking.
    
    Can execute predefined action plans or dynamically recover from injected errors.
    """

    def __init__(self, step_handlers: Optional[List[Callable[[TaskState], AgentDecision]]] = None):
        self.step_handlers = step_handlers or []
        self._custom_solver: Optional[Callable[[TaskState], AgentDecision]] = None

    def set_custom_solver(self, solver: Callable[[TaskState], AgentDecision]):
        self._custom_solver = solver

    def plan_step(
        self,
        state: TaskState,
        tool_schemas: List[Dict[str, Any]],
        system_prompt: str,
    ) -> AgentDecision:
        if self._custom_solver:
            return self._custom_solver(state)

        step_idx = len(state.history)
        if step_idx < len(self.step_handlers):
            return self.step_handlers[step_idx](state)

        # Default completion when all handlers consumed
        return AgentDecision(
            thought="All planned steps completed. Finalizing task.",
            tool_name="",
            tool_params={},
            is_finish=True,
            final_answer="Task successfully completed according to plan.",
        )


class PretrainedHuggingFaceLLM(BaseAgentLLM):
    """Pretrained causal instruction-following model backend (e.g. Qwen2.5-0.5B-Instruct)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            print(f"Loading pretrained agent model: {self.model_id} on {self.device}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self._initialized = True

    def plan_step(
        self,
        state: TaskState,
        tool_schemas: List[Dict[str, Any]],
        system_prompt: str,
    ) -> AgentDecision:
        self._lazy_init()

        context = state.get_context_summary()
        tools_str = json.dumps(tool_schemas, indent=2)

        prompt = (
            f"<|im_start|>system\n{system_prompt}\n"
            f"Available tools:\n{tools_str}\n"
            f"You MUST respond ONLY with a JSON object in this exact format:\n"
            f'{{"thought": "...", "tool": "tool_name", "params": {{...}}, "is_finish": false, "final_answer": null}}<|im_end|>\n'
            f"<|im_start|>user\nCurrent task state:\n{context}\nWhat is your next action?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        gen_text = outputs[0]["generated_text"][len(prompt) :].strip()
        return parse_agent_json(gen_text)


def parse_agent_json(text: str) -> AgentDecision:
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return AgentDecision(
                thought=data.get("thought", "Proceeding with selected action."),
                tool_name=data.get("tool", ""),
                tool_params=data.get("params", {}),
                is_finish=data.get("is_finish", False),
                final_answer=data.get("final_answer"),
            )
        except json.JSONDecodeError:
            pass

    return AgentDecision(
        thought=f"Failed to generate valid JSON: {text[:100]}",
        tool_name="",
        tool_params={},
        is_finish=True,
        final_answer=f"Execution halted due to formatting error: {text}",
    )


class HubRuntimePlanner(BaseAgentLLM):
    """Plans the next tool call using any loaded hub engine (HF, Ollama, or SmallGPT)."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id

    def plan_step(
        self,
        state: TaskState,
        tool_schemas: List[Dict[str, Any]],
        system_prompt: str,
    ) -> AgentDecision:
        from src.hub.runtime import generate_text

        context = state.get_context_summary()
        tools_str = json.dumps(tool_schemas, indent=2)
        prompt = (
            f"{system_prompt}\n\n"
            f"Available tools:\n{tools_str}\n\n"
            "Respond ONLY with a JSON object:\n"
            '{"thought":"...","tool":"tool_name","params":{},"is_finish":false,"final_answer":null}\n\n'
            f"Current task state:\n{context}\n\n"
            "Next action JSON:"
        )
        result = generate_text(
            prompt,
            model_id=self.model_id,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        return parse_agent_json(result["text"])
