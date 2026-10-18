"""Tool schema definitions, execution results, and registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """Standardized result returned by tool execution."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}\nDetails: {self.output}"


class BaseTool:
    """Base class for all sandboxed agent tools."""

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {}

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class ToolRegistry:
    """Registry maintaining approved tools and schemas."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def execute(self, name: str, params: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"ToolNotFound: No registered tool named '{name}'. Available: {list(self._tools.keys())}",
            )
        try:
            return tool.execute(**params)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"ExecutionException: {type(e).__name__}: {str(e)}",
            )

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.get_schema() for tool in self._tools.values()]

    def format_tool_prompt_description(self) -> str:
        lines = ["Available Tools:"]
        for tool in self._tools.values():
            schema = tool.get_schema()
            lines.append(f"- {schema['name']}: {schema['description']}")
            lines.append(f"  Schema: {json.dumps(schema['parameters'])}")
        return "\n".join(lines)
