"""Concrete implementations of sandboxed agent tools."""

from __future__ import annotations

import ast
import math
import operator
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.safety.boundary import SecurityError, WorkspaceSandbox
from src.tools.base import BaseTool, ToolResult


class SafeMathEvaluator:
    """Safe mathematical expression evaluator using AST without eval()."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {node.value}")

        if isinstance(node, ast.Name):
            if node.id in self.ALLOWED_CONSTANTS:
                return float(self.ALLOWED_CONSTANTS[node.id])
            raise ValueError(f"Undefined variable or constant: {node.id}")

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](self._eval_node(node.operand))
            raise ValueError(f"Unsupported unary operator: {op_type}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                if op_type == ast.Pow and (right > 100 or left > 1000):
                    raise ValueError("Exponent too large (preventing resource exhaustion)")
                return self.ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.ALLOWED_FUNCTIONS:
                func = self.ALLOWED_FUNCTIONS[node.func.id]
                args = [self._eval_node(arg) for arg in node.args]
                return float(func(*args))
            raise ValueError(f"Unsupported or dangerous function call: {ast.dump(node.func)}")

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Safely calculates mathematical expressions."""

    name = "calculator"
    description = "Compute mathematical expressions safely (e.g. '2 + 2', 'sqrt(144) * 3', '2**8')."
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
    }

    def __init__(self):
        self.evaluator = SafeMathEvaluator()

    def execute(self, expression: str) -> ToolResult:
        try:
            result = self.evaluator.evaluate(expression)
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "result": result},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"CalculationError: {type(e).__name__}: {str(e)}",
            )


class ListDirectoryTool(BaseTool):
    """Lists files and directories inside the approved workspace."""

    name = "list_directory"
    description = "List files and subdirectories within the sandboxed workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "subpath": {
                "type": "string",
                "description": "Relative subdirectory path to list (default: root '').",
                "default": "",
            }
        },
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(self, subpath: str = "") -> ToolResult:
        try:
            target_dir = self.sandbox.validate_path(subpath)
            if not target_dir.exists():
                return ToolResult(success=False, output="", error=f"Directory does not exist: {subpath}")
            if not target_dir.is_dir():
                return ToolResult(success=False, output="", error=f"Path is not a directory: {subpath}")

            entries = []
            for item in sorted(target_dir.iterdir()):
                rel = item.relative_to(self.sandbox.workspace_root).as_posix()
                kind = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else 0
                entries.append(f"{kind:<5} {size:>8} B  {rel}")

            output = "\n".join(entries) if entries else "(empty directory)"
            return ToolResult(success=True, output=output, metadata={"count": len(entries)})
        except SecurityError as se:
            return ToolResult(success=False, output="", error=str(se))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"ListDirectoryError: {str(e)}")


class ReadFileTool(BaseTool):
    """Reads content from an approved workspace file with optional line bounds."""

    name = "read_file"
    description = "Read text from an approved file inside the workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file."},
            "start_line": {"type": "integer", "description": "1-based starting line number (optional)."},
            "end_line": {"type": "integer", "description": "1-based ending line number (optional)."},
        },
        "required": ["path"],
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> ToolResult:
        try:
            target_file = self.sandbox.validate_path(path)
            if not target_file.exists():
                return ToolResult(success=False, output="", error=f"FileNotFound: {path}")
            if not target_file.is_file():
                return ToolResult(success=False, output="", error=f"NotAFile: {path}")

            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            s = max(1, start_line) if start_line is not None else 1
            e = min(total_lines, end_line) if end_line is not None else total_lines

            selected_lines = lines[s - 1 : e]
            output = "".join(selected_lines)
            return ToolResult(
                success=True,
                output=output,
                metadata={"total_lines": total_lines, "start_line": s, "end_line": e},
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=str(se))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"ReadFileError: {str(e)}")


class WriteFileTool(BaseTool):
    """Writes text content to an approved workspace file."""

    name = "write_file"
    description = "Write or overwrite text content to an approved file inside the workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to file."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            target_file = self.sandbox.validate_path(path)
            if target_file.exists() and not self.sandbox.check_approval("write_overwrite", path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"ApprovalRequired: Overwriting existing file '{path}' requires approval.",
                )

            target_file.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to '{path}'.",
                metadata={"path": path, "bytes_written": len(content.encode("utf-8"))},
            )
        except SecurityError as se:
            return ToolResult(success=False, output="", error=str(se))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"WriteFileError: {str(e)}")


class SearchFilesTool(BaseTool):
    """Searches pattern or text inside workspace files."""

    name = "search_files"
    description = "Search for a regex or text pattern inside files in the workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex pattern to search for."},
            "file_extension": {
                "type": "string",
                "description": "Optional file extension filter (e.g. '.py').",
                "default": "",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(self, pattern: str, file_extension: str = "") -> ToolResult:
        import re

        try:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []

            for root, _, files in os.walk(self.sandbox.workspace_root):
                for fname in files:
                    if file_extension and not fname.endswith(file_extension):
                        continue
                    full_p = Path(root) / fname
                    rel_p = full_p.relative_to(self.sandbox.workspace_root).as_posix()
                    try:
                        with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, start=1):
                                if regex.search(line):
                                    matches.append(f"{rel_p}:{idx}: {line.strip()}")
                                    if len(matches) >= 50:
                                        break
                    except Exception:
                        continue

            output = "\n".join(matches) if matches else "No matching files found."
            return ToolResult(success=True, output=output, metadata={"match_count": len(matches)})
        except Exception as e:
            return ToolResult(success=False, output="", error=f"SearchError: {str(e)}")


class PythonExecuteTool(BaseTool):
    """Executes a Python script in a sandboxed subprocess with strict resource bounds."""

    name = "execute_python"
    description = "Run a python script within the workspace sandbox and observe stdout, stderr, and exit code."
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 10).",
                "default": 10,
            },
        },
        "required": ["code"],
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(self, code: str, timeout: int = 10) -> ToolResult:
        # Create a temporary runner script inside the workspace
        runner_file = self.sandbox.workspace_root / ".sandbox_exec.py"
        try:
            with open(runner_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Execute subprocess with current python interpreter
            proc = subprocess.run(
                [sys.executable, str(runner_file)],
                cwd=str(self.sandbox.workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            exit_code = proc.returncode

            output = f"[Exit Code {exit_code}]\n"
            if stdout:
                output += f"STDOUT:\n{stdout}\n"
            if stderr:
                output += f"STDERR:\n{stderr}\n"
            if not stdout and not stderr:
                output += "(no output produced)"

            success = exit_code == 0
            error = None if success else f"Subprocess exited with non-zero status code: {exit_code}"

            return ToolResult(
                success=success,
                output=output,
                error=error,
                metadata={"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"TimeoutExpired: Execution exceeded {timeout} seconds limit.",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"SubprocessError: {str(e)}")
        finally:
            if runner_file.exists():
                try:
                    runner_file.unlink()
                except Exception:
                    pass


class RunUnitTestsTool(BaseTool):
    """Runs pytest on a specified test file within the workspace and reports structured results."""

    name = "run_unit_tests"
    description = "Run pytest on a test file inside workspace to verify correctness and get assertion results."
    parameters_schema = {
        "type": "object",
        "properties": {
            "test_file": {
                "type": "string",
                "description": "Relative path to test file (e.g. 'tests/test_solution.py').",
            }
        },
        "required": ["test_file"],
    }

    def __init__(self, sandbox: WorkspaceSandbox):
        self.sandbox = sandbox

    def execute(self, test_file: str) -> ToolResult:
        try:
            target_path = self.sandbox.validate_path(test_file)
            if not target_path.exists():
                return ToolResult(success=False, output="", error=f"Test file not found: {test_file}")

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(target_path), "-v", "--no-header"],
                cwd=str(self.sandbox.workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            exit_code = proc.returncode

            summary = stdout if stdout else stderr
            success = exit_code == 0

            return ToolResult(
                success=success,
                output=summary,
                error=None if success else f"Tests failed with exit code {exit_code}",
                metadata={"exit_code": exit_code, "test_file": test_file},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Test run timed out after 20 seconds.")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"TestRunnerError: {str(e)}")
