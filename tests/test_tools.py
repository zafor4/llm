"""Unit tests for sandboxed tools and security boundaries."""

import pytest
from pathlib import Path
from src.safety.boundary import SecurityError, WorkspaceSandbox
from src.tools.implementations import (
    CalculatorTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    SearchFilesTool,
    PythonExecuteTool,
)


def test_calculator_basic_and_functions():
    calc = CalculatorTool()

    res1 = calc.execute("2 + 3 * 4")
    assert res1.success
    assert float(res1.output) == 14.0

    res2 = calc.execute("sqrt(144) + 8")
    assert res2.success
    assert float(res2.output) == 20.0

    res3 = calc.execute("2 ** 10")
    assert res3.success
    assert float(res3.output) == 1024.0

    # Test error cases
    res_div0 = calc.execute("10 / 0")
    assert not res_div0.success
    assert "ZeroDivisionError" in res_div0.error

    # Test dangerous or unbounded operations
    res_large_exp = calc.execute("2 ** 500")
    assert not res_large_exp.success


def test_sandbox_path_traversal_prevention(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path)

    # Valid relative path inside workspace
    valid_p = sandbox.validate_path("sub/file.txt")
    assert valid_p == tmp_path / "sub" / "file.txt"

    # Malicious traversal attack
    with pytest.raises(SecurityError):
        sandbox.validate_path("../outside.txt")

    with pytest.raises(SecurityError):
        sandbox.validate_path("../../windows/system32/cmd.exe")


def test_file_read_and_write_tools(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path, allow_destructive=True)
    writer = WriteFileTool(sandbox)
    reader = ReadFileTool(sandbox)

    # Write a test file
    write_res = writer.execute(path="test.py", content="print('hello agent')\nline2\nline3\n")
    assert write_res.success

    # Read the full file
    read_res = reader.execute(path="test.py")
    assert read_res.success
    assert "hello agent" in read_res.output

    # Read line slice
    read_slice = reader.execute(path="test.py", start_line=2, end_line=3)
    assert read_slice.success
    assert read_slice.output == "line2\nline3\n"


def test_search_tool(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path, allow_destructive=True)
    writer = WriteFileTool(sandbox)
    searcher = SearchFilesTool(sandbox)

    writer.execute("a.py", "def calculate_loss(): pass")
    writer.execute("b.txt", "nothing here")

    res = searcher.execute("calculate_loss")
    assert res.success
    assert "a.py:1: def calculate_loss" in res.output


def test_python_executor(tmp_path):
    sandbox = WorkspaceSandbox(workspace_root=tmp_path, allow_destructive=True)
    executor = PythonExecuteTool(sandbox)

    # Successful run
    code_success = "import math\nprint('Value:', math.factorial(5))"
    res = executor.execute(code_success)
    assert res.success
    assert "Value: 120" in res.output

    # Failing run (AssertionError)
    code_fail = "assert 1 == 2, 'One is not two'"
    res_fail = executor.execute(code_fail)
    assert not res_fail.success
    assert "AssertionError: One is not two" in res_fail.output
