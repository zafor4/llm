"""Reproducible multi-step benchmark suite for autonomous software engineering agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.agent.loop import AgentControlLoop
from src.agent.state import TaskState
from src.safety.boundary import WorkspaceSandbox
from src.tools.base import ToolRegistry
from src.tools.implementations import (
    CalculatorTool,
    ListDirectoryTool,
    PythonExecuteTool,
    ReadFileTool,
    RunUnitTestsTool,
    SearchFilesTool,
    WriteFileTool,
)


@dataclass
class BenchmarkTask:
    """Definition of a multi-step benchmark task with setup and verification."""

    task_id: str
    name: str
    category: str
    goal: str
    setup_fn: Callable[[Path], None]
    verify_fn: Callable[[TaskState, WorkspaceSandbox], Tuple[bool, str]]
    max_steps: int = 10


def setup_task_1_bug_fix(workspace: Path) -> None:
    """Setup for Task 1: Fix off-by-one bug in factorial."""
    src_dir = workspace / "src"
    tests_dir = workspace / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Buggy implementation: factorial(0) returns 0 instead of 1, and factorial(n) calculates n * fact(n-2)
    buggy_code = """def factorial(n: int) -> int:
    \"\"\"Compute factorial of n.\"\"\"
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n == 0:
        return 0  # BUG: should be 1
    result = 1
    for i in range(1, n):  # BUG: range(1, n) misses n
        result *= i
    return result
"""
    (src_dir / "math_utils.py").write_text(buggy_code, encoding="utf-8")

    test_code = """import pytest
from src.math_utils import factorial

def test_factorial_base():
    assert factorial(0) == 1
    assert factorial(1) == 1

def test_factorial_values():
    assert factorial(3) == 6
    assert factorial(5) == 120

def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)
"""
    (tests_dir / "test_math_utils.py").write_text(test_code, encoding="utf-8")


def verify_task_1(state: TaskState, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
    """Verify that all tests in test_math_utils.py pass."""
    tool = RunUnitTestsTool(sandbox)
    result = tool.execute("tests/test_math_utils.py")
    if result.success and "passed" in result.output and "failed" not in result.output:
        return True, "All unit tests in test_math_utils.py passed successfully."
    return False, f"Unit tests failed:\n{result.output}"


def setup_task_2_repo_search(workspace: Path) -> None:
    """Setup for Task 2: Search multiple service configs and aggregate port metrics."""
    services_dir = workspace / "services"
    services_dir.mkdir(parents=True, exist_ok=True)

    (services_dir / "auth_service.json").write_text(
        json.dumps({"name": "auth", "port": 8001, "status": "active"}), encoding="utf-8"
    )
    (services_dir / "payment_service.json").write_text(
        json.dumps({"name": "payment", "port": 8002, "status": "active"}), encoding="utf-8"
    )
    (services_dir / "inventory_service.json").write_text(
        json.dumps({"name": "inventory", "port": 8003, "status": "inactive"}), encoding="utf-8"
    )
    (services_dir / "gateway_service.json").write_text(
        json.dumps({"name": "gateway", "port": 8000, "status": "active"}), encoding="utf-8"
    )


def verify_task_2(state: TaskState, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
    """Verify summary.json or summary.txt was generated with active ports."""
    target_summary = sandbox.workspace_root / "summary.json"
    if not target_summary.exists():
        return False, "summary.json was not created in the workspace root."
    try:
        data = json.loads(target_summary.read_text(encoding="utf-8"))
        active_ports = data.get("active_ports", [])
        if sorted(active_ports) == [8000, 8001, 8002]:
            return True, "Active ports correctly identified and aggregated in summary.json."
        return False, f"Active ports mismatch: expected [8000, 8001, 8002], got {active_ports}"
    except Exception as e:
        return False, f"Failed to parse summary.json: {str(e)}"


def setup_task_3_data_pipeline(workspace: Path) -> None:
    """Setup for Task 3: Multi-step numerical calculation from CSV data."""
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_content = """transaction_id,amount,fee,status
tx_101,150.0,3.5,SUCCESS
tx_102,240.0,5.0,SUCCESS
tx_103,90.0,2.0,FAILED
tx_104,400.0,8.0,SUCCESS
tx_105,120.0,2.5,SUCCESS
"""
    (data_dir / "transactions.csv").write_text(csv_content, encoding="utf-8")


def verify_task_3(state: TaskState, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
    """Verify net revenue of SUCCESS transactions (150-3.5 + 240-5 + 400-8 + 120-2.5 = 891.0)."""
    report_file = sandbox.workspace_root / "net_revenue.txt"
    if not report_file.exists():
        return False, "net_revenue.txt was not created."
    content = report_file.read_text(encoding="utf-8").strip()
    if "891" in content:
        return True, "Net revenue correctly calculated as 891.0 in net_revenue.txt."
    return False, f"Calculated net revenue incorrect. Expected 891.0, found: {content}"


def setup_task_4_binary_search(workspace: Path) -> None:
    """Setup for Task 4: Implement and test binary search."""
    # Empty workspace; agent must implement search.py and tests/test_search.py
    pass


def verify_task_4(state: TaskState, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
    """Verify search.py exists and passes comprehensive test cases."""
    test_runner = PythonExecuteTool(sandbox)
    verification_script = """
from search import binary_search
assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 1) == 0
assert binary_search([1, 3, 5, 7, 9], 9) == 4
assert binary_search([1, 3, 5, 7, 9], 4) == -1
assert binary_search([], 5) == -1
print("VERIFICATION_PASS")
"""
    res = test_runner.execute(verification_script)
    if res.success and "VERIFICATION_PASS" in res.output:
        return True, "binary_search implementation passes all verification test cases."
    return False, f"Verification failed:\n{res.output}"


def setup_task_5_regression_fix(workspace: Path) -> None:
    """Setup for Task 5: Refactor user data formatter."""
    src_dir = workspace / "src"
    tests_dir = workspace / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    buggy_code = """def format_user_profile(user_data: dict) -> str:
    # BUG: Fails when age is missing or email is empty
    return f"{user_data['first_name']} {user_data['last_name']} (Age: {user_data['age']}) - {user_data['email']}"
"""
    (src_dir / "user_service.py").write_text(buggy_code, encoding="utf-8")

    test_code = """from src.user_service import format_user_profile

def test_complete_user():
    user = {"first_name": "Alice", "last_name": "Smith", "age": 30, "email": "alice@example.com"}
    assert format_user_profile(user) == "Alice Smith (Age: 30) - alice@example.com"

def test_missing_age():
    user = {"first_name": "Bob", "last_name": "Jones", "email": "bob@example.com"}
    assert format_user_profile(user) == "Bob Jones (Age: Unknown) - bob@example.com"

def test_missing_email():
    user = {"first_name": "Carol", "last_name": "White", "age": 25}
    assert format_user_profile(user) == "Carol White (Age: 25) - No email"
"""
    (tests_dir / "test_user_service.py").write_text(test_code, encoding="utf-8")


def verify_task_5(state: TaskState, sandbox: WorkspaceSandbox) -> Tuple[bool, str]:
    """Verify that all tests in test_user_service.py pass."""
    tool = RunUnitTestsTool(sandbox)
    result = tool.execute("tests/test_user_service.py")
    if result.success and "passed" in result.output and "failed" not in result.output:
        return True, "Refactored user_service.py passes all regression tests."
    return False, f"Regression tests failed:\n{result.output}"


def get_all_benchmark_tasks() -> List[BenchmarkTask]:
    """Return the suite of 5 standardized multi-step agent benchmark tasks."""
    return [
        BenchmarkTask(
            task_id="TASK_01",
            name="Bug Fix & Unit Test Verification",
            category="Code Repair",
            goal="Identify and fix the off-by-one and base-case bugs in src/math_utils.py. Run tests/test_math_utils.py to verify all assertions pass.",
            setup_fn=setup_task_1_bug_fix,
            verify_fn=verify_task_1,
            max_steps=8,
        ),
        BenchmarkTask(
            task_id="TASK_02",
            name="Repository Search & Config Aggregation",
            category="Repo Exploration",
            goal="Search through all JSON configuration files in the services/ folder. Find all services with status 'active', extract their ports, and write a summary.json file containing {'active_ports': [...]}.",
            setup_fn=setup_task_2_repo_search,
            verify_fn=verify_task_2,
            max_steps=8,
        ),
        BenchmarkTask(
            task_id="TASK_03",
            name="Data Pipeline & Financial Metric",
            category="Data Analytics",
            goal="Read data/transactions.csv, calculate the total net revenue (amount - fee) for all transactions where status is 'SUCCESS', and write the final net revenue value into net_revenue.txt.",
            setup_fn=setup_task_3_data_pipeline,
            verify_fn=verify_task_3,
            max_steps=8,
        ),
        BenchmarkTask(
            task_id="TASK_04",
            name="Algorithm Implementation & Edge Cases",
            category="Software Development",
            goal="Implement binary_search(arr, target) in search.py that searches sorted list arr for target, returning the index or -1 if not found, correctly handling empty arrays and edge elements.",
            setup_fn=setup_task_4_binary_search,
            verify_fn=verify_task_4,
            max_steps=8,
        ),
        BenchmarkTask(
            task_id="TASK_05",
            name="Legacy Code Refactor & Regression Defense",
            category="Refactoring",
            goal="Refactor format_user_profile in src/user_service.py so that it gracefully handles missing 'age' (displaying 'Age: Unknown') and missing 'email' (displaying 'No email') without raising KeyError, passing all tests in tests/test_user_service.py.",
            setup_fn=setup_task_5_regression_fix,
            verify_fn=verify_task_5,
            max_steps=8,
        ),
    ]
