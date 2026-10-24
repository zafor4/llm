"""Quick Interactive Demonstration Script for SmallGPT and the Autonomous Agent.

Run with:
    python demo.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.tokenizer import BPETokenizerWrapper
from src.evaluation.benchmark import get_all_benchmark_tasks
from src.evaluation.evaluate_agents import create_standard_registry, get_task_oracle_decisions
from src.agent.backends import DeterministicOraclePlanner
from src.agent.loop import AgentControlLoop
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.safety.boundary import WorkspaceSandbox

console = Console()


def show_menu():
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]Antigravity AI Platform[/bold cyan]\n"
            "[green]Production CLI Terminal Runner[/green]",
            border_style="cyan",
        )
    )
    console.print("[bold]Select an action to run and inspect:[/bold]")
    console.print("  [1] [bold green]Live Autonomous Agent Demo[/bold green] (Watch agent solve a bug fix & test task step-by-step)")
    console.print("  [2] [bold yellow]SmallGPT Generation Demo[/bold yellow] (Load trained checkpoint & generate code completions)")
    console.print("  [3] [bold magenta]View Benchmark Results[/bold magenta] (Inspect real empirical results across all 5 tasks)")
    console.print("  [4] [bold blue]Run All 16 Unit Tests[/bold blue] (Verify model, sandbox & agent behavior)")
    console.print("  [q] Exit")


def demo_agent():
    console.print("\n[bold cyan]=== Live Autonomous Agent Demonstration ===[/bold cyan]")
    console.print("[dim]Task: Fix off-by-one bug in factorial and verify with unit tests.[/dim]\n")

    tasks = get_all_benchmark_tasks()
    task = tasks[0]  # Task 1: Factorial bug fix

    with tempfile.TemporaryDirectory(prefix="agent_demo_") as tmp_dir:
        workspace = Path(tmp_dir)
        task.setup_fn(workspace)
        sandbox = WorkspaceSandbox(workspace_root=workspace, allow_destructive=True)
        tools = create_standard_registry(sandbox)
        decisions = get_task_oracle_decisions(task.task_id)

        planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])
        loop = AgentControlLoop(
            llm=planner,
            tools=tools,
            sandbox=sandbox,
            mode="verified",
            verifier=task.verify_fn,
        )

        state = loop.run_task(task.goal, max_steps=task.max_steps)

        for step in state.history:
            status = "[bold green]PASS[/bold green]" if step.observation_success else "[bold red]FAIL[/bold red]"
            console.print(f"[bold yellow]Step {step.step_index}:[/bold yellow] {step.thought}")
            console.print(f"  [cyan]Action:[/cyan] [bold]{step.action_tool}[/bold]({json.dumps(step.action_params)}) -> {status}")
            obs_preview = step.observation[:160].replace("\n", " ")
            console.print(f"  [dim]Observation: {obs_preview}...[/dim]\n")

        console.print(Panel(state.final_report, title="Final Execution Report", border_style="green"))


def demo_small_gpt():
    console.print("\n[bold yellow]=== SmallGPT Autoregressive Generation ===[/bold yellow]")
    ckpt_path = Path("results/checkpoints/GPT-Mini-256/best_model.pt")
    tok_path = Path("results/bpe_500.json")

    if not ckpt_path.exists() or not tok_path.exists():
        console.print("[red]Checkpoint or tokenizer not found in `results/checkpoints/`.[/red]")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizerWrapper.load(tok_path)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(checkpoint["config"])
    model = SmallGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    console.print(f"[green]Loaded checkpoint from '{ckpt_path}' on {device}[/green]")
    console.print(f"Parameters: {model.get_num_params():,} | Context window: {config.block_size}\n")

    prompt = "def binary_search(arr: list[int], target: int) -> int:\n"
    console.print(f"[bold cyan]Input Prompt:[/bold cyan]\n[dim]{prompt}[/dim]")

    prompt_tokens = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out_tokens = model.generate(idx, max_new_tokens=40, temperature=0.7, top_p=0.9)

    completion = tokenizer.decode(out_tokens[0].tolist())
    console.print(Panel(completion, title="SmallGPT Generation", border_style="yellow"))


def demo_results():
    console.print("\n[bold magenta]=== Empirical Benchmark Summary ===[/bold magenta]")
    csv_path = Path("results/agent_benchmark_results.csv")
    if not csv_path.exists():
        console.print("[red]Results CSV not found in `results/`.[/red]")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    table = Table(title="Autonomous Agent Benchmark Results across 5 Software Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Mode", style="yellow")
    table.add_column("Success", style="green")
    table.add_column("Steps", style="white")
    table.add_column("Latency (s)", style="blue")

    for _, row in df.iterrows():
        table.add_row(
            str(row["task_id"]),
            str(row["category"]),
            str(row["agent_mode"]),
            str(row["success"]),
            str(row["steps_taken"]),
            f"{row['latency_sec']:.4f}",
        )

    console.print(table)


def run_tests():
    console.print("\n[bold blue]=== Executing Test Suite ===[/bold blue]")
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def main():
    while True:
        show_menu()
        choice = input("\nEnter choice [1-4, q]: ").strip().lower()
        if choice == "1":
            demo_agent()
        elif choice == "2":
            demo_small_gpt()
        elif choice == "3":
            demo_results()
        elif choice == "4":
            run_tests()
        elif choice in ("q", "quit", "exit"):
            console.print("\n[dim]Exiting demonstration. Happy researching![/dim]")
            break
        else:
            console.print("[red]Invalid selection. Try again.[/red]")


if __name__ == "__main__":
    main()
