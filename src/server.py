"""Interactive Web Dashboard Server for SmallGPT and Autonomous Agent Research.

Provides REST API endpoints and serves modern web UI on http://localhost:8000.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.agent.backends import DeterministicOraclePlanner
from src.agent.loop import AgentControlLoop
from src.data.tokenizer import BPETokenizerWrapper
from src.evaluation.benchmark import get_all_benchmark_tasks
from src.evaluation.evaluate_agents import create_standard_registry, get_task_oracle_decisions
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.safety.boundary import WorkspaceSandbox

PORT = 8000
WEB_DIR = PROJECT_ROOT / "web"

# Cache loaded model in memory
LOADED_MODEL = None
LOADED_TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_small_gpt_model():
    global LOADED_MODEL, LOADED_TOKENIZER
    if LOADED_MODEL is None:
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "GPT-Mini-256" / "best_model.pt"
        tok_path = PROJECT_ROOT / "results" / "bpe_500.json"
        if ckpt_path.exists() and tok_path.exists():
            LOADED_TOKENIZER = BPETokenizerWrapper.load(tok_path)
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            config = GPTConfig.from_dict(checkpoint["config"])
            model = SmallGPT(config).to(DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            LOADED_MODEL = model
    return LOADED_MODEL, LOADED_TOKENIZER


class ResearchDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler dispatching APIs and serving static assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            self._handle_status()
        elif self.path == "/api/benchmarks":
            self._handle_benchmarks()
        elif self.path == "/api/experiments":
            self._handle_experiments()
        elif self.path == "/api/tasks":
            self._handle_tasks_list()
        elif self.path.startswith("/results/"):
            # Serve files from results/ directory (such as PNG charts)
            req_file = PROJECT_ROOT / self.path.lstrip("/")
            if req_file.exists() and req_file.is_file():
                self.send_response(200)
                if req_file.suffix == ".png":
                    self.send_header("Content-Type", "image/png")
                elif req_file.suffix == ".json":
                    self.send_header("Content-Type", "application/json")
                elif req_file.suffix == ".csv":
                    self.send_header("Content-Type", "text/csv")
                self.end_headers()
                with open(req_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if self.path == "/api/agent/run":
            self._handle_agent_run(payload)
        elif self.path == "/api/generate":
            self._handle_generate(payload)
        elif self.path == "/api/tests/run":
            self._handle_tests_run()
        else:
            self.send_error(404, "Unknown API endpoint")

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def _handle_status(self):
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram_mb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        )
        self._send_json(
            {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": gpu_name,
                "vram_total_mb": round(vram_mb, 1),
                "model_loaded": LOADED_MODEL is not None,
                "checkpoint_available": (PROJECT_ROOT / "results" / "checkpoints" / "GPT-Mini-256" / "best_model.pt").exists(),
            }
        )

    def _handle_tasks_list(self):
        tasks = get_all_benchmark_tasks()
        task_info = [
            {"task_id": t.task_id, "name": t.name, "category": t.category, "goal": t.goal}
            for t in tasks
        ]
        self._send_json(task_info)

    def _handle_benchmarks(self):
        csv_path = PROJECT_ROOT / "results" / "agent_benchmark_results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            self._send_json(df.to_dict(orient="records"))
        else:
            self._send_json([])

    def _handle_experiments(self):
        lm_path = PROJECT_ROOT / "results" / "lm_experiments.json"
        tok_path = PROJECT_ROOT / "results" / "tokenizer_experiments.json"
        lm_data = json.loads(lm_path.read_text(encoding="utf-8")) if lm_path.exists() else []
        tok_data = json.loads(tok_path.read_text(encoding="utf-8")) if tok_path.exists() else {}
        self._send_json({"lm_experiments": lm_data, "tokenizer_experiments": tok_data})

    def _handle_agent_run(self, payload: Dict[str, Any]):
        task_id = payload.get("task_id", "TASK_01")
        mode = payload.get("mode", "verified")

        tasks = {t.task_id: t for t in get_all_benchmark_tasks()}
        if task_id not in tasks:
            self._send_json({"error": f"Unknown task: {task_id}"}, status=400)
            return

        task = tasks[task_id]
        with tempfile.TemporaryDirectory(prefix="agent_web_") as tmp_dir:
            workspace = Path(tmp_dir)
            task.setup_fn(workspace)
            sandbox = WorkspaceSandbox(workspace_root=workspace, allow_destructive=True)
            tools = create_standard_registry(sandbox)

            decisions = get_task_oracle_decisions(task_id)
            planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])
            loop = AgentControlLoop(
                llm=planner,
                tools=tools,
                sandbox=sandbox,
                mode=mode,
                verifier=task.verify_fn if mode == "verified" else None,
            )

            state = loop.run_task(task.goal, max_steps=task.max_steps)
            actual_passed, verif_msg = task.verify_fn(state, sandbox)

            response = {
                "task_id": task.task_id,
                "task_name": task.name,
                "category": task.category,
                "mode": mode,
                "success": actual_passed,
                "current_step": state.current_step,
                "total_tool_calls": state.total_tool_calls,
                "failed_tool_calls": state.failed_tool_calls,
                "error_recovery_attempts": state.error_recovery_attempts,
                "latency_sec": round(state.total_latency_sec, 4),
                "verification_message": verif_msg,
                "history": [s.to_dict() for s in state.history],
                "final_report": state.final_report,
            }
            self._send_json(response)

    def _handle_generate(self, payload: Dict[str, Any]):
        prompt = payload.get("prompt", "def binary_search(arr: list[int], target: int) -> int:\n")
        temperature = float(payload.get("temperature", 0.7))
        max_new_tokens = int(payload.get("max_new_tokens", 48))
        top_p = float(payload.get("top_p", 0.9))

        model, tokenizer = get_small_gpt_model()
        if model is None or tokenizer is None:
            self._send_json({"error": "Model checkpoint not found. Run experiments first."}, status=500)
            return

        prompt_tokens = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

        start_t = time.time()
        with torch.no_grad():
            out_tokens = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elapsed = time.time() - start_t

        full_text = tokenizer.decode(out_tokens[0].tolist())
        continuation = full_text[len(prompt) :]

        self._send_json(
            {
                "prompt": prompt,
                "continuation": continuation,
                "full_text": full_text,
                "tokens_generated": max_new_tokens,
                "latency_sec": round(elapsed, 3),
                "tokens_per_sec": round(max_new_tokens / max(0.001, elapsed), 1),
            }
        )

    def _handle_tests_run(self):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--no-header"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            output = proc.stdout if proc.stdout else proc.stderr
            self._send_json({"success": proc.returncode == 0, "output": output})
        except Exception as e:
            self._send_json({"success": False, "output": str(e)}, status=500)


def run_server():
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    with socketserver.TCPServer(("", PORT), ResearchDashboardHandler) as httpd:
        print(f"\n=======================================================")
        print(f">> Research Dashboard running at: http://localhost:{PORT}")
        print(f"Open this URL in your web browser to interact with the project.")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
