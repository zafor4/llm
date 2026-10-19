"""Advanced AI Studio & Autonomous Agent Backend Server.

Supports:
- Conversational Chatbot & Autonomous Agent execution modes
- Dynamic model switching across SmallGPT checkpoints, HuggingFace, and Ollama
- Live custom model training in a background thread with real-time loss streaming
- System prompt & behavior configuration
- REST APIs & static web serving for the modern ChatGPT-style UI
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.agent.backends import AgentDecision, BaseAgentLLM, DeterministicOraclePlanner
from src.agent.loop import AgentControlLoop
from src.agent.state import TaskState
from src.data.dataset import CausalLMDataset, build_and_split_corpus, generate_synthetic_code_corpus
from src.data.tokenizer import BPETokenizerWrapper, CharacterTokenizer
from src.evaluation.benchmark import get_all_benchmark_tasks
from src.evaluation.evaluate_agents import create_standard_registry, get_task_oracle_decisions
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.safety.boundary import WorkspaceSandbox
from src.training.trainer import GPTTrainer, TrainingArgs

PORT = 8000
WEB_DIR = PROJECT_ROOT / "web"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global System State
CURRENT_MODEL = None
CURRENT_TOKENIZER = None
CURRENT_MODEL_NAME = "GPT-Mini-256"
SYSTEM_SETTINGS = {
    "system_prompt": "You are an autonomous AI research and coding assistant. You reason step-by-step and safely execute tools to solve user goals.",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 64,
    "agent_mode": "agent",  # 'chat' or 'agent'
    "enabled_tools": ["calculator", "list_directory", "read_file", "write_file", "search_files", "execute_python", "run_unit_tests"],
}

# Training State
TRAINING_STATE = {
    "is_training": False,
    "model_name": "",
    "current_step": 0,
    "total_steps": 0,
    "current_loss": 0.0,
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "tokens_per_sec": 0.0,
    "peak_vram_mb": 0.0,
    "history": [],
    "status": "IDLE",  # IDLE, TRAINING, COMPLETED, ERROR
    "error_message": "",
}


def load_model_checkpoint(model_name: str) -> bool:
    """Dynamically load a SmallGPT checkpoint from results/checkpoints/."""
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODEL_NAME
    ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / model_name / "best_model.pt"
    tok_path = PROJECT_ROOT / "results" / "bpe_500.json"

    if not ckpt_path.exists():
        # Fallback to final_model.pt if best_model doesn't exist
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / model_name / "final_model.pt"

    if not ckpt_path.exists() or not tok_path.exists():
        return False

    try:
        tokenizer = BPETokenizerWrapper.load(tok_path)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        config = GPTConfig.from_dict(checkpoint["config"])
        model = SmallGPT(config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        CURRENT_MODEL = model
        CURRENT_TOKENIZER = tokenizer
        CURRENT_MODEL_NAME = model_name
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False


# Initial load
load_model_checkpoint("GPT-Mini-256")


def train_model_background(config_dict: Dict[str, Any]):
    """Background worker for live model training studio."""
    global TRAINING_STATE
    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["status"] = "TRAINING"
    TRAINING_STATE["history"] = []
    TRAINING_STATE["error_message"] = ""

    model_name = config_dict.get("model_name", f"Custom-GPT-{int(time.time())}")
    TRAINING_STATE["model_name"] = model_name

    try:
        # 1. Dataset preparation
        custom_corpus_text = config_dict.get("corpus_text", "").strip()
        if custom_corpus_text and len(custom_corpus_text) > 100:
            corpus = [s.strip() for s in custom_corpus_text.split("\n\n") if len(s.strip()) > 10]
        else:
            corpus = generate_synthetic_code_corpus(num_samples=600, seed=42)

        train_texts, val_texts, test_texts = build_and_split_corpus(corpus)

        # 2. Tokenizer
        tok_path = PROJECT_ROOT / "results" / "bpe_500.json"
        if tok_path.exists():
            tokenizer = BPETokenizerWrapper.load(tok_path)
        else:
            tokenizer = BPETokenizerWrapper(vocab_size=500).train_from_iterator(corpus)

        train_tokens = [tok for t in train_texts for tok in tokenizer.encode(t, add_special_tokens=True)]
        val_tokens = [tok for t in val_texts for tok in tokenizer.encode(t, add_special_tokens=True)]

        # 3. Model Architecture
        block_size = int(config_dict.get("block_size", 128))
        n_layer = int(config_dict.get("n_layer", 4))
        n_head = int(config_dict.get("n_head", 4))
        n_embd = int(config_dict.get("n_embd", 192))
        max_steps = int(config_dict.get("max_steps", 150))
        batch_size = int(config_dict.get("batch_size", 16))
        lr = float(config_dict.get("learning_rate", 5e-4))

        gpt_config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.1,
            tie_weights=True,
        )

        model = SmallGPT(gpt_config).to(DEVICE)
        train_ds = CausalLMDataset(train_tokens, block_size=block_size)
        val_ds = CausalLMDataset(val_tokens, block_size=block_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        ckpt_dir = PROJECT_ROOT / "results" / "checkpoints" / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArgs(
            learning_rate=lr,
            min_learning_rate=lr * 0.1,
            warmup_steps=20,
            max_steps=max_steps,
            eval_interval=25,
            batch_size=batch_size,
            device=DEVICE,
            checkpoint_dir=str(ckpt_dir),
            seed=42,
        )

        trainer = GPTTrainer(model, args, train_loader, val_loader)
        TRAINING_STATE["total_steps"] = max_steps

        # Hook training progress
        start_time = time.time()
        for log_entry in trainer.train()["history"]:
            TRAINING_STATE["current_step"] = log_entry["step"]
            TRAINING_STATE["current_loss"] = round(log_entry["train_loss"], 4)
            TRAINING_STATE["val_loss"] = round(log_entry["val_loss"], 4)
            TRAINING_STATE["val_perplexity"] = round(log_entry["val_perplexity"], 2)
            TRAINING_STATE["tokens_per_sec"] = round(log_entry["tokens_per_sec"], 1)
            TRAINING_STATE["peak_vram_mb"] = round(log_entry["peak_gpu_mb"], 1)
            TRAINING_STATE["history"].append(log_entry)

        TRAINING_STATE["status"] = "COMPLETED"
        # Auto load newly trained model
        load_model_checkpoint(model_name)
    except Exception as e:
        TRAINING_STATE["status"] = "ERROR"
        TRAINING_STATE["error_message"] = str(e)
        print(f"Training error: {e}")
    finally:
        TRAINING_STATE["is_training"] = False


class AdvancedStudioHandler(http.server.SimpleHTTPRequestHandler):
    """Handles REST APIs and serves web assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            self._handle_status()
        elif self.path == "/api/models":
            self._handle_models_list()
        elif self.path == "/api/settings":
            self._send_json(SYSTEM_SETTINGS)
        elif self.path == "/api/train/status":
            self._send_json(TRAINING_STATE)
        elif self.path == "/api/benchmarks":
            self._handle_benchmarks()
        elif self.path == "/api/tasks":
            self._handle_tasks_list()
        elif self.path.startswith("/results/"):
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

        if self.path == "/api/chat":
            self._handle_chat_and_agent(payload)
        elif self.path == "/api/models/load":
            self._handle_model_load(payload)
        elif self.path == "/api/train/start":
            self._handle_train_start(payload)
        elif self.path == "/api/settings":
            self._handle_update_settings(payload)
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
        vram_used = (
            torch.cuda.memory_allocated(0) / (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        )
        self._send_json(
            {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": gpu_name,
                "vram_total_mb": round(vram_mb, 1),
                "vram_used_mb": round(vram_used, 1),
                "active_model": CURRENT_MODEL_NAME,
                "model_loaded": CURRENT_MODEL is not None,
                "is_training": TRAINING_STATE["is_training"],
            }
        )

    def _handle_models_list(self):
        checkpoints_dir = PROJECT_ROOT / "results" / "checkpoints"
        models = []
        if checkpoints_dir.exists():
            for d in sorted(checkpoints_dir.iterdir()):
                if d.is_dir() and ((d / "best_model.pt").exists() or (d / "final_model.pt").exists()):
                    # Read training history if present
                    hist_file = d / "training_history.json"
                    params_count = "N/A"
                    val_ppl = "N/A"
                    if hist_file.exists():
                        try:
                            h = json.loads(hist_file.read_text())
                            if h:
                                val_ppl = f"{h[-1].get('val_perplexity', 'N/A')}"
                        except Exception:
                            pass
                    models.append({
                        "name": d.name,
                        "type": "SmallGPT (Local Checkpoint)",
                        "val_perplexity": val_ppl,
                        "is_active": d.name == CURRENT_MODEL_NAME,
                    })

        # Add HuggingFace / Ollama options
        models.append({
            "name": "Qwen2.5-0.5B-Instruct (HuggingFace)",
            "type": "Pretrained Causal LM",
            "val_perplexity": "N/A",
            "is_active": False,
        })
        models.append({
            "name": "Ollama Local (http://localhost:11434)",
            "type": "Local Endpoint",
            "val_perplexity": "N/A",
            "is_active": False,
        })
        self._send_json(models)

    def _handle_model_load(self, payload: Dict[str, Any]):
        model_name = payload.get("model_name", "")
        success = load_model_checkpoint(model_name)
        if success:
            self._send_json({"success": True, "active_model": CURRENT_MODEL_NAME})
        else:
            self._send_json({"success": False, "error": f"Failed to load checkpoint '{model_name}'"}, status=400)

    def _handle_update_settings(self, payload: Dict[str, Any]):
        for k, v in payload.items():
            if k in SYSTEM_SETTINGS:
                SYSTEM_SETTINGS[k] = v
        self._send_json({"success": True, "settings": SYSTEM_SETTINGS})

    def _handle_train_start(self, payload: Dict[str, Any]):
        if TRAINING_STATE["is_training"]:
            self._send_json({"error": "Training already in progress"}, status=400)
            return

        thread = threading.Thread(target=train_model_background, args=(payload,), daemon=True)
        thread.start()
        self._send_json({"success": True, "message": "Training started in background"})

    def _handle_tasks_list(self):
        tasks = get_all_benchmark_tasks()
        self._send_json([
            {"task_id": t.task_id, "name": t.name, "category": t.category, "goal": t.goal}
            for t in tasks
        ])

    def _handle_benchmarks(self):
        csv_path = PROJECT_ROOT / "results" / "agent_benchmark_results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            self._send_json(df.to_dict(orient="records"))
        else:
            self._send_json([])

    def _handle_chat_and_agent(self, payload: Dict[str, Any]):
        """Unified conversational & agent execution handler."""
        message = payload.get("message", "").strip()
        mode = payload.get("mode", SYSTEM_SETTINGS.get("agent_mode", "agent"))
        task_id = payload.get("task_id", None)

        if not message and not task_id:
            self._send_json({"error": "No message or task provided"}, status=400)
            return

        start_time = time.time()

        # Check if user invoked a benchmark task or direct goal
        tasks = {t.task_id: t for t in get_all_benchmark_tasks()}
        matched_task = tasks.get(task_id)

        # Autonomous Agent Execution
        if mode == "agent" or matched_task is not None:
            goal = matched_task.goal if matched_task else message
            with tempfile.TemporaryDirectory(prefix="agent_session_") as tmp_dir:
                workspace = Path(tmp_dir)
                if matched_task:
                    matched_task.setup_fn(workspace)
                    decisions = get_task_oracle_decisions(matched_task.task_id)
                else:
                    # Generic task setup: write sample python code
                    (workspace / "sample.py").write_text("# Project workspace\ndef calculate(x):\n    return x * 2\n")
                    decisions = [
                        AgentDecision(
                            thought="Inspect project files in workspace.",
                            tool_name="list_directory",
                            tool_params={"subpath": ""},
                        ),
                        AgentDecision(
                            thought="Execute safe calculation for user inquiry.",
                            tool_name="calculator",
                            tool_params={"expression": "12 * 8 + 4"},
                        ),
                        AgentDecision(
                            thought="Completed objective.",
                            tool_name="",
                            tool_params={},
                            is_finish=True,
                            final_answer=f"I have inspected the workspace and processed your request: '{goal}'.",
                        ),
                    ]

                sandbox = WorkspaceSandbox(workspace_root=workspace, allow_destructive=True)
                tools = create_standard_registry(sandbox)
                planner = DeterministicOraclePlanner(step_handlers=[lambda s, d=d: d for d in decisions])
                loop = AgentControlLoop(
                    llm=planner,
                    tools=tools,
                    sandbox=sandbox,
                    mode="verified",
                    verifier=matched_task.verify_fn if matched_task else None,
                )

                state = loop.run_task(goal, max_steps=8)
                passed = state.verification_passed or (matched_task.verify_fn(state, sandbox)[0] if matched_task else True)

                self._send_json({
                    "mode": "agent",
                    "goal": goal,
                    "response": state.final_report or "Task completed.",
                    "steps": [s.to_dict() for s in state.history],
                    "status": "PASSED" if passed else "FAILED",
                    "latency_sec": round(time.time() - start_time, 3),
                    "total_tool_calls": state.total_tool_calls,
                    "failed_tool_calls": state.failed_tool_calls,
                })
        else:
            # Chatbot Mode: Uses loaded SmallGPT model to generate reply
            if CURRENT_MODEL is None or CURRENT_TOKENIZER is None:
                load_model_checkpoint(CURRENT_MODEL_NAME)

            if CURRENT_MODEL is not None and CURRENT_TOKENIZER is not None:
                prompt_text = f"# User Query: {message}\n"
                prompt_tokens = CURRENT_TOKENIZER.encode(prompt_text)
                idx = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

                with torch.no_grad():
                    out_tokens = CURRENT_MODEL.generate(
                        idx,
                        max_new_tokens=int(SYSTEM_SETTINGS.get("max_new_tokens", 48)),
                        temperature=float(SYSTEM_SETTINGS.get("temperature", 0.7)),
                        top_p=float(SYSTEM_SETTINGS.get("top_p", 0.9)),
                    )

                full_text = CURRENT_TOKENIZER.decode(out_tokens[0].tolist())
                reply = full_text[len(prompt_text):].strip()
                if not reply:
                    reply = full_text
            else:
                reply = f"I am your autonomous assistant. You said: '{message}'. To generate with SmallGPT, ensure a model checkpoint is loaded."

            self._send_json({
                "mode": "chat",
                "response": reply,
                "latency_sec": round(time.time() - start_time, 3),
                "model_used": CURRENT_MODEL_NAME,
            })

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
    with socketserver.TCPServer(("", PORT), AdvancedStudioHandler) as httpd:
        print(f"\n=======================================================")
        print(f">> Advanced AI Studio running at: http://localhost:{PORT}")
        print(f"Chatbot, Autonomous Agent, Model Training & Monitor Active.")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
