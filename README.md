# Antigravity AI Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg?logo=vite)](https://vite.dev/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-10b981.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-ready full-stack autonomous AI platform combining **custom compact causal Transformers (`SmallGPT`)**, an **autonomous SWE-bench-style coding agent** with self-correcting sandboxed execution, a **conversational ChatGPT-style studio**, and a **live GPU model training studio** — managed from a unified mission control dashboard.

---

## ⚡ Core Architecture

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │            FRONTEND: React 19 + Vite (Modern Dark Glassmorphism)       │
   │           Port: 3000 (Zero-CORS Proxied, Responsive Design)            │
   └───────┬──────────────┬──────────────┬──────────────┬────────────┬──────┘
           │              │              │              │            │
  ┌────────▼──────┐┌──────▼──────┐┌──────▼──────┐┌──────▼────┐┌──────▼───────┐
  │ Platform Hub  ││ChatbotStudio││ Coding Agent││ Training  ││Models Fleet  │
  │  (Dashboard)  ││(ChatGPT-UI) ││ (SWE-bench) ││  Studio   ││  & Gateways  │
  └────────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬────┘└──────┬───────┘
           │              │              │              │            │
   ────────┴──────────────┴──────────────┴──────────────┴────────────┴────────
           ▲ REST / SSE / Telemetry Stream (JSON API)
   ────────┬──────────────────────────────────────────────────────────────────
           │
   ┌───────▼────────────────────────────────────────────────────────────────┐
   │                 BACKEND: FastAPI + Uvicorn Async Gateway               │
   │                 Port: 8000 (Modular Routers, Sandboxed Runner)         │
   └───────┬─────────────────────────────┬───────────────────────────┬──────┘
           │                             │                           │
  ┌────────▼──────────────┐   ┌──────────▼────────────┐   ┌──────────▼───────┐
  │  SmallGPT Engine      │   │  Workspace Sandbox    │   │ Background Train │
  │  Pre-LN + FlashAttn   │   │  Self-Correcting Loop │   │ Thread Telemetry │
  │  NVIDIA RTX 3050 CUDA │   │  Safe Execution Tools │   │ Loss / Perplexity│
  └───────────────────────┘   └───────────────────────┘   └──────────────────┘
```

---

## 🚀 Key Applications

### 1. 🎛️ Platform Control Hub (`Platform Hub`)
* **Real-time Vitals:** Live hardware monitoring tracking dedicated GPU VRAM (NVIDIA RTX 3050, 6GB), CPU utilization, and System RAM.
* **System Health:** Continuous monitoring of background training workers and active model checkpoints.
* **1-Click Launchers:** Instant access to all sub-applications.

### 2. 💬 Conversational Chatbot Studio
* **Interactive UI:** Clean ChatGPT-style conversational assistant.
* **Model Hot-Swapping:** Select between active checkpoints (`GPT-Micro-128`, `GPT-Mini-256`, `GPT-Small-256`, `GPT-Mini-512`).
* **Generation Controls:** Live temperature and max token sliders with pre-configured developer prompts.

### 3. 🤖 Autonomous Coding Agent
* **Reason-Act-Verify Loop:** Executes multi-step software goals in a secure, isolated workspace.
* **Tool Invocation:** Built-in sandboxed tools (`list_directory`, `read_file`, `write_file`, `run_unit_tests`, `run_python`).
* **Automated Self-Correction:** When tests fail, the agent reads the traceback, discovers the root cause, modifies the file, and verifies the fix.

### 4. 🔥 Model Training Studio
* **Train from Scratch:** Train custom causal GPT models directly on your GPU from the browser.
* **Hyperparameter Configurator:** Interactive controls for layers (2–12), heads (2–16), embedding dimension (64–1024), learning rate, and batch size.
* **Live Telemetry:** Real-time polling tracking training loss, validation perplexity, throughput (tokens/sec), and peak VRAM.

### 5. ⚡ Models Fleet & External Gateways
* **Checkpoint Inventory:** View model parameters, vocab size, and disk usage.
* **1-Click VRAM Swapping:** Instantly load any checkpoint into GPU memory.
* **Provider Gateways:** Ready-to-connect bridges for Ollama (`localhost:11434`), Hugging Face Hub, and standard OpenAI `/v1/chat/completions` API compatibility.

### 6. 📊 System Benchmarks & Diagnostics
* **Throughput & Latency:** Quantitative model benchmarks across scales.
* **Automated Test Suite:** 1-click execution of the entire 16-test unit testing suite directly via the UI.

---

## 🛠️ Getting Started

### Prerequisites
* **Python**: 3.10+ (Recommended: 3.11)
* **Node.js**: v18+ (Recommended: v20 or v22)
* **NVIDIA GPU**: RTX 3050 or higher with CUDA (or CPU fallback)

### Quickstart

1. **Clone & Setup Environment:**
   ```bash
   git clone <repo-url>
   cd llm
   python -m venv .venv
   .\.venv\Scripts\activate       # On Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Setup Frontend:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Launch Backend (Port 8000):**
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```

4. **Launch Frontend (Port 3000):**
   ```bash
   cd frontend
   npm run dev
   ```

Open your browser at **`http://localhost:3000`** to access the complete platform.
Interactive API documentation is available at **`http://127.0.0.1:8000/docs`**.

---

## 🧪 Running Unit Tests

Run the full automated test suite (16 tests across Model, Tokenizer, Sandbox, and Agent):

```bash
pytest tests/ -v
```

---

## 📁 Project Directory Structure

```
├── docs/                     # Production documentation
│   ├── ARCHITECTURE.md       # High-level architecture and system design
│   ├── API_REFERENCE.md      # REST API endpoints and data schemas
│   └── DEPLOYMENT.md         # Production setup and operational guide
├── frontend/                 # React 19 + Vite Web Application
│   ├── src/
│   │   ├── components/       # Dedicated Sub-App components
│   │   │   ├── Sidebar.jsx           # App switcher navigation
│   │   │   ├── DashboardApp.jsx      # Mission Control & GPU vitals
│   │   │   ├── ChatbotApp.jsx        # Conversational assistant
│   │   │   ├── CodingAgentApp.jsx    # Autonomous SWE-bench agent
│   │   │   ├── TrainingStudioApp.jsx # Live model trainer
│   │   │   ├── ModelsFleetApp.jsx    # Checkpoints & external providers
│   │   │   └── BenchmarksApp.jsx     # System benchmarks & unit tests
│   │   ├── App.jsx           # Root layout and global state
│   │   └── index.css         # Production dark glassmorphism design tokens
│   └── vite.config.js        # Port 3000 config with proxy to Port 8000
├── results/                  # Checkpoints, tokenizers, and benchmark data
│   ├── checkpoints/          # Saved model weights (.pt)
│   ├── bpe_2048.json         # High-compression BPE tokenizer
│   └── agent_benchmark_results.csv # Agent evaluation data
├── src/                      # Core Python backend and AI engine
│   ├── agent/                # Autonomous Agent control loop and planning
│   ├── api/                  # FastAPI web services and routers
│   ├── data/                 # Tokenization and dataset pipelines
│   ├── evaluation/           # Benchmarking tasks and verification
│   ├── model/                # Transformer architecture (Pre-LN + FlashAttn)
│   ├── safety/               # Execution boundary and isolation
│   ├── tools/                # Secure agent tool implementations
│   └── training/             # Model training pipelines
├── tests/                    # 16 Comprehensive Unit Tests
├── demo.py                   # Terminal CLI runner
├── pyproject.toml            # Build system and pytest configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Primary platform documentation
```

---

## 📄 License

This project is licensed under the MIT License.
