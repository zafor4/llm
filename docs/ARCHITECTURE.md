# System Architecture & Technical Design

## 1. Overview

The **Antigravity AI Platform** is a full-stack, production-grade autonomous intelligence platform combining:
1. **Frontend Layer**: A modular React 19 single-page application built with Vite and pure CSS tokens.
2. **Backend Services Layer**: An asynchronous FastAPI service exposing REST and streaming endpoints.
3. **AI Inference & Training Engine**: A decoder-only causal Transformer (`SmallGPT`) written in PyTorch with FlashAttention SDPA and BPE tokenization.
4. **Autonomous Agent Sandbox**: An execution runtime that provides tool calling, real-time code execution, and self-correcting unit testing loops.

```
┌─────────────────────────────────────────────────────────────┐
│                 Client Layer (Browser / SPA)                │
│             React 19 + Vite (Port: 3000 / Custom CSS)        │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON API / Static Proxy
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Gateway (Port: 8000)             │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Dashboard   │ Chat Studio  │ Coding Agent  │ Model Trainer │
│  Telemetry   │  Inference   │  Loop Runner  │ Worker Thread │
└──────┬───────┴──────┬───────┴───────┬───────┴───────┬───────┘
       │              │               │               │
┌──────▼──────┐┌──────▼───────┐┌──────▼──────┐┌──────▼────────┐
│ Hardware &  ││ PyTorch Causal││  Sandboxed  ││ Dataset & BPE │
│ GPU Monitors││ SmallGPT      ││ Environment ││ Optimization  │
│ (psutil)    ││ (CUDA/Torch)  ││ (Subprocess)││ (AdamW/Cosine)│
└─────────────┘└──────────────┘└─────────────┘└───────────────┘
```

---

## 2. Directory Structure

```
├── docs/                     # System design, API specifications, and deployment guides
│   ├── ARCHITECTURE.md       # High-level architecture and system design
│   ├── API_REFERENCE.md      # REST API endpoints and data schemas
│   └── DEPLOYMENT.md         # Production setup, Docker, and hardware configuration
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
│   │   ├── backends.py       # Pluggable planners (Oracle, Local LLM)
│   │   ├── loop.py           # Reason-Act-Verify state machine
│   │   └── state.py          # Execution traces and telemetry
│   ├── api/                  # FastAPI web services
│   │   ├── routers/          # Modular API routers
│   │   │   ├── agent.py      # Goal execution and task endpoints
│   │   │   ├── benchmarks.py # Benchmark metrics
│   │   │   ├── chat.py       # Conversational text generation
│   │   │   ├── dashboard.py  # System and GPU vitals
│   │   │   ├── models.py     # Checkpoint catalog and hot-swapping
│   │   │   ├── tests.py      # Automated pytest runner
│   │   │   └── trainer.py    # Background training thread and status
│   │   └── main.py           # FastAPI entrypoint
│   ├── data/                 # Tokenization and dataset pipelines
│   │   ├── dataset.py        # Causal LM dataset loader
│   │   └── tokenizer.py      # Byte-Pair Encoding & Character Tokenizers
│   ├── evaluation/           # Benchmarking harnesses and verification tasks
│   │   ├── benchmark.py      # Task definitions (TASK_01 to TASK_05)
│   │   └── evaluate_agents.py# Comparative evaluation runners
│   ├── model/                # Transformer architecture
│   │   ├── config.py         # Hyperparameter configurations
│   │   └── transformer.py    # Pre-LN causal Transformer with SDPA
│   ├── safety/               # Execution boundary and isolation
│   │   └── boundary.py       # Path containment and safe sandbox
│   ├── tools/                # Secure agent tool implementations
│   │   ├── base.py           # Tool interface
│   │   └── standard.py       # File, search, calculator, python tools
│   └── training/             # Model training pipelines
│       └── trainer.py        # PyTorch training loop with cosine warmup
├── tests/                    # 16 Comprehensive Unit Tests
│   ├── test_agent.py         # Autonomous loop validation
│   ├── test_model.py         # Transformer forward/backward checks
│   ├── test_tokenizer.py     # Tokenizer roundtrip checks
│   └── test_tools.py         # Sandbox tool execution checks
├── demo.py                   # Terminal CLI runner
├── pyproject.toml            # Build system and pytest configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Primary platform documentation
```

---

## 3. Core Subsystems

### 3.1 Causal Transformer (`SmallGPT`)
- **Pre-LayerNorm Architecture**: Normalized residual streams prevent gradient vanishing or explosion during deep training.
- **Scaled Dot-Product Attention (SDPA)**: Utilizes PyTorch 2.0+ `F.scaled_dot_product_attention` for memory-efficient FlashAttention kernels.
- **RoPE & Positional Embeddings**: Supports rotary position embeddings or learned absolute embeddings.
- **Weight Tying**: Shares weights between input token embeddings and final projection head, reducing total parameters by up to 25%.

### 3.2 Autonomous Coding Agent
- **Reason-Act-Verify Loop**:
  $$\text{Task Goal} \longrightarrow \text{Thought} \longrightarrow \text{Action} \longrightarrow \text{Execution} \longrightarrow \text{Observation} \longrightarrow \text{Self-Correction}$$
- **Path Confinement**: Prevents path traversal vulnerabilities (`..`) by resolving all file paths within a dedicated sandboxed workspace.
- **Automated Repair**: When tests fail with a non-zero exit code, the agent inspects the traceback, applies targeted code edits, and re-runs tests until verification succeeds.
