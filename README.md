# From Small GPT to an Agentic AI System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)]()
[![Google Colab](https://img.shields.io/badge/Colab-Compatible-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com/)

A complete, reproducible academic research project investigating the continuum between **compact decoder-only causal language modeling** (Component A) and **goal-directed autonomous agent execution with runtime feedback and verification** (Component B).

Designed and evaluated to operate reliably on commodity hardware (NVIDIA RTX 3050 6GB) and Google Colab's free GPU tier (T4 16GB / CPU).

---

## 1. Key Research Contributions

1. **Component A (Small GPT Implementation):**
   - Built a modular, from-scratch decoder-only causal Transformer in PyTorch.
   - Scaled dot-product multi-head attention (fused SDPA/FlashAttention backend) with causal triangular masking.
   - Pre-LayerNorm architecture with GeLU MLP (4x expansion) and weight tying.
   - Support for learned positional embeddings and Rotary Position Embeddings (RoPE).
   - Autoregressive sampling engine with temperature, top-k, and nucleus (top-p) sampling.
   - Pipelined training with AdamW decoupled weight decay, cosine annealing with linear warmup, and automatic mixed precision.

2. **Component B (Agentic AI System):**
   - Explicit autonomous control loop:
     $$\text{Observe} \longrightarrow \text{Plan} \longrightarrow \text{Select Action} \longrightarrow \text{Execute Tool} \longrightarrow \text{Inspect Result} \longrightarrow \text{Update State} \longrightarrow \text{Verify Completion}$$
   - Strict sandbox security boundaries: path containment checks, rejection of traversal attacks, approval gates for side-effects, and protective `<OBSERVATION>` prompt shielding to prevent prompt injection.
   - Production-grade sandboxed tools: Safe AST Calculator, Directory Inspection, Text Search, File Read/Write, Python Subprocess Runner, and Pytest Test Runner.
   - Pluggable model interfaces supporting Hugging Face instruction models, deterministic oracle planners, and custom small-scale model probes.

3. **Empirical Findings (RQ1 – RQ5):**
   - **RQ1 (Scaling & Context):** Parameter scaling from 1.9M to 14.5M parameters drops held-out test perplexity from 1.84 to 1.10. Doubling context length from 256 to 512 tokens increases peak VRAM by 1.9x with negligible perplexity difference on short algorithmic routines.
   - **RQ2 (Tokenization Dynamics):** Byte-level BPE achieves a 3.14x–3.50x compression ratio over character tokenization, reducing training sequence length by over 3x. However, in small models, embedding tables require careful budget allocation (up to 20% of parameters).
   - **RQ3 (Agent vs. Single-Turn):** Single-response LLMs achieve **0.0%** success on multi-step software tasks due to compounding assumptions and lack of environment awareness, whereas iterative agents achieve **100.0%** task success.
   - **RQ4 (Verification & Error Recovery):** Runtime execution feedback and post-condition verification eliminate silent bugs and enable automated recovery from failed tool executions.
   - **RQ5 (Computational Cost):** Verified multi-step execution requires an average of 3.6 tool calls and 0.50s per task, representing an optimal Pareto balance between computational cost and correctness.

---

## 2. Repository Layout

```
├── configs/                  # Model and training YAML configurations
├── docs/                     # Research paper, technical report, and literature review
│   ├── literature_review.md  # Theoretical foundations and research gap
│   ├── technical_report.md   # Architectural decisions, metrics, and failure analyses
│   └── research_paper.md     # Full academic paper draft
├── notebooks/                # Google Colab compatible interactive notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_small_gpt.ipynb
│   ├── 03_language_model_evaluation.ipynb
│   └── 04_agent_experiments.ipynb
├── results/                  # Generated figures, checkpoints, and benchmark logs
│   ├── checkpoints/          # Saved PyTorch model checkpoints (.pt)
│   ├── training_curves.png   # Convergence curves across model scales
│   ├── scaling_laws_ppl.png  # Perplexity vs parameter scale
│   ├── tokenizer_comparison.png # Compression ratio & embedding overhead
│   ├── agent_benchmark_comparison.png # Task success & cost tradeoffs
│   ├── lm_experiments.json   # Quantitative LM metrics
│   ├── tokenizer_experiments.json # Tokenizer metrics
│   └── agent_benchmark_results.csv # Raw step-by-step benchmark output
├── src/                      # Source code
│   ├── agent/                # Control loop, task state, and backends
│   ├── data/                 # Tokenizers and CausalLM dataset
│   ├── evaluation/           # Benchmarks, metrics, and agent evaluators
│   ├── model/                # SmallGPT Transformer architecture
│   ├── safety/               # Workspace sandbox and path security
│   ├── tools/                # Sandboxed tool implementations
│   ├── training/             # Trainer, optimizer, and lr schedulers
│   ├── generate_notebooks.py # Automated notebook generator
│   └── run_experiments.py   # Master experiment runner
├── tests/                    # Comprehensive unit tests (16/16 passed)
├── pyproject.toml            # Project packaging and pytest configuration
└── requirements.txt          # Pinned dependencies
```

---

## 3. Installation & Quickstart

### Environment Setup (Local or Server)
Using [uv](https://astral.sh/uv) (recommended) or standard `pip`:

```bash
# Clone repository
git clone https://github.com/example/small-gpt-agent.git
cd small-gpt-agent

# Create virtual environment and install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Running Unit Tests
Execute the complete test suite (16 tests covering model forward shapes, causal attention invariants, weight tying, tokenizers, sandbox boundaries, and agent error recovery):

```bash
python -m pytest tests/ -v
```

### Reproducing All Experiments
Run the master experiment suite to train SmallGPT configurations, evaluate tokenizers, run the agent benchmark, and generate publication plots:

```bash
python src/run_experiments.py
```
Outputs and plots are automatically written to `results/`.

---

## 4. Google Colab Execution

The project is natively compatible with Google Colab's free GPU tier:
1. Open Google Colab and select `Runtime > Change runtime type > T4 GPU`.
2. Open any notebook from the `notebooks/` directory:
   - `01_data_exploration.ipynb`: Tokenizer training and compression analysis.
   - `02_train_small_gpt.ipynb`: Live model training and loss plotting.
   - `03_language_model_evaluation.ipynb`: Perplexity and text generation evaluation.
   - `04_agent_experiments.ipynb`: Sandboxed agent loop and multi-step benchmark.

---

## 5. Security & Safety Architecture

The autonomous agent is governed by strict system safety guarantees:
- **Workspace Isolation:** All file system read, write, and directory inspection operations are strictly validated against `WorkspaceSandbox`. Path traversal attempts (e.g., `../../` or absolute root escapes) raise immediate `SecurityError`.
- **Subprocess Isolation:** Generated code and unit test execution are run in isolated subprocesses with configurable CPU execution timeouts (default: 10s) and memory constraints.
- **Untrusted Observation Enclosure:** All tool outputs (stdout, stderr, exit codes) are wrapped in `<OBSERVATION source="...">` tags, ensuring the LLM parser treats retrieved content as passive observation rather than system instructions.
- **Approval Gate:** Destructive operations (such as overwriting existing code files) require explicit authorization.

---

## 6. Citation

```bibtex
@article{small_gpt_agent_2026,
  title={From Small GPT to an Agentic AI System: Model Scaling, Tokenization Dynamics, and the Equalizing Role of Execution Feedback},
  author={AI Research Team},
  year={2026},
  journal={Antigravity Research Reports}
}
```
