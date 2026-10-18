"""Generates self-contained, reproducible Google Colab compatible Jupyter notebooks."""

from __future__ import annotations

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def code_cell(source):
    if isinstance(source, list):
        src = [line + "\n" for line in source[:-1]] + [source[-1]]
    else:
        src = [line + "\n" for line in source.strip().split("\n")]
        if src:
            src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def markdown_cell(source):
    if isinstance(source, list):
        src = [line + "\n" for line in source[:-1]] + [source[-1]]
    else:
        src = [line + "\n" for line in source.strip().split("\n")]
        if src:
            src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def generate_all_notebooks(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Notebook 01: Data Exploration & Tokenization
    # ----------------------------------------------------
    nb1_cells = [
        markdown_cell(
            """# 01. Data Exploration, Curation & Tokenization Dynamics
**Project:** From Small GPT to an Agentic AI System  
**Objective:** Curate an algorithmic Python corpus, analyze sequence length distributions, and compare Character vs. BPE tokenization compression ratios (RQ2). Compatible with Google Colab (free GPU/CPU)."""
        ),
        code_cell(
            """# Setup & Environment Check
!pip install -q tokenizers transformers matplotlib seaborn
import os
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")"""
        ),
        markdown_cell("## 1. Corpus Generation\nWe generate and inspect clean algorithmic Python functions covering sorting, search, graph traversal, and dynamic programming."),
        code_cell(
            """import sys
from pathlib import Path
if not Path("src").exists():
    !git clone https://github.com/example/small-gpt-agent.git . 2>/dev/null || true

from src.data.dataset import generate_synthetic_code_corpus, build_and_split_corpus
corpus = generate_synthetic_code_corpus(num_samples=1000, seed=42)
train_set, val_set, test_set = build_and_split_corpus(corpus, train_ratio=0.85, val_ratio=0.075)

print(f"Dataset split: Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")
print("\\n--- Sample Corpus Snippet ---")
print(corpus[0][:300] + "\\n...")"""
        ),
        markdown_cell("## 2. Tokenizer Comparison: Character vs. BPE\nWe train Character and Byte-Level BPE tokenizers to measure compression efficiency."),
        code_cell(
            """from src.data.tokenizer import CharacterTokenizer, BPETokenizerWrapper

# 1. Character Tokenizer
char_tok = CharacterTokenizer().fit(corpus)

# 2. BPE Tokenizers
bpe_500 = BPETokenizerWrapper(vocab_size=500).train_from_iterator(corpus)
bpe_2048 = BPETokenizerWrapper(vocab_size=2048).train_from_iterator(corpus)

sample_doc = "\\n".join(test_set[:20])
raw_bytes = len(sample_doc.encode('utf-8'))

char_tokens = char_tok.encode(sample_doc)
bpe_500_tokens = bpe_500.encode(sample_doc)
bpe_2048_tokens = bpe_2048.encode(sample_doc)

print(f"Raw Byte Size: {raw_bytes} bytes")
print(f"Char Tokens:   {len(char_tokens):<6} | Compression: {raw_bytes/len(char_tokens):.2f} bytes/tok | Vocab: {char_tok.vocab_size}")
print(f"BPE-500:       {len(bpe_500_tokens):<6} | Compression: {raw_bytes/len(bpe_500_tokens):.2f} bytes/tok | Vocab: {bpe_500.vocab_size}")
print(f"BPE-2048:      {len(bpe_2048_tokens):<6} | Compression: {raw_bytes/len(bpe_2048_tokens):.2f} bytes/tok | Vocab: {bpe_2048.vocab_size}")"""
        ),
        markdown_cell("## 3. Visualizing Compression & Sequence Length Reduction"),
        code_cell(
            """import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
tok_names = ["Character", "BPE-500", "BPE-2048"]
ratios = [raw_bytes/len(char_tokens), raw_bytes/len(bpe_500_tokens), raw_bytes/len(bpe_2048_tokens)]
ax.bar(tok_names, ratios, color=["#4A90E2", "#50E3C2", "#B8E986"], edgecolor="black")
ax.set_ylabel("Compression Ratio (Bytes / Token)")
ax.set_title("Tokenizer Compression Efficiency on Python Code (RQ2)")
ax.grid(True, linestyle="--", alpha=0.6)
plt.show()"""
        ),
    ]
    with open(dest_dir / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 02: Train Small GPT
    # ----------------------------------------------------
    nb2_cells = [
        markdown_cell(
            """# 02. Small GPT Training Pipeline & Optimization
**Project:** From Small GPT to an Agentic AI System  
**Objective:** Implement a compact causal Transformer from scratch in PyTorch, train with Cosine LR scheduling and mixed precision on Colab GPU, and save checkpoints."""
        ),
        code_cell(
            """# Setup & Dependencies
!pip install -q tokenizers transformers accelerate matplotlib
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")"""
        ),
        markdown_cell("## 1. Model Configuration & Architecture"),
        code_cell(
            """from src.model.config import GPTConfig
from src.model.transformer import SmallGPT

# Configure a 6.2M parameter GPT-Mini
config = GPTConfig(
    vocab_size=500,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=288,
    dropout=0.1,
    tie_weights=True
)
model = SmallGPT(config).to(device)
print(f"Initialized SmallGPT with {model.get_num_params():,} non-embedding parameters.")"""
        ),
        markdown_cell("## 2. Dataset Preparation & Dataloaders"),
        code_cell(
            """from src.data.dataset import generate_synthetic_code_corpus, build_and_split_corpus, CausalLMDataset
from src.data.tokenizer import BPETokenizerWrapper
from torch.utils.data import DataLoader

corpus = generate_synthetic_code_corpus(num_samples=1000, seed=42)
train_set, val_set, test_set = build_and_split_corpus(corpus)
tokenizer = BPETokenizerWrapper(vocab_size=500).train_from_iterator(corpus)

train_tokens = [tok for t in train_set for tok in tokenizer.encode(t, add_special_tokens=True)]
val_tokens = [tok for t in val_set for tok in tokenizer.encode(t, add_special_tokens=True)]

train_loader = DataLoader(CausalLMDataset(train_tokens, block_size=config.block_size), batch_size=16, shuffle=True)
val_loader = DataLoader(CausalLMDataset(val_tokens, block_size=config.block_size), batch_size=16, shuffle=False)
print(f"Train chunks: {len(train_loader)} | Val chunks: {len(val_loader)}")"""
        ),
        markdown_cell("## 3. Training Loop with Mixed Precision & Cosine Warmup"),
        code_cell(
            """from src.training.trainer import GPTTrainer, TrainingArgs

args = TrainingArgs(
    learning_rate=5e-4,
    min_learning_rate=5e-5,
    warmup_steps=30,
    max_steps=200,
    eval_interval=50,
    batch_size=16,
    device=device,
    checkpoint_dir="checkpoints/colab_run"
)
trainer = GPTTrainer(model, args, train_loader, val_loader)
summary = trainer.train()"""
        ),
        markdown_cell("## 4. Plotting Loss Convergence"),
        code_cell(
            """import matplotlib.pyplot as plt
history = summary["history"]
steps = [h["step"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]

plt.figure(figsize=(8, 4))
plt.plot(steps, train_losses, label="Train Loss", linestyle="--")
plt.plot(steps, val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("SmallGPT Training Loss Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()"""
        ),
    ]
    with open(dest_dir / "02_train_small_gpt.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 03: Language Model Evaluation
    # ----------------------------------------------------
    nb3_cells = [
        markdown_cell(
            """# 03. Language Model Evaluation & Scaling Analysis
**Project:** From Small GPT to an Agentic AI System  
**Objective:** Evaluate held-out perplexity, context window scaling (RQ1), and generation syntax quality across trained model checkpoints."""
        ),
        code_cell(
            """# Load Results & Checkpoints
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("results/lm_experiments.json", "r") as f:
    lm_results = json.load(f)

df = pd.DataFrame(lm_results)
df[["model_name", "total_params", "block_size", "test_perplexity", "syntax_validity_rate", "tokens_per_sec"]]"""
        ),
        markdown_cell("## 1. Perplexity vs. Parameter Scale (RQ1)"),
        code_cell(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(df["total_params"]/1e6, df["test_perplexity"], marker="o", color="#4A90E2", linewidth=2)
ax1.set_xlabel("Parameters (Millions)")
ax1.set_ylabel("Held-Out Test Perplexity")
ax1.set_title("Scaling Laws: Perplexity vs. Scale")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.bar(df["model_name"], df["tokens_per_sec"], color="#50E3C2", edgecolor="black")
ax2.set_ylabel("Throughput (Tokens / Sec)")
ax2.set_title("Training Throughput across Configurations")
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""
        ),
        markdown_cell("## 2. Qualitative Autoregressive Code Generation"),
        code_cell(
            """# Inspect Generated Completions from Checkpoints
for model_entry in lm_results:
    print(f"\\n=== {model_entry['model_name']} Generation Samples ===")
    for sample in model_entry["sample_completions"][:2]:
        print(f"Prompt: {sample['prompt'].strip()}")
        print(f"Continuation: {sample['continuation'].strip()[:150]}")
        print(f"Syntax Valid: {sample['is_syntax_valid']}")"""
        ),
    ]
    with open(dest_dir / "03_language_model_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 04: Agent Experiments & Verification Benchmark
    # ----------------------------------------------------
    nb4_cells = [
        markdown_cell(
            """# 04. Autonomous Agentic AI System & Benchmark
**Project:** From Small GPT to an Agentic AI System  
**Objective:** Evaluate autonomous tool use, observe-plan-act control loop, runtime execution feedback, error recovery, and verification across 5 multi-step software tasks (RQ3, RQ4, RQ5)."""
        ),
        code_cell(
            """# Load Agent Benchmark Results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_agent = pd.read_csv("results/agent_benchmark_results.csv")
print("Agent Benchmark Raw Records (Sample):")
df_agent.head(10)"""
        ),
        markdown_cell("## 1. Quantitative Success Rate by Configuration (RQ3 & RQ4)"),
        code_cell(
            """summary = df_agent.groupby("agent_mode").agg(
    total_tasks=("task_id", "count"),
    passed_tasks=("success", "sum"),
    mean_steps=("steps_taken", "mean"),
    mean_tool_calls=("tool_calls", "mean"),
    mean_errors=("failed_tool_calls", "mean"),
    mean_latency=("latency_sec", "mean")
).reset_index()
summary["success_rate_pct"] = (summary["passed_tasks"] / summary["total_tasks"]) * 100
summary"""
        ),
        markdown_cell("## 2. Visualizing Agent Mode Trade-offs (RQ5)"),
        code_cell(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=summary, x="agent_mode", y="success_rate_pct", palette={"single_turn": "#E94E77", "react": "#F6A623", "verified": "#50E3C2"}, ax=ax1, edgecolor="black")
ax1.set_ylabel("Task Success Rate (%)")
ax1.set_title("Task Completion by Agent Mode (RQ3 & RQ4)")
ax1.set_ylim(0, 110)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2), ha='center')

sns.barplot(data=summary, x="agent_mode", y="mean_tool_calls", palette={"single_turn": "#E94E77", "react": "#F6A623", "verified": "#50E3C2"}, ax=ax2, edgecolor="black")
ax2.set_ylabel("Mean Tool Calls per Task")
ax2.set_title("Computational Cost per Task (RQ5)")
plt.tight_layout()
plt.show()"""
        ),
    ]
    with open(dest_dir / "04_agent_experiments.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    print("Successfully generated all 4 Google Colab notebooks in notebooks/")


if __name__ == "__main__":
    generate_all_notebooks(Path("notebooks"))
