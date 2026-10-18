"""Master Experiment Runner for Language Modeling and Agentic Evaluation.

Executes:
1. Experiment A: Scaling laws and context window lengths on SmallGPT.
2. Experiment B: Tokenizer compression and vocabulary trade-offs (Char vs BPE).
3. Experiment C: Autonomous Agent Benchmark (Single-Turn vs ReAct vs Verified).
4. Generates publication-ready plots and structured JSON/CSV result tables.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CausalLMDataset,
    build_and_split_corpus,
    generate_synthetic_code_corpus,
)
from src.data.tokenizer import BPETokenizerWrapper, CharacterTokenizer
from src.evaluation.evaluate_agents import run_comparative_agent_benchmark
from src.evaluation.metrics import evaluate_generation_quality, evaluate_perplexity
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT
from src.training.trainer import GPTTrainer, TrainingArgs

# Styling for scientific figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def run_experiment_b_tokenizers(corpus: List[str], results_dir: Path) -> Dict[str, Any]:
    """Experiment B: Compare Character vs BPE tokenization strategies."""
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Tokenization Dynamics & Vocabulary Trade-offs")
    print("=" * 60)

    # 1. Character Tokenizer
    char_tok = CharacterTokenizer()
    char_tok.fit(corpus)
    char_save = results_dir / "char_tok.json"
    char_tok.save(char_save)

    # 2. BPE Small (V=500)
    bpe_500 = BPETokenizerWrapper(vocab_size=500)
    bpe_500.train_from_iterator(corpus)
    bpe_500_save = results_dir / "bpe_500.json"
    bpe_500.save(bpe_500_save)

    # 3. BPE Medium (V=2048)
    bpe_2048 = BPETokenizerWrapper(vocab_size=2048)
    bpe_2048.train_from_iterator(corpus)
    bpe_2048_save = results_dir / "bpe_2048.json"
    bpe_2048.save(bpe_2048_save)

    # Sample evaluation on held-out text
    sample_text = "\n".join(corpus[:50])
    raw_bytes = len(sample_text.encode("utf-8"))

    char_tokens = char_tok.encode(sample_text)
    bpe_500_tokens = bpe_500.encode(sample_text)
    bpe_2048_tokens = bpe_2048.encode(sample_text)

    char_ratio = raw_bytes / len(char_tokens)
    bpe_500_ratio = raw_bytes / len(bpe_500_tokens)
    bpe_2048_ratio = raw_bytes / len(bpe_2048_tokens)

    # Embedding parameter calculations assuming d_model = 288
    d_model = 288
    metrics = {
        "Character": {
            "vocab_size": char_tok.vocab_size,
            "tokens_count": len(char_tokens),
            "compression_ratio_bytes_per_tok": round(char_ratio, 3),
            "embedding_params": char_tok.vocab_size * d_model,
        },
        "BPE-500": {
            "vocab_size": bpe_500.vocab_size,
            "tokens_count": len(bpe_500_tokens),
            "compression_ratio_bytes_per_tok": round(bpe_500_ratio, 3),
            "embedding_params": bpe_500.vocab_size * d_model,
        },
        "BPE-2048": {
            "vocab_size": bpe_2048.vocab_size,
            "tokens_count": len(bpe_2048_tokens),
            "compression_ratio_bytes_per_tok": round(bpe_2048_ratio, 3),
            "embedding_params": bpe_2048.vocab_size * d_model,
        },
    }

    print(f"Character: Vocab {char_tok.vocab_size}, Compression: {char_ratio:.2f} B/tok")
    print(f"BPE-500:   Vocab {bpe_500.vocab_size}, Compression: {bpe_500_ratio:.2f} B/tok (Seq reduction: {len(char_tokens)/len(bpe_500_tokens):.2f}x)")
    print(f"BPE-2048:  Vocab {bpe_2048.vocab_size}, Compression: {bpe_2048_ratio:.2f} B/tok (Seq reduction: {len(char_tokens)/len(bpe_2048_tokens):.2f}x)")

    # Plot Tokenizer Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    names = list(metrics.keys())
    ratios = [metrics[k]["compression_ratio_bytes_per_tok"] for k in names]
    emb_params_k = [metrics[k]["embedding_params"] / 1000 for k in names]

    ax1.bar(names, ratios, color=["#4A90E2", "#50E3C2", "#B8E986"], edgecolor="black")
    ax1.set_ylabel("Compression Ratio (Bytes / Token)")
    ax1.set_title("Tokenizer Compression Efficiency")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.bar(names, emb_params_k, color=["#E94E77", "#F6A623", "#F8E71C"], edgecolor="black")
    ax2.set_ylabel("Embedding Parameters (Thousands)")
    ax2.set_title("Embedding Parameter Overhead (d=288)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(results_dir / "tokenizer_comparison.png", dpi=300)
    plt.close()

    with open(results_dir / "tokenizer_experiments.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_experiment_a_scaling(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    tokenizer: BPETokenizerWrapper,
    results_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Experiment A: Scaling laws and context window lengths on SmallGPT."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Model Capacity & Context Window Scaling")
    print("=" * 60)

    # Tokenize partitions once
    train_tokens = []
    for t in train_texts:
        train_tokens.extend(tokenizer.encode(t, add_special_tokens=True))
    val_tokens = []
    for t in val_texts:
        val_tokens.extend(tokenizer.encode(t, add_special_tokens=True))
    test_tokens = []
    for t in test_texts:
        test_tokens.extend(tokenizer.encode(t, add_special_tokens=True))

    print(f"Tokenized Corpus: Train={len(train_tokens):,} | Val={len(val_tokens):,} | Test={len(test_tokens):,} tokens")

    # Configurations to train and compare
    configs_to_test = [
        {
            "name": "GPT-Micro-128",
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 192,
            "block_size": 128,
            "max_steps": 300,
        },
        {
            "name": "GPT-Mini-256",
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 288,
            "block_size": 256,
            "max_steps": 300,
        },
        {
            "name": "GPT-Small-256",
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 384,
            "block_size": 256,
            "max_steps": 300,
        },
        {
            "name": "GPT-Mini-512",
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 288,
            "block_size": 512,
            "max_steps": 300,
        },
    ]

    eval_prompts = [
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n",
        "def quick_sort(arr: list[int]) -> list[int]:\n",
        "def binary_search(arr: list[int], target: int) -> int:\n",
    ]

    experiment_results = []
    all_histories = {}

    for cfg_dict in configs_to_test:
        model_name = cfg_dict["name"]
        print(f"\n--- Training Configuration: {model_name} ---")

        gpt_config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg_dict["block_size"],
            n_layer=cfg_dict["n_layer"],
            n_head=cfg_dict["n_head"],
            n_embd=cfg_dict["n_embd"],
            dropout=0.1,
            bias=True,
            tie_weights=True,
        )

        model = SmallGPT(gpt_config)
        total_params = sum(p.numel() for p in model.parameters())
        non_emb_params = model.get_num_params(non_embedding=True)
        print(f"Model Parameters: {total_params:,} (Non-embedding: {non_emb_params:,})")

        # Datasets & Loaders
        train_ds = CausalLMDataset(train_tokens, block_size=gpt_config.block_size)
        val_ds = CausalLMDataset(val_tokens, block_size=gpt_config.block_size)
        test_ds = CausalLMDataset(test_tokens, block_size=gpt_config.block_size)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        ckpt_dir = results_dir / "checkpoints" / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArgs(
            learning_rate=5e-4,
            min_learning_rate=5e-5,
            warmup_steps=30,
            max_steps=cfg_dict["max_steps"],
            eval_interval=50,
            batch_size=16,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            seed=42,
        )

        trainer = GPTTrainer(
            model=model,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        train_summary = trainer.train()
        all_histories[model_name] = train_summary["history"]

        # Objective Test Set Evaluation
        test_metrics = evaluate_perplexity(model, test_loader, device=torch.device(device))
        gen_metrics = evaluate_generation_quality(
            model, tokenizer, eval_prompts, device=torch.device(device), max_new_tokens=48
        )

        record = {
            "model_name": model_name,
            "n_layer": gpt_config.n_layer,
            "n_head": gpt_config.n_head,
            "n_embd": gpt_config.n_embd,
            "block_size": gpt_config.block_size,
            "total_params": total_params,
            "non_emb_params": non_emb_params,
            "train_time_sec": round(train_summary["total_time_sec"], 2),
            "tokens_per_sec": round(train_summary["total_tokens"] / max(0.1, train_summary["total_time_sec"]), 1),
            "val_loss": round(train_summary["best_val_loss"], 4),
            "val_perplexity": round(train_summary["best_perplexity"], 2),
            "test_loss": round(test_metrics["loss"], 4),
            "test_perplexity": round(test_metrics["perplexity"], 2),
            "syntax_validity_rate": round(gen_metrics["syntax_validity_rate"], 3),
            "sample_completions": gen_metrics["completions"],
        }
        experiment_results.append(record)
        print(f"Results for {model_name}: Test PPL: {record['test_perplexity']:.2f} | Syntax Valid: {record['syntax_validity_rate']:.1%}")

    # Save Experiment A results
    with open(results_dir / "lm_experiments.json", "w") as f:
        json.dump(experiment_results, f, indent=2)

    # 1. Plot Training & Validation Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4A90E2", "#50E3C2", "#E94E77", "#F6A623"]

    for idx, (m_name, hist) in enumerate(all_histories.items()):
        steps = [entry["step"] for entry in hist]
        t_loss = [entry["train_loss"] for entry in hist]
        v_loss = [entry["val_loss"] for entry in hist]
        v_ppl = [entry["val_perplexity"] for entry in hist]

        ax1.plot(steps, t_loss, label=f"{m_name} (Train)", color=colors[idx], linestyle="--", alpha=0.7)
        ax1.plot(steps, v_loss, label=f"{m_name} (Val)", color=colors[idx], linewidth=2)
        ax2.plot(steps, v_ppl, label=m_name, color=colors[idx], linewidth=2)

    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss Convergence")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Validation Perplexity (exp(Loss))")
    ax2.set_title("Validation Perplexity vs Steps")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=300)
    plt.close()

    # 2. Plot Parameter Scale vs Test Perplexity
    fig, ax = plt.subplots(figsize=(7, 4.5))
    params_m = [r["total_params"] / 1e6 for r in experiment_results]
    ppls = [r["test_perplexity"] for r in experiment_results]
    labels = [r["model_name"] for r in experiment_results]

    ax.scatter(params_m, ppls, s=120, c="#4A90E2", edgecolors="black", zorder=3)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (params_m[i] + 0.05, ppls[i] + 0.1), fontsize=9)

    ax.set_xlabel("Model Parameters (Millions)")
    ax.set_ylabel("Held-Out Test Perplexity")
    ax.set_title("Model Parameter Scaling vs. Test Perplexity (RQ1)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "scaling_laws_ppl.png", dpi=300)
    plt.close()

    return experiment_results


def run_experiment_c_agent(results_dir: Path) -> pd.DataFrame:
    """Experiment C: Comparative Agent Benchmark across 5 tasks and 3 configurations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Agentic AI Benchmark (Single-Response vs ReAct vs Verified)")
    print("=" * 60)

    df = run_comparative_agent_benchmark()
    csv_path = results_dir / "agent_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved agent benchmark results to: {csv_path}")

    # Summary table by agent mode
    summary = (
        df.groupby("agent_mode")
        .agg(
            tasks_total=("task_id", "count"),
            tasks_passed=("success", "sum"),
            mean_steps=("steps_taken", "mean"),
            mean_tool_calls=("tool_calls", "mean"),
            mean_errors=("failed_tool_calls", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    summary["success_rate_pct"] = (summary["tasks_passed"] / summary["tasks_total"]) * 100
    print("\nBenchmark Summary by Agent Configuration:")
    print(summary.to_string(index=False))

    # Plot Comparative Benchmark Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    mode_order = ["single_turn", "react", "verified"]
    palette = {"single_turn": "#E94E77", "react": "#F6A623", "verified": "#50E3C2"}

    # Success Rate Bar Plot
    sns.barplot(
        data=summary,
        x="agent_mode",
        y="success_rate_pct",
        order=mode_order,
        palette=palette,
        edgecolor="black",
        ax=ax1,
    )
    ax1.set_ylabel("Task Completion Success Rate (%)")
    ax1.set_xlabel("Agent Architecture")
    ax1.set_title("Task Success Rate by Mode (RQ3 & RQ4)")
    ax1.set_ylim(0, 110)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2.0, p.get_height() + 2), ha="center")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Tool Calls vs Latency Tradeoff (RQ5)
    sns.barplot(
        data=summary,
        x="agent_mode",
        y="mean_tool_calls",
        order=mode_order,
        palette=palette,
        edgecolor="black",
        ax=ax2,
    )
    ax2.set_ylabel("Mean Tool Calls per Task")
    ax2.set_xlabel("Agent Architecture")
    ax2.set_title("Computational Tool Expenditure (RQ5)")
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.1), ha="center")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "agent_benchmark_comparison.png", dpi=300)
    plt.close()

    return df


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running research experiments on device: {device}")

    # 1. Dataset Generation
    corpus = generate_synthetic_code_corpus(num_samples=1000, seed=42)
    train_texts, val_texts, test_texts = build_and_split_corpus(corpus, train_ratio=0.85, val_ratio=0.075, seed=42)

    # 2. Experiment B: Tokenizer Dynamics
    run_experiment_b_tokenizers(corpus, results_dir)

    # Load fitted BPE for Experiment A
    tokenizer = BPETokenizerWrapper.load(results_dir / "bpe_500.json")

    # 3. Experiment A: Language Model Scaling & Context Length
    run_experiment_a_scaling(train_texts, val_texts, test_texts, tokenizer, results_dir, device)

    # 4. Experiment C: Agentic Evaluation Benchmark
    run_experiment_c_agent(results_dir)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print(f"Artifacts and plots saved to: {results_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
