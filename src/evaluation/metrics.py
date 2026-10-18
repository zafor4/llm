"""Language Model Evaluation Metrics: Perplexity, Cross-Entropy Loss, and Generation Quality."""

from __future__ import annotations

import ast
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.tokenizer import BaseTokenizer
from src.model.transformer import SmallGPT


@torch.no_grad()
def evaluate_perplexity(
    model: SmallGPT,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute exact cross-entropy loss and perplexity across a full dataset split."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits, loss, _ = model(x, targets=y)
        # Multiply by batch size * seq_len to get total weighted loss
        num_tokens = y.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    mean_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(mean_loss, 20.0))
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "total_tokens_evaluated": total_tokens,
    }


def evaluate_generation_quality(
    model: SmallGPT,
    tokenizer: BaseTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
    """Evaluate code generation syntax validity, diversity, and completion quality."""
    model.eval()
    completions = []
    syntax_valid_count = 0
    all_trigrams = []
    unique_trigrams = set()

    for prompt_text in prompts:
        prompt_tokens = tokenizer.encode(prompt_text)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        gen_tokens = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
        )

        full_text = tokenizer.decode(gen_tokens[0].tolist())
        continuation = full_text[len(prompt_text) :]

        # Test syntactic validity via Python AST parsing
        is_syntax_valid = False
        try:
            ast.parse(full_text)
            is_syntax_valid = True
            syntax_valid_count += 1
        except SyntaxError:
            is_syntax_valid = False

        # Measure 3-gram diversity in continuation
        words = continuation.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                tri = (words[i], words[i + 1], words[i + 2])
                all_trigrams.append(tri)
                unique_trigrams.add(tri)

        completions.append(
            {
                "prompt": prompt_text,
                "continuation": continuation,
                "is_syntax_valid": is_syntax_valid,
            }
        )

    syntax_valid_ratio = syntax_valid_count / max(1, len(prompts))
    trigram_diversity = (
        len(unique_trigrams) / max(1, len(all_trigrams)) if all_trigrams else 0.0
    )

    return {
        "syntax_validity_rate": syntax_valid_ratio,
        "trigram_diversity": trigram_diversity,
        "num_prompts": len(prompts),
        "completions": completions,
    }
