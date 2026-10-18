"""Small GPT Transformer Architecture and Autoregressive Text Generation Engine."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import CausalSelfAttention
from src.model.config import GPTConfig


class MLP(nn.Module):
    """Feed-Forward Network with GeLU non-linearity (4x expansion)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-LayerNorm and Residual Skip Connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attention_weights=return_attention_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class SmallGPT(nn.Module):
    """Compact Decoder-Only Transformer Causal Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=(
                    None
                    if config.use_rotary
                    else nn.Embedding(config.block_size, config.n_embd)
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying between token embeddings and language model head
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count.
        
        If non_embedding is True, the position and token embeddings are subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            if self.transformer.wpe is not None:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
            )

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        # Positional embeddings
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
            pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        # Forward through Transformer blocks
        all_attns = [] if return_attention_weights else None
        for block in self.transformer.h:
            x, attn_weights = block(x, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                all_attns.append(attn_weights)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Full sequence logits and cross-entropy loss
            logits = self.lm_head(x)  # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: compute logits only for the last position
            logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
            loss = None

        return logits, loss, all_attns

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with temperature, top-k, and nucleus (top-p) sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if sequence exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            # Focus on last token
            logits = logits[:, -1, :]  # (b, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b_idx in range(idx.shape[0]):
                    tokens_generated = idx[b_idx].tolist()
                    for t in set(tokens_generated):
                        if logits[b_idx, t] < 0:
                            logits[b_idx, t] *= repetition_penalty
                        else:
                            logits[b_idx, t] /= repetition_penalty

            # Apply temperature
            if temperature == 0.0:
                # Greedy argmax sampling
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # Append generated token
            idx = torch.cat((idx, idx_next), dim=1)

            # Early stopping on EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx
