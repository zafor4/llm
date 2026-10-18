"""Configuration class for Small GPT Transformer architectures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Hyperparameter configuration for SmallGPT causal language model."""

    vocab_size: int = 2048
    block_size: int = 256        # Maximum context window length
    n_layer: int = 6             # Number of Transformer decoder layers
    n_head: int = 6              # Number of attention heads
    n_embd: int = 384            # Embedding dimension
    dropout: float = 0.1         # Dropout probability
    bias: bool = True            # Include bias in Linears and LayerNorms (like GPT-2)
    use_rotary: bool = False     # Use Rotary Position Embeddings (RoPE) instead of learned
    tie_weights: bool = True     # Tie token embedding and output projection weights

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimate_flops_per_token(self) -> int:
        """Estimate forward-pass FLOPs per token (approx 2 * non_emb_params + attention)."""
        # Linear projections in attention: 4 * n_embd * n_embd (q, k, v, out)
        # Attention score and value: 2 * n_embd * block_size
        # MLP: 2 * n_embd * (4 * n_embd) = 8 * n_embd^2
        flops_per_layer = 12 * (self.n_embd**2) + 2 * self.n_embd * self.block_size
        return 2 * self.n_layer * flops_per_layer
