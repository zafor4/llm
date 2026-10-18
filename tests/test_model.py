"""Unit tests for SmallGPT Transformer architecture, causality, and generation."""

import torch
import pytest
from src.model.config import GPTConfig
from src.model.transformer import SmallGPT


@pytest.fixture
def test_config():
    return GPTConfig(
        vocab_size=128,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=True,
        tie_weights=True,
    )


def test_model_forward_pass_shapes(test_config):
    model = SmallGPT(test_config)
    model.eval()

    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    logits, loss, _ = model(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, test_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0


def test_causal_mask_strictly_enforces_autoregressive_property(test_config):
    """Verify that token at position i is completely unaffected by any token at position j > i."""
    model = SmallGPT(test_config)
    model.eval()

    seq_len = 8
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    # Sequence B has the identical prefix up to position 4, but differs at positions 5..7
    seq_b = torch.tensor([[10, 20, 30, 40, 99, 101, 102, 103]])

    with torch.no_grad():
        logits_a, _, _ = model(seq_a, targets=seq_a)
        logits_b, _, _ = model(seq_b, targets=seq_b)

    # Logits at positions 0..3 (prefix) MUST BE IDENTICAL between seq_a and seq_b
    diff_prefix = torch.max(torch.abs(logits_a[:, :4, :] - logits_b[:, :4, :]))
    assert diff_prefix.item() < 1e-5, f"Causal attention leakage detected! Max diff: {diff_prefix.item()}"

    # Logits at position 4 onwards should differ because context at or after position 4 changed
    diff_suffix = torch.max(torch.abs(logits_a[:, 4:, :] - logits_b[:, 4:, :]))
    assert diff_suffix.item() > 1e-4, "Expected suffix logits to differ"


def test_generation_shape_and_length(test_config):
    model = SmallGPT(test_config)
    model.eval()

    prompt = torch.tensor([[5, 12, 19]])
    max_new_tokens = 10
    generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    assert generated.shape == (1, prompt.shape[1] + max_new_tokens)
    # The prompt prefix should remain unchanged
    assert torch.equal(generated[:, : prompt.shape[1]], prompt)


def test_weight_tying(test_config):
    model = SmallGPT(test_config)
    assert model.transformer.wte.weight is model.lm_head.weight
