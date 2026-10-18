"""Unit tests for Character and BPE tokenizers."""

import pytest
from src.data.tokenizer import CharacterTokenizer, BPETokenizerWrapper


def test_character_tokenizer_roundtrip():
    sample_text = "def add(a: int, b: int) -> int:\n    return a + b"
    tok = CharacterTokenizer()
    tok.fit([sample_text])
    
    encoded = tok.encode(sample_text)
    decoded = tok.decode(encoded)
    
    assert decoded == sample_text
    assert tok.vocab_size > len(set(sample_text))
    assert tok.pad_token_id == 0


def test_character_tokenizer_special_tokens():
    sample_text = "x = 42"
    tok = CharacterTokenizer()
    tok.fit([sample_text])
    
    encoded_special = tok.encode(sample_text, add_special_tokens=True)
    assert encoded_special[0] == tok.char2idx[tok.bos_token]
    assert encoded_special[-1] == tok.char2idx[tok.eos_token]
    
    # Decoding with skip_special_tokens=True should return original text
    decoded = tok.decode(encoded_special, skip_special_tokens=True)
    assert decoded == sample_text


def test_bpe_tokenizer_training_and_roundtrip(tmp_path):
    corpus = [
        "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "def quicksort(arr): return arr if len(arr) <= 1 else arr",
        "for i in range(10):\n    print(i * 2)",
    ]
    bpe = BPETokenizerWrapper(vocab_size=300)
    bpe.train_from_iterator(corpus)
    
    test_str = "def fibonacci(n: int) -> int:"
    encoded = bpe.encode(test_str)
    decoded = bpe.decode(encoded)
    
    assert decoded == test_str
    assert bpe.vocab_size <= 300
    
    # Test save and reload
    save_file = tmp_path / "bpe_test.json"
    bpe.save(save_file)
    reloaded = BPETokenizerWrapper.load(save_file)
    assert reloaded.decode(reloaded.encode(test_str)) == test_str
