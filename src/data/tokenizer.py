"""Tokenization modules supporting Character-level and Byte-Pair Encoding (BPE).

Enables empirical study of tokenization effects on sequence length, vocabulary size,
compression ratio, and training throughput.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BaseTokenizer:
    """Abstract base class for all project tokenizers."""

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer for compact, zero-unknown tokenization."""

    def __init__(self, chars: Optional[List[str]] = None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        if chars is not None:
            self._build_vocab(chars)
        else:
            self.char2idx: Dict[str, int] = {}
            self.idx2char: Dict[int, str] = {}

    def _build_vocab(self, chars: List[str]) -> None:
        unique_chars = sorted(list(set(chars)))
        all_tokens = self.special_tokens + [c for c in unique_chars if c not in self.special_tokens]
        self.char2idx = {ch: idx for idx, ch in enumerate(all_tokens)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def fit(self, texts: List[str]) -> CharacterTokenizer:
        all_chars = set("".join(texts))
        self._build_vocab(list(all_chars))
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        unk_idx = self.char2idx.get(self.unk_token, 1)
        tokens = [self.char2idx.get(ch, unk_idx) for ch in text]
        if add_special_tokens:
            bos_idx = self.char2idx[self.bos_token]
            eos_idx = self.char2idx[self.eos_token]
            tokens = [bos_idx] + tokens + [eos_idx]
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for idx in tokens:
            ch = self.idx2char.get(idx, "")
            if skip_special_tokens and ch in self.special_tokens:
                continue
            chars.append(ch)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    @property
    def pad_token_id(self) -> int:
        return self.char2idx.get(self.pad_token, 0)

    @property
    def eos_token_id(self) -> int:
        return self.char2idx.get(self.eos_token, 3)

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": {str(k): v for k, v in self.idx2char.items()}}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CharacterTokenizer:
        tok = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return tok


class BPETokenizerWrapper(BaseTokenizer):
    """Byte-Level BPE Tokenizer with configurable vocabulary size."""

    def __init__(self, vocab_size: int = 2048):
        self.target_vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self._fitted = False

    def train_from_iterator(self, text_iterator, vocab_size: Optional[int] = None) -> BPETokenizerWrapper:
        target_vocab = vocab_size or self.target_vocab_size
        trainer = BpeTrainer(
            vocab_size=target_vocab,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2,
            show_progress=False,
        )
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        self._fitted = True
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id("<bos>")
            eos_id = self.tokenizer.token_to_id("<eos>")
            tokens = [bos_id] + tokens + [eos_id]
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("<pad>")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.token_to_id("<eos>")

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> BPETokenizerWrapper:
        tok = cls()
        tok.tokenizer = Tokenizer.from_file(str(filepath))
        tok._fitted = True
        return tok
