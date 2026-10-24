"""Turn common dataset files into a list of training texts.

This is the ingest layer for the hub library. New sectors (vision, audio)
can add parsers later without changing how training loads samples.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv"}
SECTORS = ("general", "language", "code", "chat")
MAX_PREVIEW_CHARS = 240
MAX_SAMPLE_CHARS = 20_000
LONG_DOC_CHARS = 2_400
LONG_DOC_OVERLAP = 200


@dataclass
class IngestResult:
    samples: List[str]
    format: str
    schema: str
    source_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def char_count(self) -> int:
        return sum(len(s) for s in self.samples)

    def preview(self, n: int = 3) -> List[str]:
        out = []
        for sample in self.samples[:n]:
            text = " ".join(sample.split())
            if len(text) > MAX_PREVIEW_CHARS:
                text = text[: MAX_PREVIEW_CHARS - 1] + "…"
            out.append(text)
        return out


def format_chat_turns(messages: Sequence[dict], include_system: bool = True) -> str:
    """Serialize chat messages the same way the hub chat runtime prompts."""
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").lower()
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if not content:
            continue
        if role == "system":
            if include_system:
                lines.append(f"System: {content}")
        elif role in {"user", "human"}:
            lines.append(f"User: {content}")
        elif role in {"assistant", "model", "bot", "gpt"}:
            lines.append(f"Assistant:\n{content}")
        else:
            lines.append(content)
    return "\n".join(lines).strip()


def _clean_sample(text: str) -> str:
    text = (text or "").replace("\x00", "").strip()
    if len(text) > MAX_SAMPLE_CHARS:
        text = text[:MAX_SAMPLE_CHARS]
    return text


def _chunk_long(text: str) -> List[str]:
    text = text.strip()
    if len(text) <= LONG_DOC_CHARS * 2:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + LONG_DOC_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + LONG_DOC_CHARS // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - LONG_DOC_OVERLAP, start + 1)
    return chunks


def _split_plain(text: str, split: str = "auto") -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    mode = (split or "auto").lower()
    if mode == "whole":
        return _chunk_long(text)
    if mode == "line":
        return [ln.strip() for ln in text.split("\n") if ln.strip()]
    if mode == "blank_line" or (mode == "auto" and "\n\n" in text):
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return parts
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if mode == "auto" and len(lines) >= 20 and sum(len(x) for x in lines) / len(lines) < 220:
        return lines
    return _chunk_long(text)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _from_record(obj: Any, sector: str = "general") -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _clean_sample(obj)
    if not isinstance(obj, dict):
        return _clean_sample(str(obj))

    if isinstance(obj.get("messages"), list):
        return _clean_sample(format_chat_turns(obj["messages"]))

    text = obj.get("text") or obj.get("content") or obj.get("document") or obj.get("body")
    if isinstance(text, str) and text.strip():
        return _clean_sample(text)

    instruction = obj.get("instruction") or obj.get("instruct")
    inp = obj.get("input") or obj.get("context") or ""
    output = obj.get("output") or obj.get("response") or obj.get("completion")
    if instruction and output:
        user = str(instruction).strip()
        if str(inp).strip():
            user = f"{user}\n{str(inp).strip()}"
        if sector == "chat":
            return _clean_sample(f"User: {user}\nAssistant:\n{output}")
        return _clean_sample(f"{user}\n{output}")

    prompt = obj.get("prompt") or obj.get("query") or obj.get("question")
    completion = obj.get("completion") or obj.get("answer") or obj.get("output")
    if prompt and completion:
        if sector == "chat":
            return _clean_sample(f"User: {prompt}\nAssistant:\n{completion}")
        return _clean_sample(f"{prompt}\n{completion}")

    if prompt:
        return _clean_sample(str(prompt))
    return None


def _parse_json_payload(text: str, sector: str) -> List[str]:
    data = json.loads(text)
    rows: Iterable[Any]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("data", "samples", "rows", "conversations", "messages", "items"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            sample = _from_record(data, sector)
            return [sample] if sample else []
    else:
        return []

    samples: List[str] = []
    for row in rows:
        sample = _from_record(row, sector)
        if sample:
            samples.append(sample)
    return samples


def _parse_jsonl(text: str, sector: str) -> List[str]:
    samples: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            samples.append(_clean_sample(line))
            continue
        sample = _from_record(obj, sector)
        if sample:
            samples.append(sample)
    return samples


def _parse_table(text: str, dialect: str, sector: str) -> List[str]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t" if dialect == "tsv" else ",")
    samples: List[str] = []
    for row in reader:
        if not row:
            continue
        lowered = {str(k).strip().lower(): (v if v is not None else "") for k, v in row.items() if k}
        sample = _from_record(lowered, sector)
        if sample:
            samples.append(sample)
            continue
        values = [str(v).strip() for v in row.values() if v and str(v).strip()]
        if values:
            samples.append(_clean_sample("\n".join(values)))
    return samples


def parse_text(
    text: str,
    *,
    filename: str = "dataset.txt",
    format_hint: str = "auto",
    split: str = "auto",
    sector: str = "general",
) -> IngestResult:
    suffix = Path(filename).suffix.lower()
    hint = (format_hint or "auto").lower()
    warnings: List[str] = []
    fmt = hint if hint != "auto" else {
        ".jsonl": "jsonl",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "tsv",
        ".md": "text",
        ".txt": "text",
    }.get(suffix, "text")

    samples: List[str] = []
    schema = "text"
    try:
        if fmt == "jsonl":
            samples = _parse_jsonl(text, sector)
            schema = "jsonl"
        elif fmt == "json":
            samples = _parse_json_payload(text, sector)
            schema = "json"
        elif fmt in {"csv", "tsv"}:
            samples = _parse_table(text, fmt, sector)
            schema = fmt
        else:
            samples = _split_plain(text, split)
            schema = "text"
    except json.JSONDecodeError as exc:
        warnings.append(f"JSON parse failed ({exc}); treating as plain text.")
        samples = _split_plain(text, split)
        fmt = "text"
        schema = "text"

    samples = [s for s in samples if s and len(s.strip()) >= 2]
    if not samples:
        warnings.append("No training samples were found in this file.")
    return IngestResult(
        samples=samples,
        format=fmt,
        schema=schema,
        source_files=[filename],
        warnings=warnings,
    )


def parse_bytes(raw: bytes, filename: str, **kwargs) -> IngestResult:
    return parse_text(_decode(raw), filename=filename, **kwargs)


def parse_path(path: Path, **kwargs) -> IngestResult:
    path = Path(path)
    if path.is_dir():
        merged = IngestResult(samples=[], format="mixed", schema="mixed", source_files=[], warnings=[])
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            merged.warnings.append("Folder has no .txt, .md, .json, .jsonl, .csv, or .tsv files.")
            return merged
        for file in files:
            part = parse_bytes(file.read_bytes(), file.name, **kwargs)
            merged.samples.extend(part.samples)
            merged.source_files.append(str(file))
            merged.warnings.extend(part.warnings)
        return merged
    return parse_bytes(path.read_bytes(), path.name, **kwargs)
