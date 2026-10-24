"""External model providers: Ollama HTTP and Hugging Face transformers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class ProviderError(RuntimeError):
    pass


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 90) -> Any:
    headers = {"Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider returned invalid JSON") from exc


def ollama_ping(host: str = "http://127.0.0.1:11434") -> Dict[str, Any]:
    host = host.rstrip("/")
    try:
        tags = _http_json(f"{host}/api/tags", timeout=5)
        models = [m.get("name") for m in tags.get("models", []) if m.get("name")]
        return {"ok": True, "host": host, "models": models}
    except ProviderError as exc:
        return {"ok": False, "host": host, "models": [], "error": str(exc)}


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 128,
    timeout: int = 120,
) -> str:
    host = host.rstrip("/")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    data = _http_json(f"{host}/api/generate", payload, timeout=timeout)
    return (data.get("response") or "").strip()


def load_huggingface(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def huggingface_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max(1, min(max_tokens, 512)),
            temperature=max(temperature, 0.01) if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()
