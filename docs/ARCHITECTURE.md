# Architecture

Intelligence Hub is a **control plane** around models you train here and models you include (Hugging Face, Ollama). The React UI is a workshop. Other apps should use the OpenAI-compatible `/v1` gateway.

```
Browser (Vite, port 3000)
    proxies /api, /v1, /results
            │
            ▼
FastAPI (port 8000)
    routers: dashboard, models, datasets, resources,
             trainer, chat, agent, gateway, benchmarks, tests
            │
            ▼
Hub core
    registry   catalog of local + remote models
    datasets   include / normalize / load samples
    runtime    load, activate, generate
    resources  host snapshot + memory assignment
    keys       gateway secrets
            │
     ┌──────┼──────────────┐
     ▼      ▼              ▼
 SmallGPT  HF Transformers  Ollama HTTP
 (train +  (inference)      (inference)
  infer)
```

---

## Layers

### UI (`frontend/`)

Single-page app. Pages map 1:1 to workshop jobs: Home, Models, Datasets, Resources, Train, Chat, Tasks, Connect, Health.

Global state in `App.jsx`: active page, active model, catalog, resource vitals. Vite proxies API calls to `127.0.0.1:8000` so the browser stays same-origin.

### API (`src/api/`)

`src/api/main.py` mounts routers and `/results` as static files. On startup it syncs the model registry, ensures the built-in dataset exists, and loads a default local checkpoint if one is present.

### Hub (`src/hub/`)

| Module | Role |
|---|---|
| `paths.py` | `results/` locations (checkpoints, datasets, jobs, keys, activity) |
| `registry.py` | Persist catalog; scan local checkpoint folders |
| `datasets.py` | Dataset library; write `samples.jsonl`; load texts for training |
| `runtime.py` | In-memory engines; `generate_text`; chat prompt format |
| `resources.py` | Device + RAM/VRAM snapshot |
| `providers.py` | Ollama ping / generate |
| `keys.py` | Issue and check gateway keys |
| `activity.py` | Append-only event log |

### Training (`src/training/`, `src/data/`, `src/model/`)

- **SmallGPT** — Pre-LN causal Transformer, SDPA attention, optional weight tying (`src/model/`).
- **Ingest** — `src/data/ingest.py` turns files into a list of strings. New modalities can add parsers without changing the trainer’s load path.
- **Dataset** — `CausalLMDataset` slices token ids into next-token blocks.
- **Trainer** — AdamW + cosine warmup. One background thread. Checkpoints: `results/checkpoints/<name>/best_model.pt` and `final_model.pt`.

Scratch training builds a new `GPTConfig`. Continue/improve loads an existing checkpoint and ignores the size fields from the request.

### Agent (`src/agent/`, `src/tools/`, `src/safety/`)

Reason–act–verify loop in a `WorkspaceSandbox`. Planners:

- **Oracle** — scripted decisions for built-in benchmark tasks (default for those tasks).
- **Hub runtime** — the active model plans when planner is `active` / `hub` (needs a model that can follow the JSON tool protocol; instruct models are the realistic choice).

---

## Data on disk

All durable hub state lives under `results/`:

| Path | Contents |
|---|---|
| `checkpoints/<name>/` | Local SmallGPT weights + `training_history.json` |
| `datasets/<id>/` | `samples.jsonl` and optional original file |
| `hub_registry.json` | Model catalog (local + HF + Ollama) |
| `hub_datasets.json` | Dataset catalog |
| `training_jobs.json` | Recent train jobs |
| `hub_activity.json` | Event log |
| `api_keys.json` | Gateway keys |
| `bpe_500.json` | Default local tokenizer |

---

## Request path: chat

1. UI posts `{ message, history, … }` to `POST /api/chat`.
2. Runtime formats `System` / `User` / `Assistant:` the same way conversation datasets are stored.
3. The **active** (or requested) engine generates: SmallGPT, Hugging Face, or Ollama.
4. The playground and `/v1/chat/completions` share this path.

---

## What this is not

- Not a hosted multi-tenant SaaS.
- Not a fine-tune service for Hugging Face or Ollama weights.
- Not an always-correct autonomous SWE agent; practice tasks are oracle-backed unless you attach an instruct model and choose the live planner.
- Chat quality is a property of the **model and dataset**, not of the UI.
