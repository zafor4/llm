# API reference

Base URL when running locally: `http://127.0.0.1:8000`.

The Vite UI proxies `/api`, `/v1`, and `/results` to that host, so the browser can call the same paths on `http://localhost:3000`.

Interactive schema: `http://127.0.0.1:8000/docs`.

---

## Service

### `GET /`

```json
{
  "platform": "Intelligence Hub",
  "status": "ONLINE",
  "version": "3.0.0",
  "docs_url": "/docs",
  "hub": "/api/dashboard/overview",
  "openai_compat": "/v1/chat/completions"
}
```

---

## Dashboard

### `GET /api/dashboard/vitals`

Host snapshot plus hub counts: CUDA, VRAM, RAM, CPU, registered models, datasets, loaded count, active model, training status.

### `GET /api/dashboard/overview`

Vitals, catalog, datasets, loaded engines, current training state, recent jobs, recent activity.

---

## Models

### `GET /api/models`  (`/`, `/list` also work)

Returns `models`, `loaded_models`, `active_model`.

Each catalog row includes `id`, `provider` (`local` | `huggingface` | `ollama`), `is_active`, `is_loaded`, and metrics when known.

### `POST /api/models/load`

```json
{ "model_name": "GPT-Mini-256", "activate": true }
```

### `POST /api/models/unload`

```json
{ "model_name": "GPT-Mini-256" }
```

### `POST /api/models/activate`

```json
{ "model_name": "GPT-Mini-256" }
```

### `POST /api/models/import/huggingface`

```json
{ "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "name": null, "load_now": false }
```

Registers the repo. Set `load_now` true only if you want weights downloaded immediately.

### `POST /api/models/import/ollama`

```json
{ "model_name": "qwen2.5:0.5b", "host": "http://127.0.0.1:11434", "load_now": true }
```

Fails if Ollama is down or the tag is not pulled.

### `GET /api/models/providers/ollama?host=http://127.0.0.1:11434`

Connectivity probe and tag list.

### `PATCH /api/models/{model_id}`

`name`, `notes`, `tags`.

### `DELETE /api/models/{model_id}?delete_files=false`

Removes the catalog row. `delete_files=true` also deletes a **local** checkpoint folder.

---

## Datasets

### `GET /api/datasets`

`{ "datasets": [...], "sectors": ["general", "language", "code", "chat"] }`

### `GET /api/datasets/{id}`

Metadata, preview, sample count, `used_by`.

### `GET /api/datasets/{id}/samples?offset=0&limit=20`

Paginated raw texts (limit capped at 100).

### `POST /api/datasets/include`

Paste text into the library.

```json
{
  "name": "Support chats",
  "text": "User: hi\nAssistant:\nHello.\n\nUser: bye\nAssistant:\nGoodbye.",
  "sector": "chat",
  "notes": "",
  "tags": [],
  "format_hint": "auto",
  "split": "auto",
  "filename": "pasted.txt"
}
```

`split`: `auto` | `blank_line` | `line` | `whole`.

### `POST /api/datasets/include-path`

```json
{
  "name": "Mixed folder",
  "path": "C:\\\\data\\\\corpus",
  "sector": "language",
  "notes": "",
  "split": "auto"
}
```

A folder merges every `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.tsv` file.

### `POST /api/datasets/upload`

`multipart/form-data`: `file` (required), `name`, `sector`, `notes`, `format_hint`, `split`. Max 80 MB. Needs `python-multipart` installed.

### `PATCH /api/datasets/{id}`

`name`, `notes`, `tags`, `sector`.

### `DELETE /api/datasets/{id}`

Built-in dataset cannot be deleted.

---

## Resources

### `GET /api/resources`

`host` snapshot, `assigned` (in memory), `available` (library but not loaded), `active_model`, training slice.

### `POST /api/resources/include`

```json
{ "model_name": "GPT-Mini-256", "activate": false }
```

### `POST /api/resources/release`

```json
{ "model_name": "GPT-Mini-256" }
```

### `POST /api/resources/release-all`

### `POST /api/resources/activate`

```json
{ "model_name": "GPT-Mini-256" }
```

### `POST /api/resources/free-cache`

---

## Trainer

One job at a time.

### `POST /api/trainer/start`

```json
{
  "model_name": "My-First-Model",
  "mode": "scratch",
  "base_model": null,
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 192,
  "block_size": 128,
  "learning_rate": 0.0005,
  "batch_size": 16,
  "max_steps": 100,
  "dataset_id": "builtin-code-algorithms",
  "corpus_text": ""
}
```

| Field | Notes |
|---|---|
| `mode` | `scratch` or `continue` |
| `base_model` | Required when `mode` is `continue` (local checkpoint name) |
| `dataset_id` | Library id. If omitted and `corpus_text` is empty, uses the built-in code set |
| `n_layer` / `n_head` / `n_embd` / `block_size` | Used only for scratch jobs |

Only **local SmallGPT** is trained.

### `POST /api/trainer/cancel`

### `GET /api/trainer/status`

Live telemetry: step, losses, perplexity, tokens/sec, VRAM, `dataset_id`, `error_message`.

### `GET /api/trainer/jobs`

Recent jobs, newest first.

---

## Chat (playground)

### `POST /api/chat`  or  `POST /api/chat/generate`

```json
{
  "message": "hi",
  "history": [{ "role": "user", "content": "hello" }, { "role": "assistant", "content": "Hi." }],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 64,
  "system_prompt": null,
  "model": null
}
```

`model` optional; default is the active engine.

```json
{
  "role": "assistant",
  "content": "…",
  "latency_sec": 0.12,
  "tokens_generated": 64,
  "model_used": "GPT-Mini-256",
  "provider": "local"
}
```

If nothing is loaded, the handler returns a plain message instead of 500.

---

## Agent (playground)

### `GET /api/agent/tasks`

Built-in benchmark tasks: `task_id`, `name`, `category`, `goal`.

### `POST /api/agent/run`

```json
{
  "goal": "",
  "task_id": "TASK_01",
  "mode": "verified",
  "max_steps": 8,
  "planner": "auto",
  "model": null
}
```

`planner`: `auto` | `oracle` | `active` | `hub`.  
`auto` + a known `task_id` uses the oracle. Live planning needs an instruct model and `active` / `hub`.

---

## OpenAI-compatible gateway

### `GET /v1/models`

OpenAI list shape. `id` is the hub catalog id.

### `POST /v1/chat/completions`

```json
{
  "model": "GPT-Mini-256",
  "messages": [
    { "role": "system", "content": "Be brief." },
    { "role": "user", "content": "hi" }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 128
}
```

`model` optional → active model. Streaming is accepted in the body but not implemented; the reply is a single completion.

**Auth:** if the request has no `Authorization` header, it is allowed (local default). If it sends `Bearer …`, the token must match a key from Connect.

### `GET /api/gateway/status`

Base URL, active model, loaded models, key previews.

### `POST /api/gateway/keys`

```json
{ "label": "website" }
```

Returns the full secret once. Store it.

### `DELETE /api/gateway/keys/{key_id}`

---

## Health

### `GET /api/benchmarks/results`

Rows from `results/agent_benchmark_results.csv`, or `[]`.

### `GET /api/benchmarks/scaling-experiments`

`lm_experiments` and `tokenizer_experiments` JSON from `results/`.

### `POST /api/tests/run`

Runs `pytest tests/ -v` with a 30s timeout. Returns `success`, `exit_code`, `output`.
