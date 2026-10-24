# REST API Reference & Endpoints

All endpoints are hosted at `http://127.0.0.1:8000/api` and proxied through the frontend at `http://localhost:3000/api`.

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## 1. Dashboard & System Health

### `GET /api/dashboard/vitals`
Returns real-time host hardware metrics and GPU utilization.

**Response Schema:**
```json
{
  "status": "HEALTHY",
  "pytorch_version": "2.6.0+cu124",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 3050",
  "vram_total_mb": 6143.4,
  "vram_used_mb": 25.2,
  "ram_total_gb": 15.9,
  "ram_used_gb": 14.8,
  "cpu_usage_percent": 18.2,
  "checkpoints_count": 4
}
```

---

## 2. Chatbot Studio Inference

### `POST /api/chat/generate`
Generates conversational completions using the currently loaded model.

**Request Schema:**
```json
{
  "prompt": "def fibonacci(n):",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response Schema:**
```json
{
  "prompt": "def fibonacci(n):",
  "generation": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
  "tokens_generated": 42,
  "model_used": "GPT-Mini-256",
  "elapsed_seconds": 0.18
}
```

---

## 3. Autonomous Coding Agent

### `GET /api/agent/tasks`
Lists all predefined benchmark goals.

**Response Schema:**
```json
[
  {
    "id": "TASK_01",
    "name": "Factorial Bug Fix",
    "description": "Fix off-by-one bug in factorial and verify with tests."
  }
]
```

### `POST /api/agent/run`
Executes an autonomous goal within a sandboxed directory.

**Request Schema:**
```json
{
  "task_id": "TASK_01",
  "custom_goal": null,
  "mode": "verified"
}
```

**Response Schema:**
```json
{
  "task_id": "TASK_01",
  "goal": "Fix off-by-one bug in factorial and verify with tests.",
  "success": true,
  "steps_taken": 4,
  "duration_seconds": 0.48,
  "execution_trace": [
    {
      "step": 1,
      "thought": "First let me list the files in the workspace.",
      "tool": "list_directory",
      "args": {},
      "status": "PASS",
      "output": "['src/math_utils.py', 'tests/test_math.py']"
    }
  ],
  "verified_post_condition": true
}
```

---

## 4. Models Fleet & Hot Swapping

### `GET /api/models/`
Lists all available checkpoint weights in `results/checkpoints/` and the active model.

### `POST /api/models/load`
Hot-swaps a model into GPU VRAM.

**Request Schema:**
```json
{
  "model_name": "GPT-Small-256"
}
```

---

## 5. Model Training Studio

### `POST /api/trainer/start`
Starts a background training job using custom architecture parameters.

**Request Schema:**
```json
{
  "model_name": "Custom-GPT-Micro",
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 192,
  "block_size": 128,
  "learning_rate": 0.0005,
  "batch_size": 16,
  "max_steps": 100,
  "raw_text": "Custom corpus text..."
}
```

### `GET /api/trainer/status`
Polls live training telemetry (loss, perplexity, step progress, tokens/sec).

---

## 6. Automated Unit Tests

### `POST /api/tests/run`
Executes `pytest tests/ -v` and returns execution status, test count, and stdout.
