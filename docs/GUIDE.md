# Using Intelligence Hub

This is a local workshop. You include models and datasets, train local checkpoints, watch this machine’s memory, and try the result in Chat, Tasks, or another app via `/v1`.

Open **http://localhost:3000** after both the API (port 8000) and the UI (port 3000) are running. See [DEPLOYMENT.md](DEPLOYMENT.md) for install steps.

---

## Three states of a model

| State | Meaning |
|---|---|
| **In the library** | Listed under Models. Local checkpoints live in `results/checkpoints/`. Hugging Face and Ollama entries are registered here too. |
| **In memory** | Loaded so it can answer. Uses RAM and, if present, GPU. Manage this on **Resources**. |
| **In use (active)** | Chat, Tasks, and `/v1` go through this one until you change it. |

**Use this** on Models (or include + activate on Resources) loads it and makes it active.

---

## Include a model

**Models → Add a model**

- **Hugging Face** — paste a repo id such as `Qwen/Qwen2.5-0.5B-Instruct`. That adds it to the library. Then **Use this** when you want it in memory. First load downloads weights.
- **Ollama** — Ollama must already be running. **Check** the host, then add a tag you have pulled (for example `qwen2.5:0.5b`).

Local SmallGPT checkpoints appear automatically when `best_model.pt` or `final_model.pt` exists under `results/checkpoints/<name>/`.

---

## Include a dataset

**Datasets → Include a dataset**

You can paste text, upload a file, or point at a file or folder on this computer.

| Accepted | Notes |
|---|---|
| `.txt`, `.md` | Split by blank lines, one line per example, or as one long document |
| `.jsonl`, `.json` | `text`, chat `messages`, `prompt`/`completion`, or `instruction`/`output` |
| `.csv`, `.tsv` | Same field names as columns |
| Folder | Every supported file inside it is merged |

Mark the **kind** so later sectors can stay separate: General, Language, Code, Conversation.

Included data is stored under `results/datasets/<id>/samples.jsonl`. **Train with this** opens Train with that dataset selected.

The library always has **Built-in code algorithms**. Use your own data when you care about the topic.

---

## Train a local model

**Train** only trains **SmallGPT** (this hub’s compact Transformer). It does not fine-tune Hugging Face or Ollama weights.

1. **New model** — pick size (extra settings) and a name for the result.
2. **Improve one I have** — start from a local checkpoint. Architecture comes from that checkpoint; only the dataset and steps change.
3. Choose a **dataset** from the library.
4. Start the job. Watch step, validation perplexity, and speed. **Stop** cancels the run.

When it finishes, the new folder is in the model library. **Use this** before you chat.

A small model learns what the dataset contains. Code-only data continues as code. Conversation data is required if you want greetings and short replies — quality will still be limited on a few-million-parameter network. For product-style chat, include an instruct model.

---

## Chat and Tasks

**Chat** talks to the **active** model. Switch models from the header or the Chat dropdown.

**Tasks** runs a sandboxed agent:

- Built-in practice jobs use a **deterministic oracle** unless you set the planner to **Active** and have an instruct model loaded.
- A custom goal with no capable planner only inspects the workspace and tells you to load an instruct model.

Chat and Tasks are playgrounds. Other products should call the gateway.

---

## Share the model with another app

**Connect**

1. Create a key (optional for local calls with no `Authorization` header).
2. Point the client at `http://127.0.0.1:8000/v1`.
3. Use `/v1/chat/completions` with the same shape as OpenAI.

If you send a Bearer token, it must match a key issued here.

---

## Resources

**Resources** shows CPU, RAM, and GPU, what is loaded, and whether a training job is running.

- **Include** — load a library model into memory
- **Activate** — make a loaded model the one in use
- **Release** / **Release all** — free memory
- **Free cache** — drop unused GPU cache

---

## Honest limits

- One training job at a time.
- Dataset uploads are text-only today (max 80 MB). Vision and audio sectors are reserved, not implemented.
- Hugging Face / Ollama models are for inference here, not for in-hub fine-tuning.
- The built-in SmallGPT tokenizer is BPE-500 unless you train a new one when the default file is missing.
