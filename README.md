# Intelligence Hub

Local workshop for models you train and models you include. Catalog them, attach datasets, train SmallGPT, watch this machine’s memory, try Chat and Tasks, and expose an OpenAI-compatible `/v1` API.

This repository is **Intelligence Hub**. It is not an Antigravity product.

---

## Docs

| | |
|---|---|
| [How to use it](docs/GUIDE.md) | Models, datasets, train, chat, connect |
| [Architecture](docs/ARCHITECTURE.md) | Control plane, runtime, training, disk layout |
| [API reference](docs/API_REFERENCE.md) | REST + `/v1` |
| [Run and host](docs/DEPLOYMENT.md) | Install, `results/`, hosting notes |
| Live OpenAPI | http://127.0.0.1:8000/docs |

---

## What you can do

```
UI (React + Vite, :3000)  →  FastAPI hub (:8000)
                              ├── Models     local SmallGPT, Hugging Face, Ollama
                              ├── Datasets   include text / files / folders
                              ├── Train      scratch or improve a local checkpoint
                              ├── Resources  RAM / GPU, include and release
                              ├── Chat/Tasks playgrounds on the active model
                              └── /v1        other apps
```

Local training always produces **SmallGPT**. Hugging Face and Ollama entries are for inference until a later training backend exists.

A model only answers the way it was trained. A code-only checkpoint continues as code. Conversation needs conversation data — or an instruct model you include.

---

## Quick start

Python 3.10+, Node 18+, optional NVIDIA GPU.

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Second terminal:

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000**.

```bash
pytest tests/ -v
```

---

## Repository

```
docs/           Guide, architecture, API, deployment
frontend/       Workshop UI
results/        Checkpoints, datasets, catalogs, keys (local state)
src/api/        FastAPI routers
src/hub/        Registry, runtime, datasets, resources, keys
src/data/       Ingest, tokenizers, causal dataset
src/model/      SmallGPT
src/training/   Train loop
src/agent/      Task loop and planners
tests/
```

---

## License

MIT. See [LICENSE](LICENSE).
