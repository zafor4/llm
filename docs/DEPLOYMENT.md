# Run and host Intelligence Hub

The hub is a **process on a machine you control**, plus a browser UI. Training, checkpoints, and GPU stay on that machine. There is no official Docker image or cloud installer yet.

---

## Prerequisites

- Python 3.10+ (3.11 is a good default)
- Node.js 18+
- Optional NVIDIA GPU + a CUDA build of PyTorch (CPU works; training and larger HF models are slower)
- For file upload: `python-multipart` (listed in `requirements.txt`)
- For Ollama models: [Ollama](https://ollama.com) running locally

---

## Local setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cd frontend
npm install
cd ..
```

**Terminal 1 — API**

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — UI**

```bash
cd frontend
npm run dev
```

- UI: **http://localhost:3000** (use `localhost`, not `127.0.0.1`, if Vite bound to IPv6)
- API / OpenAPI: **http://127.0.0.1:8000/docs**

Optional CLI demos: `python demo.py`.

```bash
pytest tests/ -v
```

---

## What you must keep

Everything important is under `results/`. Back this folder up if you care about models and datasets.

| Path | Keep? |
|---|---|
| `results/checkpoints/` | Yes — trained weights |
| `results/datasets/` | Yes — included training data |
| `results/hub_registry.json` | Yes — catalog |
| `results/hub_datasets.json` | Yes — dataset index |
| `results/api_keys.json` | Yes — treat as secrets |
| `results/training_jobs.json` | Optional history |
| `results/hub_activity.json` | Optional log |

Do not commit secrets or large weight files unless you intend to.

---

## Production-style local serving

Build the UI once:

```bash
cd frontend
npm run build
```

Output is `frontend/dist/`. Serve it with Caddy, nginx, or FastAPI `StaticFiles`, and keep the API on port 8000. The UI expects `/api`, `/v1`, and `/results` on the same origin (Vite does this in dev; you must proxy the same paths in production).

Bind the API only to localhost unless you have a tunnel or firewall in front:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

`--host 0.0.0.0` exposes the hub on the LAN. The `/v1` gateway allows requests **without** a Bearer token. Do not put that on the public internet as-is.

---

## Hosting (guidance only)

This stack is a workshop, not a static site.

| Goal | Practical approach |
|---|---|
| Private use on your PC | Run both processes locally. Tailscale or Cloudflare Tunnel if you need access from another device. |
| Public **UI only** | Cloudflare Pages / Netlify for a built frontend **only if** the API is a stable URL you proxy. Training will not run there. |
| Cheap always-on API | A small CPU VPS can serve the catalog and tiny SmallGPT inference. Not enough for comfortable HF 0.5B+ or GPU training. |
| Occasional GPU train | Hourly GPU (RunPod, vast.ai, etc.), copy `results/` in and out. |
| Free GPU notebooks | Colab/Kaggle are sessions, not a 24/7 hub. |

Do not expect Vercel/Netlify to host the Python API, long train jobs, or multi-gigabyte weights.

---

## Hardware notes

- **RTX-class 6–8 GB** is enough for SmallGPT train/infer and cautious 0.5B instruct loads. Unload unused engines on Resources.
- Training is single-job. Close other GPU apps if VRAM errors appear.
- First Hugging Face load downloads the repo into the Hugging Face cache (not `results/checkpoints/`).

---

## Environment

No required `.env` for the default local loop. Optional later: Hugging Face tokens in the usual `HF_TOKEN` / `huggingface-cli login` if you include gated models.
