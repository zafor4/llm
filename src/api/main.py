"""House of Intelligence — model hub, training, resources, and playgrounds."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import agent, benchmarks, chat, dashboard, gateway, models, resources, tests, trainer
from src.hub.activity import log_event
from src.hub.paths import RESULTS_DIR
from src.hub.registry import sync_registry
from src.hub.runtime import bootstrap_default

app = FastAPI(
    title="House of Intelligence",
    description="Control plane for training, cataloging, and serving models, plus playgrounds and an OpenAI-compatible gateway.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(resources.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(trainer.router)
app.include_router(benchmarks.router)
app.include_router(tests.router)
app.include_router(gateway.router)

if RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.on_event("startup")
def on_startup():
    sync_registry()
    loaded = bootstrap_default()
    log_event("hub", f"Control plane online. Active model: {loaded or 'none'}")


@app.get("/")
def root():
    return {
        "platform": "House of Intelligence",
        "status": "ONLINE",
        "version": "3.0.0",
        "docs_url": "/docs",
        "hub": "/api/dashboard/overview",
        "openai_compat": "/v1/chat/completions",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
