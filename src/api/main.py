"""Production-grade FastAPI application powering the Antigravity AI Platform."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import agent, benchmarks, chat, dashboard, models, tests, trainer

app = FastAPI(
    title="Antigravity AI Platform API",
    description="Production-grade asynchronous API for Causal Language Modeling, Autonomous Coding Agents, and Model Training.",
    version="2.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(dashboard.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(trainer.router)
app.include_router(benchmarks.router)
app.include_router(tests.router)

# Mount static results directory for images and charts
results_dir = PROJECT_ROOT / "results"
if results_dir.exists():
    app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")


@app.get("/")
def root():
    return {
        "platform": "Antigravity AI Platform",
        "status": "ONLINE",
        "version": "2.0.0",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
