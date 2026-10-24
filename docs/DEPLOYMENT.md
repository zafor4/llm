# Production Deployment & Operations Guide

## 1. Prerequisites

- **Python**: 3.10+ (Recommended: 3.11)
- **Node.js**: v18+ (Recommended: v20 or v22)
- **NVIDIA GPU**: RTX 3050 or higher with CUDA 12+ (or CPU mode fallback)

---

## 2. Local Setup & Quickstart

### Step 1: Install Python Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 3: Run the Application Services

**Terminal 1 (Backend API Service):**
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend Client Service):**
```bash
cd frontend
npm run dev
```

Visit `http://localhost:3000` to access the full application platform.

---

## 3. Production Build & Serving

To compile the React client into static assets for production:

```bash
cd frontend
npm run build
```

This compiles optimized bundles to `frontend/dist/`. You can serve them using Nginx, Caddy, or directly mount `frontend/dist` in FastAPI using `StaticFiles`.

---

## 4. Running Automated Tests

Run the full 16-test suite:
```bash
pytest tests/ -v
```
