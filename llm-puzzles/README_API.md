
# LLM Query API (FastAPI)

A minimal HTTP wrapper around the repository's `CallLLM.llm_query`, designed to work with g4f-backed models out of the box.

This adds two endpoints:

- `GET /health` — simple liveness check.
- `GET /models` — returns the list of currently-available models as reported by `get_models_list(CONFIG)`.
- `POST /ask` — runs a single prompt against a selected model via `llm_query`.

> **Note:** Authentication is optional. If the environment variable `API_KEY` is set, every request to `/models` and `/ask` must include the header `X-API-Key: <value>`.

---

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or with autoreload for development:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

If you want to require a key:

```bash
export API_KEY="secret-123"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3) Check it's alive

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

---

## Endpoints

### `GET /models`

Lists models from `CallLLM.get_models_list(CONFIG)`.

**Request**

- Optional header `X-API-Key: <value>` if `API_KEY` is set in env.

**Response**

```json
{
  "models": [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3.5-sonnet",
    "..."
  ]
}
```

**Errors**

- `500` — `CallLLM` module is unavailable or raised an exception.

---

### `POST /ask`

Invokes `CallLLM.llm_query` with a specific `model` and `prompt`.

**Request**

- Optional header `X-API-Key: <value>` if `API_KEY` is set in env.
- JSON body:

```json
{
  "model": "gpt-4o-mini",
  "prompt": "Write a haiku about FastAPI",
  "max_retries": 2,
  "backoff_factor": 1.25,
  "stage": "api"
}
```

**Response**

```json
{
  "model": "gpt-4o-mini",
  "output": "A short poem..."
}
```

**Errors**

- `401` — Missing or invalid API key (when `API_KEY` is set).
- `500` — `CallLLM` errors, misconfiguration, or provider failures.
- `502` — Provider returned empty response.

---

## Python usage example

```python
import requests

BASE = "http://localhost:8000"
headers = {}
# If API_KEY is set server-side:
# headers["X-API-Key"] = "secret-123"

# list models
print(requests.get(f"{BASE}/models", headers=headers).json())

# ask
payload = {
    "model": "gpt-4o-mini",
    "prompt": "Return literally: OK",
    "max_retries": 1,
    "backoff_factor": 1.0,
}
print(requests.post(f"{BASE}/ask", json=payload, headers=headers).json())
```

---

## Docker (optional)

A minimal Dockerfile is included. Build and run:

```bash
docker build -t llm-api .
docker run --rm -p 8000:8000 -e API_KEY=secret-123 llm-api
```

---

## Notes

- The `CallLLM.py` file is part of the repository and is imported directly by the API. It contains all the logic to pick providers, retry requests, and list models (including g4f AnyProvider logic).
- GUI dependencies (`tkinter`) in `CallLLM.py` are optionalized for server environments. If tkinter is not available, the module falls back to a headless mode.
- The API does **not** maintain conversation history or sessions; each `/ask` call is a single prompt invocation.


---

## Local Hugging Face models (optional)

You can run local models via `transformers`. This repo adds a *lightweight* dispatcher that recognizes model names prefixed with `hf:` and routes them to a local pipeline.

### Install local dependencies

> These are **heavy** and optional. Keep the base `requirements.txt` for g4f-only usage.

```bash
pip install -r requirements-local.txt
# Install PyTorch suitable for your platform:
# CPU only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Configure available local models

List your local models using an environment variable (comma-separated Hugging Face repo ids or aliases):

```bash
export HF_LOCAL_MODELS="meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct"
# optional: Hugging Face token if required by a gated model
# export HUGGINGFACEHUB_API_TOKEN=hf_xxx
```

You can also define aliases in `CallLLM.CONFIG['LOCAL']['ALIASES']`, for example:

```python
'ALIASES': {
    'llama3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct'
}
```

The `/models` endpoint merges:
- **g4f** models scraped from `https://github.com/xtekky/gpt4free/blob/main/g4f/models.py` (raw file),
- **local** HF models from `HF_LOCAL_MODELS` and the alias map.

HF models are returned **prefixed** with `hf:` (e.g., `hf:meta-llama/Llama-3.1-8B-Instruct`).

### Use a local model

```bash
curl -X POST "http://localhost:8000/ask"   -H "Content-Type: application/json"   -d '{
    "model":"hf:meta-llama/Llama-3.1-8B-Instruct",
    "prompt":"Return literally: OK"
  }'
```

> The local runner uses `pipeline("text-generation")` with sensible defaults (`max_new_tokens`, `temperature`), configurable under `CONFIG['LOCAL']` in `CallLLM.py`.

### Docker notes

Local models need additional dependencies and GPU drivers if you want acceleration.
For convenience, you can modify the `Dockerfile` to install from `requirements-local.txt`,
or build locally and mount your HF cache into the container (e.g. `-v ~/.cache/huggingface:/root/.cache/huggingface`).

