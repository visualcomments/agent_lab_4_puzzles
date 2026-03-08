# Low-RAM optimization changes applied

This archive includes practical optimizations based on the provided audit PDF for the g4f / remote-LLM path.

## Implemented

### 1. Streamed + bounded g4f response capture
- `AgentLaboratory/inference.py`
- `llm-puzzles/CallLLM.py`

Changes:
- switched g4f response handling to streamed capture when supported;
- replaced list accumulation + `"".join(parts)` with `io.StringIO()`;
- added hard response bounds via:
  - `AGENTLAB_MAX_RESPONSE_CHARS` (default `60000`)
  - `LLM_PUZZLES_MAX_RESPONSE_CHARS` (default `60000`)

### 2. Optional low-RAM token counting policy
- `AgentLaboratory/inference.py`

Changes:
- token counting for g4f-backed calls is skipped by default in low-RAM mode;
- can be re-enabled with `AGENTLAB_ENABLE_TOKEN_COUNT_FOR_G4F=1`;
- global disable remains available through `AGENTLAB_DISABLE_TOKEN_COUNT=1`.

### 3. Remote LLM subprocess isolation generalized
- `AgentLaboratory/inference.py`
- `AgentLaboratory/query_model_worker.py`
- `AgentLaboratory/agents.py`
- `AgentLaboratory/mlesolver.py`
- `AgentLaboratory/ai_lab_repo.py`

Changes:
- added a generic `query_model_stable(...)` wrapper;
- remote models use isolated worker processes by default (`AGENTLAB_REMOTE_SUBPROCESS=1`);
- main agent code paths now use the stable wrapper.

### 4. Reduced baseline imports on the orchestration path
- `AgentLaboratory/common_imports.py`
- `AgentLaboratory/ai_lab_repo.py`

Changes:
- heavy libraries such as TensorFlow / diffusers / spaCy / plotly are no longer imported eagerly by default;
- they are now gated by `AGENTLAB_HEAVY_IMPORTS=1`;
- `ai_lab_repo.py` no longer imports the Flask app at module import time.

### 5. Lazy embeddings + batch search in the local paper search app
- `AgentLaboratory/app.py`

Changes:
- `SentenceTransformer` is loaded lazily;
- document search now iterates through DB rows in batches with `yield_per(...)`;
- embeddings are computed batch-by-batch instead of caching the whole corpus in RAM by default;
- results are bounded with `SEARCH_TOP_K`.

### 6. Dependency cleanup
- `AgentLaboratory/requirements.txt`
- `AgentLaboratory/requirements-llm.txt`

Changes:
- added missing web-app/runtime dependencies to the main requirements file;
- added a new lightweight requirements file for the low-RAM g4f path.

### 7. Regression tests added
- `tests_test_low_ram_optimizations.py`

Covered:
- bounded streamed g4f capture in `inference.py`;
- bounded streamed capture in `CallLLM.py`;
- subprocess wrapper success path.

## Validation performed

Executed successfully:
- `python -m py_compile ...` for all modified Python files
- `pytest -q tests_test_codegen_pipeline.py tests_test_submission_pipeline.py tests_test_low_ram_optimizations.py`

Result:
- `12 passed`

Additional smoke check executed:
- imported `AgentLaboratory/inference.py`
- imported `llm-puzzles/CallLLM.py`
- verified bounded stream helpers on synthetic iterators

## Notes

Not fully implemented in code (architectural / larger-scope items from the audit):
- dedicated g4f Interference API deployment;
- on-disk / mmap vector index for RAG;
- artifact spilling of every large agent field to disk with transparent reload.

These are good next steps, but the current patch set already reduces peak RAM on the active g4f path and lowers baseline memory pressure.
