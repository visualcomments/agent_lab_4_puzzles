# Local model memory optimization changes applied

This patch extends the previous g4f/RAM optimizations with targeted improvements for the local Transformers backend (`model_str=local:<hf_model>`).

## Implemented

### 1. Quantization knobs for local HF inference
- File: `AgentLaboratory/inference.py`
- New env: `AGENTLAB_LOCAL_QUANT=none|8bit|4bit`
- Uses `BitsAndBytesConfig` when requested.
- 4-bit mode uses NF4 + double quantization.

### 2. Attention backend selection
- File: `AgentLaboratory/inference.py`
- New env: `AGENTLAB_ATTN_IMPL=sdpa|flash_attention_2|eager`
- Passed through `attn_implementation` when supported by the model.

### 3. Offload and max-memory controls
- File: `AgentLaboratory/inference.py`
- New envs:
  - `AGENTLAB_ENABLE_OFFLOAD=1`
  - `AGENTLAB_OFFLOAD_DIR=...`
  - `AGENTLAB_LOCAL_MAX_MEMORY=cuda:6GiB,cpu:24GiB`
  - `AGENTLAB_OFFLOAD_STATE_DICT=1`
  - `AGENTLAB_OFFLOAD_BUFFERS=0`
- These are wired into `device_map="auto"`, `offload_folder`, `max_memory` and related flags.

### 4. Optional torch.compile for experiments only
- File: `AgentLaboratory/inference.py`
- New envs:
  - `AGENTLAB_TORCH_COMPILE=1`
  - `AGENTLAB_TORCH_COMPILE_MODE=reduce-overhead`
- Kept strictly optional because memory impact must be benchmarked.

### 5. Input-token bounding for local inference
- File: `AgentLaboratory/inference.py`
- New env: `AGENTLAB_LOCAL_MAX_INPUT_TOKENS`
- Tokenizer now uses `truncation=True` + `max_length=...` when configured.

### 6. Bounded local model cache
- File: `AgentLaboratory/inference.py`
- Replaced unbounded local cache with an OrderedDict-based bounded cache.
- New envs:
  - `AGENTLAB_LOCAL_CACHE_MAX_ITEMS` (default `1`)
  - `AGENTLAB_LOCAL_CACHE_TTL_S` (default `3600`)

### 7. Runtime config introspection
- File: `AgentLaboratory/inference.py`
- Added `local_model_runtime_config()` for debugging / smoke tests.

### 8. Validation assets
- New test file: `tests_test_local_model_optimizations.py`
- New bench script: `scripts/bench_local_model_memory.py`
- New optional dependency file: `AgentLaboratory/requirements-local-llm.txt`

## Validation performed

Executed successfully:
- `python -m py_compile AgentLaboratory/inference.py tests_test_local_model_optimizations.py scripts/bench_local_model_memory.py`
- `pytest -q tests_test_codegen_pipeline.py tests_test_submission_pipeline.py tests_test_low_ram_optimizations.py tests_test_local_model_optimizations.py`

Result:
- `15 passed`

## Notes

Intentionally not implemented in this patch:
- GPTQ / AWQ / llama.cpp-specific branches;
- vLLM / TGI service split for the local backend;
- a persistent mmap-backed vector index in the RAG path.

Those are good next steps, but the current patch set already adds the main low-risk memory knobs for local Transformers inference.
