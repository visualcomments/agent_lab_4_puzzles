# Memory patch plan implemented

## Priority order used

1. `AgentLaboratory/inference.py`
   - kept remote/g4f calls subprocess-friendly and made g4f/tiktoken imports lazy;
   - bounded streaming response assembly now supports early stop after a completed ` ```python ` fence;
   - token counting now skips automatically for very large texts via `AGENTLAB_TOKEN_COUNT_MAX_CHARS`.

2. `AgentLaboratory/agents.py`
   - large long-lived artifacts (`report`, `plan`, `dataset_code`, `results_code`, `exp_results`, etc.) are spilled to disk once they exceed `AGENTLAB_ARTIFACT_SPILL_CHARS`;
   - only a compact preview stays in memory/state JSON, while attribute access transparently reloads full text from disk.

3. `AgentLaboratory/ai_lab_repo.py`
   - removed `from common_imports import *` from the hot path;
   - made `MLESolver` and `PyPDF2` lazy imports;
   - AgentRxiv paper text and summaries are cached on disk instead of retained in RAM.

4. `AgentLaboratory/common_imports.py`
   - reduced to a minimal low-RAM stdlib/optional-lightweight set for orchestrator code.

5. `pipeline_cli.py`
   - codegen launches now inject a conservative low-RAM environment for remote/g4f runs:
     - `AGENTLAB_REMOTE_SUBPROCESS=1`
     - `AGENTLAB_DISABLE_TOKEN_COUNT=1`
     - `AGENTLAB_MAX_RESPONSE_CHARS=40000`
     - `AGENTLAB_G4F_STOP_AT_PYTHON_FENCE=1`
     - `AGENTLAB_ARTIFACT_SPILL_CHARS=8000`
     - `AGENTLAB_HEAVY_IMPORTS=0`
     - `MALLOC_ARENA_MAX=2`

## Main practical effect

These changes target the highest-probability RAM spikes reported in the uploaded audits:
- lower baseline RSS before any LLM call;
- less retained memory across long fixer loops;
- bounded response size for g4f/codegen;
- less duplication of large strings in agent state.
