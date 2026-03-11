# Changes made on 2026-03-11

## Main fixes

1. **Precise Python extraction for coder/fixer outputs**
   - `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
   - Added multi-candidate fenced-block extraction.
   - Added scoring so blocks containing `solve(...)` and valid Python are preferred.
   - Added raw-code fallback extraction for unfenced Python replies.

2. **Automatic cleanup of extracted Python**
   - Strips Python `# comments` and docstrings from extracted LLM code before compile/validation.
   - Prevents trailing markdown explanations from leaking into saved solver files.

3. **Earlier contract validation**
   - Added `validate_solver_contract()`.
   - Fails fast when generated code is missing `solve(...)` or has the wrong callable shape.
   - Provides the fixer loop with a clearer report before running the external validator.

4. **Stricter generation prompts**
   - Added shared strict output requirements to generation/fixer prompts.
   - Updated prompt JSON files so coder/fixer agents are reminded to return exactly one Python file, without explanations, and to avoid comments/docstrings unless necessary.

5. **Safer remote g4f defaults**
   - `pipeline_cli.py` now defaults `AGENTLAB_G4F_STOP_AT_PYTHON_FENCE=1` for remote codegen runs.
   - This reduces the chance that long post-code explanations pollute the captured response.

## Validation performed

- Python compile check for modified core files.
- Focused extraction demo on a noisy fixer response with comments + markdown explanation.
- Test suite run:

  ```bash
  pytest -q tests_test_agent_model_routing.py \
           tests_test_codegen_pipeline.py \
           tests_test_g4f_model_check.py \
           tests_test_kaggle_auth.py \
           tests_test_local_model_optimizations.py \
           tests_test_low_ram_optimizations.py \
           tests_test_rapapport_guardrails.py \
           tests_test_submission_pipeline.py \
           tests_test_cli_syntax.py
  ```

- Result: **54 passed**.

## Most relevant changed files

- `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
- `pipeline_cli.py`
- `AgentLaboratory/perm_pipeline/default_prompts.json`
- `prompts/custom_prompts_template.json`
- `competitions/*/prompts/*.json`
- `tests_test_codegen_pipeline.py`
- `tests_test_g4f_model_check.py`
