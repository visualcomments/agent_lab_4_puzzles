# Megaminx self-improvement breakthrough scenarios

This repository now contains two explicit scenarios for self-improving prompt runs.
They address the failure mode observed in the prompt sweep archive: provider/codegen
failures were silently converted into the baseline solver, then scored as if they
were real candidates.

## Basic scenario: honest self-improvement loop

Goal: stop treating baseline fallback as a valid candidate.

Implemented changes:

- `pipeline_cli.py` passes `--strict` to `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
  whenever `--keep-improving`, prompt self-improvement, or novelty guards are enabled.
- `run_perm_pipeline.py` now accepts `--attempt-archive-dir` and writes per-round
  raw prompt/response/extracted-code artifacts plus `strict_failure_report.json`.
- `pipeline_cli.py` rejects byte-identical solver candidates before validator/scoring,
  with `failure_kind=no_novelty_identical_solver_pre_score`.
- The outer loop stops early after 3 consecutive codegen/no-novelty failures before
  the first live non-identical candidate. Override with:

```bash
SELF_IMPROVE_MAX_CODEGEN_FAILURES_BEFORE_LIVE=5 python pipeline_cli.py run ...
```

Recommended baseline command:

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --keep-improving \
  --self-improve-prompts \
  --improvement-rounds 30 \
  --reject-identical-candidates \
  --prompt-variant failure_aware_self_improvement
```

Expected behavior:

- credentials/rate-limit/timeouts become failed rounds, not baseline submissions;
- identical solver files are not scored;
- per-round artifacts appear under `<out_stem>_candidate_archive/round_XXXX/`;
- `run_log.json`/history clearly separates provider failure, no novelty, no per-row
  improvement, and real score regression.

## Advanced scenario: evaluator-driven lane evolution

Goal: make progress by generating small, auditable, exact-valid algorithmic deltas.

Implemented prompt/runtime primitives:

- `competitions/cayley-py-megaminx/self_improvement_scenarios.py`
  defines:
  - `BasicScenarioPolicy`;
  - `CandidateManifest`;
  - `validate_candidate_manifest`;
  - `build_hard_row_micro_pack`;
  - `append_candidate_archive_entry`;
  - `rejection_bucket_from_history_entry`.
- `prompt_self_improver.py` now injects:
  - diff-first / patch discipline;
  - candidate manifest contract;
  - five advanced lanes: `lane_patch`, `lane_fresh`, `lane_params`,
    `lane_hard_row_micro`, `lane_portfolio`.
- `row_profile_memory.py` now builds `hard_row_micro_pack` with compact row tasks,
  path prefixes/suffixes, repeated motifs and rewrite windows.

Advanced candidate contract:

```python
CANDIDATE_MANIFEST = {
    "lane_id": "lane_hard_row_micro",
    "changed_mechanism": "added exact replay guarded local window rewrite bank",
    "target_rows": [123, 456],
    "expected_improved_rows": 1,
    "fallback_policy": "rollback per row if candidate_len >= baseline_len or exact replay fails",
    "novelty_claim": "changes optimizer core; does not replay optimized_submission lookup wrapper",
}
```

Recommended advanced operating pattern:

1. Start with the basic scenario enabled.
2. Run with `--prompt-variant failure_aware_self_improvement` and `--self-improve-prompts`.
3. Inspect `<out_stem>_candidate_archive` and the prompt-round metadata.
4. Promote only candidates with:
   - non-identical solver digest;
   - non-identical submission digest;
   - `improved_rows > 0`;
   - zero or explicitly rolled-back regressions;
   - a meaningful `CANDIDATE_MANIFEST`.
5. Use `hard_row_micro_pack` to force the next prompt round to shorten one concrete
   hard row instead of asking for broad solver rewrites.

## Why this should change outcomes

The previous sweep repeatedly produced the same baseline file. These changes make
that impossible to misinterpret as progress. The outer loop now has enough structure
to distinguish:

- provider/backend failure;
- no Python returned;
- baseline fallback/no novelty;
- valid but no per-row improvement;
- score regression;
- partial row-level win;
- real accepted improvement.

That separation is the prerequisite for useful self-improving prompts.
