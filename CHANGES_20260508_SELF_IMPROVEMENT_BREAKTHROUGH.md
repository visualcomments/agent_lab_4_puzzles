# 2026-05-08 self-improvement breakthrough hardening

Added basic and advanced scenarios for Megaminx self-improving prompt runs.

## Basic scenario

- `pipeline_cli.py` now runs AgentLaboratory in strict mode during improvement loops.
- Identical solver candidates are rejected before scoring.
- Consecutive provider/no-novelty failures before the first live candidate stop early.
- Per-round codegen artifacts are passed to `run_perm_pipeline.py` via `--attempt-archive-dir`.
- `run_perm_pipeline.py` writes strict failure reports and raw generation artifacts.

## Advanced scenario

- Added `competitions/cayley-py-megaminx/self_improvement_scenarios.py`.
- `prompt_self_improver.py` now injects diff-first, manifest, and evaluator-driven lane contracts.
- `row_profile_memory.py` now emits hard-row micro packs with path motifs and rewrite windows.
- Added `competitions/cayley-py-megaminx/SELF_IMPROVEMENT_BREAKTHROUGH_SCENARIOS.md`.
