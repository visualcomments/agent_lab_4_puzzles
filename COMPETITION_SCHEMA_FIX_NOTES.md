# Competition submission schema fix

## Problem
For `cayleypy-pancake`, the pipeline generated a submission using the wrong schema.
The local repo expected `initial_state_id,path`, but the uploaded competition bundle's
`sample_submission.csv` requires:

- `id`
- `permutation`
- `solution`

That mismatch leads to Kaggle-style validation errors such as:
`ID column id not found in submission`.

## What was changed

- Added generic format `format/id+permutation+solution` in `llm-puzzles/src/comp_registry.py`.
- Switched `cayleypy-pancake`, `cayleypy-glushkov`, and `cayleypy-rapapport-m2` to that format.
- Updated `pipeline_registry.py` so these competitions default to the correct format.
- Added dynamic sample-schema inference in `pipeline_cli.py`.
- Made `pipeline_cli.py` prefer `sample_submission.csv` from the competition ZIP when available.
- Synced bundled `sample_submission.csv` files for the affected competitions with the uploaded ZIPs.
- Updated submission-schema tests.

## Verified

Command executed successfully in baseline mode:

```bash
python pipeline_cli.py run \
  --competition cayleypy-pancake \
  --output submissions/submission.csv \
  --no-llm \
  --schema-check \
  --max-iters 100
```

Produced header:

```csv
id,permutation,solution
```
