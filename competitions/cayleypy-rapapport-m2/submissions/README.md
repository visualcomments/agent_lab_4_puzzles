# Submissions

Recommended location for generated submissions.

Build locally (baseline only, no LLM):

    python ../../pipeline_cli.py run --competition cayleypy-rapapport-m2 --output submissions/submission.csv --no-llm

The pipeline now writes to `submission.csv.candidate`, validates that every move token is legal for RapaportM2 (`I`, `S`, `K`), optionally checks the schema against `sample_submission.csv`, and only then atomically replaces the final `submission.csv`.

Submit to Kaggle (optional):

    python ../../pipeline_cli.py run --competition cayleypy-rapapport-m2 --output submissions/submission.csv --no-llm --submit --message baseline --kaggle-json /path/to/kaggle.json --submit-via auto

If Kaggle returns `401 Unauthorized`, make sure the account has joined the competition, accepted the rules, and that the token in `kaggle.json` / `~/.kaggle/access_token` is valid.
