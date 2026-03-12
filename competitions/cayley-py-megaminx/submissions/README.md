# Submissions

Recommended location for generated submissions.

Build locally (baseline only, no LLM):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm

The bundled baseline can replay the official paths from `data/sample_submission.csv` for any state present in `data/test.csv`.

Submit to Kaggle (optional):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm --submit --message baseline --kaggle-json /path/to/kaggle.json --submit-via auto
