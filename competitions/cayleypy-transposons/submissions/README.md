# Submissions

Recommended location for generated submissions.

Build (offline):

    python ../../pipeline_cli.py run --competition cayleypy-transposons --output submissions/submission.csv --no-llm

Submit to Kaggle (optional):

    python ../../pipeline_cli.py run --competition cayleypy-transposons --output submissions/submission.csv --no-llm --submit --message baseline --kaggle-json /path/to/kaggle.json --submit-via api
