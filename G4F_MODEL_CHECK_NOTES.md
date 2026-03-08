# G4F model-check CLI

Added a dedicated CLI command:

```bash
python pipeline_cli.py check-g4f-models --list-only
python pipeline_cli.py check-g4f-models --timeout 12
python pipeline_cli.py check-g4f-models --models gpt-4o-mini,command-r,aria
```

## What it does

- autodiscovers candidate text-capable g4f models from the bundled `gpt4free/g4f/models.py` registry;
- can optionally fetch the candidate list from a running g4f backend API via `/backend-api/v2/models`;
- probes each candidate through the repository's own `query_model_stable(...)` path;
- prints the final working subset and can emit JSON.

## Files changed

- `pipeline_cli.py`
- `README.md`
- `docs/USAGE.md`
- `tests_test_g4f_model_check.py`

## Notes

- Discovery is fully local and works with the bundled `gpt4free` checkout.
- Live probing depends on external provider availability / credentials, so timeouts are possible; the command now reports them cleanly instead of crashing the whole workflow.
