# G4F async model-check changes

Implemented an async probe path for `pipeline_cli.py check-g4f-models`.

## What changed

- Switched model probing from the synchronous `query_model_stable(...)` path to `g4f.client.AsyncClient`.
- Added `_load_g4f_async_client_class()` to import AsyncClient from the bundled `gpt4free` checkout or site-packages.
- Added `_probe_g4f_model_async(...)` for one async probe.
- Added `_probe_g4f_models_async(...)` for concurrent probing with `asyncio` and a concurrency limit.
- Added CLI option `--concurrency`.
- Kept `--list-only` semantics: it prints only models that returned a non-empty response to the probe prompt.
- Preserved `--discover-only` for candidate discovery without probing.
- Updated README and docs/USAGE.md.
- Expanded tests to cover AsyncClient-based probing and concurrency behavior.

## Example

```bash
python pipeline_cli.py check-g4f-models --list-only
python pipeline_cli.py check-g4f-models --timeout 12 --concurrency 5
python pipeline_cli.py check-g4f-models --discover-only
```
