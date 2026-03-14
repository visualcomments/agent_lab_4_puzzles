# LLM (g4f) — кратко

Установите зависимости (минимум `g4f` и `requests`):
```bash
pip install -r requirements.txt
pip install g4f requests
```

Переменные окружения:
- `G4F_MODELS="gpt-4o-mini,claude-3-haiku"` — вручной белый список и порядок.
- `G4F_SELFCHECK=1` / `G4F_SELFCHECK_TOP=2` / `G4F_SELFCHECK_PROMPT="ok?"` — быстрая проверка отклика 1–2 моделей.
- `STRICT_LLM=1` — только LLM (без fallback на sample).
- `PREFER_SAMPLE=1` — всегда брать пути из `sample_submission.csv`.
- `LLM_OFFLINE=1` — отключить реальные вызовы LLM (dry run).

Примеры запуска см. в README.md.
