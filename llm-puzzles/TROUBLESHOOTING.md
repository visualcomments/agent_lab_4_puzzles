# TROUBLESHOOTING

## LLM «молчит» или пустые ответы
- Укажите белый список: `export G4F_MODELS="gpt-4o-mini,claude-3-haiku"`.
- Включите быстрый self-check: `export G4F_SELFCHECK=1`.
- Увеличены таймауты/ретраи в коде; при необходимости поднимите их в CONFIG.
- Для диагностики выключите `--strict-llm` и включите fallback на sample (`--prefer-sample`).

## Ошибки импортов
- Установите зависимости: `pip install g4f requests`.

## Ошибки формата CSV
- Для CayleyPy: `initial_state_id,path`. Используйте `--prefer-sample` для baseline.
