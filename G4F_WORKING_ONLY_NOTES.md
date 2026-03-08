# G4F working-only list update

## Что изменено

Цель: команда `pipeline_cli.py check-g4f-models` должна выводить в режиме списка только те g4f-модели, которые реально ответили на тестовый запрос (по умолчанию `ping`).

### Изменения

- `--list-only` больше **не** означает «просто показать все найденные кандидаты».
  Теперь этот режим:
  - запускает probe для каждой candidate model;
  - печатает **только** модели, которые вернули непустой ответ;
  - по умолчанию проверяет запросом `ping`.

- добавлен новый флаг `--discover-only` для старого поведения:
  - показывает все обнаруженные candidate models;
  - не делает probe.

- JSON-вывод для `--list-only` теперь содержит:
  - `working_models`
  - `working_count`
  - `checked_count`
  - `probe_prompt`
  - `results`

- default probe prompt изменён на `ping`.

## Примеры

```bash
# Только реально ответившие модели
python pipeline_cli.py check-g4f-models --list-only

# Подробная проверка с построчным статусом
python pipeline_cli.py check-g4f-models --timeout 12

# Все найденные кандидаты без probe
python pipeline_cli.py check-g4f-models --discover-only
```

## Изменённые файлы

- `pipeline_cli.py`
- `tests_test_g4f_model_check.py`
- `README.md`
- `docs/USAGE.md`
