# Анализ файлов соревнований Kaggle и внесённые изменения (2026-03-11)

## Что было проверено

Я сопоставил содержимое `competition_files/*.zip` с локальными каталогами `competitions/*/data/`.
Для всех загруженных соревнований были отдельно сверены:

- `test.csv`
- `sample_submission.csv`
- соответствие формата сабмита настройкам пайплайна и `llm-puzzles/src/comp_registry.py`
- наличие в `test.csv` хотя бы одной колонки из `PipelineSpec.state_columns`

В репозитории добавлены тесты, которые теперь автоматически проверяют эти инварианты.

## Выявленные форматы сабмита

### Формат `id,permutation,solution`

- `cayleypy-pancake`
- `cayleypy-glushkov`
- `cayleypy-rapapport-m2`

### Формат `initial_state_id,path`

- `cayley-py-444-cube`
- `cayleypy-christophers-jewel`
- `cayleypy-reversals`
- `cayleypy-transposons`

### Формат `permutation,solution`

- `lrx-discover-math-gods-algorithm`

### Формат `n,solution`

- `lrx-oeis-a-186783-brainstorm-math-conjecture`

## Что улучшено в коде

### 1. Kaggle preflight перед live-submit

Добавлено две явные проверки перед отправкой:

- проверка версии установленного Kaggle client/package;
- проверка доступа к `competition submissions`, чтобы заранее ловить случаи:
  - правила соревнования не приняты;
  - аккаунт ещё не joined competition;
  - Kaggle CLI/API недоступен или слишком старый.

### 2. Новый CLI-командный режим

Добавлена команда:

```bash
python pipeline_cli.py kaggle-preflight --competition <slug> --submit-via auto
```

Она не загружает файл на Kaggle, а только проверяет готовность окружения к submit.

### 3. Логирование результата submit/preflight

В `run_log.json` теперь сохраняется сводка по Kaggle submit:

- какой канал использовался (`api` / `cli`),
- результат preflight,
- статус сабмита после best-effort polling.

### 4. Усилены регрессионные тесты

Добавлены проверки, что:

- архивы из `competition_files/` совпадают с checked-in `competitions/*/data/*`;
- формат `sample_submission.csv` согласован с объявленным config;
- `test.csv` реально содержит одну из ожидаемых колонок состояния.

## Что это даёт practically

- меньше ложных Kaggle submit failures;
- быстрее понятно, что проблема в версии клиента или в непринятых правилах, а не в самом `submission.csv`;
- меньше риска рассинхронизации между архивом соревнования и локальным пайплайном;
- лучше воспроизводимость для AgentLaboratory / llm-puzzles конвейера.

## Дополнительно исправлено по итогам реального запуска

Во время живой проверки обнаружился ещё один практический edge case:

- импорт `kaggle` / `KaggleApi` у новых версий клиента может сразу пытаться аутентифицироваться;
- из-за этого даже простая проверка версии через `import kaggle` могла падать раньше времени.

Это исправлено: версия Python-пакета теперь определяется через `importlib.metadata.version("kaggle")`, без раннего импорта `kaggle`, а сам `KaggleApi` импортируется лениво только внутри `ensure_auth()` после подготовки переменных окружения и временного `KAGGLE_CONFIG_DIR`.

Для этого добавлен отдельный regression-test.

## Результат живой проверки Kaggle в этом окружении

Была выполнена реальная проверка с пользовательским `kaggle.json`:

```bash
python pipeline_cli.py kaggle-preflight \
  --competition cayleypy-rapapport-m2 \
  --kaggle-json /mnt/data/kaggle.json \
  --submit-via auto --json
```

Preflight корректно отработал и показал, что код проверки живой, но сам доступ к `api.kaggle.com` из текущего контейнера упёрся в сетевое ограничение среды (`Failed to resolve api.kaggle.com`).

То есть логика preflight и submit-path теперь работает до реального сетевого вызова, а live-submit в данном контейнере блокируется не кодом репозитория, а отсутствием внешней DNS/HTTP-доступности до Kaggle.
