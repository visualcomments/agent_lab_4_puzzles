# Pipeline CLI — подробный туториал

Этот документ описывает **все команды** `pipeline_cli.py` и **все ключевые опции**, а также типовые сценарии: сбор сабмита, генерация солвера, проверка схемы `submission.csv`, запись `run_log.json`, и отправка на Kaggle.

> Коротко: `--competition <slug>` выбирает пайплайн (соревнование). Дальше CLI умеет сгенерировать/проверить солвер, собрать `submission.csv` и (опционально) сделать сабмит на Kaggle.

---

## 0) Быстрый старт

### 0.1 Проверить, какие пайплайны доступны

```bash
python pipeline_cli.py list-pipelines
```

### 0.2 Посмотреть подробности конкретного пайплайна

```bash
python pipeline_cli.py show-pipeline --competition lrx-discover-math-gods-algorithm
```

Можно в JSON:

```bash
python pipeline_cli.py show-pipeline --competition lrx-discover-math-gods-algorithm --json
```

### 0.3 Собрать сабмит baseline (без LLM)

```bash
python pipeline_cli.py run \
  --competition lrx-discover-math-gods-algorithm \
  --output competitions/lrx-discover-math-gods-algorithm/submissions/submission.csv \
  --no-llm
```

Если `--puzzles` не указан, CLI попробует взять bundled `competitions/<slug>/data/test.csv`.

---

## 1) Как устроен репозиторий

Типичная структура пайплайна:

```text
competitions/<slug>/
  solve_module.py                 # baseline solver
  validate_solve_output.py         # валидатор солвера (smoke-check)
  prompts/                         # промпты для AgentLaboratory (если есть)
  data/
    test.csv                       # test из архива соревнования (опционально)
    sample_submission.csv          # sample_submission из архива (опционально)
  submissions/
    sample_submission.csv          # копия шаблона для удобства
    README.md
```

Параметры пайплайна описаны в `pipeline_registry.py` (PipelineSpec):

- `key` — ключ пайплайна (обычно равен slug соревнования)
- `competition` — slug для Kaggle submit (`kaggle competitions submit -c ...`)
- `format_slug` — формат сабмита для `llm-puzzles/src/comp_registry.py`
- `baseline_solver`, `validator`, `prompt_file`, `custom_prompts_file`
- `state_columns` — какие колонки в `test.csv` пробовать как “состояние”
- `smoke_vector` — маленький пример для быстрой проверки `solve()`

---

## 2) Установка зависимостей

Минимальная установка (без Kaggle submit и без LLM-генерации):

```bash
pip install -r requirements-min.txt
```

Полная установка (Kaggle submit + AgentLaboratory/g4f):

```bash
pip install -r requirements-full.txt
```

---

## 3) Команды `pipeline_cli.py`

Посмотреть список команд:

```bash
python pipeline_cli.py --help
```

### Список команд

- `list-pipelines` — список пайплайнов
- `show-pipeline` — подробности пайплайна (пути, bundled файлы, expected schema)
- `generate-solver` — генерация/починка солвера (LLM или baseline)
- `validate-solver` — валидация солвера
- `build-submission` — сбор `submission.csv`
- `run` — end-to-end (generate → validate → build → schema-check → submit)
- `selftest` — оффлайн smoke-tests

---

## 4) `list-pipelines`

Печатает все зарегистрированные пайплайны.

```bash
python pipeline_cli.py list-pipelines
```

Опций нет.

---

## 5) `show-pipeline`

Команда выводит подробную «паспортную карточку» пайплайна:

- ключ, Kaggle slug, format slug
- **ожидаемые колонки сабмита** (из `llm-puzzles/src/comp_registry.py`)
- пути к baseline/validator/prompts (+ exists)
- где лежат bundled `test.csv` и `sample_submission.csv` (+ header sample)
- список файлов в `competitions/<slug>/{data,submissions,prompts}`

### Использование

```bash
python pipeline_cli.py show-pipeline --competition <slug>
```

JSON:

```bash
python pipeline_cli.py show-pipeline --competition <slug> --json
```

Override format (для диагностики):

```bash
python pipeline_cli.py show-pipeline --competition <slug> --format format/moves-dot
```

**Опции**:

- `--competition <slug>` (обяз.)
- `--format <slug>` (опц.)
- `--json` (флаг)

---

## 6) `generate-solver`

Генерирует (или «чинит») солвер `solve_module.py`.

### 6.1 Генерация через AgentLaboratory (LLM)

```bash
python pipeline_cli.py generate-solver \
  --competition cayleypy-rapapport-m2 \
  --out generated/solve_rapapport_m2.py \
  --models gpt-4o-mini \
  --max-iters 8
```

### 6.2 Без LLM (просто копировать baseline)

```bash
python pipeline_cli.py generate-solver \
  --competition lrx-discover-math-gods-algorithm \
  --out generated/solve_lrx.py \
  --no-llm
```

### Опции

- `--competition <slug>` (обяз.)
- `--out <path>` (обяз.)
- `--prompt-file <path>` (опц.) — переопределяет prompt
- `--custom-prompts <path>` (опц.) — переопределяет custom prompts
- `--models <comma-separated>` (опц.) — список моделей для AgentLab
- `--max-iters <int>` (опц.) — число итераций
- `--allow-baseline` (флаг) — хук/совместимость
- `--no-llm` (флаг) — копировать baseline

После генерации команда делает **smoke-валидацию** через `validate_solve_output.py`.

---

## 7) `validate-solver`

Проверяет, что солвер:

- импортируется
- имеет `solve(vec)`
- не падает на smoke-векторе
- возвращает результат в допустимом формате

### Использование

```bash
python pipeline_cli.py validate-solver \
  --competition lrx-discover-math-gods-algorithm \
  --solver generated/solve_lrx.py
```

С кастомным вектором:

```bash
python pipeline_cli.py validate-solver \
  --competition cayleypy-rapapport-m2 \
  --solver generated/solve_rapapport_m2.py \
  --vector "[3,0,1,4,2]"
```

### Опции

- `--competition <slug>` (обяз.)
- `--solver <path>` (обяз.)
- `--vector "[... ]"` (опц.) — JSON-список

---

## 8) `build-submission`

Собирает `submission.csv` на основе:

- `test.csv` (или `--puzzles`)
- солвера `--solver`
- формата сабмита (`format_slug`)

### Пример (bundled test.csv)

```bash
python pipeline_cli.py build-submission \
  --competition lrx-discover-math-gods-algorithm \
  --solver competitions/lrx-discover-math-gods-algorithm/solve_module.py \
  --output competitions/lrx-discover-math-gods-algorithm/submissions/submission.csv
```

### Пример (с внешним puzzles/test.csv)

```bash
python pipeline_cli.py build-submission \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --solver generated/solve_rapapport_m2.py \
  --output submissions/submission.csv
```

### Опции (все)

- `--competition <slug>` (обяз.)
- `--puzzles <path>` (опц.) — если не указан, ищет bundled `competitions/<slug>/data/test.csv`
- `--solver <path>` (обяз.)
- `--output <path>` (обяз.)
- `--format <slug>` (опц.) — override format
- `--vector-col <column>` (опц.) — override колонки состояния (если автодетект не угадал)
- `--max-rows <int>` (опц.) — ограничить строки для дебага
- `--no-progress` (флаг) — выключить progress bar

#### Schema check

- `--schema-check` — сравнить `submission.csv` с bundled `sample_submission.csv`
- `--no-schema-check` — выключить проверку (даже если `--schema-check` задан)
- `--no-schema-check-ids` — не сравнивать множество id (только колонки + число строк)

#### Run log

- `--run-log <path>` — путь до `run_log.json` (по умолчанию `<output_dir>/run_log.json`)
- `--no-run-log` — отключить лог

---

## 9) `run` (end-to-end)

Делает все этапы одной командой:

1) generate solver (LLM или baseline)
2) validate solver
3) build submission
4) schema-check (авто перед submit, или вручную через `--schema-check`)
5) Kaggle submit (если `--submit`)
6) запись `run_log.json`

### 9.1 Без LLM, без submit

```bash
python pipeline_cli.py run \
  --competition lrx-oeis-a-186783-brainstorm-math-conjecture \
  --output competitions/lrx-oeis-a-186783-brainstorm-math-conjecture/submissions/submission.csv \
  --no-llm
```

### 9.2 С LLM-генерацией

```bash
python pipeline_cli.py run \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --output submissions/submission.csv \
  --models gpt-4o-mini \
  --max-iters 10
```

### 9.3 С submit на Kaggle

```bash
python pipeline_cli.py run \
  --competition lrx-discover-math-gods-algorithm \
  --output submissions/submission.csv \
  --no-llm \
  --submit --message "baseline" \
  --kaggle-json /path/to/kaggle.json \
  --submit-via api
```

> Перед `--submit` schema-check включается автоматически (если есть sample_submission).

### Опции `run` (суммарно)

- `--competition <slug>` (обяз.)
- `--puzzles <path>` (опц.)
- `--output <path>` (обяз.)

Генерация солвера:
- `--prompt-file <path>`
- `--custom-prompts <path>`
- `--models <list>`
- `--max-iters <int>`
- `--allow-baseline`
- `--no-llm`

Сбор сабмита:
- `--format <slug>`
- `--vector-col <column>`
- `--max-rows <int>`
- `--no-progress`

Schema check:
- `--schema-check`
- `--no-schema-check`
- `--no-schema-check-ids`

Kaggle submit:
- `--submit`
- `--message "text"` (обязательно с `--submit`)
- `--kaggle-json <path>`
- `--kaggle-config-dir <dir>`
- `--submit-via auto|api|cli`
- `--submit-competition <slug>`

Run log:
- `--run-log <path>`
- `--no-run-log`

---

## 10) `run_log.json`: что внутри

Лог пишется по умолчанию в:

```text
<директория output>/run_log.json
```

Это **JSON-массив** записей (каждый запуск добавляет новую запись, не затирает).

Запись включает:

- `status`: `ok` / `error`
- `stages`: тайминги этапов (`seconds`)
- `files`: размеры файлов + строки/колонки для puzzles/submission/sample
- `schema`: статистика schema-check (если был)
- `error`: `type`, `message`, `stacktrace` (если упало)

Полезно для сравнения подходов/итераций.

---

## 11) Kaggle submit: практические заметки

1) Убедитесь, что вы приняли правила соревнования на Kaggle (частая причина ошибок submit).
2) Установите `kaggle` пакет:

```bash
pip install kaggle
```

3) Рекомендуемый путь: передавать ключ через `--kaggle-json`, чтобы не класть его в `~/.kaggle`.

---

## 12) Частые проблемы и быстрые решения

### 12.1 Schema-check падает на row count

Если вы используете `--max-rows`, то `submission.csv` будет короче sample — это ожидаемо.

Решения:
- убрать `--schema-check`
- или добавить `--no-schema-check`

### 12.2 Не угадалась колонка состояния

Если `test.csv` нестандартный, задайте явно:

```bash
--vector-col permutation
```

### 12.3 Хотите быстро понять, что CLI “думает” про пайплайн

Используйте:

```bash
python pipeline_cli.py show-pipeline --competition <slug>
```

Это покажет фактические пути, bundled файлы и expected columns.

---

## 13) Selftest

Оффлайн-проверка, что базовые пайплайны и сборка сабмита не сломаны:

```bash
python pipeline_cli.py selftest
```

Опций нет.

---

## 14) Справка по конкретной команде

У каждой команды есть собственный help:

```bash
python pipeline_cli.py run --help
python pipeline_cli.py build-submission --help
python pipeline_cli.py show-pipeline --help
```

