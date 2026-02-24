# Unified AgentLaboratory + g4f + llm-puzzles Kaggle Pipeline

Этот репозиторий объединяет три вещи в **единый CLI-пайплайн**:

1) **AgentLaboratory** — мульти-агентный цикл (planner → coder → fixer), который генерирует/чинит `solve_module.py` под заданный пользовательский промпт.
2) **g4f (GPT4Free)** — LLM backend по умолчанию для агентов (можно указывать несколько моделей и использовать их как fallback по очереди).
3) **llm-puzzles** — универсальный адаптер для генерации `submission.csv` и (опционально) автоматического сабмита на Kaggle.

Цель: вы даёте **пользовательский промпт** (например, про сортировку вектора ходами `L/R/X`), а пайплайн:

- генерирует solver,
- прогоняет локальную валидацию,
- собирает `submission.csv` из `puzzles.csv` соревнования,
- (опционально) сабмитит на Kaggle и пытается прочитать score.

---

## Быстрый старт (без интернета)

1) Проверка, что всё компилируется и baseline solver валиден:

```bash
python pipeline_cli.py selftest
```

Это:
- компилирует весь репозиторий,
- проверяет `solve_module.py` на случайных тестах,
- создаёт маленький `puzzles.csv` и собирает `submission.csv`.

---

## Установка

### Вариант A — использовать **встроенный** g4f
В репозитории есть папка `./gpt4free/` (vendor checkout). В этом случае `import g4f` будет работать автоматически.

### Вариант B — установить g4f через pip

```bash
pip install g4f
```

### Kaggle (опционально)
Чтобы сабмитить решения и читать результаты, нужен Kaggle API:

```bash
pip install kaggle
```

Дальше настройте `kaggle.json` (обычно `~/.kaggle/kaggle.json` на Linux/macOS или `C:\Users\<you>\.kaggle\kaggle.json` на Windows).

---

## Структура репозитория

- `pipeline_cli.py` — **единая CLI-точка входа**.
- `solve_module.py` — baseline solver (LRX сортировка за полиномиальное время).
- `validate_solve_output.py` — валидатор: симулирует ходы и проверяет `moves → sorted_array`.
- `AgentLaboratory/` — исходный репозиторий AgentLaboratory (пропатчен под g4f).
  - `AgentLaboratory/perm_pipeline/run_perm_pipeline.py` — генератор/чинитель solver.
  - `AgentLaboratory/perm_pipeline/default_prompts.json` — дефолтные системные промпты для planner/coder/fixer.
- `llm-puzzles/` — исходный репозиторий llm-puzzles + добавленные entrypoints.
  - `llm-puzzles/examples/agentlab_sort/solver.py` — адаптер «строка CSV → moves».

---

## Команды CLI

Посмотреть все команды:

```bash
python pipeline_cli.py -h
```

### 1) Сгенерировать solver

```bash
python pipeline_cli.py generate-solver \
  --prompt-file prompts/example_user_prompt.txt \
  --models "gpt-4o-mini,command-r,aria" \
  --out generated/solve_module.py
```

Опции:
- `--models` — CSV списка g4f моделей. Пайплайн будет пробовать их по очереди.
- `--custom-prompts` — JSON override системных промптов (см. ниже).
- `--no-llm` — **не вызывать LLM**, а просто сохранить baseline solver.

### 2) Провалидировать solver на одном векторе

```bash
python pipeline_cli.py validate-solver --solver generated/solve_module.py --vector "[3,1,2,5,4]"
```

### 3) Собрать submission.csv из puzzles.csv

```bash
python pipeline_cli.py build-submission \
  --puzzles /path/to/puzzles.csv \
  --out submission.csv \
  --format format/moves-dot \
  --solver generated/solve_module.py \
  --vector-col vector
```

Опции:
- `--format` — формат вывода для llm-puzzles:
  - `format/moves-dot` (по умолчанию) — moves соединяются точками
  - `format/moves-space` — moves соединяются пробелами
  - `format/initial_state_id+path` — колонки `initial_state_id,path`
- `--vector-col` — имя колонки, где лежит JSON-вектор (иначе пытаемся угадать автоматически).
- `--add-config` — добавить кастомный формат (JSON) в runtime.

### 4) End-to-end run: prompt → solver → validate → submission → (submit)

```bash
python pipeline_cli.py run \
  --competition <kaggle-slug> \
  --puzzles /path/to/puzzles.csv \
  --format format/moves-dot \
  --prompt-file prompts/example_user_prompt.txt \
  --models "gpt-4o-mini,command-r,aria" \
  --out submission.csv
```

Добавить сабмит:

```bash
python pipeline_cli.py run \
  --competition <kaggle-slug> \
  --puzzles /path/to/puzzles.csv \
  --format format/moves-dot \
  --prompt-file prompts/example_user_prompt.txt \
  --submit --message "auto" --print-score
```

Офлайн режим (без LLM):

```bash
python pipeline_cli.py run \
  --competition <kaggle-slug> \
  --puzzles /path/to/puzzles.csv \
  --format format/moves-dot \
  --no-llm
```

---

## Как добавлять/модифицировать системные промпты (planner/coder/fixer)

Дефолтные промпты лежат тут:

- `AgentLaboratory/perm_pipeline/default_prompts.json`

Чтобы переопределить часть промптов — создайте свой JSON:

```json
{
  "planner": "...",
  "coder": "...",
  "fixer": "..."
}
```

И передайте:

```bash
python pipeline_cli.py run ... --custom-prompts prompts/custom_prompts.json
```

Рекомендация по стилю промптов:
- **Planner**: запрещать BFS/DFS/перебор и требовать конструктивную схему.
- **Coder**: требовать полный self-contained файл с CLI и строгими ограничениями операций.
- **Fixer**: давать отчёт валидатора + требовать вернуть полный исправленный файл.

---

## Как выбирать g4f модели

- Через `--models "modelA,modelB,modelC"`.
- Порядок важен: пайплайн пробует по очереди.

Также можно задать env:

```bash
export G4F_MODELS="modelA,modelB"
```

---

## Как добавить кастомный формат submission

Если ваш Kaggle competition требует нестандартные колонки/формат moves:

1) Создайте JSON, например:

```json
{
  "slug": "my-format",
  "submission_headers": ["id","moves"],
  "header_keys": ["id","moves"],
  "puzzles_id_field": "id",
  "moves_key": "moves",
  "move_joiner": "."
}
```

2) Запускайте с:

```bash
python pipeline_cli.py build-submission --add-config my_format.json --format my-format ...
```

---

## Troubleshooting

### 1) Solver не проходит валидацию
- Запустите `python pipeline_cli.py validate-solver ...` и посмотрите stderr.
- Увеличьте `--max-iters` (если используете LLM).
- Убедитесь, что solver **строго** записывает ходы в список и возвращает JSON-совместимый вывод.

### 2) g4f не работает
- Проверьте, что доступна папка `./gpt4free/` или установлен `pip install g4f`.
- Некоторые провайдеры могут требовать cookies/токены (зависит от выбранного провайдера/модели).

### 3) Kaggle submit/score
- Установите `pip install kaggle`.
- Настройте `kaggle.json` и права доступа.
- Учтите: некоторые соревнования не принимают файл-сабмиты и требуют notebook-only.

---

## Лицензии и происхождение кода

Внутри папок `AgentLaboratory/`, `llm-puzzles/`, `gpt4free/` находятся исходные репозитории со своими лицензиями. Этот репозиторий добавляет glue-код и CLI.
