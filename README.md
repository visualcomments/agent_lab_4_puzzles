# agent_lab_4_puzzles

This repository is a template for solving **Kaggle CayleyPy**-style puzzle competitions using:

- **AgentLaboratory** (perm_pipeline) to generate / repair a `solve_module.py` with an LLM
- **llm-puzzles** utilities to format a Kaggle **submission.csv**

It now supports **multiple competitions** via a single CLI entrypoint (`pipeline_cli.py`).

---

## What changed (high level)

- Each supported competition now has its own:
  - baseline `solve_module.py`
  - validator `validate_solve_output.py`
  - (optionally) prompt bundle for AgentLaboratory

  under `competitions/<competition>/...`

- `pipeline_cli.py` selects the correct validator / baseline / prompt bundle based on **`--competition`**.

- `llm-puzzles` formatting registry (`llm-puzzles/src/comp_registry.py`) now includes:
  - `cayleypy-rapapport-m2` (special submission schema)
  - `CayleyPy-pancake` (and lowercase alias)
  - other CayleyPy competitions (default `initial_state_id,path` schema)

---

## Install

```bash
pip install -r requirements-min.txt
# or, with Kaggle + g4f
pip install -r requirements-full.txt
```

> Notes:
> - `AgentLaboratory/perm_pipeline` uses `g4f` for model access.
> - If you do **not** want to use an LLM, you can run everything with `--no-llm`.

---

## List available pipelines

```bash
python pipeline_cli.py list-pipelines
```

---


## Full usage guide

For a complete, detailed tutorial (all commands and flags, schema checks, run logs, Kaggle submit), see:

- `docs/USAGE.md`

Quickly inspect a pipeline (paths, bundled files, expected submission columns):

```bash
python pipeline_cli.py show-pipeline --competition <slug>
```

### Проверка g4f-моделей

```bash
# Проверить модели асинхронным ping-probe через g4f AsyncClient
python pipeline_cli.py check-g4f-models --list-only

# То же, но с подробным статусом и ограничением конкуренции
python pipeline_cli.py check-g4f-models --timeout 12 --concurrency 5

# Проверить свой белый список
python pipeline_cli.py check-g4f-models --models gpt-4o-mini,command-r,aria

# При необходимости отдельно показать все найденные кандидаты без probe
python pipeline_cli.py check-g4f-models --discover-only
```

---

## End-to-end pipeline

### LRX (Discover / OEIS)

These LRX competitions are bundled with `test.csv` and `sample_submission.csv` under each competition folder,
so you can omit `--puzzles` to use the bundled `competitions/<slug>/data/test.csv`.

```bash
python pipeline_cli.py run \
  --competition lrx-discover-math-gods-algorithm \
  --output competitions/lrx-discover-math-gods-algorithm/submissions/submission.csv \
  --no-llm

python pipeline_cli.py run \
  --competition lrx-oeis-a-186783-brainstorm-math-conjecture \
  --output competitions/lrx-oeis-a-186783-brainstorm-math-conjecture/submissions/submission.csv \
  --no-llm
```


### 1) Build a submission with the baseline solver (no LLM)

#### CayleyPy RapaportM2

```bash
python pipeline_cli.py run \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm
```

#### CayleyPy Pancake

```bash
python pipeline_cli.py run \
  --competition CayleyPy-pancake \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm
```

### 2) Generate a solver with AgentLaboratory (custom prompt + per-competition validator)

#### RapaportM2 (custom prompt bundle included)

```bash
python pipeline_cli.py generate-solver \
  --competition cayleypy-rapapport-m2 \
  --out generated/solve_rapapport_m2.py \
  --models gpt-4o-mini \
  --max-iters 8
```

#### Pancake (custom prompt bundle included)

```bash
python pipeline_cli.py generate-solver \
  --competition CayleyPy-pancake \
  --out generated/solve_pancake.py \
  --models gpt-4o-mini \
  --max-iters 8
```

### 3) Build submission from an existing solver

```bash
python pipeline_cli.py build-submission \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --solver generated/solve_rapapport_m2.py \
  --output submission.csv
```

---

## Competition assets layout

```
competitions/
  lrx-sort/
    solve_module.py
    validate_solve_output.py
    prompts/
  cayleypy-rapapport-m2/
    solve_module.py
    validate_solve_output.py
    prompts/
      user_prompt.txt
      custom_prompts.json
  cayleypy-pancake/
    solve_module.py
    validate_solve_output.py
    prompts/
      user_prompt.txt
      custom_prompts.json
  ...
```

---

## Offline self-test

Runs `compileall`, validates several baseline solvers, and builds tiny dummy submissions.

```bash
python pipeline_cli.py selftest
```

---

## Kaggle submit (optional)

Before a live submit, you can run an explicit preflight that checks the installed Kaggle client version and whether the current account can access competition submissions for that slug:

```bash
python pipeline_cli.py kaggle-preflight   --competition cayleypy-rapapport-m2   --kaggle-json /path/to/kaggle.json   --submit-via auto
```

Notes:

- The repository now rejects submit attempts when the installed Kaggle client is older than `1.5.0`.
- The preflight also probes competition submission access, so "rules not accepted / account not joined" is reported before upload.
- The preferred CLI form stays aligned with the official docs: `kaggle competitions submit <competition> -f ... -m ...`.

If you have the Kaggle API installed and configured (either via `~/.kaggle/kaggle.json` or by passing `--kaggle-json`), you can submit directly from `run`:

```bash
python pipeline_cli.py run   --competition cayleypy-rapapport-m2   --puzzles /path/to/test.csv   --output submission.csv   --no-llm   --submit   --message "baseline"

# Or pass kaggle.json explicitly
python pipeline_cli.py run   --competition lrx-discover-math-gods-algorithm   --output submission.csv   --no-llm   --submit   --message "baseline"   --kaggle-json /path/to/kaggle.json   --submit-via api
```

---

## Notes about validators

- Every pipeline has its own `competitions/<pipeline>/validate_solve_output.py`.
- For **RapaportM2** and **Pancake**, validators fully simulate the moves.
- For complex twisty-puzzle competitions in the CayleyPy series (cube/minx/etc.), the included baseline returns `UNSOLVED` and the validator is a lightweight smoke-check (format + runtime). Use AgentLaboratory prompts + competition files (`graphs_info.json`, etc.) to implement full solvers.


---

## Progress logging

- Submission building shows a lightweight progress bar by default.
- Disable it with `--no-progress`.


## Demo: bubble sort sanity-check (g4f vs local)

В репозитории есть мини-соревнование **demo-bubble-sort** — оно специально сделано, чтобы быстро проверить:

- baseline (без LLM) генерирует корректный `submission.csv`
- AgentLaboratory может сгенерировать решатель (и валидатор его примет)
- одинаково работает режим `g4f:*` и `local:*` (локальная модель может использовать GPU)

### Baseline (без LLM)

```bash
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv --no-llm
```

### g4f

```bash
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv --models gpt-4 --max-iters 3
```

### Local (GPU)

```bash
AGENTLAB_DEVICE=cuda AGENTLAB_USE_GPU=1 \
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv \
  --models "local:Qwen/Qwen2.5-0.5B-Instruct" --max-iters 3
```
