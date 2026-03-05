# demo-bubble-sort

Мини-пайплайн для проверки, что:

- baseline работает без LLM
- генерация решателя через AgentLaboratory работает
- `g4f:*` и `local:*` режимы дают валидный `submission.csv`

## Быстрый прогон (baseline)

```bash
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv --no-llm
```

## Прогон с g4f

```bash
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv --models gpt-4 --max-iters 3
```

## Прогон с локальной моделью (GPU)

```bash
AGENTLAB_DEVICE=cuda AGENTLAB_USE_GPU=1 \
python pipeline_cli.py run --competition demo-bubble-sort --output submissions/demo-bubble-sort.csv \
  --models "local:Qwen/Qwen2.5-0.5B-Instruct" --max-iters 3
```
